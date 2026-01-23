# nrg_core/models_v2.py
"""
Architecture v2.0 Data Models - Two-tier analysis system with rubric-based scoring.

Design Decisions:
- Pydantic BaseModel for runtime validation and JSON serialization
- Finding requires at least one Quote to enforce evidence-backed claims
- RubricScore captures full audit trail (dimension, score, rationale, evidence, anchor)
- Separate PrimaryAnalysis and JudgeValidation to cleanly separate Tier 1 and Tier 2 outputs

Why These Models:
- Quote: Enforces verbatim evidence extraction with source location
- Finding: Prevents hallucinations by requiring supporting quotes
- RubricScore: Enables audit trail for compliance and explainability
- TwoTierAnalysisResult: Aggregates full pipeline output for downstream use
"""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone


class Quote(BaseModel):
    """
    Supporting evidence extracted verbatim from bill text.
    
    Why section is required: Enables judge to verify quote exists at stated location.
    Why page is optional: Not all bill formats include page numbers.
    """
    text: str = Field(..., min_length=10, description="Exact quote from bill")
    section: str = Field(..., description="Section reference (e.g., '2.1', '4.3a')")
    page: Optional[int] = Field(None, description="Page number if available")


class Finding(BaseModel):
    """
    Single analytical finding with mandatory supporting evidence.
    
    Design: Every claim must have a quote to prevent hallucination.
    The quotes list enforces this at the model level rather than relying on prompts.
    
    impact_estimate: Initial estimate from analyst (0-10), refined by judge scoring.
    """
    statement: str = Field(..., min_length=20, description="Clear, specific finding")
    quotes: List[Quote] = Field(..., min_length=1, description="Supporting quotes")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analyst confidence 0-1")
    impact_estimate: int = Field(..., ge=0, le=10, description="Initial impact estimate 0-10")
    
    @field_validator('quotes')
    @classmethod
    def validate_quotes(cls, v):
        """Enforce at least one supporting quote to prevent unsupported claims."""
        if not v:
            raise ValueError("Finding must have at least one quote")
        return v


class RubricScore(BaseModel):
    """
    Single rubric dimension score with full audit trail.
    
    Design: Captures everything needed to explain and audit the score:
    - dimension: Which aspect is being scored
    - score: Numeric value for ranking/filtering
    - rationale: Human-readable explanation
    - evidence: Supporting quotes from bill
    - rubric_anchor: Reference to scale definition used
    
    Phase 1 dimensions: legal_risk, financial_impact
    Phase 2 adds: operational_complexity, ambiguity
    """
    dimension: Literal["legal_risk", "financial_impact", "operational_disruption", "ambiguity_risk"] = Field(
        ..., description="Rubric dimension - 4 dimensions in Phase 2"
    )
    score: int = Field(..., ge=0, le=10, description="Score 0-10")
    rationale: str = Field(..., min_length=50, description="Detailed explanation")
    evidence: List[Quote] = Field(default_factory=list, description="Supporting evidence")
    rubric_anchor: str = Field(..., description="Reference to rubric scale")


class PrimaryAnalysis(BaseModel):
    """
    Output from Tier 1: Primary Analyst.
    
    requires_multi_sample: Flag for Tier 1.5 consistency check.
    Triggered when: impact >= 6 OR confidence < 0.7
    This ensures high-stakes or uncertain findings get extra validation.
    """
    bill_id: str
    findings: List[Finding] = Field(default_factory=list)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    requires_multi_sample: bool = Field(
        default=False,
        description="True if impact>=6 OR confidence<0.7"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JudgeValidation(BaseModel):
    """
    Output from Tier 2: Judge Model validation of a single finding.
    
    Validation checks:
    - quote_verified: Do quotes exist verbatim in bill text?
    - hallucination_detected: Does statement claim things not in quotes?
    - evidence_quality: How well do quotes support the statement?
    - ambiguity: How much interpretation is required?
    
    Design: Separate validation per finding enables granular filtering.
    """
    finding_id: int = Field(..., description="Index of finding being validated")
    quote_verified: bool = Field(..., description="All quotes exist verbatim in bill?")
    hallucination_detected: bool = Field(default=False, description="Claims not in bill text?")
    evidence_quality: float = Field(..., ge=0.0, le=1.0, description="Quality of evidence 0-1")
    ambiguity: float = Field(..., ge=0.0, le=1.0, description="Interpretive uncertainty 0-1")
    judge_confidence: float = Field(..., ge=0.0, le=1.0, description="Judge's confidence 0-1")


class ResearchInsight(BaseModel):
    """
    External research insight from deep research agent (Phase 4).
    
    Captures external context that enriches analysis:
    - claim: What the research found
    - source_url: Where it came from (for citation)
    - snippet: Relevant excerpt from source
    - relevance: How relevant to the finding ("high", "medium", "low")
    - checker_validated: Whether checker agent confirmed the claim
    - trust: Confidence in the research (0-1)
    
    Design: Separate model enables filtering by trust/validation status.
    """
    claim: str = Field(..., description="Research claim or insight")
    source_url: str = Field(..., description="Source URL for citation")
    snippet: str = Field(..., description="Relevant excerpt from source")
    relevance: str = Field(..., description="Relevance level: high, medium, low")
    checker_validated: bool = Field(default=False, description="Validated by checker agent")
    trust: float = Field(default=0.0, ge=0.0, le=1.0, description="Trust score 0-1")


class TwoTierAnalysisResult(BaseModel):
    """
    Complete output from two-tier analysis pipeline.
    
    Aggregates:
    - primary_analysis: Tier 1 findings
    - judge_validations: Tier 2 validation per finding
    - rubric_scores: Scored dimensions for validated findings
    - audit_trails: Compliance-ready documentation per finding (Phase 2)
    - multi_sample_agreement: Consistency score from Tier 1.5 (spec line 505)
    - second_model_reviewed: Whether fallback model was used
    - research_insights: External context from deep research (Phase 4)
    - cross_bill_references: Detected statutory references (Phase 4)
    - route: Whether STANDARD or ENHANCED path was used
    - cost_estimate: LLM API cost for tracking
    
    Design: Single object captures full analysis for storage/retrieval.
    """
    bill_id: str
    primary_analysis: PrimaryAnalysis
    judge_validations: List[JudgeValidation] = Field(default_factory=list)
    rubric_scores: List[RubricScore] = Field(default_factory=list)
    audit_trails: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Audit trails per finding for compliance (Phase 2)"
    )
    multi_sample_agreement: Optional[float] = Field(
        default=None,
        description="Consistency score from Tier 1.5 multi-sample check (0-1)"
    )
    second_model_reviewed: bool = Field(
        default=False,
        description="Whether fallback second model was consulted"
    )
    research_insights: List[ResearchInsight] = Field(
        default_factory=list,
        description="External research insights (Phase 4)"
    )
    cross_bill_references: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detected cross-bill references (Phase 4)"
    )
    route: Literal["STANDARD", "ENHANCED"] = Field(default="STANDARD")
    cost_estimate: float = Field(default=0.0, description="Total LLM cost in USD")

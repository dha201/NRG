# Phase 1: Core Two-Tier Analysis System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the core two-tier analysis system (Supervisor routing + Primary Analyst + Judge) from Architecture v2.0

**Architecture:** Code-based supervisor routes bills to primary analyst (GPT-4o), judge validates findings with rubric scoring (2 dimensions: legal_risk, financial_impact)

**Tech Stack:** Python 3.12, OpenAI GPT-4o, Pydantic for validation, pytest for testing

---

## Task 1: Create V2 Data Models

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/models_v2.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_models_v2.py`

**Step 1: Write failing test for Finding model**

Create test file:

```python
# tests/test_models_v2.py
import pytest
from nrg_core.models_v2 import Finding, Quote, RubricScore


def test_finding_creation():
    """Finding should store statement, quotes, and confidence."""
    quote = Quote(text="Section 2.1: Tax applies", section="2.1", page=5)
    finding = Finding(
        statement="Tax applies to facilities >50MW",
        quotes=[quote],
        confidence=0.85,
        impact_estimate=7
    )
    
    assert finding.statement == "Tax applies to facilities >50MW"
    assert len(finding.quotes) == 1
    assert finding.quotes[0].section == "2.1"
    assert finding.confidence == 0.85
    assert finding.impact_estimate == 7


def test_finding_requires_at_least_one_quote():
    """Finding must have supporting evidence."""
    with pytest.raises(ValueError, match="at least one quote"):
        Finding(
            statement="Tax applies",
            quotes=[],
            confidence=0.5,
            impact_estimate=3
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_v2.py::test_finding_creation -v`

Expected: `ModuleNotFoundError: No module named 'nrg_core.models_v2'`

**Step 3: Create V2 models with Pydantic validation**

Create model file:

```python
# nrg_core/models_v2.py
"""
Architecture v2.0 Data Models
Two-tier analysis system with rubric-based scoring
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class Quote(BaseModel):
    """Supporting evidence from bill text."""
    text: str = Field(..., min_length=10, description="Exact quote from bill")
    section: str = Field(..., description="Section reference (e.g., '2.1', '4.3a')")
    page: Optional[int] = Field(None, description="Page number if available")


class Finding(BaseModel):
    """Single analytical finding with evidence."""
    statement: str = Field(..., min_length=20, description="Clear, specific finding")
    quotes: List[Quote] = Field(..., min_length=1, description="Supporting quotes")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analyst confidence 0-1")
    impact_estimate: int = Field(..., ge=0, le=10, description="Initial impact estimate 0-10")
    
    @field_validator('quotes')
    @classmethod
    def validate_quotes(cls, v):
        if not v:
            raise ValueError("Finding must have at least one quote")
        return v


class RubricScore(BaseModel):
    """Single rubric dimension score with audit trail."""
    dimension: Literal["legal_risk", "financial_impact"] = Field(..., description="Rubric dimension")
    score: int = Field(..., ge=0, le=10, description="Score 0-10")
    rationale: str = Field(..., min_length=50, description="Detailed explanation")
    evidence: List[Quote] = Field(default_factory=list, description="Supporting evidence")
    rubric_anchor: str = Field(..., description="Reference to rubric scale")
    
    
class PrimaryAnalysis(BaseModel):
    """Output from Tier 1: Primary Analyst."""
    bill_id: str
    findings: List[Finding] = Field(default_factory=list)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    requires_multi_sample: bool = Field(
        default=False,
        description="True if impact>=6 OR confidence<0.7"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JudgeValidation(BaseModel):
    """Output from Tier 2: Judge Model."""
    finding_id: int = Field(..., description="Index of finding being validated")
    quote_verified: bool = Field(..., description="All quotes exist verbatim in bill?")
    hallucination_detected: bool = Field(default=False, description="Claims not in bill text?")
    evidence_quality: float = Field(..., ge=0.0, le=1.0, description="Quality of evidence 0-1")
    ambiguity: float = Field(..., ge=0.0, le=1.0, description="Interpretive uncertainty 0-1")
    judge_confidence: float = Field(..., ge=0.0, le=1.0, description="Judge's confidence 0-1")


class TwoTierAnalysisResult(BaseModel):
    """Complete output from two-tier analysis."""
    bill_id: str
    primary_analysis: PrimaryAnalysis
    judge_validations: List[JudgeValidation] = Field(default_factory=list)
    rubric_scores: List[RubricScore] = Field(default_factory=list)
    route: Literal["STANDARD", "ENHANCED"] = Field(default="STANDARD")
    cost_estimate: float = Field(default=0.0, description="Total LLM cost in USD")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models_v2.py -v`

Expected: All tests PASS

**Step 5: Add test for rubric score validation**

Add to test file:

```python
def test_rubric_score_validation():
    """Rubric score requires rationale and anchor."""
    quote = Quote(text="Section 4.2: $50/MW annual tax", section="4.2", page=12)
    
    score = RubricScore(
        dimension="financial_impact",
        score=7,
        rationale="Estimated $1.14M annual cost based on Section 4.2 tax rate ($50/MW) applied to NRG's 60% fossil portfolio (23 GW)",
        evidence=[quote],
        rubric_anchor="6-8: Major obligations, $500K-$5M exposure"
    )
    
    assert score.score == 7
    assert score.dimension == "financial_impact"
    assert len(score.evidence) == 1
```

Run: `pytest tests/test_models_v2.py::test_rubric_score_validation -v`

Expected: PASS

**Step 6: Commit models**

```bash
git add nrg_core/models_v2.py tests/test_models_v2.py
git commit -m "feat(v2): add pydantic models for two-tier analysis"
```

---

## Task 2: Create Supervisor Router Module

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/supervisor.py`
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/__init__.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_supervisor.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/__init__.py`

**Step 1: Write failing test for complexity scoring**

```python
# tests/test_v2/__init__.py
# Empty init file

# tests/test_v2/test_supervisor.py
import pytest
from nrg_core.v2.supervisor import SupervisorRouter, Route


def test_simple_bill_routes_to_standard():
    """Short bills with <20 pages, <2 versions route to STANDARD."""
    router = SupervisorRouter()
    
    bill_metadata = {
        "bill_id": "HB123",
        "page_count": 15,
        "version_count": 1,
        "domain": "general"
    }
    
    route = router.assess_complexity(bill_metadata)
    
    assert route == Route.STANDARD
    assert router.complexity_score == 1  # Only 1 point (1 version)


def test_complex_bill_routes_to_enhanced():
    """Long bills, multiple versions, energy domain route to ENHANCED."""
    router = SupervisorRouter()
    
    bill_metadata = {
        "bill_id": "HB456",
        "page_count": 55,
        "version_count": 6,
        "domain": "energy"
    }
    
    route = router.assess_complexity(bill_metadata)
    
    assert route == Route.ENHANCED
    assert router.complexity_score >= 3  # Length(2) + Versions(2) + Domain(2) = 6
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_v2/test_supervisor.py -v`

Expected: `ModuleNotFoundError: No module named 'nrg_core.v2'`

**Step 3: Implement supervisor router**

```python
# nrg_core/v2/__init__.py
"""Architecture v2.0 components."""
from nrg_core.v2.supervisor import SupervisorRouter, Route

__all__ = ["SupervisorRouter", "Route"]
```

```python
# nrg_core/v2/supervisor.py
"""
Supervisor Router - Code-based complexity assessment and routing
Architecture v2.0 - Phase 1
"""
from enum import Enum
from typing import Dict, Any, Literal


class Route(str, Enum):
    """Analysis route selection."""
    STANDARD = "STANDARD"  # Simple bills, single-pass analysis
    ENHANCED = "ENHANCED"  # Complex bills, full two-tier pipeline


class SupervisorRouter:
    """
    Code-based router for bill analysis.
    
    Complexity Scoring:
    - Length: >50 pages = +2, 20-50 pages = +1
    - Versions: >5 = +2, 2-5 = +1
    - Domain: Energy/Tax = +2, Environmental = +1
    
    Route Decision:
    - 0-2 points: STANDARD (80% of bills)
    - 3+ points: ENHANCED (20% of bills)
    """
    
    def __init__(self):
        self.complexity_score = 0
        self.score_breakdown = {}
    
    def assess_complexity(self, bill_metadata: Dict[str, Any]) -> Route:
        """
        Assess bill complexity using deterministic rules.
        
        Args:
            bill_metadata: Dict with keys: page_count, version_count, domain
        
        Returns:
            Route.STANDARD or Route.ENHANCED
        """
        score = 0
        breakdown = {}
        
        # Length scoring
        page_count = bill_metadata.get("page_count", 0)
        if page_count > 50:
            score += 2
            breakdown["length"] = 2
        elif page_count >= 20:
            score += 1
            breakdown["length"] = 1
        else:
            breakdown["length"] = 0
        
        # Version count scoring
        version_count = bill_metadata.get("version_count", 1)
        if version_count > 5:
            score += 2
            breakdown["versions"] = 2
        elif version_count >= 2:
            score += 1
            breakdown["versions"] = 1
        else:
            breakdown["versions"] = 0
        
        # Domain scoring
        domain = bill_metadata.get("domain", "general").lower()
        if domain in ["energy", "tax"]:
            score += 2
            breakdown["domain"] = 2
        elif domain == "environmental":
            score += 1
            breakdown["domain"] = 1
        else:
            breakdown["domain"] = 0
        
        self.complexity_score = score
        self.score_breakdown = breakdown
        
        # Route decision
        if score >= 3:
            return Route.ENHANCED
        else:
            return Route.STANDARD
    
    def get_budget_constraints(self, route: Route) -> Dict[str, int]:
        """Get token and time budgets for route."""
        if route == Route.STANDARD:
            return {
                "token_budget": 50000,
                "time_budget_seconds": 30,
                "min_evidence_count": 1
            }
        else:  # ENHANCED
            return {
                "token_budget": 100000,
                "time_budget_seconds": 300,
                "min_evidence_count": 2
            }
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_v2/test_supervisor.py -v`

Expected: All tests PASS

**Step 5: Add boundary test**

```python
def test_boundary_at_3_points():
    """Exactly 3 points routes to ENHANCED."""
    router = SupervisorRouter()
    
    # 20 pages (1) + 2 versions (1) + environmental (1) = 3 points
    bill_metadata = {
        "bill_id": "HB789",
        "page_count": 20,
        "version_count": 2,
        "domain": "environmental"
    }
    
    route = router.assess_complexity(bill_metadata)
    
    assert route == Route.ENHANCED
    assert router.complexity_score == 3
```

Run: `pytest tests/test_v2/test_supervisor.py::test_boundary_at_3_points -v`

Expected: PASS

**Step 6: Commit**

```bash
git add nrg_core/v2/ tests/test_v2/
git commit -m "feat(v2): add supervisor router with complexity scoring"
```

---

## Task 3: Create Primary Analyst Agent

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/primary_analyst.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_primary_analyst.py`

**Step 1: Write failing test for primary analyst**

```python
# tests/test_v2/test_primary_analyst.py
import pytest
from unittest.mock import Mock, patch
from nrg_core.v2.primary_analyst import PrimaryAnalyst
from nrg_core.models_v2 import PrimaryAnalysis, Finding


def test_primary_analyst_extracts_findings():
    """Primary analyst should extract findings with quotes."""
    analyst = PrimaryAnalyst(model="gpt-4o", api_key="test-key")
    
    bill_text = """
    Section 2.1: Energy Tax Collection
    A tax of $50 per megawatt shall be assessed annually on all 
    fossil fuel generation facilities exceeding 50 megawatts capacity.
    
    Section 2.2: Exemptions
    Renewable energy facilities are exempt from this tax.
    """
    
    nrg_context = "NRG operates 23 GW fossil (60%), 15 GW renewable (40%)"
    
    # Mock OpenAI response
    mock_response = {
        "findings": [
            {
                "statement": "Tax applies to fossil fuel facilities >50MW at $50/MW annually",
                "quotes": [
                    {"text": "A tax of $50 per megawatt shall be assessed annually on all fossil fuel generation facilities exceeding 50 megawatts capacity", "section": "2.1", "page": None}
                ],
                "confidence": 0.95,
                "impact_estimate": 8
            }
        ],
        "overall_confidence": 0.95,
        "requires_multi_sample": True  # impact >= 6
    }
    
    with patch.object(analyst, '_call_llm', return_value=mock_response):
        result = analyst.analyze(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context=nrg_context
        )
    
    assert isinstance(result, PrimaryAnalysis)
    assert len(result.findings) == 1
    assert result.findings[0].impact_estimate == 8
    assert result.requires_multi_sample is True
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_v2/test_primary_analyst.py::test_primary_analyst_extracts_findings -v`

Expected: `ModuleNotFoundError: No module named 'nrg_core.v2.primary_analyst'`

**Step 3: Implement primary analyst**

```python
# nrg_core/v2/primary_analyst.py
"""
Primary Analyst Agent (Tier 1)
Extracts findings from bill text with supporting quotes
"""
from typing import Dict, Any
from openai import OpenAI
from nrg_core.models_v2 import PrimaryAnalysis, Finding, Quote


PRIMARY_ANALYST_PROMPT = """You are a legislative analyst for NRG Energy.

BUSINESS CONTEXT:
{nrg_context}

TASK:
Analyze the following bill and identify ALL provisions that could impact NRG Energy's business.

BILL TEXT:
{bill_text}

REQUIREMENTS:
1. Extract specific findings (not general observations)
2. Each finding MUST have at least one supporting quote from the bill
3. Quote the exact text (verbatim) with section reference
4. Estimate impact 0-10 (0=no impact, 10=existential threat)
5. Provide confidence 0-1 for each finding

OUTPUT FORMAT (JSON):
{{
  "findings": [
    {{
      "statement": "Clear, specific finding",
      "quotes": [
        {{"text": "Exact quote from bill", "section": "2.1", "page": null}}
      ],
      "confidence": 0.85,
      "impact_estimate": 7
    }}
  ],
  "overall_confidence": 0.80
}}

CRITICAL:
- Every statement must be supported by a direct quote
- Do not infer provisions not explicitly stated
- Mark low confidence (<0.7) if interpretation is uncertain
"""


class PrimaryAnalyst:
    """
    Tier 1: Primary Analyst
    Single-pass analysis extracting findings with evidence
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def analyze(
        self,
        bill_id: str,
        bill_text: str,
        nrg_context: str
    ) -> PrimaryAnalysis:
        """
        Analyze bill and extract findings.
        
        Args:
            bill_id: Bill identifier
            bill_text: Full bill text
            nrg_context: NRG business context
        
        Returns:
            PrimaryAnalysis with findings and confidence
        """
        # Call LLM
        llm_output = self._call_llm(bill_text, nrg_context)
        
        # Parse findings
        findings = [
            Finding(
                statement=f["statement"],
                quotes=[Quote(**q) for q in f["quotes"]],
                confidence=f["confidence"],
                impact_estimate=f["impact_estimate"]
            )
            for f in llm_output["findings"]
        ]
        
        # Determine if multi-sample needed
        requires_multi_sample = any(
            f.impact_estimate >= 6 or f.confidence < 0.7
            for f in findings
        )
        
        return PrimaryAnalysis(
            bill_id=bill_id,
            findings=findings,
            overall_confidence=llm_output["overall_confidence"],
            requires_multi_sample=requires_multi_sample
        )
    
    def _call_llm(self, bill_text: str, nrg_context: str) -> Dict[str, Any]:
        """Call OpenAI API with structured output."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        prompt = PRIMARY_ANALYST_PROMPT.format(
            bill_text=bill_text,
            nrg_context=nrg_context
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        return json.loads(response.choices[0].message.content)
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_v2/test_primary_analyst.py::test_primary_analyst_extracts_findings -v`

Expected: PASS

**Step 5: Add test for multi-sample trigger**

```python
def test_multi_sample_trigger_on_low_confidence():
    """Multi-sample should trigger when confidence < 0.7."""
    analyst = PrimaryAnalyst(model="gpt-4o", api_key="test-key")
    
    mock_response = {
        "findings": [
            {
                "statement": "Ambiguous provision may apply",
                "quotes": [{"text": "Section 5.2 references undefined term", "section": "5.2", "page": None}],
                "confidence": 0.65,  # Low confidence
                "impact_estimate": 4
            }
        ],
        "overall_confidence": 0.65
    }
    
    with patch.object(analyst, '_call_llm', return_value=mock_response):
        result = analyst.analyze("HB456", "bill text", "nrg context")
    
    assert result.requires_multi_sample is True  # Triggered by low confidence
```

Run: `pytest tests/test_v2/test_primary_analyst.py::test_multi_sample_trigger_on_low_confidence -v`

Expected: PASS

**Step 6: Commit**

```bash
git add nrg_core/v2/primary_analyst.py tests/test_v2/test_primary_analyst.py
git commit -m "feat(v2): add primary analyst with LLM integration"
```

---

## Task 4: Create Judge Model

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/judge.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_judge.py`

**Step 1: Write failing test for quote verification**

```python
# tests/test_v2/test_judge.py
import pytest
from unittest.mock import Mock, patch
from nrg_core.v2.judge import JudgeModel
from nrg_core.models_v2 import Finding, Quote, JudgeValidation


def test_judge_verifies_quotes_exist():
    """Judge should verify quotes exist verbatim in bill text."""
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    
    bill_text = """
    Section 2.1: Tax Rate
    The annual tax rate is $50 per megawatt.
    """
    
    finding = Finding(
        statement="Tax is $50/MW annually",
        quotes=[Quote(text="The annual tax rate is $50 per megawatt", section="2.1", page=None)],
        confidence=0.9,
        impact_estimate=7
    )
    
    mock_validation = {
        "quote_verified": True,
        "hallucination_detected": False,
        "evidence_quality": 0.95,
        "ambiguity": 0.1,
        "judge_confidence": 0.95
    }
    
    with patch.object(judge, '_call_llm', return_value=mock_validation):
        result = judge.validate(finding_id=0, finding=finding, bill_text=bill_text)
    
    assert isinstance(result, JudgeValidation)
    assert result.quote_verified is True
    assert result.hallucination_detected is False
    assert result.evidence_quality == 0.95


def test_judge_detects_hallucination():
    """Judge should flag claims not in bill text."""
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    
    bill_text = """
    Section 2.1: Tax Rate
    The annual tax rate is $50 per megawatt.
    """
    
    finding = Finding(
        statement="Tax includes quarterly reporting requirement",  # NOT in bill
        quotes=[Quote(text="The annual tax rate is $50 per megawatt", section="2.1", page=None)],
        confidence=0.8,
        impact_estimate=5
    )
    
    mock_validation = {
        "quote_verified": True,
        "hallucination_detected": True,  # Quote exists but statement adds claim
        "evidence_quality": 0.4,
        "ambiguity": 0.3,
        "judge_confidence": 0.75
    }
    
    with patch.object(judge, '_call_llm', return_value=mock_validation):
        result = judge.validate(finding_id=0, finding=finding, bill_text=bill_text)
    
    assert result.hallucination_detected is True
    assert result.evidence_quality < 0.5
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_v2/test_judge.py -v`

Expected: `ModuleNotFoundError: No module named 'nrg_core.v2.judge'`

**Step 3: Implement judge model**

```python
# nrg_core/v2/judge.py
"""
Judge Model (Tier 2)
Validates findings, detects hallucinations, scores evidence quality
"""
from typing import Dict, Any
from openai import OpenAI
from nrg_core.models_v2 import Finding, JudgeValidation


JUDGE_VALIDATION_PROMPT = """You are a validation judge for legislative analysis.

BILL TEXT:
{bill_text}

FINDING TO VALIDATE:
Statement: {statement}
Quotes: {quotes}

TASK:
Validate this finding by checking:
1. Do all quoted texts exist VERBATIM in the bill? (exact match required)
2. Does the statement make claims NOT supported by the quotes?
3. How strong is the evidence (0-1)?
4. How ambiguous is the interpretation (0-1)?

OUTPUT (JSON):
{{
  "quote_verified": true/false,
  "hallucination_detected": true/false,
  "evidence_quality": 0.0-1.0,
  "ambiguity": 0.0-1.0,
  "judge_confidence": 0.0-1.0
}}

CRITICAL:
- quote_verified = false if ANY quote is not exact match
- hallucination_detected = true if statement claims things not in quotes
- evidence_quality = how well quotes support the statement
- ambiguity = how much interpretation is required
"""


class JudgeModel:
    """
    Tier 2: Judge
    Validates findings from primary analyst
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def validate(
        self,
        finding_id: int,
        finding: Finding,
        bill_text: str
    ) -> JudgeValidation:
        """
        Validate a single finding.
        
        Args:
            finding_id: Index of finding
            finding: Finding to validate
            bill_text: Full bill text for verification
        
        Returns:
            JudgeValidation with verification results
        """
        # Call LLM for validation
        validation_output = self._call_llm(finding, bill_text)
        
        return JudgeValidation(
            finding_id=finding_id,
            quote_verified=validation_output["quote_verified"],
            hallucination_detected=validation_output["hallucination_detected"],
            evidence_quality=validation_output["evidence_quality"],
            ambiguity=validation_output["ambiguity"],
            judge_confidence=validation_output["judge_confidence"]
        )
    
    def score_rubric(
        self,
        dimension: str,
        finding: Finding,
        bill_text: str,
        nrg_context: str,
        rubric_anchors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Score a finding on a rubric dimension.
        
        Args:
            dimension: "legal_risk" or "financial_impact"
            finding: Finding to score
            bill_text: Full bill text
            nrg_context: NRG business context
            rubric_anchors: Scale definitions
        
        Returns:
            Dict with score, rationale, evidence, rubric_anchor
        """
        # Will implement in next task
        pass
    
    def _call_llm(self, finding: Finding, bill_text: str) -> Dict[str, Any]:
        """Call OpenAI for validation."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        quotes_text = "\n".join([f"- {q.text} (Section {q.section})" for q in finding.quotes])
        
        prompt = JUDGE_VALIDATION_PROMPT.format(
            bill_text=bill_text,
            statement=finding.statement,
            quotes=quotes_text
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1  # Lower temperature for validation
        )
        
        import json
        return json.loads(response.choices[0].message.content)
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_v2/test_judge.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add nrg_core/v2/judge.py tests/test_v2/test_judge.py
git commit -m "feat(v2): add judge model for validation"
```

---

## Task 5: Add Rubric Scoring to Judge

**Files:**
- Modify: `/Users/thamac/Documents/NRG/nrg_core/v2/judge.py`
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/rubrics.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_rubrics.py`

**Step 1: Write failing test for rubric scoring**

```python
# tests/test_v2/test_rubrics.py
import pytest
from unittest.mock import patch
from nrg_core.v2.judge import JudgeModel
from nrg_core.v2.rubrics import LEGAL_RISK_RUBRIC, FINANCIAL_IMPACT_RUBRIC
from nrg_core.models_v2 import Finding, Quote, RubricScore


def test_judge_scores_legal_risk():
    """Judge should score findings on legal risk dimension."""
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    
    finding = Finding(
        statement="Tax applies with $10K penalty per violation",
        quotes=[Quote(text="Penalty of $10,000 per violation", section="4.3", page=15)],
        confidence=0.9,
        impact_estimate=7
    )
    
    mock_score_output = {
        "dimension": "legal_risk",
        "score": 7,
        "rationale": "Score=7 because bill creates new tax obligation with significant penalties ($10K per violation). Section 4.3 establishes financial consequences for non-compliance.",
        "evidence": [{"text": "Penalty of $10,000 per violation", "section": "4.3", "page": 15}],
        "rubric_anchor": "6-8: Significant obligations + penalties"
    }
    
    with patch.object(judge, '_call_llm_for_rubric', return_value=mock_score_output):
        score = judge.score_rubric(
            dimension="legal_risk",
            finding=finding,
            bill_text="...",
            nrg_context="NRG operates 60% fossil, 40% renewable",
            rubric_anchors=LEGAL_RISK_RUBRIC
        )
    
    assert isinstance(score, RubricScore)
    assert score.dimension == "legal_risk"
    assert score.score == 7
    assert "penalties" in score.rationale.lower()
    assert score.rubric_anchor == "6-8: Significant obligations + penalties"


def test_judge_scores_financial_impact():
    """Judge should estimate financial impact with business context."""
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    
    finding = Finding(
        statement="$50/MW annual tax on fossil generation >50MW",
        quotes=[Quote(text="Annual tax of $50 per megawatt", section="2.1", page=3)],
        confidence=0.95,
        impact_estimate=8
    )
    
    mock_score_output = {
        "dimension": "financial_impact",
        "score": 7,
        "rationale": "Score=7 because $50/MW tax on NRG's 23 GW fossil portfolio = $1.15M annual cost. Material to P&L but not existential.",
        "evidence": [{"text": "Annual tax of $50 per megawatt", "section": "2.1", "page": 3}],
        "rubric_anchor": "6-8: $500K-$5M material to P&L"
    }
    
    with patch.object(judge, '_call_llm_for_rubric', return_value=mock_score_output):
        score = judge.score_rubric(
            dimension="financial_impact",
            finding=finding,
            bill_text="...",
            nrg_context="NRG: 23 GW fossil, 15 GW renewable",
            rubric_anchors=FINANCIAL_IMPACT_RUBRIC
        )
    
    assert score.score == 7
    assert "$1.15M" in score.rationale or "$1,150,000" in score.rationale
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_v2/test_rubrics.py -v`

Expected: `ModuleNotFoundError: No module named 'nrg_core.v2.rubrics'`

**Step 3: Create rubric definitions**

```python
# nrg_core/v2/rubrics.py
"""
Rubric Definitions for Phase 1
Two dimensions: Legal Risk, Financial Impact
"""

LEGAL_RISK_RUBRIC = {
    "0-2": "No new obligations - cosmetic language changes only",
    "3-5": "Minor compliance - reporting, training, disclosure requirements",
    "6-8": "Significant obligations + penalties - new taxes, operational requirements, financial consequences",
    "9-10": "Existential threats - bans, license revocations, structural prohibitions"
}


FINANCIAL_IMPACT_RUBRIC = {
    "0-2": "<$100K annual - negligible impact",
    "3-5": "$100K-$500K - minor but trackable cost",
    "6-8": "$500K-$5M - material to P&L",
    "9-10": ">$5M or revenue at risk - major financial exposure"
}


RUBRIC_SCORING_PROMPT = """You are scoring a legislative finding on the {dimension} dimension.

RUBRIC SCALE:
{rubric_scale}

NRG BUSINESS CONTEXT:
{nrg_context}

BILL TEXT:
{bill_text}

FINDING:
Statement: {statement}
Quotes: {quotes}

TASK:
Score this finding 0-10 on {dimension} using the rubric scale above.

REQUIREMENTS:
1. Select score that best matches rubric anchor
2. Provide detailed rationale (minimum 50 chars) explaining the score
3. Reference specific quotes as evidence
4. Cite the rubric anchor used

OUTPUT (JSON):
{{
  "dimension": "{dimension}",
  "score": 0-10,
  "rationale": "Detailed explanation with business impact calculation",
  "evidence": [{{"text": "quote", "section": "X.X", "page": null}}],
  "rubric_anchor": "X-Y: anchor text from rubric"
}}
"""


def format_rubric_scale(rubric: dict) -> str:
    """Format rubric dict into readable scale."""
    return "\n".join([f"{k}: {v}" for k, v in rubric.items()])
```

**Step 4: Implement rubric scoring in Judge**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/judge.py`:

```python
# Add to imports
from nrg_core.models_v2 import Finding, JudgeValidation, RubricScore, Quote
from nrg_core.v2.rubrics import RUBRIC_SCORING_PROMPT, format_rubric_scale

# Replace score_rubric method
    def score_rubric(
        self,
        dimension: str,
        finding: Finding,
        bill_text: str,
        nrg_context: str,
        rubric_anchors: Dict[str, str]
    ) -> RubricScore:
        """
        Score a finding on a rubric dimension.
        
        Args:
            dimension: "legal_risk" or "financial_impact"
            finding: Finding to score
            bill_text: Full bill text
            nrg_context: NRG business context
            rubric_anchors: Scale definitions
        
        Returns:
            RubricScore with score, rationale, evidence
        """
        # Call LLM for rubric scoring
        score_output = self._call_llm_for_rubric(
            dimension=dimension,
            finding=finding,
            bill_text=bill_text,
            nrg_context=nrg_context,
            rubric_anchors=rubric_anchors
        )
        
        return RubricScore(
            dimension=score_output["dimension"],
            score=score_output["score"],
            rationale=score_output["rationale"],
            evidence=[Quote(**q) for q in score_output["evidence"]],
            rubric_anchor=score_output["rubric_anchor"]
        )
    
    def _call_llm_for_rubric(
        self,
        dimension: str,
        finding: Finding,
        bill_text: str,
        nrg_context: str,
        rubric_anchors: Dict[str, str]
    ) -> Dict[str, Any]:
        """Call OpenAI for rubric scoring."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        quotes_text = "\n".join([f"- {q.text} (Section {q.section})" for q in finding.quotes])
        rubric_scale = format_rubric_scale(rubric_anchors)
        
        prompt = RUBRIC_SCORING_PROMPT.format(
            dimension=dimension,
            rubric_scale=rubric_scale,
            nrg_context=nrg_context,
            bill_text=bill_text[:5000],  # Truncate for token limit
            statement=finding.statement,
            quotes=quotes_text
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        import json
        return json.loads(response.choices[0].message.content)
```

**Step 5: Run tests to verify pass**

Run: `pytest tests/test_v2/test_rubrics.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add nrg_core/v2/rubrics.py nrg_core/v2/judge.py tests/test_v2/test_rubrics.py
git commit -m "feat(v2): add rubric scoring to judge (legal risk + financial impact)"
```

---

## Task 6: Create Two-Tier Orchestrator

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/two_tier.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_two_tier.py`

**Step 1: Write failing integration test**

```python
# tests/test_v2/test_two_tier.py
import pytest
from unittest.mock import patch, Mock
from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.models_v2 import TwoTierAnalysisResult


def test_two_tier_orchestrator_full_pipeline():
    """Integration test: orchestrator runs primary analyst → judge validation → rubric scoring."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test-primary",
        judge_api_key="test-judge"
    )
    
    bill_text = """
    Section 2.1: Annual tax of $50 per megawatt on fossil fuel facilities >50MW.
    Section 4.3: Penalty of $10,000 per violation for non-compliance.
    """
    
    nrg_context = "NRG operates 23 GW fossil (60%), 15 GW renewable (40%)"
    
    # Mock primary analyst response
    mock_primary_findings = {
        "findings": [
            {
                "statement": "Tax of $50/MW on fossil >50MW",
                "quotes": [{"text": "Annual tax of $50 per megawatt on fossil fuel facilities >50MW", "section": "2.1", "page": None}],
                "confidence": 0.95,
                "impact_estimate": 8
            }
        ],
        "overall_confidence": 0.95
    }
    
    # Mock judge validation
    mock_judge_validation = {
        "quote_verified": True,
        "hallucination_detected": False,
        "evidence_quality": 0.95,
        "ambiguity": 0.1,
        "judge_confidence": 0.95
    }
    
    # Mock rubric scores
    mock_legal_score = {
        "dimension": "legal_risk",
        "score": 7,
        "rationale": "New tax obligation with penalties",
        "evidence": [{"text": "Annual tax of $50 per megawatt", "section": "2.1", "page": None}],
        "rubric_anchor": "6-8: Significant obligations + penalties"
    }
    
    mock_financial_score = {
        "dimension": "financial_impact",
        "score": 7,
        "rationale": "$1.15M annual cost on 23 GW fossil portfolio",
        "evidence": [{"text": "Annual tax of $50 per megawatt", "section": "2.1", "page": None}],
        "rubric_anchor": "6-8: $500K-$5M material to P&L"
    }
    
    with patch.object(orchestrator.primary_analyst, '_call_llm', return_value=mock_primary_findings), \
         patch.object(orchestrator.judge, '_call_llm', return_value=mock_judge_validation), \
         patch.object(orchestrator.judge, '_call_llm_for_rubric', side_effect=[mock_legal_score, mock_financial_score]):
        
        result = orchestrator.analyze(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context=nrg_context
        )
    
    assert isinstance(result, TwoTierAnalysisResult)
    assert len(result.primary_analysis.findings) == 1
    assert len(result.judge_validations) == 1
    assert len(result.rubric_scores) == 2  # legal_risk + financial_impact
    assert result.judge_validations[0].quote_verified is True
    assert result.rubric_scores[0].dimension == "legal_risk"
    assert result.rubric_scores[1].dimension == "financial_impact"
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_v2/test_two_tier.py -v`

Expected: `ModuleNotFoundError: No module named 'nrg_core.v2.two_tier'`

**Step 3: Implement two-tier orchestrator**

```python
# nrg_core/v2/two_tier.py
"""
Two-Tier Analysis Orchestrator
Coordinates Primary Analyst (Tier 1) → Judge (Tier 2) → Rubric Scoring
"""
from typing import List
from nrg_core.v2.primary_analyst import PrimaryAnalyst
from nrg_core.v2.judge import JudgeModel
from nrg_core.v2.rubrics import LEGAL_RISK_RUBRIC, FINANCIAL_IMPACT_RUBRIC
from nrg_core.models_v2 import TwoTierAnalysisResult, RubricScore


class TwoTierOrchestrator:
    """
    Orchestrates two-tier analysis pipeline:
    1. Primary Analyst extracts findings
    2. Judge validates each finding
    3. Judge scores findings on rubrics
    """
    
    def __init__(
        self,
        primary_model: str = "gpt-4o",
        judge_model: str = "gpt-4o",
        primary_api_key: str = None,
        judge_api_key: str = None
    ):
        self.primary_analyst = PrimaryAnalyst(
            model=primary_model,
            api_key=primary_api_key
        )
        self.judge = JudgeModel(
            model=judge_model,
            api_key=judge_api_key
        )
    
    def analyze(
        self,
        bill_id: str,
        bill_text: str,
        nrg_context: str
    ) -> TwoTierAnalysisResult:
        """
        Run full two-tier analysis.
        
        Args:
            bill_id: Bill identifier
            bill_text: Full bill text
            nrg_context: NRG business context
        
        Returns:
            TwoTierAnalysisResult with analysis, validations, scores
        """
        # Tier 1: Primary Analyst
        primary_analysis = self.primary_analyst.analyze(
            bill_id=bill_id,
            bill_text=bill_text,
            nrg_context=nrg_context
        )
        
        # Tier 2: Judge Validation
        judge_validations = []
        for idx, finding in enumerate(primary_analysis.findings):
            validation = self.judge.validate(
                finding_id=idx,
                finding=finding,
                bill_text=bill_text
            )
            judge_validations.append(validation)
        
        # Tier 2: Rubric Scoring (only for validated findings)
        rubric_scores: List[RubricScore] = []
        for idx, finding in enumerate(primary_analysis.findings):
            validation = judge_validations[idx]
            
            # Skip scoring if hallucination detected
            if validation.hallucination_detected:
                continue
            
            # Score on both dimensions
            for dimension, rubric in [
                ("legal_risk", LEGAL_RISK_RUBRIC),
                ("financial_impact", FINANCIAL_IMPACT_RUBRIC)
            ]:
                score = self.judge.score_rubric(
                    dimension=dimension,
                    finding=finding,
                    bill_text=bill_text,
                    nrg_context=nrg_context,
                    rubric_anchors=rubric
                )
                rubric_scores.append(score)
        
        return TwoTierAnalysisResult(
            bill_id=bill_id,
            primary_analysis=primary_analysis,
            judge_validations=judge_validations,
            rubric_scores=rubric_scores,
            route="ENHANCED",  # Phase 1 uses enhanced path only
            cost_estimate=0.0  # TODO: track actual costs
        )
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_v2/test_two_tier.py::test_two_tier_orchestrator_full_pipeline -v`

Expected: PASS

**Step 5: Add test for hallucination handling**

```python
def test_two_tier_skips_rubric_scoring_for_hallucinations():
    """Findings with hallucinations should not be scored on rubrics."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test-primary",
        judge_api_key="test-judge"
    )
    
    mock_primary_findings = {
        "findings": [
            {
                "statement": "Bill requires quarterly reporting",  # Hallucination
                "quotes": [{"text": "Annual tax of $50", "section": "2.1", "page": None}],
                "confidence": 0.8,
                "impact_estimate": 5
            }
        ],
        "overall_confidence": 0.8
    }
    
    mock_judge_validation = {
        "quote_verified": True,
        "hallucination_detected": True,  # Statement not supported by quote
        "evidence_quality": 0.3,
        "ambiguity": 0.5,
        "judge_confidence": 0.7
    }
    
    with patch.object(orchestrator.primary_analyst, '_call_llm', return_value=mock_primary_findings), \
         patch.object(orchestrator.judge, '_call_llm', return_value=mock_judge_validation):
        
        result = orchestrator.analyze("HB456", "bill text", "nrg context")
    
    # Validation should exist
    assert len(result.judge_validations) == 1
    assert result.judge_validations[0].hallucination_detected is True
    
    # But no rubric scores (skipped due to hallucination)
    assert len(result.rubric_scores) == 0
```

Run: `pytest tests/test_v2/test_two_tier.py::test_two_tier_skips_rubric_scoring_for_hallucinations -v`

Expected: PASS

**Step 6: Commit**

```bash
git add nrg_core/v2/two_tier.py tests/test_v2/test_two_tier.py
git commit -m "feat(v2): add two-tier orchestrator for complete pipeline"
```

---

## Task 7: Create CLI Entry Point for V2

**Files:**
- Create: `/Users/thamac/Documents/NRG/run_v2_analysis.py`
- Modify: `/Users/thamac/Documents/NRG/README.md`

**Step 1: Write CLI script**

```python
# run_v2_analysis.py
"""
CLI for Architecture v2.0 Two-Tier Analysis
Usage: python run_v2_analysis.py --bill-id HB123 --bill-text-file path/to/bill.txt
"""
import os
import sys
import argparse
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from nrg_core.v2.supervisor import SupervisorRouter
from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.config import load_nrg_context

load_dotenv()
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run Architecture v2.0 Analysis")
    parser.add_argument("--bill-id", required=True, help="Bill identifier (e.g., HB123)")
    parser.add_argument("--bill-text-file", required=True, help="Path to bill text file")
    parser.add_argument("--output", default="v2_analysis.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load bill text
    with open(args.bill_text_file, 'r') as f:
        bill_text = f.read()
    
    # Load NRG context
    nrg_context = load_nrg_context()
    
    # Step 1: Supervisor routing
    console.print("\n[cyan]Step 1: Assessing Complexity...[/cyan]")
    router = SupervisorRouter()
    
    bill_metadata = {
        "bill_id": args.bill_id,
        "page_count": len(bill_text) // 3000,  # Rough estimate
        "version_count": 1,
        "domain": "general"
    }
    
    route = router.assess_complexity(bill_metadata)
    console.print(f"Route: [bold]{route}[/bold]")
    console.print(f"Complexity Score: {router.complexity_score}")
    console.print(f"Breakdown: {router.score_breakdown}")
    
    # Step 2: Two-tier analysis
    console.print("\n[cyan]Step 2: Running Two-Tier Analysis...[/cyan]")
    
    orchestrator = TwoTierOrchestrator(
        primary_api_key=os.getenv("OPENAI_API_KEY"),
        judge_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    result = orchestrator.analyze(
        bill_id=args.bill_id,
        bill_text=bill_text,
        nrg_context=nrg_context
    )
    
    # Display results
    console.print(f"\n[green]✓ Analysis Complete[/green]")
    console.print(f"Findings: {len(result.primary_analysis.findings)}")
    console.print(f"Validations: {len(result.judge_validations)}")
    console.print(f"Rubric Scores: {len(result.rubric_scores)}")
    
    # Table of findings
    table = Table(title="Findings", show_lines=True)
    table.add_column("#", style="cyan")
    table.add_column("Statement", style="white")
    table.add_column("Impact", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Verified", justify="center")
    
    for idx, finding in enumerate(result.primary_analysis.findings):
        validation = result.judge_validations[idx]
        verified = "✓" if validation.quote_verified and not validation.hallucination_detected else "✗"
        
        table.add_row(
            str(idx),
            finding.statement[:80] + "...",
            str(finding.impact_estimate),
            f"{finding.confidence:.2f}",
            verified
        )
    
    console.print(table)
    
    # Rubric scores table
    if result.rubric_scores:
        rubric_table = Table(title="Rubric Scores")
        rubric_table.add_column("Dimension", style="cyan")
        rubric_table.add_column("Score", justify="center")
        rubric_table.add_column("Rationale", style="white")
        
        for score in result.rubric_scores:
            rubric_table.add_row(
                score.dimension,
                str(score.score),
                score.rationale[:100] + "..."
            )
        
        console.print(rubric_table)
    
    # Save to JSON
    output_data = {
        "bill_id": result.bill_id,
        "route": result.route,
        "findings_count": len(result.primary_analysis.findings),
        "findings": [
            {
                "statement": f.statement,
                "quotes": [{"text": q.text, "section": q.section} for q in f.quotes],
                "confidence": f.confidence,
                "impact_estimate": f.impact_estimate
            }
            for f in result.primary_analysis.findings
        ],
        "validations": [
            {
                "finding_id": v.finding_id,
                "quote_verified": v.quote_verified,
                "hallucination_detected": v.hallucination_detected,
                "evidence_quality": v.evidence_quality,
                "judge_confidence": v.judge_confidence
            }
            for v in result.judge_validations
        ],
        "rubric_scores": [
            {
                "dimension": s.dimension,
                "score": s.score,
                "rationale": s.rationale,
                "rubric_anchor": s.rubric_anchor
            }
            for s in result.rubric_scores
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"\n[green]✓ Results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
```

**Step 2: Test CLI manually**

Create test bill file `/Users/thamac/Documents/NRG/test_bill.txt`:

```
Section 2.1: Energy Tax
An annual tax of $50 per megawatt shall be assessed on all fossil fuel 
generation facilities exceeding 50 megawatts capacity.

Section 4.3: Penalties
Non-compliance results in penalty of $10,000 per violation.
```

Run: `python run_v2_analysis.py --bill-id TEST123 --bill-text-file test_bill.txt --output test_output.json`

Expected: Analysis runs, displays tables, saves JSON

**Step 3: Update README**

Add to `/Users/thamac/Documents/NRG/README.md`:

```markdown
## Architecture v2.0 Usage

Run two-tier analysis:

```bash
python run_v2_analysis.py \
  --bill-id HB123 \
  --bill-text-file path/to/bill.txt \
  --output analysis_result.json
```

Output includes:
- Primary analyst findings with quotes
- Judge validations (quote verification, hallucination detection)
- Rubric scores (legal risk, financial impact)
```

**Step 4: Commit**

```bash
git add run_v2_analysis.py README.md test_bill.txt
git commit -m "feat(v2): add CLI entry point for two-tier analysis"
```

---

## Summary

**Phase 1 Complete:** Core two-tier system implemented with:

✅ Task 1: V2 data models (Pydantic validation)  
✅ Task 2: Supervisor router (complexity scoring, routing)  
✅ Task 3: Primary analyst (GPT-4o, findings extraction)  
✅ Task 4: Judge model (quote verification, hallucination detection)  
✅ Task 5: Rubric scoring (legal risk, financial impact)  
✅ Task 6: Two-tier orchestrator (full pipeline)  
✅ Task 7: CLI entry point

**Next Steps:**
- Phase 2: Self-consistency sampling, fallback model, sequential evolution
- Phase 3: Evaluation pipeline (silver set, LLM-judge ensemble)
- Phase 4: Deep research integration

**Verification:**
```bash
# Run all tests
pytest tests/test_v2/ -v

# Run end-to-end
python run_v2_analysis.py --bill-id TEST --bill-text-file test_bill.txt
```

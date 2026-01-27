"""
Two-Tier Validation Orchestrator - Validates findings from Sequential Evolution.

Architecture v2.0 - Phase 1

Design Decisions:
- Receives pre-extracted findings from Sequential Evolution (no duplicate extraction)
- Orchestrator is stateless - all state flows through function calls
- Hallucinated findings are validated but NOT scored on rubrics
- All 4 rubric dimensions scored for each validated finding

Pipeline Flow:
1. Receive findings_registry from Sequential Evolution (extraction already done)
2. Tier 1.5: Multi-sample check (conditional, for uncertain findings)
3. Tier 2: Judge validates each finding (quote verification, hallucination detection)
4. Tier 2.5: Fallback model (conditional, for high-impact uncertain findings)
5. Judge scores validated findings on rubrics
6. Results aggregated into TwoTierAnalysisResult

Why This Structure:
- Sequential Evolution handles extraction, Two-Tier handles validation only
- Filtering at validation: hallucinations don't waste rubric scoring
- Complete audit trail: every finding has validation + scores
"""
from typing import List, Dict, Any
import logging
import os
from nrg_core.v2.config import DEFAULT_CONFIG
from nrg_core.v2.judge import JudgeModel
from nrg_core.v2.rubrics import ALL_RUBRICS
from nrg_core.v2.multi_sample import MultiSampleChecker, ConsensusResult
from nrg_core.v2.fallback import FallbackAnalyst
from nrg_core.v2.audit_trail import AuditTrailGenerator
from nrg_core.v2.deep_research import DeepResearchAgent
from nrg_core.v2.cross_bill import ReferenceDetector
from nrg_core.models_v2 import TwoTierAnalysisResult, RubricScore, ResearchInsight, FallbackResult, PrimaryAnalysis, Finding, Quote


class TwoTierOrchestrator:
    """
    Orchestrates two-tier validation pipeline.
    
    Pipeline (receives findings from Sequential Evolution):
    1. Tier 1.5 → Multi-sample check (conditional)
    2. Tier 2 → Judge validates each finding
    3. Tier 2.5 → Fallback model (conditional)
    4. Judge → scores validated findings on rubrics
    
    Usage:
        orchestrator = TwoTierOrchestrator(
            judge_api_key=os.getenv("OPENAI_API_KEY")
        )
        result = orchestrator.validate(
            bill_id=bill_id,
            bill_text=bill_text,
            nrg_context=nrg_context,
            findings_registry=evolution_result.findings_registry,
            stability_scores=evolution_result.stability_scores
        )
    """
    
    def __init__(
        self,
        judge_model: str = "gpt-4o",
        judge_api_key: str | None = None,
        fallback_api_key: str | None = None,
        enable_multi_sample: bool = True,
        enable_fallback: bool = True,
        enable_deep_research: bool = False,
        enable_cross_bill_refs: bool = False,
        # Dependency injection - pass None to use defaults
        judge: JudgeModel | None = None,
        multi_sample: MultiSampleChecker | None = None,
        fallback: FallbackAnalyst | None = None,
        audit_generator: AuditTrailGenerator | None = None,
        research_agent: DeepResearchAgent | None = None,
        reference_detector: ReferenceDetector | None = None
    ) -> None:
        """
        Initialize orchestrator with optional dependency injection for testing.

        Pass pre-instantiated objects to override defaults (useful for mocks).

        Args:
            judge_model: Model for judge (default: gpt-4o)
            judge_api_key: API key for judge and optional components
            fallback_api_key: Anthropic API key for fallback (Claude). If None, fallback is disabled.
            enable_multi_sample: Enable multi-sample consistency check
            enable_fallback: Enable fallback second model
            enable_deep_research: Enable deep research for external context (Phase 4)
            enable_cross_bill_refs: Enable cross-bill reference detection (Phase 4)
            judge: Pre-instantiated JudgeModel (for testing/mocking)
            multi_sample: Pre-instantiated MultiSampleChecker (for testing/mocking)
            fallback: Pre-instantiated FallbackAnalyst (for testing/mocking)
            audit_generator: Pre-instantiated AuditTrailGenerator (for testing/mocking)
            research_agent: Pre-instantiated DeepResearchAgent (for testing/mocking)
            reference_detector: Pre-instantiated ReferenceDetector (for testing/mocking)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Two-Tier Orchestrator initialized for model={judge_model}")

        self.judge = judge or JudgeModel(
            model=judge_model,
            api_key=judge_api_key
        )
        self.multi_sample = multi_sample if multi_sample is not None else (
            MultiSampleChecker(model=judge_model, api_key=judge_api_key) if enable_multi_sample else None
        )

        # Fallback uses Anthropic Claude API - requires separate API key
        # Gracefully disable if no Anthropic key available
        if fallback is not None:
            self.fallback = fallback
        elif enable_fallback and fallback_api_key:
            self.fallback = FallbackAnalyst(api_key=fallback_api_key)
        else:
            if enable_fallback and not fallback_api_key:
                self.logger.warning("Fallback disabled: ANTHROPIC_API_KEY not set. Two-tier validation will proceed without Claude second opinion.")
            self.fallback = None
        self.audit_generator = audit_generator or AuditTrailGenerator()

        # Phase 4: Deep research agent for external context
        # Uses environment variables for API keys (OpenStates, BillTrack50)
        self.research_agent = research_agent if research_agent is not None else (
            DeepResearchAgent(
                openstates_key=os.getenv("OPENSTATES_API_KEY"),
                billtrack_key=os.getenv("BILLTRACK50_API_KEY"),
                openai_key=judge_api_key
            ) if enable_deep_research else None
        )

        # Phase 4: Cross-bill reference detector
        self.reference_detector = reference_detector if reference_detector is not None else (
            ReferenceDetector() if enable_cross_bill_refs else None
        )
    
    def validate(
        self,
        bill_id: str,
        bill_text: str,
        nrg_context: str,
        findings_registry: Dict[str, Dict[str, Any]],
        stability_scores: Dict[str, float]
    ) -> TwoTierAnalysisResult:
        """
        Validate findings from Sequential Evolution.
        
        Process:
        1. Convert findings_registry to Finding objects
        2. Multi-sample check (conditional)
        3. Judge validates each finding
        4. Fallback model (conditional)
        5. Score validated findings on rubrics
        
        Args:
            bill_id: Bill identifier (e.g., "HB123")
            bill_text: Full bill text (latest version)
            nrg_context: NRG business context for relevance
            findings_registry: From Sequential Evolution - {id: {statement, origin_version, ...}}
            stability_scores: From Sequential Evolution - {id: score}
        
        Returns:
            TwoTierAnalysisResult with validations and scores
        """
        # Convert findings_registry to Finding objects
        findings = self._convert_findings(findings_registry, stability_scores)
        
        # Build PrimaryAnalysis wrapper for compatibility
        primary_analysis = PrimaryAnalysis(
            bill_id=bill_id,
            findings=findings,
            overall_confidence=sum(f.confidence for f in findings) / len(findings) if findings else 0.0,
            requires_multi_sample=any(
                f.impact_estimate >= DEFAULT_CONFIG.MULTI_SAMPLE_IMPACT_THRESHOLD or f.confidence < DEFAULT_CONFIG.MULTI_SAMPLE_LOW_CONFIDENCE
                for f in findings
            )
        )
        
        # Tier 1.5: Multi-sample check (conditional)
        # Spec: Run 2-3x with different prompts, compare outputs for agreement
        multi_sample_agreement: float | None = None
        if self.multi_sample and primary_analysis.requires_multi_sample:
            consensus_result = self.multi_sample.check_consistency(
                bill_id=bill_id,
                bill_text=bill_text,
                nrg_context=nrg_context
            )
            multi_sample_agreement = consensus_result.consistency_score
            # Use consensus findings if available - reconstruct to trigger validation
            if consensus_result.consensus_findings:
                primary_analysis = PrimaryAnalysis(
                    bill_id=primary_analysis.bill_id,
                    findings=consensus_result.consensus_findings,
                    overall_confidence=sum(f.confidence for f in consensus_result.consensus_findings) / len(consensus_result.consensus_findings) if consensus_result.consensus_findings else 0.0,
                    requires_multi_sample=primary_analysis.requires_multi_sample
                )
        
        # Tier 2: Judge validates each finding
        judge_validations = []
        for idx, finding in enumerate(primary_analysis.findings):
            validation = self.judge.validate(
                finding_id=idx,
                finding=finding,
                bill_text=bill_text
            )
            judge_validations.append(validation)
        
        # Tier 2.5: Fallback second model (conditional)
        # Spec: Different provider (Claude) for architectural diversity
        second_model_reviewed = False
        fallback_results: List[FallbackResult] = []
        if self.fallback:
            for idx, finding in enumerate(primary_analysis.findings):
                validation = judge_validations[idx]

                # Trigger: uncertain judge + high impact
                # Spec (Architecture_v2.md:462-463): impact >= 7 (not 6)
                if DEFAULT_CONFIG.JUDGE_LOW_CONFIDENCE <= validation.judge_confidence <= DEFAULT_CONFIG.JUDGE_HIGH_CONFIDENCE and finding.impact_estimate >= DEFAULT_CONFIG.FALLBACK_IMPACT_THRESHOLD:
                    second_opinion = self.fallback.get_second_opinion(finding, bill_text)
                    second_model_reviewed = True

                    # Store the fallback result for audit trail and downstream use
                    fallback_results.append(FallbackResult(
                        finding_index=idx,
                        agrees=second_opinion.agrees,
                        alternative_interpretation=second_opinion.alternative_interpretation,
                        rationale=second_opinion.rationale,
                        confidence=second_opinion.confidence
                    ))
        
        # Tier 2: Rubric scoring (only for validated findings without hallucinations)
        # Why skip hallucinations: Scoring false claims wastes cost and pollutes results
        rubric_scores: List[RubricScore] = []
        for idx, finding in enumerate(primary_analysis.findings):
            validation = judge_validations[idx]
            
            # Skip scoring if hallucination detected
            if validation.hallucination_detected:
                continue
            
            # Score on all 4 dimensions (Phase 2: added operational_disruption, ambiguity_risk)
            for dimension, rubric in ALL_RUBRICS.items():
                score = self.judge.score_rubric(
                    dimension=dimension,
                    finding=finding,
                    bill_text=bill_text,
                    nrg_context=nrg_context,
                    rubric_anchors=rubric
                )
                rubric_scores.append(score)
        
        # Phase 2: Generate audit trails for compliance documentation
        # Why batch generation: Efficient handling of multiple findings with their scores
        # Filter to only non-hallucinated findings (they have rubric scores)
        non_hallucinated = [(f, v) for f, v in zip(primary_analysis.findings, judge_validations)
                            if not v.hallucination_detected]
        audit_trails = self.audit_generator.generate_batch(
            findings=[f for f, _ in non_hallucinated],
            validations=[v for _, v in non_hallucinated],
            all_rubric_scores=rubric_scores,
            dimensions_per_finding=len(ALL_RUBRICS)  # 4 dimensions
        )
        
        # Phase 4: Deep research for external context (optional)
        # Why per-finding: Each finding may need different external context
        research_insights: List[ResearchInsight] = []
        if self.research_agent:
            for finding in findings:
                research = self.research_agent.research(
                    finding={
                        "statement": finding.statement,
                        "quotes": [{"text": q.text, "section": q.section} for q in finding.quotes]
                    },
                    bill_text=bill_text
                )
                # Convert top 3 sources to ResearchInsight models
                for source in research.sources[:3]:
                    insight = ResearchInsight(
                        claim=research.summary,
                        source_url=source.url,
                        snippet=source.snippet,
                        relevance=source.relevance,
                        checker_validated=source.checker_validated or False,
                        trust=research.research_confidence
                    )
                    research_insights.append(insight)
        
        # Phase 4: Cross-bill reference detection (optional)
        # Why separate from research: References are bill-level, not finding-level
        cross_bill_refs = {}
        if self.reference_detector:
            detected = self.reference_detector.detect(bill_text)
            cross_bill_refs = {
                "detected": [
                    {"citation": r.citation, "type": r.reference_type, "location": r.location}
                    for r in detected
                ],
                "count": len(detected)
            }
        
        return TwoTierAnalysisResult(
            bill_id=bill_id,
            primary_analysis=primary_analysis,
            judge_validations=judge_validations,
            rubric_scores=rubric_scores,
            audit_trails=audit_trails,
            multi_sample_agreement=multi_sample_agreement,
            second_model_reviewed=second_model_reviewed,
            fallback_results=fallback_results,
            research_insights=research_insights,
            cross_bill_references=cross_bill_refs,
            route="ENHANCED",  # Phase 1 uses enhanced path only
            cost_estimate=0.0  # Placeholder - actual cost tracked at runtime
        )
    
    def _convert_findings(
        self,
        findings_registry: Dict[str, Dict[str, Any]],
        stability_scores: Dict[str, float]
    ) -> List[Finding]:
        """
        Convert findings_registry from Sequential Evolution to Finding objects.
        
        Args:
            findings_registry: {id: {statement, origin_version, modification_count, ...}}
            stability_scores: {id: float}
        
        Returns:
            List of Finding objects for validation
        """
        findings = []
        for finding_id, data in findings_registry.items():
            # Extract quotes if present, or create placeholder
            quotes_data = data.get("quotes", [])
            if not quotes_data:
                logging.getLogger(__name__).warning(f"Finding {finding_id} has no quotes - skipping")
                continue  # Skip this finding entirely
            
            quotes = [Quote(**q) if isinstance(q, dict) else q for q in quotes_data]
            
            # Use stability score as confidence proxy
            stability = stability_scores.get(finding_id, 0.5)
            
            finding = Finding(
                finding_id=finding_id,  # Preserve ID from Sequential Evolution for traceability
                statement=data.get("statement", ""),
                quotes=quotes,
                confidence=stability,
                impact_estimate=data.get("impact_estimate", 5)
            )
            findings.append(finding)
        
        return findings

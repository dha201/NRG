# nrg_core/v2/two_tier.py
"""
Two-Tier Analysis Orchestrator - Coordinates Primary Analyst → Judge → Rubric Scoring.

Architecture v2.0 - Phase 1

Design Decisions:
- Orchestrator is stateless - all state flows through function calls
- Hallucinated findings are validated but NOT scored on rubrics
- Both rubric dimensions scored for each validated finding
- Cost tracking placeholder for future optimization

Pipeline Flow:
1. Primary Analyst extracts findings with quotes
2. Judge validates each finding (quote verification, hallucination detection)
3. Judge scores validated findings on rubrics (legal_risk, financial_impact)
4. Results aggregated into TwoTierAnalysisResult

Why This Structure:
- Clear separation: extraction vs validation vs scoring
- Filtering at validation: hallucinations don't waste rubric scoring cost
- Complete audit trail: every finding has validation + scores
"""
from typing import List
import os
from nrg_core.v2.primary_analyst import PrimaryAnalyst
from nrg_core.v2.judge import JudgeModel
from nrg_core.v2.rubrics import ALL_RUBRICS
from nrg_core.v2.multi_sample import MultiSampleChecker, ConsensusResult
from nrg_core.v2.fallback import FallbackAnalyst
from nrg_core.v2.audit_trail import AuditTrailGenerator
from nrg_core.v2.deep_research import DeepResearchAgent
from nrg_core.v2.cross_bill import ReferenceDetector
from nrg_core.models_v2 import TwoTierAnalysisResult, RubricScore, ResearchInsight


class TwoTierOrchestrator:
    """
    Orchestrates two-tier analysis pipeline.
    
    Pipeline:
    1. Primary Analyst → extracts findings with quotes
    2. Judge → validates each finding (quote verification, hallucination)
    3. Judge → scores validated findings on rubrics
    
    Usage:
        orchestrator = TwoTierOrchestrator(
            primary_api_key=os.getenv("OPENAI_API_KEY"),
            judge_api_key=os.getenv("OPENAI_API_KEY")
        )
        result = orchestrator.analyze(bill_id, bill_text, nrg_context)
    """
    
    def __init__(
        self,
        primary_model: str = "gpt-4o",
        judge_model: str = "gpt-4o",
        primary_api_key: str = None,
        judge_api_key: str = None,
        enable_multi_sample: bool = True,
        enable_fallback: bool = True,
        enable_deep_research: bool = False,
        enable_cross_bill_refs: bool = False
    ):
        """
        Initialize orchestrator with analyst and judge.
        
        Args:
            primary_model: Model for primary analyst (default: gpt-4o)
            judge_model: Model for judge (default: gpt-4o)
            primary_api_key: API key for primary analyst
            judge_api_key: API key for judge (can be same as primary)
            enable_multi_sample: Enable multi-sample consistency check
            enable_fallback: Enable fallback second model
            enable_deep_research: Enable deep research for external context (Phase 4)
            enable_cross_bill_refs: Enable cross-bill reference detection (Phase 4)
        """
        self.primary_analyst = PrimaryAnalyst(
            model=primary_model,
            api_key=primary_api_key
        )
        self.judge = JudgeModel(
            model=judge_model,
            api_key=judge_api_key
        )
        self.multi_sample = MultiSampleChecker(model=primary_model, api_key=primary_api_key) if enable_multi_sample else None
        self.fallback = FallbackAnalyst(api_key=primary_api_key) if enable_fallback else None
        self.audit_generator = AuditTrailGenerator()
        
        # Phase 4: Deep research agent for external context
        # Uses environment variables for API keys (OpenStates, BillTrack50)
        self.research_agent = DeepResearchAgent(
            openstates_key=os.getenv("OPENSTATES_API_KEY"),
            billtrack_key=os.getenv("BILLTRACK50_API_KEY"),
            openai_key=primary_api_key
        ) if enable_deep_research else None
        
        # Phase 4: Cross-bill reference detector
        self.reference_detector = ReferenceDetector() if enable_cross_bill_refs else None
    
    def analyze(
        self,
        bill_id: str,
        bill_text: str,
        nrg_context: str
    ) -> TwoTierAnalysisResult:
        """
        Run full two-tier analysis pipeline.
        
        Process:
        1. Primary analyst extracts findings
        2. Judge validates each finding
        3. Judge scores validated findings on rubrics
        
        Args:
            bill_id: Bill identifier (e.g., "HB123")
            bill_text: Full bill text
            nrg_context: NRG business context for relevance
        
        Returns:
            TwoTierAnalysisResult with analysis, validations, scores
        """
        # Tier 1: Primary Analyst extracts findings
        primary_analysis = self.primary_analyst.analyze(
            bill_id=bill_id,
            bill_text=bill_text,
            nrg_context=nrg_context
        )
        
        # Tier 1.5: Multi-sample check (conditional)
        consensus_result = None
        if self.multi_sample and primary_analysis.requires_multi_sample:
            consensus_result = self.multi_sample.check_consistency(
                bill_id=bill_id,
                bill_text=bill_text,
                nrg_context=nrg_context
            )
            # Use consensus findings if available
            if consensus_result.consensus_findings:
                primary_analysis.findings = consensus_result.consensus_findings
        
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
        second_opinions = []
        if self.fallback:
            for idx, finding in enumerate(primary_analysis.findings):
                validation = judge_validations[idx]
                
                # Trigger: uncertain judge + high impact
                if 0.6 <= validation.judge_confidence <= 0.8 and finding.impact_estimate >= 6:
                    opinion = self.fallback.get_second_opinion(finding, bill_text)
                    second_opinions.append((idx, opinion))
        
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
        audit_trails = self.audit_generator.generate_batch(
            findings=primary_analysis.findings,
            validations=judge_validations,
            all_rubric_scores=rubric_scores,
            dimensions_per_finding=len(ALL_RUBRICS)  # 4 dimensions
        )
        
        # Phase 4: Deep research for external context (optional)
        # Why per-finding: Each finding may need different external context
        research_insights: List[ResearchInsight] = []
        if self.research_agent:
            for finding in primary_analysis.findings:
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
            research_insights=research_insights,
            cross_bill_references=cross_bill_refs,
            route="ENHANCED",  # Phase 1 uses enhanced path only
            cost_estimate=0.0  # TODO: Track actual API costs in Phase 2
        )

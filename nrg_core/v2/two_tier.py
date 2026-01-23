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
from nrg_core.v2.primary_analyst import PrimaryAnalyst
from nrg_core.v2.judge import JudgeModel
from nrg_core.v2.rubrics import LEGAL_RISK_RUBRIC, FINANCIAL_IMPACT_RUBRIC
from nrg_core.v2.multi_sample import MultiSampleChecker, ConsensusResult
from nrg_core.models_v2 import TwoTierAnalysisResult, RubricScore


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
        enable_multi_sample: bool = True
    ):
        """
        Initialize orchestrator with analyst and judge.
        
        Args:
            primary_model: Model for primary analyst (default: gpt-4o)
            judge_model: Model for judge (default: gpt-4o)
            primary_api_key: API key for primary analyst
            judge_api_key: API key for judge (can be same as primary)
            enable_multi_sample: Enable multi-sample consistency check
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
        
        # Tier 2: Rubric scoring (only for validated findings without hallucinations)
        # Why skip hallucinations: Scoring false claims wastes cost and pollutes results
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
            cost_estimate=0.0  # TODO: Track actual API costs in Phase 2
        )

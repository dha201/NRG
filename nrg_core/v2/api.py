"""
Unified API for NRG Legislative Analysis Pipeline.

Provides programmatic entry points for agent orchestration.

Design Philosophy:
- No file I/O required - all inputs/outputs are in-memory
- All configuration via parameters - no environment assumptions
- Returns typed objects directly - no JSON parsing needed
- Composable stages - can run full pipeline or individual steps

Usage:
    from nrg_core.v2 import analyze_bill, validate_findings

    # Full pipeline (Sequential Evolution + Two-Tier Validation)
    result = analyze_bill(
        bill_id="HB123",
        bill_text="...",
        nrg_context="...",
    )

    # Validation only (when extraction is already done)
    result = validate_findings(
        bill_id="HB123",
        bill_text="...",
        nrg_context="...",
        findings_registry={...},
        stability_scores={...},
    )
"""
from typing import Dict, Any, List
from nrg_core.v2.sequential_evolution import SequentialEvolutionAgent, BillVersion, EvolutionResult
from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.models_v2 import TwoTierAnalysisResult


def analyze_bill(
    bill_id: str,
    bill_text: str,
    nrg_context: str,
    versions: List[BillVersion] | None = None,
    api_key: str | None = None,
    enable_multi_sample: bool = True,
    enable_fallback: bool = True,
) -> TwoTierAnalysisResult:
    """
    Full pipeline entry point for agent orchestration.

    Runs Sequential Evolution (extraction) + Two-Tier Validation.

    Pipeline Stages:
    1. Sequential Evolution: Walk bill versions, extract findings with stability scores
    2. Two-Tier Validation: Validate findings, score on rubrics, generate audit trails

    Args:
        bill_id: Bill identifier (e.g., "HB123")
        bill_text: Full bill text (latest version)
        nrg_context: NRG business context for relevance assessment
        versions: List of bill versions (if None, uses single current version)
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        enable_multi_sample: Enable multi-sample consistency check (Tier 1.5)
        enable_fallback: Enable fallback second model (Tier 2.5)

    Returns:
        TwoTierAnalysisResult with:
        - primary_analysis: Extracted findings
        - judge_validations: Validation results per finding
        - rubric_scores: Scored dimensions (legal_risk, financial_impact, etc.)
        - audit_trails: Compliance documentation
        - multi_sample_agreement: Consistency score if multi-sample enabled
        - second_model_reviewed: Whether fallback was triggered

    Example:
        result = analyze_bill(
            bill_id="HB123",
            bill_text=open("bill.txt").read(),
            nrg_context="NRG operates gas distribution in Texas...",
        )

        # Access findings
        for finding in result.primary_analysis.findings:
            print(f"Finding: {finding.statement}")

        # Access rubric scores
        for score in result.rubric_scores:
            print(f"{score.dimension}: {score.score}/10")
    """
    import os
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    # Stage 1: Sequential Evolution (extraction)
    evolution_agent = SequentialEvolutionAgent(api_key=api_key)
    if versions is None:
        versions = [BillVersion(version_number=1, text=bill_text, name="Current")]
    evolution_result = evolution_agent.walk_versions(bill_id=bill_id, versions=versions)

    # Stage 2: Two-Tier Validation
    orchestrator = TwoTierOrchestrator(
        judge_api_key=api_key,
        enable_multi_sample=enable_multi_sample,
        enable_fallback=enable_fallback
    )
    return orchestrator.validate(
        bill_id=bill_id,
        bill_text=bill_text,
        nrg_context=nrg_context,
        findings_registry=evolution_result.findings_registry,
        stability_scores=evolution_result.stability_scores
    )


def validate_findings(
    bill_id: str,
    bill_text: str,
    nrg_context: str,
    findings_registry: Dict[str, Dict[str, Any]],
    stability_scores: Dict[str, float],
    api_key: str | None = None,
) -> TwoTierAnalysisResult:
    """
    Validation-only entry point (when extraction is already done).

    Use this when you have pre-extracted findings from Sequential Evolution
    or another extraction source and only need validation + scoring.

    Args:
        bill_id: Bill identifier
        bill_text: Full bill text (latest version)
        nrg_context: NRG business context for relevance assessment
        findings_registry: Pre-extracted findings from Sequential Evolution
            Format: {id: {statement, origin_version, modification_count, ...}}
        stability_scores: Stability scores from Sequential Evolution
            Format: {id: score} where score is 0-1
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)

    Returns:
        TwoTierAnalysisResult with validations and scores

    Example:
        # After running Sequential Evolution separately
        evolution_result = evolution_agent.walk_versions(...)

        result = validate_findings(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context="NRG operates gas distribution...",
            findings_registry=evolution_result.findings_registry,
            stability_scores=evolution_result.stability_scores,
        )
    """
    import os
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    orchestrator = TwoTierOrchestrator(judge_api_key=api_key)
    return orchestrator.validate(
        bill_id=bill_id,
        bill_text=bill_text,
        nrg_context=nrg_context,
        findings_registry=findings_registry,
        stability_scores=stability_scores
    )

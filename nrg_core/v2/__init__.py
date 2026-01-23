# nrg_core/v2/__init__.py
"""
Architecture v2.0 components.

This package contains the two-tier validation system:
- SequentialEvolutionAgent: Stage 1 finding extraction with version tracking
- TwoTierOrchestrator: Stage 2 validation and rubric scoring
- JudgeModel: Finding validation and rubric scoring

Agent Orchestration API:
- analyze_bill: Full pipeline entry point (Sequential Evolution + Two-Tier Validation)
- validate_findings: Validation-only entry point (when extraction is already done)
"""
from nrg_core.v2.api import analyze_bill, validate_findings

__all__ = ["analyze_bill", "validate_findings"]

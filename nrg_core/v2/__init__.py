# nrg_core/v2/__init__.py
"""
Architecture v2.0 components.

This package contains the two-tier analysis system:
- SupervisorRouter: Code-based complexity assessment and routing
- PrimaryAnalyst: Tier 1 LLM-based finding extraction
- JudgeModel: Tier 2 validation and rubric scoring
- TwoTierOrchestrator: Pipeline coordinator
"""
from nrg_core.v2.supervisor import SupervisorRouter, Route

__all__ = ["SupervisorRouter", "Route"]

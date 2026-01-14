# NRG Legislative Tracker Core Library
# Main entry point: from nrg_core.orchestrator import run_analysis

from .config import load_config, load_nrg_context
from .orchestrator import run_analysis

from .models import (
    Bill,
    BillVersion,
    Analysis,
    ChangeData,
    AnalysisResult,
    VERSION_NORMALIZATION,
    normalize_version_type,
)

from .utils import (
    CostTracker,
    LLMUsage,
    log_llm_cost,
    get_cost_summary,
    reset_cost_tracker,
    estimate_tokens,
)

__all__ = [
    # Main entry point
    "run_analysis",
    "load_config",
    "load_nrg_context",
    # Models
    "Bill",
    "BillVersion", 
    "Analysis",
    "ChangeData",
    "AnalysisResult",
    "VERSION_NORMALIZATION",
    "normalize_version_type",
    # Utils
    "CostTracker",
    "LLMUsage",
    "log_llm_cost",
    "get_cost_summary",
    "reset_cost_tracker",
    "estimate_tokens",
]

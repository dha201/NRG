from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# =============================================================================
# VERSION TERMINOLOGY NORMALIZATION
# =============================================================================
#
# Federal bills (Congress.gov) and state bills (Open States) use different
# terminology for the same legislative stages:
#
#   Federal (Congress.gov)     State (Open States/Texas)
#   ----------------------     -------------------------
#   IH (Introduced House)      "Introduced"
#   EH (Engrossed House)       "Engrossed"  
#   ENR (Enrolled)             "Enrolled"
#
# Without normalization, reports show inconsistent labels:
#   - Federal: "Version IH --> Version EH --> Version ENR"
#   - State:   "Version Introduced --> Version Engrossed --> Version Enrolled"
#
# This breaks:
#   1. Cross-bill comparisons (can't filter by stage across sources)
#   2. Analytics/dashboards (grouping by stage requires matching both terms)
#   3. Report readability (users see different terms for same concept)
#
# Normalize state terms to federal codes as the canonical format.
# =============================================================================

# Map state terminology (Open States) to federal codes (Congress.gov)
# Federal codes: IH/IS (Introduced), RH/RS (Reported), EH/ES (Engrossed), ENR (Enrolled)
VERSION_TYPE_MAP: dict[str, str] = {
    "Introduced": "IH",       # Initial filing - same as federal "Introduced in House"
    "As Filed": "IH",         # Texas terminology for initial filing
    "Filed": "IH",            # Alternative state terminology
    "Committee Report": "RH", # After committee review - maps to "Reported in House"
    "House Committee Report": "RH",
    "Senate Committee Report": "RS",
    "Reported": "RH",         # Generic "reported out of committee"
    "Engrossed": "EH",        # Passed originating chamber - "Engrossed in House"
    "Enrolled": "ENR",        # Final version passed both chambers
    "Chaptered": "ENR",       # California terminology for enacted law
    "Signed": "ENR",          # Governor signed - effectively enrolled
}

# Reverse map for display: federal code → human-readable label
# Used when generating reports to show consistent, readable labels
VERSION_DISPLAY_MAP: dict[str, str] = {
    "IH": "Introduced (House)",
    "IS": "Introduced (Senate)",
    "RH": "Reported (House)",
    "RS": "Reported (Senate)",
    "EH": "Engrossed (House)",
    "ES": "Engrossed (Senate)",
    "ENR": "Enrolled",
}


def normalize_version_type(raw_type: str, source: str = "Open States") -> str:
    """
    Normalize version terminology to federal codes for consistency.
    
    Args:
        raw_type: Raw version type from API (e.g., "Introduced", "IH")
        source: Data source ("Congress.gov" or "Open States")
        
    Returns:
        Normalized version code (e.g., "IH", "EH", "ENR")
    """
    if not raw_type:
        return "Unknown"
    
    # Federal bills already use codes
    if source == "Congress.gov":
        return raw_type
    
    # State bills need mapping
    return VERSION_TYPE_MAP.get(raw_type, raw_type)


def get_version_display_name(version_code: str) -> str:
    """
    Get human-readable display name for a version code.
    
    Args:
        version_code: Normalized version code (e.g., "IH", "ENR")
        
    Returns:
        Human-readable label (e.g., "Introduced (House)", "Enrolled")
    """
    return VERSION_DISPLAY_MAP.get(version_code, version_code)


# =============================================================================
# LLM COST TRACKING
# =============================================================================

LLM_COSTS = {
    "gpt-5": {"input": 0.01, "output": 0.03},
    "gemini-3-pro-preview": {"input": 0.00125, "output": 0.005},
}


@dataclass
class LLMUsage:
    """Track LLM usage per call."""
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def estimated_cost(self) -> float:
        # Place Holder, TODO: Update the following with accurate resources or find existing opensource lib (e.g, tiktoken)
        costs = LLM_COSTS.get(self.model, {"input": 0.01, "output": 0.03})
        input_cost = (self.input_tokens / 1000) * costs["input"]
        output_cost = (self.output_tokens / 1000) * costs["output"]
        return input_cost + output_cost


@dataclass 
class CostTracker:
    """Accumulate LLM costs across a session."""
    usages: list = field(default_factory=list)
    
    def add(self, model: str, input_tokens: int, output_tokens: int) -> LLMUsage:
        usage = LLMUsage(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        self.usages.append(usage)
        return usage
    
    @property
    def total_cost(self) -> float:
        return sum(u.estimated_cost for u in self.usages)
    
    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usages)
    
    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usages)
    
    def summary(self) -> dict:
        return {
            "total_calls": len(self.usages),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "by_model": self._by_model()
        }
    
    def _by_model(self) -> dict:
        models = {}
        for u in self.usages:
            if u.model not in models:
                models[u.model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0}
            models[u.model]["calls"] += 1
            models[u.model]["input_tokens"] += u.input_tokens
            models[u.model]["output_tokens"] += u.output_tokens
            models[u.model]["cost"] += u.estimated_cost
        return models


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimation (4 chars ≈ 1 token for English).
    """
    if not text:
        return 0
    return len(text) // 4


# Global cost tracker instance
cost_tracker = CostTracker()


def log_llm_cost(model: str, prompt: str, response_text: str) -> LLMUsage:
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(response_text)
    return cost_tracker.add(model, input_tokens, output_tokens)


def get_cost_summary() -> dict:
    return cost_tracker.summary()


def reset_cost_tracker():
    global cost_tracker
    cost_tracker = CostTracker()

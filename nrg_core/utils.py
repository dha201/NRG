from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


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
        """Estimate cost in USD based on token counts."""
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
        """Record a new LLM call."""
        usage = LLMUsage(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        self.usages.append(usage)
        return usage
    
    @property
    def total_cost(self) -> float:
        """Total estimated cost for all tracked calls."""
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
        """Break down usage by model."""
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

"""
Configuration constants for the Two-Tier Validation Pipeline.

All thresholds and magic numbers are centralized here for easy tuning.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdConfig:
    """Configurable thresholds for the validation pipeline."""

    # Judge confidence thresholds
    JUDGE_LOW_CONFIDENCE: float = 0.6
    JUDGE_HIGH_CONFIDENCE: float = 0.8

    # Impact thresholds
    FALLBACK_IMPACT_THRESHOLD: int = 7
    MULTI_SAMPLE_IMPACT_THRESHOLD: int = 6

    # Multi-sample check
    MULTI_SAMPLE_LOW_CONFIDENCE: float = 0.7

    # Stability scoring
    STABILITY_HIGH: float = 0.95
    STABILITY_MEDIUM: float = 0.85
    STABILITY_LOW: float = 0.40
    STABILITY_LAST_MINUTE: float = 0.20


# Default configuration instance
DEFAULT_CONFIG = ThresholdConfig()

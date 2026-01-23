"""
Custom exceptions for the NRG v2 analysis pipeline.

These exceptions provide actionable error information for debugging and monitoring.
"""


class NRGAnalysisError(Exception):
    """Base exception for NRG analysis errors."""
    pass


class LLMResponseError(NRGAnalysisError):
    """LLM returned an invalid or unexpected response."""
    pass


class ValidationError(NRGAnalysisError):
    """Validation of input data failed."""
    pass


class QuoteMissingError(NRGAnalysisError):
    """Finding is missing required quote evidence."""
    pass


class APIKeyMissingError(NRGAnalysisError):
    """Required API key is not configured."""
    pass

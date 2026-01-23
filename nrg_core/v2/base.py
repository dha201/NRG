"""
Base classes for LLM-powered agents in the NRG v2 pipeline.
"""
from openai import OpenAI
from nrg_core.v2.exceptions import APIKeyMissingError


class BaseLLMAgent:
    """Base class for agents that use OpenAI LLM."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        """
        Initialize LLM agent.

        Args:
            model: Model name (default: gpt-4o)
            api_key: OpenAI API key (uses env var if None)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None

    def _ensure_client(self) -> OpenAI:
        """Ensure OpenAI client is available, raise if not."""
        if not self.client:
            raise APIKeyMissingError(
                f"{self.__class__.__name__} requires an API key. "
                "Pass api_key to constructor or set OPENAI_API_KEY environment variable."
            )
        return self.client

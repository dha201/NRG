"""
Checker Agent for Research Claim Validation.

Validates that research claims are directly supported by source snippets,
flagging speculative leaps where claims extrapolate beyond evidence.

Design rationale:
- Separate validation from research to enable independent testing
- LLM-based validation captures nuanced semantic relationships
- Explicit "directly_states" flag distinguishes support from inference
- Confidence score enables threshold-based filtering

Why this matters:
- Research can retrieve relevant sources that don't actually support claims
- Speculative leaps undermine analysis credibility
- Audit trail requires clear validation status

Usage:
    checker = CheckerAgent(model="gpt-4o", api_key="...")
    result = checker.check(claim, snippet, source_url)
    if result.directly_states and result.confidence > 0.8:
        # Claim is well-supported
"""
import json
import logging
from typing import Dict, Any
from dataclasses import dataclass
from openai import OpenAI
from nrg_core.v2.exceptions import APIKeyMissingError, LLMResponseError

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """
    Result of claim validation against source.
    
    Attributes:
        directly_states: True only if source explicitly states the claim
        confidence: Checker's confidence in assessment (0-1)
        rationale: Explanation of validation decision (for audit trail)
    
    Design: directly_states is boolean (not probabilistic) because
    partial support is captured in confidence, while the flag
    answers "does source say this?" definitively.
    """
    directly_states: bool
    confidence: float
    rationale: str


# Prompt designed to catch speculative leaps
# Key instruction: "directly_states" = true ONLY if explicit
CHECKER_PROMPT = """You are validating a research claim against a source snippet.

CLAIM:
{claim}

SOURCE SNIPPET:
{snippet}

SOURCE URL: {url}

QUESTION: Does the source directly state this claim?

RULES:
- "directly_states" = true ONLY if source explicitly says what the claim says
- "directly_states" = false if claim extrapolates, generalizes, or infers beyond source
- Be strict: if source says "one company" and claim says "all companies", that's false
- Provide confidence 0-1 based on how clearly the source supports or contradicts
- Rationale must explain your reasoning

OUTPUT (JSON):
{{
  "directly_states": true/false,
  "confidence": 0.0-1.0,
  "rationale": "Explanation of your assessment"
}}"""


class CheckerAgent:
    """
    Validate research claims against source snippets.
    
    Uses LLM to assess whether a source directly supports a claim,
    or if the claim represents a speculative leap beyond the evidence.
    
    Validation criteria:
    - Explicit statement: Source must say what claim says
    - No extrapolation: "one case" doesn't support "all cases"
    - No inference: Implications don't count as direct support
    
    Why LLM-based validation:
    - Semantic matching catches paraphrasing
    - Nuanced reasoning handles complex claims
    - Rationale generation supports audit trail
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        """
        Initialize checker with LLM model.
        
        Args:
            model: OpenAI model to use (default: gpt-4o for reasoning quality)
            api_key: OpenAI API key
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def check(
        self,
        claim: str,
        snippet: str,
        source_url: str
    ) -> CheckResult:
        """
        Validate claim against source snippet.
        
        Process:
        1. Format prompt with claim, snippet, URL
        2. Call LLM for structured validation
        3. Parse response into CheckResult
        
        Args:
            claim: Research claim to validate
            snippet: Source snippet (max ~500 chars)
            source_url: URL for citation context
        
        Returns:
            CheckResult with validation status and rationale
        """
        result = self._call_llm(claim, snippet, source_url)
        
        return CheckResult(
            directly_states=result["directly_states"],
            confidence=result["confidence"],
            rationale=result["rationale"]
        )
    
    def _call_llm(self, claim: str, snippet: str, url: str) -> Dict[str, Any]:
        """
        Call LLM for claim validation.
        
        Uses JSON response format for reliable parsing.
        Low temperature (0.1) for consistent, deterministic validation.
        
        Args:
            claim: Claim text
            snippet: Source snippet
            url: Source URL
        
        Returns:
            Dict with directly_states, confidence, rationale
        
        Raises:
            ValueError: If client not initialized
        """
        if not self.client:
            raise APIKeyMissingError("OpenAI client not initialized - provide api_key")
        
        prompt = CHECKER_PROMPT.format(
            claim=claim,
            snippet=snippet,
            url=url
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1  # Low temp for consistent validation
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse checker response as JSON: {e}")
            raise LLMResponseError(f"LLM returned invalid JSON for claim validation: {e}") from e

    def check_batch(
        self,
        claims_and_snippets: list[tuple[str, str, str]]
    ) -> list[CheckResult]:
        """
        Validate multiple claims (convenience method).
        
        Args:
            claims_and_snippets: List of (claim, snippet, url) tuples
        
        Returns:
            List of CheckResult objects
        """
        return [
            self.check(claim, snippet, url)
            for claim, snippet, url in claims_and_snippets
        ]

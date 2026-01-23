# nrg_core/v2/judge.py
"""
Judge Model (Tier 2) - Validates findings, detects hallucinations, scores evidence quality.

Design Decisions:
- Separate from Primary Analyst to enable independent validation
- Lower temperature (0.1) than analyst for more deterministic validation
- Explicit hallucination detection to catch unsupported claims
- Evidence quality and ambiguity scores enable filtering/prioritization

Validation Checks:
1. quote_verified: Do all quoted texts exist VERBATIM in bill?
2. hallucination_detected: Does statement claim things not in quotes?
3. evidence_quality: How well do quotes support the statement?
4. ambiguity: How much interpretation is required?

Why This Matters:
- Primary analyst may hallucinate (claim things not in bill)
- Quotes may be paraphrased or slightly modified
- Statements may extrapolate beyond what quotes actually say
- Judge catches these issues before findings reach downstream consumers
"""
import json
import logging
import time
from functools import wraps
from typing import Dict, Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])
from openai import OpenAI
from nrg_core.models_v2 import Finding, JudgeValidation, RubricScore, Quote
from nrg_core.v2.rubrics import RUBRIC_SCORING_PROMPT, format_rubric_scale
from nrg_core.v2.exceptions import APIKeyMissingError, LLMResponseError
from nrg_core.v2.base import BaseLLMAgent

logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable[[F], F]:
    """Retry decorator for transient LLM errors."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_error
        return wrapper  # type: ignore[return-value]
    return decorator


def _sanitize_for_prompt(text: str) -> str:
    """Sanitize external text before including in LLM prompt.

    Removes or escapes patterns that could be interpreted as instructions.
    """
    if not text:
        return ""
    # Remove common injection patterns
    sanitized = text.replace("```", "'''")  # Prevent code block escapes
    sanitized = sanitized.replace("SYSTEM:", "[SYSTEM]")
    sanitized = sanitized.replace("USER:", "[USER]")
    sanitized = sanitized.replace("ASSISTANT:", "[ASSISTANT]")
    sanitized = sanitized.replace("INSTRUCTION:", "[INSTRUCTION]")
    sanitized = sanitized.replace("IGNORE", "[IGNORE]")
    # Limit length to prevent context overflow
    max_len = 50000
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "... [truncated]"
    return sanitized


# Prompt for validation - emphasizes verbatim matching and hallucination detection
JUDGE_VALIDATION_PROMPT = """You are a validation judge for legislative analysis.

BILL TEXT:
{bill_text}

FINDING TO VALIDATE:
Statement: {statement}
Quotes: {quotes}

TASK:
Validate this finding by checking:
1. Do all quoted texts exist VERBATIM in the bill? (exact match required)
2. Does the statement make claims NOT supported by the quotes?
3. How strong is the evidence (0-1)?
4. How ambiguous is the interpretation (0-1)?

OUTPUT (JSON):
{{
  "quote_verified": true/false,
  "hallucination_detected": true/false,
  "evidence_quality": 0.0-1.0,
  "ambiguity": 0.0-1.0,
  "judge_confidence": 0.0-1.0
}}

CRITICAL:
- quote_verified = false if ANY quote is not exact match
- hallucination_detected = true if statement claims things not in quotes
- evidence_quality = how well quotes support the statement
- ambiguity = how much interpretation is required
"""


class JudgeModel(BaseLLMAgent):
    """
    Tier 2: Judge - Validates findings from primary analyst.

    Responsibilities:
    - Verify quotes exist verbatim in bill text
    - Detect hallucinations (unsupported claims)
    - Score evidence quality and ambiguity
    - Score findings on rubric dimensions (Task 5)

    Usage:
        judge = JudgeModel(api_key=os.getenv("OPENAI_API_KEY"))
        validation = judge.validate(finding_id, finding, bill_text)
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        """
        Initialize judge with LLM configuration.

        Args:
            model: OpenAI model to use (default: gpt-4o)
            api_key: OpenAI API key (required for actual calls, optional for testing)
        """
        super().__init__(model=model, api_key=api_key)
    
    def validate(
        self,
        finding_id: int,
        finding: Finding,
        bill_text: str
    ) -> JudgeValidation:
        """
        Validate a single finding from primary analyst.
        
        Checks:
        - Do quotes exist verbatim in bill?
        - Does statement add claims not in quotes (hallucination)?
        - How strong is the evidence?
        - How much interpretation required?
        
        Args:
            finding_id: Index of finding in primary analysis
            finding: Finding to validate
            bill_text: Full bill text for verification
        
        Returns:
            JudgeValidation with verification results
        """
        # Call LLM for validation
        validation_output = self._call_llm(finding, bill_text)
        
        return JudgeValidation(
            finding_id=finding_id,
            quote_verified=validation_output["quote_verified"],
            hallucination_detected=validation_output["hallucination_detected"],
            evidence_quality=validation_output["evidence_quality"],
            ambiguity=validation_output["ambiguity"],
            judge_confidence=validation_output["judge_confidence"]
        )
    
    def score_rubric(
        self,
        dimension: str,
        finding: Finding,
        bill_text: str,
        nrg_context: str,
        rubric_anchors: Dict[str, str]
    ) -> RubricScore:
        """
        Score a finding on a rubric dimension.
        
        Uses LLM to score finding against defined rubric anchors.
        Returns structured RubricScore with full audit trail.
        
        Args:
            dimension: "legal_risk" or "financial_impact"
            finding: Finding to score
            bill_text: Full bill text
            nrg_context: NRG business context for impact calculation
            rubric_anchors: Scale definitions (e.g., LEGAL_RISK_RUBRIC)
        
        Returns:
            RubricScore with score, rationale, evidence, anchor
        """
        # Call LLM for rubric scoring
        score_output = self._call_llm_for_rubric(
            dimension=dimension,
            finding=finding,
            bill_text=bill_text,
            nrg_context=nrg_context,
            rubric_anchors=rubric_anchors
        )
        
        return RubricScore(
            dimension=score_output["dimension"],
            score=score_output["score"],
            rationale=score_output["rationale"],
            evidence=[Quote(**q) for q in score_output["evidence"]],
            rubric_anchor=score_output["rubric_anchor"]
        )
    
    @retry_on_error(max_retries=3, delay=1.0)
    def _call_llm_for_rubric(
        self,
        dimension: str,
        finding: Finding,
        bill_text: str,
        nrg_context: str,
        rubric_anchors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Call OpenAI for rubric scoring.
        
        Args:
            dimension: Rubric dimension to score
            finding: Finding to score
            bill_text: Full bill text
            nrg_context: NRG business context
            rubric_anchors: Scale definitions
            
        Returns:
            Parsed JSON with score, rationale, evidence, anchor
            
        Raises:
            ValueError: If client not initialized
        """
        if not self.client:
            raise APIKeyMissingError("OpenAI client not initialized - provide api_key")

        # Format quotes and rubric scale for prompt
        quotes_text = "\n".join([f"- {q.text} (Section {q.section})" for q in finding.quotes])
        rubric_scale = format_rubric_scale(rubric_anchors)

        # Sanitize external inputs to prevent prompt injection
        prompt = RUBRIC_SCORING_PROMPT.format(
            dimension=dimension,
            rubric_scale=rubric_scale,
            nrg_context=_sanitize_for_prompt(nrg_context),
            bill_text=_sanitize_for_prompt(bill_text[:5000]),  # Truncate to avoid token limits
            statement=_sanitize_for_prompt(finding.statement),
            quotes=_sanitize_for_prompt(quotes_text)
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1  # Low temperature for consistent scoring
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rubric scoring response as JSON: {e}")
            raise LLMResponseError(f"LLM returned invalid JSON for rubric scoring: {e}") from e

    @retry_on_error(max_retries=3, delay=1.0)
    def _call_llm(self, finding: Finding, bill_text: str) -> Dict[str, Any]:
        """
        Call OpenAI for validation.
        
        Args:
            finding: Finding to validate
            bill_text: Full bill text
            
        Returns:
            Parsed JSON with validation results
            
        Raises:
            ValueError: If client not initialized
        """
        if not self.client:
            raise APIKeyMissingError("OpenAI client not initialized - provide api_key")

        # Format quotes for prompt
        quotes_text = "\n".join([f"- {q.text} (Section {q.section})" for q in finding.quotes])

        # Sanitize external inputs to prevent prompt injection
        prompt = JUDGE_VALIDATION_PROMPT.format(
            bill_text=_sanitize_for_prompt(bill_text),
            statement=_sanitize_for_prompt(finding.statement),
            quotes=_sanitize_for_prompt(quotes_text)
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1  # Lower temperature for validation (more deterministic)
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation response as JSON: {e}")
            raise LLMResponseError(f"LLM returned invalid JSON for validation: {e}") from e

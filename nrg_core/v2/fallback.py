"""
Fallback Analyst - Second Model Opinion.

Uses different LLM provider (Claude) for architectural diversity.
Provides adversarial check on uncertain primary analyst findings.

Design:
- Triggered when judge_confidence in [0.6, 0.8] AND impact >= 7
- Uses Claude Opus for different model architecture
- Returns structured second opinion with agreement/alternative
- Low temperature (0.2) for consistent reasoning

Why:
- Different models have different biases and failure modes
- Architectural diversity reduces systematic errors
- Adversarial checking improves reliability for high-impact findings
"""
import json
import logging
from typing import Dict, Any
from dataclasses import dataclass
import anthropic
from nrg_core.models_v2 import Finding
from nrg_core.v2.exceptions import APIKeyMissingError, LLMResponseError

logger = logging.getLogger(__name__)


@dataclass
class SecondOpinion:
    """Result from fallback model.
    
    Attributes:
        agrees: Whether Claude agrees with primary analyst
        alternative_interpretation: Claude's alternative reading (empty if agrees)
        rationale: Explanation of agreement/disagreement
        confidence: Claude's confidence in its assessment (0-1)
    """
    agrees: bool
    alternative_interpretation: str
    rationale: str
    confidence: float


FALLBACK_PROMPT = """You are a second opinion validator for legislative analysis.

PRIMARY ANALYST'S FINDING:
{statement}

Supporting quotes:
{quotes}

BILL TEXT:
{bill_text}

QUESTION: Do you agree with this interpretation? If not, what's your alternative reading?

Respond in JSON:
{{
  "agrees": true/false,
  "alternative_interpretation": "Your interpretation (empty if agrees)",
  "rationale": "Explanation of why you agree/disagree",
  "confidence": 0.0-1.0
}}"""


class FallbackAnalyst:
    """
    Tier 2 Fallback: Different model for adversarial check.
    
    Triggered when judge_confidence in [0.6, 0.8] AND impact >= 7.
    Uses Claude Opus for architectural diversity from GPT models.
    
    Process:
    1. Receive primary analyst finding and supporting quotes
    2. Compare finding to full bill text
    3. Provide agreement or alternative interpretation
    4. Return structured second opinion with confidence
    """
    
    def __init__(self, model: str = "claude-opus-4", api_key: str | None = None):
        """
        Initialize fallback analyst with Claude.
        
        Args:
            model: Claude model to use (default: claude-opus-4)
            api_key: Anthropic API key
        """
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
    
    def get_second_opinion(
        self,
        finding: Finding,
        bill_text: str
    ) -> SecondOpinion:
        """
        Get alternative interpretation from different model.
        
        Method:
        1. Format prompt with finding, quotes, and bill text
        2. Call Claude with low temperature for consistency
        3. Parse JSON response
        4. Return structured second opinion
        
        Args:
            finding: Primary analyst's finding with quotes
            bill_text: Full bill text (truncated to 8000 chars for limits)
        
        Returns:
            SecondOpinion with agreement/alternative assessment
        """
        response = self._call_claude(finding, bill_text)
        
        return SecondOpinion(
            agrees=response["agrees"],
            alternative_interpretation=response["alternative_interpretation"],
            rationale=response["rationale"],
            confidence=response["confidence"]
        )
    
    def _call_claude(self, finding: Finding, bill_text: str) -> Dict[str, Any]:
        """
        Call Anthropic Claude API.
        
        Uses temperature=0.2 for consistent reasoning.
        Truncates bill text to stay within token limits.
        
        Args:
            finding: Primary analyst finding
            bill_text: Full bill text
        
        Returns:
            Parsed JSON response from Claude
        """
        if not self.client:
            raise APIKeyMissingError("Anthropic client not initialized - provide api_key")
        
        quotes_text = "\n".join([f"- {q.text} (Section {q.section})" for q in finding.quotes])
        
        prompt = FALLBACK_PROMPT.format(
            statement=finding.statement,
            quotes=quotes_text,
            bill_text=bill_text[:8000]  # Truncate for token limit
        )
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            return json.loads(message.content[0].text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse fallback analyst response as JSON: {e}")
            raise LLMResponseError(f"Claude returned invalid JSON for fallback analysis: {e}") from e

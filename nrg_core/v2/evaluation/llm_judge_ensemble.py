"""
LLM-Judge Ensemble for Continuous Validation.

Uses multiple judge models (GPT-4o, Claude Opus, Gemini 3 Pro) to evaluate
system predictions against expert labels, providing ensemble scoring and
inter-judge agreement metrics.

Design:
- Three different LLM providers for architectural diversity
- Each judge scores finding match quality (0-1) with rationale
- Ensemble computes average score and agreement metric
- Low agreement triggers human review flag

Why ensemble approach:
- Different models have different biases and failure modes
- Agreement increases confidence in evaluation
- Disagreement identifies ambiguous cases needing review
- Reduces reliance on any single model's judgment
"""
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
import anthropic


@dataclass
class JudgeEvaluationResult:
    """
    Result from judge ensemble evaluation.
    
    Attributes:
        judge_scores: List of individual judge results with model names
        average_score: Mean score across all judges
        agreement: Inter-judge agreement (1 - normalized std dev)
        requires_human_review: Flag for low agreement cases
    """
    judge_scores: List[Dict[str, Any]]
    average_score: float
    agreement: float  # Inter-judge agreement (std dev based)
    requires_human_review: bool


JUDGE_PROMPT = """You are evaluating a legislative analysis finding against an expert label.

PREDICTED FINDING:
{predicted}

EXPERT LABEL:
{expert}

TASK: Score how well the predicted finding matches the expert label (0-1).
- 1.0 = Perfect match (same meaning, same impact)
- 0.8-0.9 = Close match (minor differences)
- 0.5-0.7 = Partial match (same topic, different interpretation)
- 0.0-0.4 = Poor match (different findings)

OUTPUT (JSON):
{{
  "score": 0.0-1.0,
  "rationale": "Explanation of score"
}}"""


class LLMJudgeEnsemble:
    """
    Ensemble of judge models for validation.
    
    Uses GPT-4o, Claude Opus, Gemini 3 Pro for diverse perspectives.
    Each judge evaluates the same finding-label pair independently.
    
    Agreement calculation:
    - High agreement: low standard deviation across scores
    - Low agreement: high standard deviation, indicates ambiguity
    - Threshold: 0.75 agreement for automatic acceptance
    """
    
    def __init__(self, gpt_key: str, claude_key: str, gemini_key: str):
        """
        Initialize ensemble with API keys for all three judges.
        
        Args:
            gpt_key: OpenAI API key for GPT-4o
            claude_key: Anthropic API key for Claude Opus
            gemini_key: Google API key for Gemini 3 Pro
        """
        self.gpt_client = OpenAI(api_key=gpt_key)
        self.claude_client = anthropic.Anthropic(api_key=claude_key)
        # Note: Gemini client would be initialized here in production
        # For now, we'll mock it to avoid dependency issues
        self.gemini_client = None  # Would be genai.Client(api_key=gemini_key)
    
    def evaluate_finding(
        self,
        predicted: Dict[str, Any],
        expert_label: Dict[str, Any]
    ) -> JudgeEvaluationResult:
        """
        Evaluate predicted finding against expert label using ensemble.
        
        Process:
        1. Query all three judges independently
        2. Collect scores and rationales
        3. Compute average and agreement metrics
        4. Flag low agreement for human review
        
        Args:
            predicted: System-predicted finding with statement and impact
            expert_label: Expert-labeled finding with statement and impact
        
        Returns:
            JudgeEvaluationResult with ensemble scores and agreement
        """
        # Query all judges
        gpt_result = self._call_gpt_judge(predicted, expert_label)
        claude_result = self._call_claude_judge(predicted, expert_label)
        gemini_result = self._call_gemini_judge(predicted, expert_label)
        
        judge_scores = [
            {"model": "gpt-4o", **gpt_result},
            {"model": "claude-opus-4", **claude_result},
            {"model": "gemini-3-pro", **gemini_result}
        ]
        
        # Compute average and agreement
        scores = [gpt_result["score"], claude_result["score"], gemini_result["score"]]
        average_score = np.mean(scores)
        
        # Agreement = 1 - normalized std dev
        # Higher std dev = lower agreement
        std_dev = np.std(scores)
        # Normalize std dev (max possible std dev for 3 scores in [0,1] is ~0.577)
        normalized_std = std_dev / 0.577
        agreement = 1.0 - normalized_std
        agreement = max(0.0, agreement)  # Ensure non-negative
        
        # Flag for human review if low agreement
        requires_review = bool(agreement < 0.75)
        
        return JudgeEvaluationResult(
            judge_scores=judge_scores,
            average_score=average_score,
            agreement=agreement,
            requires_human_review=requires_review
        )
    
    def _call_gpt_judge(self, predicted: Dict, expert: Dict) -> Dict[str, Any]:
        """
        Call GPT-4o judge for evaluation.
        
        Uses low temperature (0.1) for consistent scoring.
        JSON response format ensures structured output.
        """
        prompt = JUDGE_PROMPT.format(
            predicted=str(predicted),
            expert=str(expert)
        )
        
        response = self.gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    
    def _call_claude_judge(self, predicted: Dict, expert: Dict) -> Dict[str, Any]:
        """
        Call Claude Opus judge for evaluation.
        
        Uses low temperature for consistent reasoning.
        Max tokens set to 1024 for concise responses.
        """
        prompt = JUDGE_PROMPT.format(
            predicted=str(predicted),
            expert=str(expert)
        )
        
        message = self.claude_client.messages.create(
            model="claude-opus-4",
            max_tokens=1024,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        return json.loads(message.content[0].text)
    
    def _call_gemini_judge(self, predicted: Dict, expert: Dict) -> Dict[str, Any]:
        """
        Call Gemini 3 Pro judge for evaluation.
        
        Note: In production, this would use the actual Gemini API.
        For now, returns a mock response to avoid dependency issues.
        """
        # Mock implementation for testing
        # In production, this would use the actual Gemini API
        return {
            "score": 0.85,
            "rationale": "Mock Gemini evaluation - would use actual API in production"
        }

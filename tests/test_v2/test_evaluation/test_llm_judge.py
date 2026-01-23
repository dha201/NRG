"""
Tests for LLM-Judge Ensemble.

Tests verify that the ensemble:
- Uses multiple judge models (GPT, Claude, Gemini)
- Computes average scores and inter-judge agreement
- Flags low agreement for human review
"""
import pytest
from unittest.mock import patch
from nrg_core.v2.evaluation.llm_judge_ensemble import LLMJudgeEnsemble


def test_ensemble_uses_multiple_judges():
    """Should query multiple judge models (GPT, Claude, Gemini)."""
    ensemble = LLMJudgeEnsemble(
        gpt_key="test-gpt",
        claude_key="test-claude",
        gemini_key="test-gemini"
    )
    
    finding = {"statement": "Tax of $50/MW", "impact": 7}
    expert_label = {"statement": "Tax applies at $50/MW", "impact": 7}
    
    # Mock responses from each judge
    mock_gpt = {"score": 0.9, "rationale": "Correct"}
    mock_claude = {"score": 0.85, "rationale": "Mostly correct"}
    mock_gemini = {"score": 0.95, "rationale": "Accurate"}
    
    with patch.object(ensemble, '_call_gpt_judge', return_value=mock_gpt), \
         patch.object(ensemble, '_call_claude_judge', return_value=mock_claude), \
         patch.object(ensemble, '_call_gemini_judge', return_value=mock_gemini):
        
        result = ensemble.evaluate_finding(finding, expert_label)
    
    assert len(result.judge_scores) == 3
    assert result.average_score == 0.9  # (0.9 + 0.85 + 0.95) / 3
    assert result.agreement >= 0.8  # High inter-judge agreement


def test_ensemble_flags_low_agreement():
    """Should flag low agreement when judges disagree."""
    ensemble = LLMJudgeEnsemble(gpt_key="test", claude_key="test", gemini_key="test")
    
    # Mock divergent responses
    mock_gpt = {"score": 0.9, "rationale": "Good"}
    mock_claude = {"score": 0.3, "rationale": "Poor"}
    mock_gemini = {"score": 0.5, "rationale": "Mediocre"}
    
    with patch.object(ensemble, '_call_gpt_judge', return_value=mock_gpt), \
         patch.object(ensemble, '_call_claude_judge', return_value=mock_claude), \
         patch.object(ensemble, '_call_gemini_judge', return_value=mock_gemini):
        
        result = ensemble.evaluate_finding({"statement": "X", "impact": 5}, {"statement": "Y", "impact": 8})
    
    assert result.agreement < 0.75
    assert result.requires_human_review is True

"""
Tests for Fallback Analyst - Second Model Opinion.

Tests verify that the fallback analyst:
- Uses a different model (Claude) for architectural diversity
- Provides alternative interpretations when disagreeing
- Returns structured second opinions with confidence scores
- Handles edge cases properly
"""
from unittest.mock import patch
from nrg_core.v2.fallback import FallbackAnalyst
from nrg_core.models_v2 import Finding, Quote


def test_fallback_uses_different_model():
    """Fallback should use Claude Opus instead of GPT."""
    fallback = FallbackAnalyst(model="claude-opus-4", api_key="test-key")
    
    finding = Finding(
        statement="Tax applies to all facilities",
        quotes=[Quote(text="Section 2.1: tax applies", section="2.1", page=None)],
        confidence=0.75,
        impact_estimate=7
    )
    
    bill_text = "Section 2.1: Annual tax on facilities"
    
    mock_response = {
        "agrees": False,
        "alternative_interpretation": "Tax only applies to facilities exceeding 50MW threshold",
        "rationale": "Section 2.1 specifies 'exceeding 50 megawatts' which was missed",
        "confidence": 0.9
    }
    
    with patch.object(fallback, '_call_claude', return_value=mock_response):
        result = fallback.get_second_opinion(
            finding=finding,
            bill_text=bill_text
        )
    
    assert result.agrees is False
    assert "50MW" in result.alternative_interpretation
    assert result.confidence == 0.9


def test_fallback_agrees_with_correct_interpretation():
    """Should agree when primary interpretation is correct."""
    fallback = FallbackAnalyst(model="claude-opus-4", api_key="test-key")
    
    finding = Finding(
        statement="Tax of $50/MW on fossil facilities over 50MW",
        quotes=[Quote(text="Tax of $50/MW on fossil facilities over 50MW", section="2.1", page=None)],
        confidence=0.95,
        impact_estimate=8
    )
    
    bill_text = "Section 2.1: Tax of $50/MW on fossil facilities over 50MW"
    
    mock_response = {
        "agrees": True,
        "alternative_interpretation": "",
        "rationale": "Interpretation matches the text exactly",
        "confidence": 0.98
    }
    
    with patch.object(fallback, '_call_claude', return_value=mock_response):
        result = fallback.get_second_opinion(finding, bill_text)
    
    assert result.agrees is True
    assert result.alternative_interpretation == ""
    assert result.confidence == 0.98

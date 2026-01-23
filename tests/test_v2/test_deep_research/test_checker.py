"""
Tests for CheckerAgent.

Validates:
- Claim verification against source snippets
- Detection of speculative leaps (claims beyond source)
- Confidence scoring for validated claims
"""
import pytest
from unittest.mock import patch
from nrg_core.v2.deep_research.checker import CheckerAgent, CheckResult


def test_checker_validates_claim():
    """Checker should verify if source supports claim."""
    checker = CheckerAgent(model="gpt-4o", api_key="test-key")
    
    claim = "Similar bills in other states resulted in $1M annual cost"
    snippet = "State X renewable tax credit bill cost utilities approximately $1.2M annually"
    
    mock_result = {
        "directly_states": True,
        "confidence": 0.9,
        "rationale": "Source explicitly mentions $1.2M annual cost, supporting the claim"
    }
    
    with patch.object(checker, '_call_llm', return_value=mock_result):
        result = checker.check(claim=claim, snippet=snippet, source_url="https://...")
    
    assert isinstance(result, CheckResult)
    assert result.directly_states is True
    assert result.confidence >= 0.8


def test_checker_flags_speculative_leap():
    """Checker should flag when claim goes beyond source."""
    checker = CheckerAgent(model="gpt-4o", api_key="test-key")
    
    claim = "All utilities faced $5M costs"
    snippet = "One utility reported increased costs"
    
    mock_result = {
        "directly_states": False,
        "confidence": 0.3,
        "rationale": "Source mentions one utility, claim extrapolates to all and adds specific dollar amount not in source"
    }
    
    with patch.object(checker, '_call_llm', return_value=mock_result):
        result = checker.check(claim, snippet, "https://...")
    
    assert result.directly_states is False
    assert result.confidence < 0.5


def test_checker_result_includes_rationale():
    """Check result should always include rationale for audit trail."""
    checker = CheckerAgent(model="gpt-4o", api_key="test-key")
    
    mock_result = {
        "directly_states": True,
        "confidence": 0.85,
        "rationale": "Source directly states the claimed fact with supporting evidence"
    }
    
    with patch.object(checker, '_call_llm', return_value=mock_result):
        result = checker.check("claim", "snippet", "https://...")
    
    assert result.rationale is not None
    assert len(result.rationale) > 10  # Non-trivial rationale

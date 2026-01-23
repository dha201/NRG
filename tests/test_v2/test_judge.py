# tests/test_v2/test_judge.py
"""
Tests for Judge Model (Tier 2).

The judge validates findings from the primary analyst:
- Verifies quotes exist verbatim in bill text
- Detects hallucinations (claims not supported by quotes)
- Assesses evidence quality and ambiguity
"""
import pytest
from unittest.mock import Mock, patch
from nrg_core.v2.judge import JudgeModel
from nrg_core.models_v2 import Finding, Quote, JudgeValidation


def test_judge_verifies_quotes_exist():
    """Judge should verify quotes exist verbatim in bill text."""
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    
    bill_text = """
    Section 2.1: Tax Rate
    The annual tax rate is $50 per megawatt.
    """
    
    finding = Finding(
        statement="Tax is $50/MW annually",
        quotes=[Quote(text="The annual tax rate is $50 per megawatt", section="2.1", page=None)],
        confidence=0.9,
        impact_estimate=7
    )
    
    mock_validation = {
        "quote_verified": True,
        "hallucination_detected": False,
        "evidence_quality": 0.95,
        "ambiguity": 0.1,
        "judge_confidence": 0.95
    }
    
    with patch.object(judge, '_call_llm', return_value=mock_validation):
        result = judge.validate(finding_id=0, finding=finding, bill_text=bill_text)
    
    assert isinstance(result, JudgeValidation)
    assert result.quote_verified is True
    assert result.hallucination_detected is False
    assert result.evidence_quality == 0.95


def test_judge_detects_hallucination():
    """Judge should flag claims not in bill text."""
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    
    bill_text = """
    Section 2.1: Tax Rate
    The annual tax rate is $50 per megawatt.
    """
    
    # Statement claims quarterly reporting which is NOT in the bill
    finding = Finding(
        statement="Tax includes quarterly reporting requirement",
        quotes=[Quote(text="The annual tax rate is $50 per megawatt", section="2.1", page=None)],
        confidence=0.8,
        impact_estimate=5
    )
    
    mock_validation = {
        "quote_verified": True,
        "hallucination_detected": True,  # Quote exists but statement adds unsupported claim
        "evidence_quality": 0.4,
        "ambiguity": 0.3,
        "judge_confidence": 0.75
    }
    
    with patch.object(judge, '_call_llm', return_value=mock_validation):
        result = judge.validate(finding_id=0, finding=finding, bill_text=bill_text)
    
    assert result.hallucination_detected is True
    assert result.evidence_quality < 0.5

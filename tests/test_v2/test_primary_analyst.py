# tests/test_v2/test_primary_analyst.py
"""
Tests for Primary Analyst Agent (Tier 1).

The primary analyst extracts findings from bill text with supporting quotes.
Multi-sample flag triggers when high-impact or low-confidence findings detected.
"""
import pytest
from unittest.mock import Mock, patch
from nrg_core.v2.primary_analyst import PrimaryAnalyst
from nrg_core.models_v2 import PrimaryAnalysis, Finding


def test_primary_analyst_extracts_findings():
    """Primary analyst should extract findings with quotes."""
    analyst = PrimaryAnalyst(model="gpt-4o", api_key="test-key")
    
    bill_text = """
    Section 2.1: Energy Tax Collection
    A tax of $50 per megawatt shall be assessed annually on all 
    fossil fuel generation facilities exceeding 50 megawatts capacity.
    
    Section 2.2: Exemptions
    Renewable energy facilities are exempt from this tax.
    """
    
    nrg_context = "NRG operates 23 GW fossil (60%), 15 GW renewable (40%)"
    
    # Mock OpenAI response
    mock_response = {
        "findings": [
            {
                "statement": "Tax applies to fossil fuel facilities >50MW at $50/MW annually",
                "quotes": [
                    {"text": "A tax of $50 per megawatt shall be assessed annually on all fossil fuel generation facilities exceeding 50 megawatts capacity", "section": "2.1", "page": None}
                ],
                "confidence": 0.95,
                "impact_estimate": 8
            }
        ],
        "overall_confidence": 0.95
    }
    
    with patch.object(analyst, '_call_llm', return_value=mock_response):
        result = analyst.analyze(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context=nrg_context
        )
    
    assert isinstance(result, PrimaryAnalysis)
    assert len(result.findings) == 1
    assert result.findings[0].impact_estimate == 8
    assert result.requires_multi_sample is True  # impact >= 6


def test_multi_sample_trigger_on_low_confidence():
    """Multi-sample should trigger when confidence < 0.7."""
    analyst = PrimaryAnalyst(model="gpt-4o", api_key="test-key")
    
    mock_response = {
        "findings": [
            {
                "statement": "Ambiguous provision may apply to certain facilities",
                "quotes": [{"text": "Section 5.2 references undefined term", "section": "5.2", "page": None}],
                "confidence": 0.65,  # Low confidence
                "impact_estimate": 4
            }
        ],
        "overall_confidence": 0.65
    }
    
    with patch.object(analyst, '_call_llm', return_value=mock_response):
        result = analyst.analyze("HB456", "bill text", "nrg context")
    
    assert result.requires_multi_sample is True  # Triggered by low confidence

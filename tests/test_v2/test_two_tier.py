# tests/test_v2/test_two_tier.py
"""
Tests for Two-Tier Orchestrator.

The orchestrator coordinates the full pipeline:
1. Primary Analyst extracts findings
2. Judge validates each finding
3. Judge scores validated findings on rubrics
"""
import pytest
from unittest.mock import patch, Mock
from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.models_v2 import TwoTierAnalysisResult


def test_two_tier_orchestrator_full_pipeline():
    """Integration test: orchestrator runs primary analyst → judge validation → rubric scoring."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test-primary",
        judge_api_key="test-judge"
    )
    
    bill_text = """
    Section 2.1: Annual tax of $50 per megawatt on fossil fuel facilities >50MW.
    Section 4.3: Penalty of $10,000 per violation for non-compliance.
    """
    
    nrg_context = "NRG operates 23 GW fossil (60%), 15 GW renewable (40%)"
    
    # Mock primary analyst response
    mock_primary_findings = {
        "findings": [
            {
                "statement": "Tax of $50/MW on fossil >50MW",
                "quotes": [{"text": "Annual tax of $50 per megawatt on fossil fuel facilities >50MW", "section": "2.1", "page": None}],
                "confidence": 0.95,
                "impact_estimate": 8
            }
        ],
        "overall_confidence": 0.95
    }
    
    # Mock judge validation
    mock_judge_validation = {
        "quote_verified": True,
        "hallucination_detected": False,
        "evidence_quality": 0.95,
        "ambiguity": 0.1,
        "judge_confidence": 0.95
    }
    
    # Mock rubric scores
    mock_legal_score = {
        "dimension": "legal_risk",
        "score": 7,
        "rationale": "New tax obligation with penalties creates significant compliance requirements for NRG.",
        "evidence": [{"text": "Annual tax of $50 per megawatt", "section": "2.1", "page": None}],
        "rubric_anchor": "6-8: Significant obligations + penalties"
    }
    
    mock_financial_score = {
        "dimension": "financial_impact",
        "score": 7,
        "rationale": "$1.15M annual cost on 23 GW fossil portfolio based on $50/MW rate.",
        "evidence": [{"text": "Annual tax of $50 per megawatt", "section": "2.1", "page": None}],
        "rubric_anchor": "6-8: $500K-$5M material to P&L"
    }
    
    with patch.object(orchestrator.primary_analyst, '_call_llm', return_value=mock_primary_findings), \
         patch.object(orchestrator.judge, '_call_llm', return_value=mock_judge_validation), \
         patch.object(orchestrator.judge, '_call_llm_for_rubric', side_effect=[mock_legal_score, mock_financial_score]):
        
        result = orchestrator.analyze(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context=nrg_context
        )
    
    assert isinstance(result, TwoTierAnalysisResult)
    assert len(result.primary_analysis.findings) == 1
    assert len(result.judge_validations) == 1
    assert len(result.rubric_scores) == 2  # legal_risk + financial_impact
    assert result.judge_validations[0].quote_verified is True
    assert result.rubric_scores[0].dimension == "legal_risk"
    assert result.rubric_scores[1].dimension == "financial_impact"


def test_two_tier_skips_rubric_scoring_for_hallucinations():
    """Findings with hallucinations should not be scored on rubrics."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test-primary",
        judge_api_key="test-judge"
    )
    
    mock_primary_findings = {
        "findings": [
            {
                "statement": "Bill requires quarterly reporting",  # Hallucination
                "quotes": [{"text": "Annual tax of $50", "section": "2.1", "page": None}],
                "confidence": 0.8,
                "impact_estimate": 5
            }
        ],
        "overall_confidence": 0.8
    }
    
    mock_judge_validation = {
        "quote_verified": True,
        "hallucination_detected": True,  # Statement not supported by quote
        "evidence_quality": 0.3,
        "ambiguity": 0.5,
        "judge_confidence": 0.7
    }
    
    with patch.object(orchestrator.primary_analyst, '_call_llm', return_value=mock_primary_findings), \
         patch.object(orchestrator.judge, '_call_llm', return_value=mock_judge_validation):
        
        result = orchestrator.analyze("HB456", "bill text", "nrg context")
    
    # Validation should exist
    assert len(result.judge_validations) == 1
    assert result.judge_validations[0].hallucination_detected is True
    
    # But no rubric scores (skipped due to hallucination)
    assert len(result.rubric_scores) == 0

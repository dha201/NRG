# tests/test_v2/test_two_tier.py
"""
Tests for Two-Tier Validation Orchestrator.

The orchestrator validates findings from Sequential Evolution:
1. Receives pre-extracted findings (no duplicate extraction)
2. Judge validates each finding
3. Judge scores validated findings on rubrics
"""
import pytest
from unittest.mock import patch, Mock
from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.models_v2 import TwoTierAnalysisResult


def test_two_tier_orchestrator_full_pipeline():
    """Integration test: orchestrator validates findings → judge validation → rubric scoring."""
    # Disable multi_sample and fallback to avoid real API calls in unit test
    orchestrator = TwoTierOrchestrator(
        judge_api_key="test-judge",
        enable_multi_sample=False,
        enable_fallback=False
    )

    bill_text = """
    Section 2.1: Annual tax of $50 per megawatt on fossil fuel facilities >50MW.
    Section 4.3: Penalty of $10,000 per violation for non-compliance.
    """

    nrg_context = "NRG operates 23 GW fossil (60%), 15 GW renewable (40%)"

    # Mock findings_registry from Sequential Evolution (extraction already done upstream)
    findings_registry = {
        "F1": {
            "statement": "Tax of $50/MW on fossil >50MW",
            "quotes": [{"text": "Annual tax of $50 per megawatt on fossil fuel facilities >50MW", "section": "2.1", "page": None}],
            "origin_version": 1,
            "affected_sections": ["2.1"],
            "modification_count": 0,
            "impact_estimate": 8
        }
    }
    stability_scores = {"F1": 0.95}

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

    # Phase 2: Added operational_disruption and ambiguity_risk
    mock_operational_score = {
        "dimension": "operational_disruption",
        "score": 4,
        "rationale": "Requires process adjustments for tax calculation and reporting procedures.",
        "evidence": [{"text": "Annual tax of $50 per megawatt", "section": "2.1", "page": None}],
        "rubric_anchor": "3-5: Process adjustments"
    }

    mock_ambiguity_score = {
        "dimension": "ambiguity_risk",
        "score": 3,
        "rationale": "Language is mostly clear but 'fossil fuel facilities' definition needs regulatory guidance.",
        "evidence": [{"text": "fossil fuel facilities >50MW", "section": "2.1", "page": None}],
        "rubric_anchor": "3-5: Some ambiguity"
    }

    with patch.object(orchestrator.judge, '_call_llm', return_value=mock_judge_validation), \
         patch.object(orchestrator.judge, '_call_llm_for_rubric', side_effect=[
             mock_legal_score, mock_financial_score, mock_operational_score, mock_ambiguity_score
         ]):

        result = orchestrator.validate(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context=nrg_context,
            findings_registry=findings_registry,
            stability_scores=stability_scores
        )

    assert isinstance(result, TwoTierAnalysisResult)
    assert len(result.primary_analysis.findings) == 1
    assert len(result.judge_validations) == 1
    assert len(result.rubric_scores) == 4  # Phase 2: all 4 dimensions
    assert result.judge_validations[0].quote_verified is True
    # Verify all 4 dimensions are scored
    scored_dims = {s.dimension for s in result.rubric_scores}
    assert scored_dims == {"legal_risk", "financial_impact", "operational_disruption", "ambiguity_risk"}
    # Verify audit trails are generated (Phase 2)
    assert len(result.audit_trails) == 1
    assert "quotes_used" in result.audit_trails[0]


def test_two_tier_skips_rubric_scoring_for_hallucinations():
    """Findings with hallucinations should not be scored on rubrics."""
    # Disable multi_sample and fallback to avoid real API calls in unit test
    orchestrator = TwoTierOrchestrator(
        judge_api_key="test-judge",
        enable_multi_sample=False,
        enable_fallback=False
    )

    # Mock findings_registry from Sequential Evolution - statement is hallucination
    findings_registry = {
        "F1": {
            "statement": "Bill requires quarterly reporting",  # Hallucination
            "quotes": [{"text": "Annual tax of $50", "section": "2.1", "page": None}],
            "origin_version": 1,
            "affected_sections": ["2.1"],
            "modification_count": 0,
            "impact_estimate": 5
        }
    }
    stability_scores = {"F1": 0.8}

    mock_judge_validation = {
        "quote_verified": True,
        "hallucination_detected": True,  # Statement not supported by quote
        "evidence_quality": 0.3,
        "ambiguity": 0.5,
        "judge_confidence": 0.7
    }

    with patch.object(orchestrator.judge, '_call_llm', return_value=mock_judge_validation):

        result = orchestrator.validate(
            bill_id="HB456",
            bill_text="bill text",
            nrg_context="nrg context",
            findings_registry=findings_registry,
            stability_scores=stability_scores
        )

    # Validation should exist
    assert len(result.judge_validations) == 1
    assert result.judge_validations[0].hallucination_detected is True

    # But no rubric scores (skipped due to hallucination)
    assert len(result.rubric_scores) == 0

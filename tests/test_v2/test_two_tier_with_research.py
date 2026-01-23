"""
Integration tests for Two-Tier Pipeline with Deep Research (Phase 4).

Validates:
- Deep research integration produces research insights
- Cross-bill reference detection integration works
- Pipeline functions correctly with both features disabled (default)
"""
import pytest
from unittest.mock import patch, MagicMock
from nrg_core.v2.two_tier import TwoTierOrchestrator
from nrg_core.models_v2 import TwoTierAnalysisResult


@pytest.fixture
def mock_findings_registry():
    """Create a mock findings registry for testing."""
    return {
        "F1": {
            "statement": "Tax of $50/MW applies to fossil fuel facilities over 50MW capacity",
            "quotes": [{"text": "Tax of $50/MW on fossil facilities", "section": "2.1", "page": None}],
            "origin_version": 1,
            "affected_sections": ["2.1"],
            "modification_count": 0,
            "impact_estimate": 7
        }
    }


@pytest.fixture
def mock_stability_scores():
    """Create mock stability scores for testing."""
    return {"F1": 0.95}


@pytest.fixture
def mock_judge_validation_dict():
    """Create a mock judge validation dict for testing."""
    return {
        "quote_verified": True,
        "hallucination_detected": False,
        "evidence_quality": 0.9,
        "ambiguity": 0.2,
        "judge_confidence": 0.85
    }


@pytest.fixture
def mock_rubric_score_dict():
    """Create a mock rubric score dict for testing."""
    return {
        "dimension": "legal_risk",
        "score": 7,
        "rationale": "The bill introduces significant new compliance requirements with penalties for non-compliance.",
        "evidence": [{"text": "Tax of $50/MW", "section": "2.1", "page": None}],
        "rubric_anchor": "6-8: Significant obligations + penalties"
    }


def test_two_tier_with_deep_research_disabled_by_default():
    """Pipeline should work with deep research disabled (default)."""
    orchestrator = TwoTierOrchestrator(
        judge_api_key="test",
        enable_multi_sample=False,
        enable_fallback=False
        # enable_deep_research defaults to False
        # enable_cross_bill_refs defaults to False
    )

    # Verify research agent is None
    assert orchestrator.research_agent is None
    assert orchestrator.reference_detector is None


def test_two_tier_with_cross_bill_refs_enabled(mock_findings_registry, mock_stability_scores, mock_judge_validation_dict, mock_rubric_score_dict):
    """Should detect cross-bill references when enabled."""
    orchestrator = TwoTierOrchestrator(
        judge_api_key="test",
        enable_multi_sample=False,
        enable_fallback=False,
        enable_cross_bill_refs=True
    )

    bill_text = """
    Section 2.1: This Act amends 26 U.S.C. 48 to modify energy tax credits.
    A tax of $50/MW on fossil facilities over 50MW capacity shall apply.
    """

    # Mock rubric scores for all 4 dimensions
    mock_financial_score = {
        "dimension": "financial_impact",
        "score": 7,
        "rationale": "$1.15M annual cost on 23 GW fossil portfolio based on $50/MW rate.",
        "evidence": [{"text": "Tax of $50/MW", "section": "2.1", "page": None}],
        "rubric_anchor": "6-8: $500K-$5M material to P&L"
    }
    mock_operational_score = {
        "dimension": "operational_disruption",
        "score": 4,
        "rationale": "Requires process adjustments for tax calculation and reporting procedures.",
        "evidence": [{"text": "Tax of $50/MW", "section": "2.1", "page": None}],
        "rubric_anchor": "3-5: Process adjustments"
    }
    mock_ambiguity_score = {
        "dimension": "ambiguity_risk",
        "score": 3,
        "rationale": "Language is mostly clear but 'fossil fuel facilities' definition needs regulatory guidance.",
        "evidence": [{"text": "fossil facilities", "section": "2.1", "page": None}],
        "rubric_anchor": "3-5: Some ambiguity"
    }

    # Mock the judge to avoid real API calls
    with patch.object(orchestrator.judge, '_call_llm', return_value=mock_judge_validation_dict), \
         patch.object(orchestrator.judge, '_call_llm_for_rubric', side_effect=[
             mock_rubric_score_dict, mock_financial_score, mock_operational_score, mock_ambiguity_score
         ]):

        result = orchestrator.validate(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context="NRG context",
            findings_registry=mock_findings_registry,
            stability_scores=mock_stability_scores
        )

    # Should have detected cross-bill references
    assert result.cross_bill_references is not None
    assert "detected" in result.cross_bill_references
    assert result.cross_bill_references["count"] >= 1

    # Should find 26 U.S.C. 48 reference
    citations = [r["citation"] for r in result.cross_bill_references["detected"]]
    assert any("26 U.S.C. 48" in c for c in citations)


def test_two_tier_with_deep_research_enabled(mock_findings_registry, mock_stability_scores, mock_judge_validation_dict, mock_rubric_score_dict):
    """Should include research insights when deep research enabled."""
    orchestrator = TwoTierOrchestrator(
        judge_api_key="test",
        enable_multi_sample=False,
        enable_fallback=False,
        enable_deep_research=True
    )

    bill_text = "Tax of $50/MW on fossil facilities over 50MW capacity"

    # Mock rubric scores for all 4 dimensions
    mock_financial_score = {
        "dimension": "financial_impact",
        "score": 7,
        "rationale": "$1.15M annual cost on 23 GW fossil portfolio based on $50/MW rate.",
        "evidence": [{"text": "Tax of $50/MW", "section": "2.1", "page": None}],
        "rubric_anchor": "6-8: $500K-$5M material to P&L"
    }
    mock_operational_score = {
        "dimension": "operational_disruption",
        "score": 4,
        "rationale": "Requires process adjustments for tax calculation and reporting procedures.",
        "evidence": [{"text": "Tax of $50/MW", "section": "2.1", "page": None}],
        "rubric_anchor": "3-5: Process adjustments"
    }
    mock_ambiguity_score = {
        "dimension": "ambiguity_risk",
        "score": 3,
        "rationale": "Language is mostly clear but 'fossil fuel facilities' definition needs regulatory guidance.",
        "evidence": [{"text": "fossil facilities", "section": "2.1", "page": None}],
        "rubric_anchor": "3-5: Some ambiguity"
    }

    # Mock research result
    mock_research_result = MagicMock()
    mock_research_result.sources = [
        MagicMock(
            url="https://openstates.org/bill/123",
            snippet="Similar tax in State X cost utilities $1M",
            relevance="high",
            checker_validated=True
        )
    ]
    mock_research_result.summary = "Found 1 similar bill"
    mock_research_result.research_confidence = 0.8

    with patch.object(orchestrator.judge, '_call_llm', return_value=mock_judge_validation_dict), \
         patch.object(orchestrator.judge, '_call_llm_for_rubric', side_effect=[
             mock_rubric_score_dict, mock_financial_score, mock_operational_score, mock_ambiguity_score
         ]), \
         patch.object(orchestrator.research_agent, 'research', return_value=mock_research_result):

        result = orchestrator.validate(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context="NRG context",
            findings_registry=mock_findings_registry,
            stability_scores=mock_stability_scores
        )

    # Should have research insights
    assert len(result.research_insights) >= 1
    assert result.research_insights[0].source_url is not None
    assert result.research_insights[0].trust == 0.8


def test_result_includes_new_phase4_fields():
    """TwoTierAnalysisResult should include Phase 4 fields."""
    orchestrator = TwoTierOrchestrator(
        judge_api_key="test",
        enable_multi_sample=False,
        enable_fallback=False
    )

    # Check that result model has new fields
    from nrg_core.models_v2 import TwoTierAnalysisResult, ResearchInsight

    # ResearchInsight should exist and have expected fields
    insight = ResearchInsight(
        claim="Test claim",
        source_url="https://example.com",
        snippet="Test snippet",
        relevance="high",
        checker_validated=True,
        trust=0.9
    )
    assert insight.claim == "Test claim"
    assert insight.trust == 0.9

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
from nrg_core.models_v2 import (
    PrimaryAnalysis,
    Finding,
    Quote,
    JudgeValidation,
    RubricScore
)


@pytest.fixture
def mock_primary_analysis():
    """Create a mock primary analysis for testing."""
    return PrimaryAnalysis(
        bill_id="HB123",
        findings=[
            Finding(
                statement="Tax of $50/MW applies to fossil fuel facilities over 50MW capacity",
                quotes=[Quote(text="Tax of $50/MW on fossil facilities", section="2.1")],
                confidence=0.9,
                impact_estimate=7
            )
        ],
        overall_confidence=0.9,
        requires_multi_sample=False
    )


@pytest.fixture
def mock_judge_validation():
    """Create a mock judge validation for testing."""
    return JudgeValidation(
        finding_id=0,
        quote_verified=True,
        hallucination_detected=False,
        evidence_quality=0.9,
        ambiguity=0.2,
        judge_confidence=0.85
    )


@pytest.fixture
def mock_rubric_score():
    """Create a mock rubric score for testing."""
    return RubricScore(
        dimension="legal_risk",
        score=7,
        rationale="The bill introduces significant new compliance requirements with penalties for non-compliance.",
        evidence=[Quote(text="Tax of $50/MW", section="2.1")],
        rubric_anchor="6-8: Significant obligations + penalties"
    )


def test_two_tier_with_deep_research_disabled_by_default():
    """Pipeline should work with deep research disabled (default)."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test",
        judge_api_key="test",
        enable_multi_sample=False,
        enable_fallback=False
        # enable_deep_research defaults to False
        # enable_cross_bill_refs defaults to False
    )
    
    # Verify research agent is None
    assert orchestrator.research_agent is None
    assert orchestrator.reference_detector is None


def test_two_tier_with_cross_bill_refs_enabled(mock_primary_analysis, mock_judge_validation, mock_rubric_score):
    """Should detect cross-bill references when enabled."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test",
        judge_api_key="test",
        enable_multi_sample=False,
        enable_fallback=False,
        enable_cross_bill_refs=True
    )
    
    bill_text = """
    Section 2.1: This Act amends 26 U.S.C. 48 to modify energy tax credits.
    A tax of $50/MW on fossil facilities over 50MW capacity shall apply.
    """
    
    # Mock the primary analyst and judge to avoid real API calls
    with patch.object(orchestrator.primary_analyst, 'analyze', return_value=mock_primary_analysis), \
         patch.object(orchestrator.judge, 'validate', return_value=mock_judge_validation), \
         patch.object(orchestrator.judge, 'score_rubric', return_value=mock_rubric_score):
        
        result = orchestrator.analyze("HB123", bill_text, "NRG context")
    
    # Should have detected cross-bill references
    assert result.cross_bill_references is not None
    assert "detected" in result.cross_bill_references
    assert result.cross_bill_references["count"] >= 1
    
    # Should find 26 U.S.C. 48 reference
    citations = [r["citation"] for r in result.cross_bill_references["detected"]]
    assert any("26 U.S.C. 48" in c for c in citations)


def test_two_tier_with_deep_research_enabled(mock_primary_analysis, mock_judge_validation, mock_rubric_score):
    """Should include research insights when deep research enabled."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test",
        judge_api_key="test",
        enable_multi_sample=False,
        enable_fallback=False,
        enable_deep_research=True
    )
    
    bill_text = "Tax of $50/MW on fossil facilities over 50MW capacity"
    
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
    
    with patch.object(orchestrator.primary_analyst, 'analyze', return_value=mock_primary_analysis), \
         patch.object(orchestrator.judge, 'validate', return_value=mock_judge_validation), \
         patch.object(orchestrator.judge, 'score_rubric', return_value=mock_rubric_score), \
         patch.object(orchestrator.research_agent, 'research', return_value=mock_research_result):
        
        result = orchestrator.analyze("HB123", bill_text, "NRG context")
    
    # Should have research insights
    assert len(result.research_insights) >= 1
    assert result.research_insights[0].source_url is not None
    assert result.research_insights[0].trust == 0.8


def test_result_includes_new_phase4_fields():
    """TwoTierAnalysisResult should include Phase 4 fields."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test",
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

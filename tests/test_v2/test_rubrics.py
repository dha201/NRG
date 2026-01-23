# tests/test_v2/test_rubrics.py
"""
Tests for Rubric Scoring.

Rubrics provide standardized scoring on specific dimensions:
- legal_risk: Severity of new obligations and penalties
- financial_impact: Estimated dollar impact to NRG

Each score includes rationale and evidence for audit trail.
"""
import pytest
from unittest.mock import patch
from nrg_core.v2.judge import JudgeModel
from nrg_core.v2.rubrics import LEGAL_RISK_RUBRIC, FINANCIAL_IMPACT_RUBRIC
from nrg_core.models_v2 import Finding, Quote, RubricScore


def test_judge_scores_legal_risk():
    """Judge should score findings on legal risk dimension."""
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    
    finding = Finding(
        statement="Tax applies with $10K penalty per violation",
        quotes=[Quote(text="Penalty of $10,000 per violation", section="4.3", page=15)],
        confidence=0.9,
        impact_estimate=7
    )
    
    mock_score_output = {
        "dimension": "legal_risk",
        "score": 7,
        "rationale": "Score=7 because bill creates new tax obligation with significant penalties ($10K per violation). Section 4.3 establishes financial consequences for non-compliance.",
        "evidence": [{"text": "Penalty of $10,000 per violation", "section": "4.3", "page": 15}],
        "rubric_anchor": "6-8: Significant obligations + penalties"
    }
    
    with patch.object(judge, '_call_llm_for_rubric', return_value=mock_score_output):
        score = judge.score_rubric(
            dimension="legal_risk",
            finding=finding,
            bill_text="...",
            nrg_context="NRG operates 60% fossil, 40% renewable",
            rubric_anchors=LEGAL_RISK_RUBRIC
        )
    
    assert isinstance(score, RubricScore)
    assert score.dimension == "legal_risk"
    assert score.score == 7
    assert "penalties" in score.rationale.lower()
    assert score.rubric_anchor == "6-8: Significant obligations + penalties"


def test_judge_scores_financial_impact():
    """Judge should estimate financial impact with business context."""
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    
    finding = Finding(
        statement="$50/MW annual tax on fossil generation >50MW",
        quotes=[Quote(text="Annual tax of $50 per megawatt", section="2.1", page=3)],
        confidence=0.95,
        impact_estimate=8
    )
    
    mock_score_output = {
        "dimension": "financial_impact",
        "score": 7,
        "rationale": "Score=7 because $50/MW tax on NRG's 23 GW fossil portfolio = $1.15M annual cost. Material to P&L but not existential.",
        "evidence": [{"text": "Annual tax of $50 per megawatt", "section": "2.1", "page": 3}],
        "rubric_anchor": "6-8: $500K-$5M material to P&L"
    }
    
    with patch.object(judge, '_call_llm_for_rubric', return_value=mock_score_output):
        score = judge.score_rubric(
            dimension="financial_impact",
            finding=finding,
            bill_text="...",
            nrg_context="NRG: 23 GW fossil, 15 GW renewable",
            rubric_anchors=FINANCIAL_IMPACT_RUBRIC
        )
    
    assert score.score == 7
    assert "$1.15M" in score.rationale or "1.15" in score.rationale

# tests/test_models_v2.py
"""
Tests for Architecture v2.0 Data Models.

These models enforce structured output for the two-tier analysis system:
- Finding: Analyst findings with mandatory supporting quotes
- Quote: Evidence extracted from bill text
- RubricScore: Scored dimensions with audit trail
"""
import pytest
from nrg_core.models_v2 import Finding, Quote, RubricScore


def test_finding_creation():
    """Finding should store statement, quotes, and confidence."""
    quote = Quote(text="Section 2.1: Tax applies", section="2.1", page=5)
    finding = Finding(
        statement="Tax applies to facilities >50MW",
        quotes=[quote],
        confidence=0.85,
        impact_estimate=7
    )
    
    assert finding.statement == "Tax applies to facilities >50MW"
    assert len(finding.quotes) == 1
    assert finding.quotes[0].section == "2.1"
    assert finding.confidence == 0.85
    assert finding.impact_estimate == 7


def test_finding_requires_at_least_one_quote():
    """Finding must have supporting evidence - Pydantic validates min_length=1 on quotes."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Finding(
            statement="Tax applies to facilities greater than 50MW",  # Meet min_length=20
            quotes=[],
            confidence=0.5,
            impact_estimate=3
        )


def test_rubric_score_validation():
    """Rubric score requires rationale and anchor."""
    quote = Quote(text="Section 4.2: $50/MW annual tax", section="4.2", page=12)
    
    score = RubricScore(
        dimension="financial_impact",
        score=7,
        rationale="Estimated $1.14M annual cost based on Section 4.2 tax rate ($50/MW) applied to NRG's 60% fossil portfolio (23 GW)",
        evidence=[quote],
        rubric_anchor="6-8: Major obligations, $500K-$5M exposure"
    )
    
    assert score.score == 7
    assert score.dimension == "financial_impact"
    assert len(score.evidence) == 1

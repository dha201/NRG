# tests/test_v2/test_audit_trail.py
"""
Tests for Audit Trail Generation.

Audit trails provide compliance-ready documentation capturing:
- Finding details and supporting quotes
- Validation results from judge
- Rubric scores with rationales
- Timestamps for temporal tracking
"""
import pytest
from nrg_core.v2.audit_trail import AuditTrailGenerator
from nrg_core.models_v2 import Finding, Quote, RubricScore, JudgeValidation


def test_audit_trail_generation():
    """Should generate complete audit trail with quotes, scores, rationales."""
    generator = AuditTrailGenerator()
    
    finding = Finding(
        statement="Tax of $50/MW applies to fossil facilities",
        quotes=[Quote(text="Annual tax of $50 per megawatt", section="2.1", page=5)],
        confidence=0.9,
        impact_estimate=7
    )
    
    validation = JudgeValidation(
        finding_id=0,
        quote_verified=True,
        hallucination_detected=False,
        evidence_quality=0.95,
        ambiguity=0.1,
        judge_confidence=0.95
    )
    
    scores = [
        RubricScore(
            dimension="financial_impact",
            score=7,
            rationale="$1.15M annual cost based on 23 GW fossil portfolio at $50/MW tax rate.",
            evidence=[Quote(text="Annual tax of $50 per megawatt", section="2.1", page=5)],
            rubric_anchor="6-8: $500K-$5M material"
        )
    ]
    
    trail = generator.generate(
        finding=finding,
        validation=validation,
        rubric_scores=scores
    )
    
    # Verify structure
    assert "quotes_used" in trail
    assert "validation_result" in trail
    assert "rubric_scores" in trail
    assert "timestamp" in trail
    assert "trail_version" in trail
    
    # Verify content
    assert trail["validation_result"]["quote_verified"] is True
    assert trail["validation_result"]["hallucination_detected"] is False
    assert len(trail["rubric_scores"]) == 1
    assert trail["rubric_scores"][0]["dimension"] == "financial_impact"
    assert trail["trail_version"] == "v2.0"


def test_audit_trail_captures_all_quotes():
    """Audit trail should include all supporting quotes from finding."""
    generator = AuditTrailGenerator()
    
    # Finding with multiple quotes
    finding = Finding(
        statement="Tax applies with exemptions for renewables",
        quotes=[
            Quote(text="Annual tax of $50 per megawatt", section="2.1", page=5),
            Quote(text="Exemption for renewable generation facilities", section="2.3", page=7)
        ],
        confidence=0.85,
        impact_estimate=6
    )
    
    validation = JudgeValidation(
        finding_id=0,
        quote_verified=True,
        hallucination_detected=False,
        evidence_quality=0.9,
        ambiguity=0.2,
        judge_confidence=0.9
    )
    
    trail = generator.generate(finding, validation, rubric_scores=[])
    
    assert len(trail["quotes_used"]) == 2
    assert trail["quotes_used"][0]["section"] == "2.1"
    assert trail["quotes_used"][1]["section"] == "2.3"


def test_audit_trail_batch_generation():
    """Should generate trails for multiple findings efficiently."""
    generator = AuditTrailGenerator()
    
    findings = [
        Finding(
            statement="Tax provision for fossil facilities",
            quotes=[Quote(text="Tax on fossil", section="2.1", page=1)],
            confidence=0.9,
            impact_estimate=7
        ),
        Finding(
            statement="Renewable exemption clause included",
            quotes=[Quote(text="Exempt renewable", section="3.1", page=5)],
            confidence=0.85,
            impact_estimate=5
        )
    ]
    
    validations = [
        JudgeValidation(finding_id=0, quote_verified=True, hallucination_detected=False,
                       evidence_quality=0.9, ambiguity=0.1, judge_confidence=0.9),
        JudgeValidation(finding_id=1, quote_verified=True, hallucination_detected=False,
                       evidence_quality=0.85, ambiguity=0.2, judge_confidence=0.85)
    ]
    
    # 4 dimensions per finding = 8 total scores
    all_scores = []
    for f_idx in range(2):
        for dim in ["legal_risk", "financial_impact", "operational_disruption", "ambiguity_risk"]:
            all_scores.append(RubricScore(
                dimension=dim,
                score=5,
                rationale=f"Rationale for finding {f_idx} on {dim} - meets 50 char minimum requirement.",
                evidence=[],
                rubric_anchor="3-5: test anchor"
            ))
    
    trails = generator.generate_batch(
        findings=findings,
        validations=validations,
        all_rubric_scores=all_scores,
        dimensions_per_finding=4
    )
    
    assert len(trails) == 2
    assert len(trails[0]["rubric_scores"]) == 4
    assert len(trails[1]["rubric_scores"]) == 4

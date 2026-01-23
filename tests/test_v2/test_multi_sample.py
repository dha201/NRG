"""
Tests for Multi-Sample Consistency Checker.

Tests verify that the checker can:
- Run multiple LLM samples with different seeds
- Compute consistency scores using cosine similarity
- Extract consensus findings from multiple samples
- Flag low consistency for human review
"""
from unittest.mock import patch
from nrg_core.v2.multi_sample import MultiSampleChecker
from nrg_core.models_v2 import Finding, Quote


def test_multi_sample_runs_multiple_times():
    """Should resample analysis 2-3x and check consistency."""
    checker = MultiSampleChecker(model="gpt-4o", api_key="test-key", num_samples=3)
    
    bill_text = "Section 2.1: Tax of $50/MW on fossil facilities >50MW"
    nrg_context = "NRG: 23 GW fossil, 15 GW renewable"
    
    # Mock 3 consistent responses
    mock_responses = [
        {
            "findings": [{
                "statement": "Tax applies to fossil >50MW at $50/MW",
                "quotes": [{"text": "Tax of $50/MW on fossil facilities >50MW", "section": "2.1", "page": None}],
                "confidence": 0.9,
                "impact_estimate": 8
            }]
        },
        {
            "findings": [{
                "statement": "Tax of $50/MW applies to fossil plants exceeding 50MW",
                "quotes": [{"text": "Tax of $50/MW on fossil facilities >50MW", "section": "2.1", "page": None}],
                "confidence": 0.92,
                "impact_estimate": 8
            }]
        },
        {
            "findings": [{
                "statement": "$50/MW annual tax on fossil generation >50MW",
                "quotes": [{"text": "Tax of $50/MW on fossil facilities >50MW", "section": "2.1", "page": None}],
                "confidence": 0.88,
                "impact_estimate": 7
            }]
        }
    ]
    
    with patch.object(checker, '_call_llm', side_effect=mock_responses):
        consensus = checker.check_consistency(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context=nrg_context
        )
    
    assert consensus.num_samples == 3
    assert consensus.consistency_score >= 0.4  # Moderate agreement for similar statements
    assert len(consensus.consensus_findings) >= 1


def test_multi_sample_detects_inconsistency():
    """Should flag low consistency when responses diverge."""
    checker = MultiSampleChecker(model="gpt-4o", api_key="test-key", num_samples=3)
    
    # Mock 3 DIVERGENT responses
    mock_responses = [
        {"findings": [{"statement": "Tax applies to all facilities", "quotes": [{"text": "Section 2.1: Tax applies to all facilities", "section": "2.1", "page": None}], "confidence": 0.6, "impact_estimate": 8}]},
        {"findings": [{"statement": "Tax only applies to coal plants", "quotes": [{"text": "Section 2.1: Tax only applies to coal plants", "section": "2.1", "page": None}], "confidence": 0.5, "impact_estimate": 3}]},
        {"findings": [{"statement": "No tax, only reporting requirement", "quotes": [{"text": "Section 2.1: No tax, only reporting requirement", "section": "2.1", "page": None}], "confidence": 0.4, "impact_estimate": 2}]}
    ]
    
    with patch.object(checker, '_call_llm', side_effect=mock_responses):
        consensus = checker.check_consistency("HB456", "bill text", "nrg context")
    
    assert consensus.consistency_score < 0.5  # Low agreement
    assert consensus.requires_human_review is True

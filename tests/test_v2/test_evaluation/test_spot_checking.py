"""
Tests for Spot-Checking Pipeline.

Tests verify that the spot-checker:
- Samples bills stratified by confidence and impact
- Oversamples risky strata (low confidence + high impact)
- Estimates error rates from human reviews
"""
import pytest
from nrg_core.v2.evaluation.spot_checking import SpotChecker


def test_stratified_sampling():
    """Should sample bills stratified by confidence and impact."""
    checker = SpotChecker(sample_size=20)
    
    # Mock bills with different confidence/impact
    bills = [
        {"bill_id": f"HB{i}", "confidence": 0.9, "impact": 8} for i in range(10)
    ] + [
        {"bill_id": f"HB{i+10}", "confidence": 0.6, "impact": 3} for i in range(10)
    ] + [
        {"bill_id": f"HB{i+20}", "confidence": 0.4, "impact": 9} for i in range(10)
    ]
    
    sample = checker.stratified_sample(bills)
    
    assert len(sample) == 20
    
    # Should have bills from each stratum
    high_conf_high_impact = [b for b in sample if b["confidence"] >= 0.8 and b["impact"] >= 7]
    low_conf_high_impact = [b for b in sample if b["confidence"] < 0.7 and b["impact"] >= 7]
    
    assert len(high_conf_high_impact) > 0
    assert len(low_conf_high_impact) > 0  # Oversample risky stratum

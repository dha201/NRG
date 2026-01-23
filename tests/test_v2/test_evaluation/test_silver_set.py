"""
Tests for Silver Set Infrastructure.

Tests verify that the silver set:
- Loads expert-labeled bills correctly
- Validates required expert labels
- Manages bill storage and retrieval
"""
import pytest
import json
import os
from pathlib import Path
from nrg_core.v2.evaluation.silver_set import SilverSet, SilverBill


def test_silver_set_loads_bills():
    """Should load silver set bills with expert labels."""
    silver_set = SilverSet(data_dir="/tmp/test_silver_set")
    
    # Create test bill
    test_bill = {
        "bill_id": "HB123",
        "bill_text": "Section 2.1: Tax of $50/MW",
        "expert_labels": {
            "findings": [
                {"statement": "Tax applies at $50/MW", "impact": 7}
            ],
            "rubric_scores": {
                "legal_risk": 7,
                "financial_impact": 7
            }
        }
    }
    
    os.makedirs("/tmp/test_silver_set", exist_ok=True)
    with open("/tmp/test_silver_set/HB123.json", "w") as f:
        json.dump(test_bill, f)
    
    bills = silver_set.load()
    
    assert len(bills) >= 1
    assert bills[0].bill_id == "HB123"
    assert len(bills[0].expert_labels["findings"]) == 1
    
    # Cleanup
    os.remove("/tmp/test_silver_set/HB123.json")
    os.rmdir("/tmp/test_silver_set")


def test_silver_set_validation():
    """Silver bills must have expert labels."""
    with pytest.raises(ValueError, match="expert_labels"):
        SilverBill(
            bill_id="HB456",
            bill_text="...",
            expert_labels=None  # Missing labels
        )

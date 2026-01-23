# tests/test_v2/test_supervisor.py
"""
Tests for Supervisor Router - code-based complexity assessment and routing.

The supervisor uses deterministic rules (no LLM) to route bills:
- STANDARD: Simple bills get single-pass analysis (80% of bills)
- ENHANCED: Complex bills get full two-tier pipeline (20% of bills)
"""
import pytest
from nrg_core.v2.supervisor import SupervisorRouter, Route


def test_simple_bill_routes_to_standard():
    """Short bills with <20 pages, <2 versions route to STANDARD."""
    router = SupervisorRouter()
    
    bill_metadata = {
        "bill_id": "HB123",
        "page_count": 15,
        "version_count": 1,
        "domain": "general"
    }
    
    route = router.assess_complexity(bill_metadata)
    
    assert route == Route.STANDARD
    assert router.complexity_score == 0  # No points: <20 pages, 1 version, general domain


def test_complex_bill_routes_to_enhanced():
    """Long bills, multiple versions, energy domain route to ENHANCED."""
    router = SupervisorRouter()
    
    bill_metadata = {
        "bill_id": "HB456",
        "page_count": 55,
        "version_count": 6,
        "domain": "energy"
    }
    
    route = router.assess_complexity(bill_metadata)
    
    assert route == Route.ENHANCED
    assert router.complexity_score >= 3  # Length(2) + Versions(2) + Domain(2) = 6


def test_boundary_at_3_points():
    """Exactly 3 points routes to ENHANCED."""
    router = SupervisorRouter()
    
    # 20 pages (1) + 2 versions (1) + environmental (1) = 3 points
    bill_metadata = {
        "bill_id": "HB789",
        "page_count": 20,
        "version_count": 2,
        "domain": "environmental"
    }
    
    route = router.assess_complexity(bill_metadata)
    
    assert route == Route.ENHANCED
    assert router.complexity_score == 3

"""
Tests for Sequential Evolution Agent.

Tests verify that the agent:
- Walks bill versions in chronological order
- Maintains structured memory across versions
- Tracks finding origins and modifications
- Computes stability scores correctly
"""
from unittest.mock import patch
from nrg_core.v2.sequential_evolution import SequentialEvolutionAgent, BillVersion


def test_sequential_agent_walks_versions():
    """Agent should process versions in order, maintaining memory."""
    agent = SequentialEvolutionAgent(model="gpt-4o", api_key="test-key")
    
    versions = [
        BillVersion(version_number=1, text="Section 2.1: Tax of $50/MW", name="Introduced"),
        BillVersion(version_number=2, text="Section 2.1: Tax of $75/MW", name="Engrossed"),
        BillVersion(version_number=3, text="Section 2.1: Tax of $75/MW. Section 2.2: Exemptions for renewable", name="Enrolled")
    ]
    
    # Mock LLM responses for each version
    with patch.object(agent, '_analyze_version') as mock_analyze:
        mock_analyze.side_effect = [
            {"findings": [{"id": "F1", "statement": "Tax $50/MW", "origin_version": 1}]},
            {"findings": [{"id": "F1", "statement": "Tax $75/MW (MODIFIED)", "origin_version": 1, "modification_count": 1}]},
            {"findings": [
                {"id": "F1", "statement": "Tax $75/MW", "origin_version": 1, "modification_count": 1},
                {"id": "F2", "statement": "Renewable exempt", "origin_version": 3}
            ]}
        ]
        
        result = agent.walk_versions(bill_id="HB123", versions=versions)
    
    assert len(result.findings_registry) >= 2  # F1, F2
    assert result.findings_registry["F1"]["modification_count"] == 1
    assert result.findings_registry["F2"]["origin_version"] == 3


def test_stability_score_calculation():
    """Should compute stability scores based on origin and modifications."""
    agent = SequentialEvolutionAgent(model="gpt-4o", api_key="test-key")
    
    # Test stability formula
    registry = {
        "F1": {"origin_version": 1, "modification_count": 0},  # Stable: 0.95
        "F2": {"origin_version": 1, "modification_count": 1},  # One change: 0.85
        "F3": {"origin_version": 3, "modification_count": 0},  # Last minute: 0.20
    }
    
    scores = agent._compute_stability(registry, num_versions=3)
    
    assert scores["F1"] == 0.95
    assert scores["F2"] == 0.85
    assert scores["F3"] == 0.20

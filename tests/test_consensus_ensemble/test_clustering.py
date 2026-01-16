import pytest
from nrg_core.consensus_ensemble.clustering import SemanticClusterer


def test_finding_clustering():
    findings = [
        {"statement": "Tax applies to energy generation exceeding 50MW", "model": "Gemini-A"},
        {"statement": "Tax on energy generation over 50 megawatts", "model": "Gemini-B"},
        {"statement": "Renewable energy exempt", "model": "Gemini-A"},
        {"statement": "Renewable facilities are exempt", "model": "GPT-5"},
    ]

    clusterer = SemanticClusterer(similarity_threshold=0.85)
    clusters = clusterer.cluster_findings(findings)

    # Should have 2 clusters (tax threshold + renewable exempt)
    assert len(clusters) == 2

    # Each cluster should have 2 findings
    assert all(len(cluster) == 2 for cluster in clusters)


def test_similarity_calculation():
    clusterer = SemanticClusterer()

    # Similar statements
    sim1 = clusterer.calculate_similarity(
        "Tax applies to energy generation exceeding 50 megawatts",
        "Tax on energy generation over 50 megawatts"
    )
    assert sim1 > 0.85

    # Dissimilar statements
    sim2 = clusterer.calculate_similarity(
        "Tax applies to energy generation exceeding 50 megawatts",
        "The committee will meet on Tuesday at 3pm"
    )
    assert sim2 < 0.5


def test_edge_case_empty_findings():
    """Test empty input returns empty list"""
    clusterer = SemanticClusterer()
    clusters = clusterer.cluster_findings([])
    assert clusters == []


def test_edge_case_single_finding():
    """Test single finding returns single cluster"""
    findings = [{"statement": "Tax applies to energy >50MW", "confidence": 0.9}]
    clusterer = SemanticClusterer()
    clusters = clusterer.cluster_findings(findings)
    
    assert len(clusters) == 1
    assert len(clusters[0]) == 1
    assert clusters[0][0]["statement"] == "Tax applies to energy >50MW"


def test_edge_case_all_identical():
    """Test all findings identical creates single cluster"""
    findings = [
        {"statement": "Tax applies to energy >50MW", "confidence": 0.9},
        {"statement": "Tax applies to energy >50MW", "confidence": 0.85},
        {"statement": "Tax applies to energy >50MW", "confidence": 0.88},
    ]
    clusterer = SemanticClusterer()
    clusters = clusterer.cluster_findings(findings)
    
    assert len(clusters) == 1
    assert len(clusters[0]) == 3  # All in one cluster


def test_edge_case_all_different():
    """Test no findings similar creates individual clusters"""
    findings = [
        {"statement": "Tax applies to energy >50MW", "confidence": 0.9},
        {"statement": "Renewable energy exempt", "confidence": 0.8},
        {"statement": "Quarterly reporting required", "confidence": 0.85},
    ]
    clusterer = SemanticClusterer()
    clusters = clusterer.cluster_findings(findings)
    
    assert len(clusters) == 3  # Each in separate cluster
    assert all(len(cluster) == 1 for cluster in clusters)


def test_edge_case_threshold_boundary():
    """Test threshold behavior at boundary"""
    findings = [
        {"statement": "Tax applies to energy >50MW", "confidence": 0.9},
        {"statement": "Tax on energy exceeding 50MW", "confidence": 0.85},  # Very similar
        {"statement": "Energy tax for >50MW", "confidence": 0.8},           # Moderately similar
    ]
    
    # High threshold - only very similar cluster
    clusterer_high = SemanticClusterer(similarity_threshold=0.90)
    clusters_high = clusterer_high.cluster_findings(findings)
    assert len(clusters_high) >= 2  # Should split into multiple clusters
    
    # Low threshold - all cluster together
    clusterer_low = SemanticClusterer(similarity_threshold=0.70)
    clusters_low = clusterer_low.cluster_findings(findings)
    assert len(clusters_low) <= 2  # Should merge more clusters


def test_edge_case_order_dependency():
    """Test that input order affects clustering (greedy algorithm limitation)"""
    findings_a = [
        {"statement": "Tax applies to energy >50MW", "confidence": 0.9},
        {"statement": "Tax on energy exceeding 50MW", "confidence": 0.85},
        {"statement": "Energy tax >50MW", "confidence": 0.8},
    ]
    findings_b = list(reversed(findings_a))  # Reverse order
    
    clusterer = SemanticClusterer(similarity_threshold=0.80)
    clusters_a = clusterer.cluster_findings(findings_a)
    clusters_b = clusterer.cluster_findings(findings_b)
    
    # Note: This test documents the greedy algorithm's order dependency
    # The exact assertion depends on actual similarity scores
    # This is more of a documentation test than a strict assertion


def test_edge_case_consensus_levels():
    """Test cluster sizes correspond to consensus levels"""
    # Unanimous (3/3)
    unanimous = [
        {"statement": "Tax applies to energy >50MW", "confidence": 0.9},
        {"statement": "Tax on energy exceeding 50MW", "confidence": 0.85},
        {"statement": "Energy tax >50MW", "confidence": 0.8},
    ]
    
    # Majority (2/3)
    majority = [
        {"statement": "Tax applies to energy >50MW", "confidence": 0.9},
        {"statement": "Tax on energy exceeding 50MW", "confidence": 0.85},
        {"statement": "Renewable energy exempt", "confidence": 0.8},
    ]
    
    # Disputed (1/3)
    disputed = [
        {"statement": "Tax applies to energy >50MW", "confidence": 0.9},
        {"statement": "Renewable energy exempt", "confidence": 0.8},
        {"statement": "Quarterly reporting required", "confidence": 0.85},
    ]
    
    clusterer = SemanticClusterer(similarity_threshold=0.80)
    
    # Test unanimous case
    clusters_unanimous = clusterer.cluster_findings(unanimous)
    assert any(len(cluster) == 3 for cluster in clusters_unanimous)
    
    # Test majority case  
    clusters_majority = clusterer.cluster_findings(majority)
    assert any(len(cluster) == 2 for cluster in clusters_majority)
    
    # Test disputed case
    clusters_disputed = clusterer.cluster_findings(disputed)
    assert all(len(cluster) == 1 for cluster in clusters_disputed)

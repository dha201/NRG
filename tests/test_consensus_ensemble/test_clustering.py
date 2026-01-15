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

    # Similar statements (semantic paraphrase)
    sim1 = clusterer.calculate_similarity(
        "Tax applies to >50MW",
        "Tax on energy exceeding 50 megawatts"
    )
    assert sim1 > 0.85

    # Dissimilar statements (different meaning)
    sim2 = clusterer.calculate_similarity(
        "Tax applies to >50MW",
        "Renewable energy exempt"
    )
    assert sim2 < 0.5

import pytest
from nrg_core.consensus_ensemble.consensus import ConsensusEngine
from nrg_core.consensus_ensemble.models import ModelResponse, Finding


def test_unanimous_consensus():
    """All 3 models found the same finding (after clustering)"""
    responses = [
        ModelResponse(
            model_name="Gemini-3-Pro-A",
            findings=[{"statement": "Tax applies to energy generation exceeding 50 megawatts", "quote": "Section 2.1b", "confidence": 0.92}]
        ),
        ModelResponse(
            model_name="Gemini-3-Pro-B",
            findings=[{"statement": "Tax on energy generation over 50 megawatts", "quote": "Section 2.1b", "confidence": 0.95}]
        ),
        ModelResponse(
            model_name="GPT-5",
            findings=[{"statement": "Tax for energy generation exceeding 50 megawatts", "quote": "Section 2.1b", "confidence": 0.90}]
        )
    ]

    engine = ConsensusEngine(similarity_threshold=0.85)
    consensus = engine.build_consensus(responses, bill_text="Sample bill")

    # Should have 1 finding with unanimous consensus (all 3 clustered together)
    assert len(consensus.findings) == 1
    assert consensus.findings[0].consensus_level == "unanimous"
    assert consensus.findings[0].confidence >= 0.95
    assert len(consensus.findings[0].found_by) == 3


def test_majority_consensus():
    """2 of 3 models found the finding, 1 hallucinated different scope"""
    responses = [
        ModelResponse(
            model_name="Gemini-3-Pro-A",
            findings=[{"statement": "Tax applies to all energy companies", "quote": "", "confidence": 0.65}]
        ),
        ModelResponse(
            model_name="Gemini-3-Pro-B",
            findings=[{"statement": "Tax applies to energy generation exceeding 50 megawatts", "quote": "Section 2.1b", "confidence": 0.92}]
        ),
        ModelResponse(
            model_name="GPT-5",
            findings=[{"statement": "Tax on energy generation over 50 megawatts", "quote": "Section 2.1b", "confidence": 0.90}]
        )
    ]

    engine = ConsensusEngine(similarity_threshold=0.85)
    consensus = engine.build_consensus(responses, bill_text="Section 2.1b: exceeding fifty megawatts")

    # Should have findings with majority consensus (2 agree, 1 disagreed)
    majority_findings = [f for f in consensus.findings if f.consensus_level == "majority"]
    assert len(majority_findings) >= 1
    assert majority_findings[0].confidence >= 0.65
    assert majority_findings[0].confidence < 0.95

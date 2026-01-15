import pytest
from nrg_core.sota.models import Finding, ConsensusAnalysis, ModelResponse


def test_finding_creation():
    finding = Finding(
        statement="Tax applies to energy generation exceeding 50MW",
        confidence=0.70,
        supporting_quotes=["Section 2.1b: exceeding fifty megawatts"],
        found_by=["GPT-4o", "Claude"]
    )
    assert finding.statement == "Tax applies to energy generation exceeding 50MW"
    assert finding.confidence == 0.70
    assert len(finding.supporting_quotes) == 1
    assert len(finding.found_by) == 2


def test_model_response_parsing():
    response_data = {
        "findings": [
            {
                "statement": "Tax applies to >50MW",
                "quote": "Section 2.1b: exceeding fifty megawatts",
                "confidence": 0.85
            }
        ]
    }
    model_response = ModelResponse.from_dict(response_data, model_name="GPT-4o")
    assert model_response.model_name == "GPT-4o"
    assert len(model_response.findings) == 1
    assert model_response.findings[0]["statement"] == "Tax applies to >50MW"

import pytest
from unittest.mock import Mock, patch, AsyncMock
from nrg_core.consensus_ensemble.ensemble import ConsensusEnsemble
from nrg_core.consensus_ensemble.models import ModelResponse


@pytest.mark.asyncio
async def test_ensemble_analysis():
    with patch('nrg_core.consensus_ensemble.ensemble.ParallelAnalyzer') as MockAnalyzer:
        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.analyze_parallel = AsyncMock(return_value=[
            ModelResponse(
                model_name="Gemini-3-Pro-A",
                findings=[{"statement": "Tax applies to energy generation exceeding 50 megawatts", "quote": "Section 2.1", "confidence": 0.9}]
            ),
            ModelResponse(
                model_name="Gemini-3-Pro-B",
                findings=[{"statement": "Tax on energy over 50 megawatts", "quote": "Section 2.1", "confidence": 0.85}]
            ),
            ModelResponse(
                model_name="GPT-5",
                findings=[{"statement": "Tax applies to generation exceeding 50MW", "quote": "Section 2.1", "confidence": 0.88}]
            )
        ])

        ensemble = ConsensusEnsemble()
        result = await ensemble.analyze("Test bill text", nrg_context="NRG business context")

        assert result.overall_confidence > 0.0
        assert len(result.findings) > 0
        assert result.findings[0].consensus_level == "unanimous"

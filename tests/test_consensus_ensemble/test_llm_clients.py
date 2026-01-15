import pytest
from unittest.mock import Mock, patch, AsyncMock
from nrg_core.consensus_ensemble.llm_clients import LLMClient, GeminiClient, OpenAIClient, ParallelAnalyzer


@pytest.mark.asyncio
async def test_gemini_client_analyze():
    """Gemini uses response_schema for structured output enforcement"""
    with patch('google.genai.Client') as mock_client:
        mock_response = Mock()
        mock_response.text = '{"findings": [{"statement": "test", "quote": "test", "confidence": 0.9}]}'
        mock_client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="test_key", model_id="gemini-3-pro")
        result = await client.analyze_bill("Test bill text", "Test prompt", nrg_context="NRG context")

        assert result.model_name == "Gemini-3-Pro-A"
        assert len(result.findings) == 1


@pytest.mark.asyncio
async def test_parallel_analysis():
    """2x Gemini + 1x GPT-5 for MVP (no Claude)"""

    # Mock responses: 2 Gemini instances + 1 GPT-5
    with patch('nrg_core.consensus_ensemble.llm_clients.GeminiClient.analyze_bill') as mock_gemini, \
         patch('nrg_core.consensus_ensemble.llm_clients.OpenAIClient.analyze_bill') as mock_openai:

        mock_response = Mock()
        mock_response.findings = [{"statement": "test"}]
        mock_response.error = None
        mock_gemini.return_value = mock_response
        mock_openai.return_value = mock_response

        analyzer = ParallelAnalyzer()
        results = await analyzer.analyze_parallel("Test bill", "Test prompt", "NRG context")

        # 2 Gemini + 1 GPT-5 = 3 total
        assert len(results) == 3

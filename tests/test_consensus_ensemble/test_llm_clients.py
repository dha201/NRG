import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from dotenv import load_dotenv
from nrg_core.consensus_ensemble.llm_clients import LLMClient, GeminiClient, OpenAIClient, ParallelAnalyzer

load_dotenv()


@pytest.mark.asyncio
async def test_gemini_client_analyze():
    """Gemini uses response_schema for structured output enforcement"""
    with patch('google.genai.Client') as mock_client:
        mock_response = Mock()
        mock_response.text = '{"findings": [{"statement": "test", "quote": "test", "confidence": 0.9}]}'
        mock_client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key=os.getenv("GOOGLE_API_KEY"), model_id="gemini-3-pro")
        result = await client.analyze_bill("Test bill text", "Test prompt", nrg_context="NRG context")

        assert result.model_name == "Gemini-3-Pro-A"
        assert len(result.findings) == 1


@pytest.mark.asyncio
async def test_parallel_analysis():
    """2x Gemini + 1x GPT-5 for MVP (no Claude)"""

    # Mock responses: 2 Gemini instances + 1 GPT-5
    with patch('nrg_core.consensus_ensemble.llm_clients.GeminiClient.analyze_bill') as mock_gemini, \
         patch('nrg_core.consensus_ensemble.llm_clients.OpenAIClient.analyze_bill') as mock_openai:

        from nrg_core.consensus_ensemble.models import ModelResponse

        mock_response = ModelResponse(
            model_name="test",
            findings=[{"statement": "test"}],
            error=None
        )
        mock_gemini.return_value = mock_response
        mock_openai.return_value = mock_response

        analyzer = ParallelAnalyzer(
            gemini_key=os.getenv("GOOGLE_API_KEY"),
            openai_key=os.getenv("OPENAI_API_KEY")
        )
        results = await analyzer.analyze_parallel("Test bill", "Test prompt", "NRG context")

        # 2 Gemini + 1 GPT-5 = 3 total
        assert len(results) == 3

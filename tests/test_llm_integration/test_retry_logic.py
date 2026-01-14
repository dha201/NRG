"""
Retry logic tests for LLM analysis functions.

Tests verify exponential backoff and retry behavior for transient errors.
"""
import pytest
import json
from unittest.mock import MagicMock, patch, call
import time

from nrg_core.analysis.llm import analyze_with_gemini, analyze_with_openai


class TestGeminiRetryLogic:
    """Tests for analyze_with_gemini() retry behavior."""
    
    @pytest.mark.llm
    def test_success_on_first_attempt(
        self, sample_bill_dict, nrg_context, sample_config, valid_analysis_response
    ):
        """Successful first attempt should return analysis without retries."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        json_text = json.dumps(valid_analysis_response)
        mock_response.text = json_text
        # Complete mock structure matching real Gemini API
        text_part = MagicMock()
        text_part.text = json_text
        text_part.thought = False
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [text_part]
        mock_client.models.generate_content.return_value = mock_response
        
        result = analyze_with_gemini(
            sample_bill_dict, nrg_context,
            gemini_client=mock_client,
            config=sample_config,
            max_retries=3
        )
        
        assert result["business_impact_score"] == 7
        assert mock_client.models.generate_content.call_count == 1
    
    @pytest.mark.llm
    def test_retry_on_transient_error(
        self, sample_bill_dict, nrg_context, sample_config, valid_analysis_response
    ):
        """Transient errors should trigger retry with eventual success."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        json_text = json.dumps(valid_analysis_response)
        mock_response.text = json_text
        text_part = MagicMock()
        text_part.text = json_text
        text_part.thought = False
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [text_part]
        
        mock_client.models.generate_content.side_effect = [
            Exception("Rate limit exceeded"),
            Exception("Temporary failure"),
            mock_response
        ]
        
        with patch('time.sleep'):
            result = analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=3,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 7
        assert mock_client.models.generate_content.call_count == 3
    
    @pytest.mark.llm
    def test_fail_after_max_retries(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """Should return error response after max retries exhausted."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Persistent failure")
        
        with patch('time.sleep'):
            result = analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=3,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 0
        assert "error" in result
        assert mock_client.models.generate_content.call_count == 3
    
    @pytest.mark.llm
    def test_exponential_backoff_delays(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """Delays should follow exponential backoff pattern."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Error")
        
        sleep_calls = []
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
        
        with patch('time.sleep', side_effect=mock_sleep):
            analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=3,
                base_delay=1.0
            )
        
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0
    
    @pytest.mark.llm
    def test_missing_client_returns_error(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """Missing gemini_client should return error response."""
        result = analyze_with_gemini(
            sample_bill_dict, nrg_context,
            gemini_client=None,
            config=sample_config
        )
        
        assert result["business_impact_score"] == 0
        assert "error" in result
    
    @pytest.mark.llm
    def test_missing_config_returns_error(
        self, sample_bill_dict, nrg_context, mock_gemini_client
    ):
        """Missing config should return error response."""
        result = analyze_with_gemini(
            sample_bill_dict, nrg_context,
            gemini_client=mock_gemini_client,
            config=None
        )
        
        assert result["business_impact_score"] == 0
        assert "error" in result


class TestOpenAIRetryLogic:
    """Tests for analyze_with_openai() retry behavior."""
    
    @pytest.mark.llm
    def test_success_on_first_attempt(
        self, sample_bill_dict, nrg_context, valid_analysis_response
    ):
        """Successful first attempt should return analysis without retries."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = json.dumps(valid_analysis_response)
        mock_client.responses.create.return_value = mock_response
        
        result = analyze_with_openai(
            sample_bill_dict, nrg_context,
            openai_client=mock_client,
            max_retries=3
        )
        
        assert result["business_impact_score"] == 7
        assert mock_client.responses.create.call_count == 1
    
    @pytest.mark.llm
    def test_retry_on_transient_error(
        self, sample_bill_dict, nrg_context, valid_analysis_response
    ):
        """Transient errors should trigger retry."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = json.dumps(valid_analysis_response)
        
        mock_client.responses.create.side_effect = [
            Exception("Rate limit"),
            mock_response
        ]
        
        with patch('time.sleep'):
            result = analyze_with_openai(
                sample_bill_dict, nrg_context,
                openai_client=mock_client,
                max_retries=3,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 7
        assert mock_client.responses.create.call_count == 2
    
    @pytest.mark.llm
    def test_missing_client_returns_error(
        self, sample_bill_dict, nrg_context
    ):
        """Missing openai_client should return error response."""
        result = analyze_with_openai(
            sample_bill_dict, nrg_context,
            openai_client=None
        )
        
        assert result["business_impact_score"] == 0
        assert "error" in result


class TestJsonDecodeRetry:
    """Tests for retry on JSON decode errors."""
    
    @pytest.mark.llm
    def test_retry_on_json_decode_error(
        self, sample_bill_dict, nrg_context, sample_config, valid_analysis_response
    ):
        """JSON decode errors should trigger retry."""
        mock_client = MagicMock()
        
        # Invalid response - complete structure with bad JSON
        invalid_response = MagicMock()
        invalid_response.text = "Not valid JSON {"
        invalid_part = MagicMock()
        invalid_part.text = "Not valid JSON {"
        invalid_part.thought = False
        invalid_response.candidates = [MagicMock()]
        invalid_response.candidates[0].content = MagicMock()
        invalid_response.candidates[0].content.parts = [invalid_part]
        
        # Valid response - complete structure
        valid_response = MagicMock()
        json_text = json.dumps(valid_analysis_response)
        valid_response.text = json_text
        valid_part = MagicMock()
        valid_part.text = json_text
        valid_part.thought = False
        valid_response.candidates = [MagicMock()]
        valid_response.candidates[0].content = MagicMock()
        valid_response.candidates[0].content.parts = [valid_part]
        
        mock_client.models.generate_content.side_effect = [
            invalid_response,
            valid_response
        ]
        
        with patch('time.sleep'):
            result = analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=3,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 7
    
    @pytest.mark.llm
    def test_json_error_after_all_retries(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """Persistent JSON errors should return error response."""
        mock_client = MagicMock()
        invalid_response = MagicMock()
        invalid_response.text = "Invalid JSON forever"
        invalid_part = MagicMock()
        invalid_part.text = "Invalid JSON forever"
        invalid_part.thought = False
        invalid_response.candidates = [MagicMock()]
        invalid_response.candidates[0].content = MagicMock()
        invalid_response.candidates[0].content.parts = [invalid_part]
        mock_client.models.generate_content.return_value = invalid_response
        
        with patch('time.sleep'):
            result = analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=3,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 0
        assert "error" in result


class TestRetryConfiguration:
    """Tests for retry configuration options."""
    
    @pytest.mark.llm
    def test_custom_max_retries(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """Custom max_retries should be respected."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Error")
        
        with patch('time.sleep'):
            analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=5,
                base_delay=0.01
            )
        
        assert mock_client.models.generate_content.call_count == 5
    
    @pytest.mark.llm
    def test_custom_base_delay(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """Custom base_delay should affect backoff calculation."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Error")
        
        sleep_calls = []
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
        
        with patch('time.sleep', side_effect=mock_sleep):
            analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=3,
                base_delay=2.0
            )
        
        assert sleep_calls[0] == 2.0
        assert sleep_calls[1] == 4.0

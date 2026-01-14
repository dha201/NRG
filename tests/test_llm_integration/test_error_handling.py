"""
Error handling tests for LLM analysis functions.

Tests verify graceful fallback behavior on various error conditions.
"""
import pytest
import json
from unittest.mock import MagicMock, patch

from nrg_core.analysis.llm import analyze_with_gemini, analyze_with_openai, ERROR_RESPONSE
from nrg_core.analysis.changes import analyze_changes_with_llm, analyze_version_changes_with_llm


class TestErrorResponseStructure:
    """Tests for ERROR_RESPONSE constant."""
    
    def test_error_response_has_required_fields(self):
        """ERROR_RESPONSE should have minimum required fields."""
        assert "business_impact_score" in ERROR_RESPONSE
        assert "impact_summary" in ERROR_RESPONSE
    
    def test_error_response_score_is_zero(self):
        """ERROR_RESPONSE should have score of 0."""
        assert ERROR_RESPONSE["business_impact_score"] == 0


class TestGeminiErrorHandling:
    """Tests for analyze_with_gemini() error handling."""
    
    @pytest.mark.llm
    def test_none_response_text_handled(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """None response.text should be handled gracefully."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response
        
        with patch('time.sleep'):
            result = analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=1
            )
        
        assert result["business_impact_score"] == 0
        assert "error" in result or "impact_summary" in result
    
    @pytest.mark.llm
    def test_invalid_json_returns_error_response(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """Invalid JSON should return error response after retries."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON {{"
        mock_client.models.generate_content.return_value = mock_response
        
        with patch('time.sleep'):
            result = analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=2,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 0
        assert "error" in result
    
    @pytest.mark.llm
    def test_api_exception_returns_error_response(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """API exceptions should return error response after retries."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        
        with patch('time.sleep'):
            result = analyze_with_gemini(
                sample_bill_dict, nrg_context,
                gemini_client=mock_client,
                config=sample_config,
                max_retries=2,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 0
        assert "error" in result
    
    @pytest.mark.llm
    def test_schema_mismatch_handled(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """Response not matching schema should still be processed."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({"unexpected": "structure"})
        mock_client.models.generate_content.return_value = mock_response
        
        result = analyze_with_gemini(
            sample_bill_dict, nrg_context,
            gemini_client=mock_client,
            config=sample_config
        )
        
        assert isinstance(result, dict)


class TestOpenAIErrorHandling:
    """Tests for analyze_with_openai() error handling."""
    
    @pytest.mark.llm
    def test_invalid_json_returns_error_response(
        self, sample_bill_dict, nrg_context
    ):
        """Invalid JSON should return error response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Invalid JSON"
        mock_client.responses.create.return_value = mock_response
        
        with patch('time.sleep'):
            result = analyze_with_openai(
                sample_bill_dict, nrg_context,
                openai_client=mock_client,
                max_retries=1,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 0
    
    @pytest.mark.llm
    def test_api_exception_returns_error_response(
        self, sample_bill_dict, nrg_context
    ):
        """API exceptions should return error response."""
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("OpenAI API Error")
        
        with patch('time.sleep'):
            result = analyze_with_openai(
                sample_bill_dict, nrg_context,
                openai_client=mock_client,
                max_retries=2,
                base_delay=0.01
            )
        
        assert result["business_impact_score"] == 0
        assert "error" in result


class TestChangeAnalysisErrorHandling:
    """Tests for change analysis error handling."""
    
    @pytest.mark.llm
    def test_no_changes_returns_none(self, sample_bill_dict, nrg_context, sample_config):
        """No changes should return None without calling LLM."""
        change_data = {"has_changes": False}
        
        result = analyze_changes_with_llm(
            sample_bill_dict,
            change_data,
            nrg_context,
            sample_config
        )
        
        assert result is None
    
    @pytest.mark.llm
    def test_llm_failure_returns_error_dict(
        self, sample_bill_dict, nrg_context, sample_config
    ):
        """LLM failure during change analysis should return error dict."""
        change_data = {
            "has_changes": True,
            "changes": [{"type": "text_change", "summary": "Text changed", "diff": ""}]
        }
        
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        
        with patch('time.sleep'):
            result = analyze_changes_with_llm(
                sample_bill_dict,
                change_data,
                nrg_context,
                sample_config,
                gemini_client=mock_client
            )
        
        assert result["change_impact_score"] == 0
        assert "error" in result


class TestVersionChangesErrorHandling:
    """Tests for version changes analysis error handling."""
    
    @pytest.mark.llm
    def test_llm_failure_returns_error_dict(self, nrg_context, sample_config):
        """LLM failure should return structured error dict."""
        old_version = {"version_type": "v1", "full_text": "Old text"}
        new_version = {"version_type": "v2", "full_text": "New text"}
        old_analysis = {"business_impact_score": 5}
        new_analysis = {"business_impact_score": 7}
        bill_info = {"number": "HB 1", "title": "Test", "source": "Test"}
        
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        
        with patch('time.sleep'):
            result = analyze_version_changes_with_llm(
                old_version, new_version,
                old_analysis, new_analysis,
                bill_info, nrg_context,
                sample_config,
                gemini_client=mock_client
            )
        
        assert "summary" in result
        assert "Error" in result["summary"] or "error" in result.get("summary", "").lower()
        assert result["key_provisions_added"] == []
        assert result["key_provisions_removed"] == []
        assert result["key_provisions_modified"] == []


class TestEdgeCaseInputs:
    """Tests for edge case inputs."""
    
    @pytest.mark.llm
    def test_empty_bill_summary(self, nrg_context, sample_config, mock_gemini_client):
        """Empty bill summary should still process."""
        bill = {
            "source": "Test",
            "type": "Bill",
            "number": "HB 1",
            "title": "Test Bill",
            "status": "Introduced",
            "summary": ""
        }
        
        result = analyze_with_gemini(
            bill, nrg_context,
            gemini_client=mock_gemini_client,
            config=sample_config
        )
        
        assert "business_impact_score" in result
    
    @pytest.mark.llm
    def test_empty_nrg_context(self, sample_bill_dict, sample_config, mock_gemini_client):
        """Empty NRG context should still process."""
        result = analyze_with_gemini(
            sample_bill_dict, "",
            gemini_client=mock_gemini_client,
            config=sample_config
        )
        
        assert "business_impact_score" in result

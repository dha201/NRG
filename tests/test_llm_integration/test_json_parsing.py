"""
JSON parsing tests for LLM responses.

Tests handle Gemini 3 "thinking" model response quirks including thought_signature parts.
"""
import pytest
import json
from unittest.mock import MagicMock

from nrg_core.analysis.llm import extract_json_from_gemini_response


class TestExtractJsonFromGeminiResponse:
    """Tests for extract_json_from_gemini_response() function."""
    
    def test_valid_text_part_extracted(self, valid_analysis_response):
        """Valid response with text part should extract JSON."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        
        text_part = MagicMock()
        text_part.text = json.dumps(valid_analysis_response)
        text_part.thought = False
        mock_response.candidates[0].content.parts = [text_part]
        
        result = extract_json_from_gemini_response(mock_response)
        
        assert result == json.dumps(valid_analysis_response)
    
    def test_thought_signature_skipped(self, valid_analysis_response):
        """Thought signature parts should be skipped."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        
        thought_part = MagicMock()
        thought_part.text = "encrypted_reasoning_trace_abc123"
        thought_part.thought = True
        
        text_part = MagicMock()
        text_part.text = json.dumps(valid_analysis_response)
        text_part.thought = False
        
        mock_response.candidates[0].content.parts = [thought_part, text_part]
        
        result = extract_json_from_gemini_response(mock_response)
        
        assert "encrypted_reasoning" not in result
        parsed = json.loads(result)
        assert parsed["business_impact_score"] == valid_analysis_response["business_impact_score"]
    
    def test_multiple_text_parts_takes_first(self, valid_analysis_response):
        """Multiple text parts should return only the first (avoid concatenation bug)."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        
        text_part1 = MagicMock()
        text_part1.text = json.dumps(valid_analysis_response)
        text_part1.thought = False
        
        text_part2 = MagicMock()
        text_part2.text = json.dumps({"extra": "data"})
        text_part2.thought = False
        
        mock_response.candidates[0].content.parts = [text_part1, text_part2]
        
        result = extract_json_from_gemini_response(mock_response)
        
        parsed = json.loads(result)
        assert "business_impact_score" in parsed
        assert "extra" not in parsed
    
    def test_no_candidates_raises_error(self):
        """Response with no candidates should raise ValueError."""
        mock_response = MagicMock()
        mock_response.candidates = []
        
        with pytest.raises(ValueError) as exc_info:
            extract_json_from_gemini_response(mock_response)
        
        assert "no candidates" in str(exc_info.value).lower()
    
    def test_no_content_raises_error(self):
        """Response with no content should raise ValueError."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = None
        
        with pytest.raises(ValueError) as exc_info:
            extract_json_from_gemini_response(mock_response)
        
        assert "no content" in str(exc_info.value).lower()
    
    def test_no_parts_raises_error(self):
        """Response with no parts should raise ValueError."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = []
        
        with pytest.raises(ValueError) as exc_info:
            extract_json_from_gemini_response(mock_response)
        
        assert "no parts" in str(exc_info.value).lower()
    
    def test_only_thought_parts_raises_error(self):
        """Response with only thought_signature parts should raise ValueError."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        
        thought_part = MagicMock()
        thought_part.text = "encrypted_reasoning"
        thought_part.thought = True
        
        mock_response.candidates[0].content.parts = [thought_part]
        
        with pytest.raises(ValueError) as exc_info:
            extract_json_from_gemini_response(mock_response)
        
        assert "no text part" in str(exc_info.value).lower()


class TestJsonParsingEdgeCases:
    """Tests for JSON parsing edge cases."""
    
    def test_empty_text_part_skipped(self, valid_analysis_response):
        """Empty text parts should be skipped."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        
        empty_part = MagicMock()
        empty_part.text = ""
        empty_part.thought = False
        
        valid_part = MagicMock()
        valid_part.text = json.dumps(valid_analysis_response)
        valid_part.thought = False
        
        mock_response.candidates[0].content.parts = [empty_part, valid_part]
        
        result = extract_json_from_gemini_response(mock_response)
        parsed = json.loads(result)
        assert "business_impact_score" in parsed
    
    def test_none_text_part_skipped(self, valid_analysis_response):
        """Parts with None text should be skipped."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        
        none_part = MagicMock()
        none_part.text = None
        none_part.thought = False
        
        valid_part = MagicMock()
        valid_part.text = json.dumps(valid_analysis_response)
        valid_part.thought = False
        
        mock_response.candidates[0].content.parts = [none_part, valid_part]
        
        result = extract_json_from_gemini_response(mock_response)
        parsed = json.loads(result)
        assert "business_impact_score" in parsed
    
    def test_whitespace_only_text_skipped(self, valid_analysis_response):
        """Whitespace-only text parts should be skipped."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        
        ws_part = MagicMock()
        ws_part.text = "   \n\t  "
        ws_part.thought = False
        
        valid_part = MagicMock()
        valid_part.text = json.dumps(valid_analysis_response)
        valid_part.thought = False
        
        mock_response.candidates[0].content.parts = [ws_part, valid_part]
        
        result = extract_json_from_gemini_response(mock_response)
        parsed = json.loads(result)
        assert "business_impact_score" in parsed


class TestGeminiResponseTextAccessor:
    """Tests validating behavior that extract_json_from_gemini_response fixes."""
    
    def test_sdk_text_accessor_concatenation_issue(self):
        """Document the SDK bug: response.text concatenates all text parts.
        
        This test documents the issue our fix addresses:
        - Gemini SDK's response.text concatenates all text parts
        - Results in '{"json1"}{"json2"}' which breaks json.loads()
        """
        mock_response = MagicMock()
        
        json1 = '{"score": 5}'
        json2 = '{"extra": "data"}'
        mock_response.text = json1 + json2
        
        with pytest.raises(json.JSONDecodeError):
            json.loads(mock_response.text)
    
    def test_sdk_text_accessor_none_issue(self):
        """Document the SDK bug: response.text returns None for thought-only responses.
        
        This test documents the issue our fix addresses:
        - When response has only thought_signature parts, response.text is None
        - json.loads(None) raises TypeError
        """
        mock_response = MagicMock()
        mock_response.text = None
        
        with pytest.raises(TypeError):
            json.loads(mock_response.text)

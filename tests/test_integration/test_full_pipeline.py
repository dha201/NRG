"""
Full pipeline integration tests.

Tests end-to-end analysis flow with mocked LLM responses.
"""
import pytest
import json
from unittest.mock import MagicMock, patch

from nrg_core.models import Bill
from nrg_core.analysis.changes import detect_bill_changes


class TestNewBillAnalysisPipeline:
    """Tests for analyzing new bills (no cache)."""
    
    @pytest.mark.llm
    def test_new_bill_detected_correctly(self, sample_bill_dict):
        """New bill without cache should be flagged as new."""
        result = detect_bill_changes(None, sample_bill_dict)
        
        assert result["is_new"] is True
        assert result["has_changes"] is False
        assert result["change_type"] == "new_bill"
    
    @pytest.mark.llm
    def test_new_bill_analysis_structure(
        self, sample_bill_dict, nrg_context, sample_config, mock_gemini_client
    ):
        """New bill analysis should return complete structure."""
        from nrg_core.analysis.llm import analyze_with_gemini
        
        result = analyze_with_gemini(
            sample_bill_dict, nrg_context,
            gemini_client=mock_gemini_client,
            config=sample_config
        )
        
        assert "business_impact_score" in result
        assert "impact_type" in result
        assert "impact_summary" in result
        assert "recommended_action" in result


class TestChangedBillAnalysisPipeline:
    """Tests for analyzing changed bills."""
    
    @pytest.mark.llm
    def test_changed_bill_detected_correctly(self, sample_cached_bill):
        """Changed bill should be flagged with correct change types."""
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Engrossed",
            "summary": "Completely revised bill text with new provisions."
        }
        
        result = detect_bill_changes(sample_cached_bill, current_bill)
        
        assert result["is_new"] is False
        assert result["has_changes"] is True
        assert result["change_type"] == "modified"
    
    @pytest.mark.llm
    def test_change_analysis_called_when_changes_exist(
        self, sample_bill_dict, sample_cached_bill, nrg_context, sample_config, valid_change_impact_response
    ):
        """Change analysis should be performed when changes detected."""
        from nrg_core.analysis.changes import analyze_changes_with_llm
        
        current_bill = {
            "number": "HB 1234",
            "source": "OpenStates",
            "title": "Energy Market Reform Act",
            "status": "Engrossed",
            "summary": "Revised text."
        }
        
        change_data = detect_bill_changes(sample_cached_bill, current_bill)
        assert change_data["has_changes"] is True
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(valid_change_impact_response)
        mock_client.models.generate_content.return_value = mock_response
        
        result = analyze_changes_with_llm(
            current_bill, change_data, nrg_context,
            sample_config, gemini_client=mock_client
        )
        
        assert result is not None
        assert "change_impact_score" in result


class TestBillWithVersionsPipeline:
    """Tests for analyzing bills with multiple versions."""
    
    def test_bill_versions_parsed(self, sample_bill_with_versions):
        """Bill with versions should have all versions accessible."""
        bill = Bill.from_dict(sample_bill_with_versions)
        
        assert len(bill.versions) == 3
        assert bill.versions[0].version_type_raw == "Introduced"
        assert bill.versions[1].version_type_raw == "House Committee Report"
        assert bill.versions[2].version_type_raw == "Engrossed"
    
    def test_version_pairs_for_comparison(self, sample_bill_with_versions):
        """Should generate correct version pairs for comparison."""
        bill = Bill.from_dict(sample_bill_with_versions)
        sorted_versions = sorted(bill.versions, key=lambda v: v.version_number)
        
        pairs = list(zip(sorted_versions[:-1], sorted_versions[1:]))
        
        assert len(pairs) == 2
        assert pairs[0][0].version_type_raw == "Introduced"
        assert pairs[0][1].version_type_raw == "House Committee Report"
        assert pairs[1][0].version_type_raw == "House Committee Report"
        assert pairs[1][1].version_type_raw == "Engrossed"


class TestPipelineErrorRecovery:
    """Tests for pipeline error recovery."""
    
    @pytest.mark.llm
    def test_pipeline_continues_on_single_version_failure(
        self, sample_bill_with_versions, nrg_context, sample_config, valid_analysis_response
    ):
        """Pipeline should continue processing when one version fails."""
        bill = Bill.from_dict(sample_bill_with_versions)
        from nrg_core.analysis.llm import analyze_with_gemini
        
        call_count = [0]
        
        def mock_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Temporary failure for v2")
            mock_resp = MagicMock()
            mock_resp.text = json.dumps(valid_analysis_response)
            return mock_resp
        
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = mock_generate
        
        results = []
        for version in bill.versions:
            temp_bill = {
                "source": bill.source,
                "type": bill.bill_type,
                "number": bill.number,
                "title": bill.title,
                "status": version.version_type_raw,
                "summary": version.full_text
            }
            with patch('time.sleep'):
                result = analyze_with_gemini(
                    temp_bill, nrg_context,
                    gemini_client=mock_client,
                    config=sample_config,
                    max_retries=1,
                    base_delay=0.01
                )
            results.append(result)
        
        successful = [r for r in results if r.get("business_impact_score", 0) > 0]
        assert len(successful) >= 2


class TestEndToEndDataFlow:
    """Tests for data flow through the pipeline."""
    
    def test_bill_dict_to_model_roundtrip(self, sample_bill_dict):
        """Bill should survive dict → model → dict roundtrip."""
        bill = Bill.from_dict(sample_bill_dict)
        result = bill.to_dict()
        
        assert result["source"] == sample_bill_dict["source"]
        assert result["number"] == sample_bill_dict["number"]
        assert result["title"] == sample_bill_dict["title"]
    
    def test_analysis_attached_to_bill_result(
        self, sample_bill_dict, valid_analysis_response
    ):
        """Analysis result should be attachable to bill."""
        from nrg_core.models import AnalysisResult
        
        result = AnalysisResult(
            item=sample_bill_dict,
            analysis=valid_analysis_response
        )
        
        assert result.item["number"] == sample_bill_dict["number"]
        assert result.analysis["business_impact_score"] == 7

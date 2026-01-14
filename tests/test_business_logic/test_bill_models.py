"""
Bill model tests.

Tests validate data models correctly parse and serialize data.
"""
import pytest

from nrg_core.models import Bill, BillVersion, Analysis, ChangeData


class TestBillFromDict:
    """Tests for Bill.from_dict() parsing."""
    
    def test_bill_from_dict_all_fields(self, sample_bill_dict):
        """Bill should parse all fields from dictionary."""
        bill = Bill.from_dict(sample_bill_dict)
        
        assert bill.source == "OpenStates"
        assert bill.bill_type == "State Bill"
        assert bill.number == "HB 1234"
        assert bill.title == "Energy Market Reform Act"
        assert bill.status == "Introduced"
        assert bill.url == "https://example.com/bill/1234"
        assert bill.summary == "An act relating to energy market reforms and consumer protection."
        assert bill.sponsor == "Rep. Smith"
        assert bill.introduced_date == "2026-01-01"
        assert bill.policy_area == "Energy"
        assert bill.state == "TX"
    
    def test_bill_from_dict_missing_optional_fields(self):
        """Bill should handle missing optional fields with None."""
        minimal_dict = {
            "source": "OpenStates",
            "number": "HB 1",
            "title": "Test Bill",
            "status": "Introduced",
            "url": "",
            "summary": ""
        }
        bill = Bill.from_dict(minimal_dict)
        
        assert bill.sponsor is None
        assert bill.introduced_date is None
        assert bill.policy_area is None
        assert bill.state is None
    
    def test_bill_from_dict_with_versions(self, sample_bill_with_versions):
        """Bill should parse versions as BillVersion objects."""
        bill = Bill.from_dict(sample_bill_with_versions)
        
        assert len(bill.versions) == 3
        assert all(isinstance(v, BillVersion) for v in bill.versions)
        assert bill.versions[0].version_number == 1
        assert bill.versions[0].version_type_raw == "Introduced"
    
    def test_bill_from_dict_empty_versions(self, sample_bill_dict):
        """Bill should handle empty versions list."""
        sample_bill_dict["versions"] = []
        bill = Bill.from_dict(sample_bill_dict)
        
        assert bill.versions == []


class TestBillToDict:
    """Tests for Bill.to_dict() serialization."""
    
    def test_bill_to_dict_roundtrip(self, sample_bill_dict):
        """to_dict should preserve all data from from_dict."""
        bill = Bill.from_dict(sample_bill_dict)
        result = bill.to_dict()
        
        assert result["source"] == sample_bill_dict["source"]
        assert result["number"] == sample_bill_dict["number"]
        assert result["title"] == sample_bill_dict["title"]
        assert result["status"] == sample_bill_dict["status"]
    
    def test_bill_to_dict_with_versions(self, sample_bill_with_versions):
        """to_dict should serialize versions correctly."""
        bill = Bill.from_dict(sample_bill_with_versions)
        result = bill.to_dict()
        
        assert len(result["versions"]) == 3
        assert result["versions"][0]["version_number"] == 1


class TestBillVersionFromDict:
    """Tests for BillVersion.from_dict() parsing."""
    
    def test_version_from_dict_all_fields(self, sample_version_dict):
        """BillVersion should parse all fields correctly."""
        version = BillVersion.from_dict(sample_version_dict)
        
        assert version.version_number == 1
        assert version.version_type_raw == "Introduced"
        assert version.version_type_normalized == "introduced"
        assert version.version_date == "2026-01-01"
        assert "Short title" in version.full_text
        assert version.text_hash == "abc123hash"
        assert version.word_count == 50
        assert version.pdf_url == "https://example.com/bill.pdf"
        assert version.text_url == "https://example.com/bill.txt"
    
    def test_version_from_dict_normalizes_version_type(self):
        """BillVersion should normalize version_type automatically."""
        version_dict = {
            "version_number": 1,
            "version_type": "Engrossed",
            "version_date": "",
            "full_text": "",
            "text_hash": "",
            "word_count": 0
        }
        version = BillVersion.from_dict(version_dict)
        
        assert version.version_type_raw == "Engrossed"
        assert version.version_type_normalized == "passed_originating_chamber"
    
    def test_version_from_dict_uses_note_fallback(self):
        """BillVersion should use 'note' field if 'version_type' missing."""
        version_dict = {
            "version_number": 1,
            "note": "Committee Report",
            "date": "2026-01-15",
            "full_text": "",
            "text_hash": "",
            "word_count": 0
        }
        version = BillVersion.from_dict(version_dict)
        
        assert version.version_type_raw == "Committee Report"
        assert version.version_type_normalized == "committee_report"


class TestAnalysisFromDict:
    """Tests for Analysis.from_dict() parsing."""
    
    def test_analysis_from_dict_all_fields(self, valid_analysis_response):
        """Analysis should parse all fields from LLM response."""
        analysis = Analysis.from_dict(valid_analysis_response)
        
        assert analysis.business_impact_score == 7
        assert analysis.impact_type == "regulatory_compliance"
        assert "reporting requirements" in analysis.impact_summary
        assert analysis.recommended_action == "engage"
        assert "Retail Commodity" in analysis.nrg_business_verticals
        assert "Regulatory Affairs" in analysis.internal_stakeholders
    
    def test_analysis_from_dict_minimal(self, minimal_valid_analysis):
        """Analysis should handle minimal response with defaults."""
        analysis = Analysis.from_dict(minimal_valid_analysis)
        
        assert analysis.business_impact_score == 5
        assert analysis.impact_type == "operational"
        assert analysis.recommended_action == "monitor"
        assert analysis.nrg_business_verticals == []
        assert analysis.internal_stakeholders == []
    
    def test_analysis_from_dict_nested_objects(self, valid_analysis_response):
        """Analysis should parse nested objects correctly."""
        analysis = Analysis.from_dict(valid_analysis_response)
        
        assert analysis.legal_code_changes is not None
        assert "Utilities Code §39.001" in analysis.legal_code_changes["sections_amended"]
        assert analysis.application_scope is not None
        assert "retail electric providers" in analysis.application_scope["applies_to"]
    
    def test_analysis_to_dict_roundtrip(self, valid_analysis_response):
        """to_dict should preserve all data."""
        analysis = Analysis.from_dict(valid_analysis_response)
        result = analysis.to_dict()
        
        assert result["business_impact_score"] == 7
        assert result["impact_type"] == "regulatory_compliance"
        assert result["recommended_action"] == "engage"


class TestChangeDataFromDict:
    """Tests for ChangeData.from_dict() parsing."""
    
    def test_change_data_from_dict_with_changes(self):
        """ChangeData should parse change information correctly."""
        data = {
            "has_changes": True,
            "is_new": False,
            "change_type": "modified",
            "changes": [
                {"type": "text_change", "summary": "Text modified"},
                {"type": "status_change", "old_value": "Introduced", "new_value": "Engrossed"}
            ]
        }
        change_data = ChangeData.from_dict(data)
        
        assert change_data.has_changes is True
        assert change_data.is_new is False
        assert change_data.change_type == "modified"
        assert len(change_data.changes) == 2
    
    def test_change_data_from_dict_new_bill(self):
        """ChangeData should handle new bill correctly."""
        data = {
            "has_changes": False,
            "is_new": True,
            "change_type": "new_bill",
            "changes": []
        }
        change_data = ChangeData.from_dict(data)
        
        assert change_data.has_changes is False
        assert change_data.is_new is True
        assert change_data.change_type == "new_bill"
    
    def test_change_data_from_dict_none_input(self):
        """ChangeData should handle None input gracefully."""
        change_data = ChangeData.from_dict(None)
        
        assert change_data.has_changes is False
        assert change_data.is_new is False
        assert change_data.change_type == "unchanged"

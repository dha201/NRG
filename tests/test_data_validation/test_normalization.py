"""
Version normalization tests.

Tests verify source-specific version labels are normalized correctly.
Critical for consistent version comparison across different data sources.
"""
import pytest

from nrg_core.models import VERSION_NORMALIZATION, normalize_version_type, BillVersion


class TestOpenStatesNormalization:
    """Tests for Open States (Texas) version normalization."""
    
    def test_introduced_normalized(self):
        """'Introduced' should normalize to 'introduced'."""
        assert normalize_version_type("Introduced") == "introduced"
    
    def test_engrossed_normalized(self):
        """'Engrossed' should normalize to 'passed_originating_chamber'."""
        assert normalize_version_type("Engrossed") == "passed_originating_chamber"
    
    def test_enrolled_normalized(self):
        """'Enrolled' should normalize to 'enrolled'."""
        assert normalize_version_type("Enrolled") == "enrolled"
    
    def test_house_committee_report_normalized(self):
        """'House Committee Report' should normalize to 'committee_report'."""
        assert normalize_version_type("House Committee Report") == "committee_report"
    
    def test_senate_committee_report_normalized(self):
        """'Senate Committee Report' should normalize to 'committee_report'."""
        assert normalize_version_type("Senate Committee Report") == "committee_report"
    
    def test_committee_report_normalized(self):
        """'Committee Report' should normalize to 'committee_report'."""
        assert normalize_version_type("Committee Report") == "committee_report"


class TestCongressGovNormalization:
    """Tests for Congress.gov version code normalization."""
    
    def test_ih_introduced_in_house(self):
        """'IH' (Introduced in House) should normalize to 'introduced'."""
        assert normalize_version_type("IH") == "introduced"
    
    def test_is_introduced_in_senate(self):
        """'IS' (Introduced in Senate) should normalize to 'introduced'."""
        assert normalize_version_type("IS") == "introduced"
    
    def test_rh_reported_in_house(self):
        """'RH' (Reported in House) should normalize to 'committee_report'."""
        assert normalize_version_type("RH") == "committee_report"
    
    def test_rs_reported_in_senate(self):
        """'RS' (Reported in Senate) should normalize to 'committee_report'."""
        assert normalize_version_type("RS") == "committee_report"
    
    def test_eh_engrossed_in_house(self):
        """'EH' (Engrossed in House) should normalize to 'passed_originating_chamber'."""
        assert normalize_version_type("EH") == "passed_originating_chamber"
    
    def test_es_engrossed_in_senate(self):
        """'ES' (Engrossed in Senate) should normalize to 'passed_originating_chamber'."""
        assert normalize_version_type("ES") == "passed_originating_chamber"
    
    def test_enr_enrolled(self):
        """'ENR' (Enrolled) should normalize to 'enrolled'."""
        assert normalize_version_type("ENR") == "enrolled"


class TestUnknownVersionHandling:
    """Tests for unknown version type handling."""
    
    def test_unknown_version_lowercased_with_underscores(self):
        """Unknown versions should be lowercased with spaces replaced by underscores."""
        assert normalize_version_type("Custom Version Type") == "custom_version_type"
    
    def test_unknown_version_preserved_if_no_spaces(self):
        """Unknown versions without spaces should just be lowercased."""
        assert normalize_version_type("CustomType") == "customtype"
    
    def test_empty_version_handled(self):
        """Empty version type should not raise error."""
        result = normalize_version_type("")
        assert result == ""
    
    def test_numeric_version_converted(self):
        """Numeric-looking versions should be handled."""
        assert normalize_version_type("Version 1") == "version_1"


class TestBillVersionNormalization:
    """Tests for BillVersion.from_dict normalization."""
    
    def test_bill_version_stores_both_raw_and_normalized(self, sample_version_dict):
        """BillVersion should store both raw and normalized version types."""
        version = BillVersion.from_dict(sample_version_dict)
        
        assert version.version_type_raw == "Introduced"
        assert version.version_type_normalized == "introduced"
    
    def test_bill_version_congress_gov_code(self):
        """BillVersion should normalize Congress.gov codes correctly."""
        version_dict = {
            "version_number": 1,
            "version_type": "EH",
            "version_date": "2026-01-15",
            "full_text": "Bill text here.",
            "text_hash": "abc123",
            "word_count": 50
        }
        version = BillVersion.from_dict(version_dict)
        
        assert version.version_type_raw == "EH"
        assert version.version_type_normalized == "passed_originating_chamber"
    
    def test_bill_version_uses_note_field_fallback(self):
        """BillVersion should fallback to 'note' field if 'version_type' missing."""
        version_dict = {
            "version_number": 1,
            "note": "Engrossed",
            "date": "2026-01-15",
            "full_text": "Bill text here.",
            "text_hash": "abc123",
            "word_count": 50
        }
        version = BillVersion.from_dict(version_dict)
        
        assert version.version_type_raw == "Engrossed"
        assert version.version_type_normalized == "passed_originating_chamber"


class TestNormalizationConsistency:
    """Tests for normalization consistency across sources."""
    
    def test_house_and_senate_introduced_same(self):
        """Both IH and IS should normalize to same value."""
        ih = normalize_version_type("IH")
        is_ = normalize_version_type("IS")
        openstates = normalize_version_type("Introduced")
        
        assert ih == is_ == openstates == "introduced"
    
    def test_house_and_senate_engrossed_same(self):
        """Both EH and ES should normalize to same value."""
        eh = normalize_version_type("EH")
        es = normalize_version_type("ES")
        openstates = normalize_version_type("Engrossed")
        
        assert eh == es == openstates == "passed_originating_chamber"
    
    def test_all_committee_reports_same(self):
        """All committee report variants should normalize the same."""
        rh = normalize_version_type("RH")
        rs = normalize_version_type("RS")
        house = normalize_version_type("House Committee Report")
        senate = normalize_version_type("Senate Committee Report")
        generic = normalize_version_type("Committee Report")
        
        assert rh == rs == house == senate == generic == "committee_report"
    
    def test_enrolled_versions_same(self):
        """ENR and Enrolled should normalize the same."""
        enr = normalize_version_type("ENR")
        enrolled = normalize_version_type("Enrolled")
        
        assert enr == enrolled == "enrolled"


class TestVersionNormalizationMapping:
    """Tests for VERSION_NORMALIZATION dictionary completeness."""
    
    def test_all_openstates_versions_mapped(self):
        """All expected Open States versions should be in mapping."""
        openstates_versions = ["Introduced", "Engrossed", "Enrolled", 
                               "House Committee Report", "Senate Committee Report", "Committee Report"]
        for version in openstates_versions:
            assert version in VERSION_NORMALIZATION
    
    def test_all_congress_codes_mapped(self):
        """All expected Congress.gov codes should be in mapping."""
        congress_codes = ["IH", "IS", "RH", "RS", "EH", "ES", "ENR"]
        for code in congress_codes:
            assert code in VERSION_NORMALIZATION

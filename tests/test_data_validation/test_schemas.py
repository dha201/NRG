"""
Schema validation tests for LLM response structures.

Tests ensure LLM outputs conform to expected JSON schemas before downstream processing.
"""
import pytest
import jsonschema
from jsonschema import validate, ValidationError

from nrg_core.analysis.prompts import GEMINI_RESPONSE_SCHEMA
from nrg_core.analysis.changes import CHANGE_IMPACT_SCHEMA, VERSION_CHANGES_SCHEMA


class TestGeminiResponseSchema:
    """Tests for GEMINI_RESPONSE_SCHEMA validation."""
    
    def test_valid_full_response_passes(self, valid_analysis_response):
        """Valid response with all fields should pass validation."""
        validate(instance=valid_analysis_response, schema=GEMINI_RESPONSE_SCHEMA)
    
    def test_minimal_valid_response_passes(self, minimal_valid_analysis):
        """Response with only required fields should pass."""
        validate(instance=minimal_valid_analysis, schema=GEMINI_RESPONSE_SCHEMA)
    
    def test_missing_business_impact_score_fails(self, minimal_valid_analysis):
        """Missing required field business_impact_score should fail."""
        del minimal_valid_analysis["business_impact_score"]
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=minimal_valid_analysis, schema=GEMINI_RESPONSE_SCHEMA)
        assert "business_impact_score" in str(exc_info.value)
    
    def test_missing_impact_type_fails(self, minimal_valid_analysis):
        """Missing required field impact_type should fail."""
        del minimal_valid_analysis["impact_type"]
        with pytest.raises(ValidationError):
            validate(instance=minimal_valid_analysis, schema=GEMINI_RESPONSE_SCHEMA)
    
    def test_missing_impact_summary_fails(self, minimal_valid_analysis):
        """Missing required field impact_summary should fail."""
        del minimal_valid_analysis["impact_summary"]
        with pytest.raises(ValidationError):
            validate(instance=minimal_valid_analysis, schema=GEMINI_RESPONSE_SCHEMA)
    
    def test_missing_recommended_action_fails(self, minimal_valid_analysis):
        """Missing required field recommended_action should fail."""
        del minimal_valid_analysis["recommended_action"]
        with pytest.raises(ValidationError):
            validate(instance=minimal_valid_analysis, schema=GEMINI_RESPONSE_SCHEMA)
    
    def test_business_impact_score_must_be_integer(self):
        """business_impact_score must be an integer."""
        invalid = {
            "business_impact_score": "seven",
            "impact_type": "regulatory_compliance",
            "impact_summary": "Test summary",
            "recommended_action": "monitor"
        }
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=invalid, schema=GEMINI_RESPONSE_SCHEMA)
        assert "integer" in str(exc_info.value).lower()
    
    def test_nrg_business_verticals_must_be_array(self, minimal_valid_analysis):
        """nrg_business_verticals must be an array of strings."""
        minimal_valid_analysis["nrg_business_verticals"] = "Retail Commodity"
        with pytest.raises(ValidationError):
            validate(instance=minimal_valid_analysis, schema=GEMINI_RESPONSE_SCHEMA)
    
    def test_nested_legal_code_changes_structure(self, valid_analysis_response):
        """Nested legal_code_changes object should validate correctly."""
        valid_analysis_response["legal_code_changes"]["sections_amended"] = "not an array"
        with pytest.raises(ValidationError):
            validate(instance=valid_analysis_response, schema=GEMINI_RESPONSE_SCHEMA)
    
    def test_effective_dates_array_of_objects(self, valid_analysis_response):
        """effective_dates must be array of objects with date and applies_to."""
        valid_analysis_response["effective_dates"] = ["2027-01-01"]
        with pytest.raises(ValidationError):
            validate(instance=valid_analysis_response, schema=GEMINI_RESPONSE_SCHEMA)


class TestChangeImpactSchema:
    """Tests for CHANGE_IMPACT_SCHEMA validation."""
    
    def test_valid_change_impact_passes(self, valid_change_impact_response):
        """Valid change impact response should pass validation."""
        validate(instance=valid_change_impact_response, schema=CHANGE_IMPACT_SCHEMA)
    
    def test_minimal_change_impact_passes(self):
        """Response with only required fields should pass."""
        minimal = {
            "change_impact_score": 5,
            "change_summary": "Minor text changes.",
            "recommended_action": "monitor"
        }
        validate(instance=minimal, schema=CHANGE_IMPACT_SCHEMA)
    
    def test_missing_change_impact_score_fails(self, valid_change_impact_response):
        """Missing required change_impact_score should fail."""
        del valid_change_impact_response["change_impact_score"]
        with pytest.raises(ValidationError):
            validate(instance=valid_change_impact_response, schema=CHANGE_IMPACT_SCHEMA)
    
    def test_missing_change_summary_fails(self, valid_change_impact_response):
        """Missing required change_summary should fail."""
        del valid_change_impact_response["change_summary"]
        with pytest.raises(ValidationError):
            validate(instance=valid_change_impact_response, schema=CHANGE_IMPACT_SCHEMA)
    
    def test_missing_recommended_action_fails(self, valid_change_impact_response):
        """Missing required recommended_action should fail."""
        del valid_change_impact_response["recommended_action"]
        with pytest.raises(ValidationError):
            validate(instance=valid_change_impact_response, schema=CHANGE_IMPACT_SCHEMA)
    
    def test_change_impact_score_must_be_integer(self):
        """change_impact_score must be an integer."""
        invalid = {
            "change_impact_score": 5.5,
            "change_summary": "Test",
            "recommended_action": "monitor"
        }
        with pytest.raises(ValidationError):
            validate(instance=invalid, schema=CHANGE_IMPACT_SCHEMA)
    
    def test_impact_increased_must_be_boolean(self, valid_change_impact_response):
        """impact_increased must be boolean if present."""
        valid_change_impact_response["impact_increased"] = "yes"
        with pytest.raises(ValidationError):
            validate(instance=valid_change_impact_response, schema=CHANGE_IMPACT_SCHEMA)
    
    def test_key_concerns_must_be_array(self, valid_change_impact_response):
        """key_concerns must be array of strings."""
        valid_change_impact_response["key_concerns"] = "single concern"
        with pytest.raises(ValidationError):
            validate(instance=valid_change_impact_response, schema=CHANGE_IMPACT_SCHEMA)


class TestVersionChangesSchema:
    """Tests for VERSION_CHANGES_SCHEMA validation."""
    
    def test_valid_version_changes_passes(self, valid_version_changes_response):
        """Valid version changes response should pass validation."""
        validate(instance=valid_version_changes_response, schema=VERSION_CHANGES_SCHEMA)
    
    def test_minimal_version_changes_passes(self):
        """Response with only required fields should pass."""
        minimal = {
            "key_provisions_added": [],
            "key_provisions_removed": [],
            "key_provisions_modified": ["Section 2 updated"],
            "summary": "Minor modifications to definitions."
        }
        validate(instance=minimal, schema=VERSION_CHANGES_SCHEMA)
    
    def test_missing_key_provisions_added_fails(self, valid_version_changes_response):
        """Missing required key_provisions_added should fail."""
        del valid_version_changes_response["key_provisions_added"]
        with pytest.raises(ValidationError):
            validate(instance=valid_version_changes_response, schema=VERSION_CHANGES_SCHEMA)
    
    def test_missing_key_provisions_removed_fails(self, valid_version_changes_response):
        """Missing required key_provisions_removed should fail."""
        del valid_version_changes_response["key_provisions_removed"]
        with pytest.raises(ValidationError):
            validate(instance=valid_version_changes_response, schema=VERSION_CHANGES_SCHEMA)
    
    def test_missing_key_provisions_modified_fails(self, valid_version_changes_response):
        """Missing required key_provisions_modified should fail."""
        del valid_version_changes_response["key_provisions_modified"]
        with pytest.raises(ValidationError):
            validate(instance=valid_version_changes_response, schema=VERSION_CHANGES_SCHEMA)
    
    def test_missing_summary_fails(self, valid_version_changes_response):
        """Missing required summary should fail."""
        del valid_version_changes_response["summary"]
        with pytest.raises(ValidationError):
            validate(instance=valid_version_changes_response, schema=VERSION_CHANGES_SCHEMA)
    
    def test_provisions_arrays_contain_strings(self):
        """Provision arrays must contain strings."""
        invalid = {
            "key_provisions_added": [123],
            "key_provisions_removed": [],
            "key_provisions_modified": [],
            "summary": "Test"
        }
        with pytest.raises(ValidationError):
            validate(instance=invalid, schema=VERSION_CHANGES_SCHEMA)


class TestSchemaRangeConstraints:
    """Tests for value range constraints (application-level validation)."""
    
    def test_business_impact_score_range_0_to_10(self, minimal_valid_analysis):
        """business_impact_score should be 0-10 (application validation)."""
        valid_scores = [0, 1, 5, 10]
        for score in valid_scores:
            minimal_valid_analysis["business_impact_score"] = score
            validate(instance=minimal_valid_analysis, schema=GEMINI_RESPONSE_SCHEMA)
    
    def test_recommended_action_valid_values(self, minimal_valid_analysis):
        """Test all valid recommended_action values."""
        valid_actions = ["ignore", "monitor", "engage", "urgent", "review"]
        for action in valid_actions:
            minimal_valid_analysis["recommended_action"] = action
            validate(instance=minimal_valid_analysis, schema=GEMINI_RESPONSE_SCHEMA)

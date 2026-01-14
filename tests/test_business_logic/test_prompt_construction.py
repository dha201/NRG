"""
Prompt construction tests.

Tests verify prompts inject context and schemas correctly.
"""
import pytest

from nrg_core.analysis.prompts import build_analysis_prompt, ANALYSIS_JSON_SCHEMA


class TestBuildAnalysisPrompt:
    """Tests for build_analysis_prompt() function."""
    
    def test_prompt_includes_nrg_context(self, sample_bill_dict, nrg_context):
        """Prompt should include NRG business context."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert "NRG Energy" in prompt
        assert "Retail Commodity" in prompt
        assert "ERCOT" in prompt
    
    def test_prompt_includes_bill_metadata(self, sample_bill_dict, nrg_context):
        """Prompt should include bill number, title, source, status."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert sample_bill_dict["number"] in prompt
        assert sample_bill_dict["title"] in prompt
        assert sample_bill_dict["source"] in prompt
        assert sample_bill_dict["status"] in prompt
    
    def test_prompt_includes_bill_type(self, sample_bill_dict, nrg_context):
        """Prompt should include bill type."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert sample_bill_dict["type"] in prompt
    
    def test_prompt_includes_bill_summary(self, sample_bill_dict, nrg_context):
        """Prompt should include bill summary/text."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert sample_bill_dict["summary"] in prompt
    
    def test_prompt_includes_json_schema(self, sample_bill_dict, nrg_context):
        """Prompt should include JSON schema for response format."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert "business_impact_score" in prompt
        assert "impact_type" in prompt
        assert "recommended_action" in prompt
        assert "nrg_business_verticals" in prompt
    
    def test_prompt_includes_analysis_instructions(self, sample_bill_dict, nrg_context):
        """Prompt should include analysis instructions."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert "legal precision" in prompt.lower() or "LEGAL PRECISION" in prompt
        assert "mandatory" in prompt.lower()
        assert "permissive" in prompt.lower()


class TestPromptWithLongText:
    """Tests for prompt handling of long bill text."""
    
    def test_long_summary_included(self, sample_bill_dict, nrg_context):
        """Long bill summary should be included in prompt."""
        sample_bill_dict["summary"] = "A" * 10000
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert "AAAA" in prompt
    
    def test_very_long_summary_handling(self, sample_bill_dict, nrg_context):
        """Very long summaries should be handled (may be truncated in LLM call)."""
        sample_bill_dict["summary"] = "Word " * 5000
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert "Word" in prompt


class TestPromptStructure:
    """Tests for prompt structure and formatting."""
    
    def test_prompt_has_clear_sections(self, sample_bill_dict, nrg_context):
        """Prompt should have clearly marked sections."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert "NRG BUSINESS CONTEXT" in prompt
        assert "LEGISLATION TO ANALYZE" in prompt or "LEGISLATION" in prompt
        assert "JSON" in prompt.upper()
    
    def test_prompt_requests_json_response(self, sample_bill_dict, nrg_context):
        """Prompt should explicitly request JSON response."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        assert "JSON" in prompt
    
    def test_prompt_includes_vertical_list(self, sample_bill_dict, nrg_context):
        """Prompt should include list of NRG business verticals."""
        prompt = build_analysis_prompt(sample_bill_dict, nrg_context)
        
        verticals = [
            "Retail Commodity",
            "Electric Generation",
            "Renewables",
            "Environmental"
        ]
        
        found_verticals = sum(1 for v in verticals if v in prompt)
        assert found_verticals >= 2


class TestAnalysisJsonSchema:
    """Tests for ANALYSIS_JSON_SCHEMA constant."""
    
    def test_schema_includes_required_fields(self):
        """Schema should document all required fields."""
        assert "business_impact_score" in ANALYSIS_JSON_SCHEMA
        assert "impact_type" in ANALYSIS_JSON_SCHEMA
        assert "impact_summary" in ANALYSIS_JSON_SCHEMA
        assert "recommended_action" in ANALYSIS_JSON_SCHEMA
    
    def test_schema_includes_optional_fields(self):
        """Schema should document optional analysis fields."""
        assert "legal_code_changes" in ANALYSIS_JSON_SCHEMA
        assert "nrg_business_verticals" in ANALYSIS_JSON_SCHEMA
        assert "affected_nrg_assets" in ANALYSIS_JSON_SCHEMA
        assert "internal_stakeholders" in ANALYSIS_JSON_SCHEMA
    
    def test_schema_documents_score_range(self):
        """Schema should document score range (0-10)."""
        assert "0-10" in ANALYSIS_JSON_SCHEMA or "integer" in ANALYSIS_JSON_SCHEMA
    
    def test_schema_documents_action_values(self):
        """Schema should document valid recommended_action values."""
        action_values = ["ignore", "monitor", "engage", "urgent"]
        for action in action_values:
            assert action in ANALYSIS_JSON_SCHEMA

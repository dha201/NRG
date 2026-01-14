"""
Pytest fixtures and configuration.

Following TDD principles:
- Fixtures provide real-world-like test data
- Mocks used only when unavoidable (LLM clients)
- Each test should be independent and fast
"""
import pytest
import json
from unittest.mock import MagicMock
from typing import Any
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# SAMPLE BILL DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_bill_dict() -> dict[str, Any]:
    """Minimal valid bill dictionary for testing."""
    return {
        "source": "OpenStates",
        "type": "State Bill",
        "number": "HB 1234",
        "title": "Energy Market Reform Act",
        "status": "Introduced",
        "url": "https://example.com/bill/1234",
        "summary": "An act relating to energy market reforms and consumer protection.",
        "sponsor": "Rep. Smith",
        "introduced_date": "2026-01-01",
        "policy_area": "Energy",
        "state": "TX",
        "versions": [],
    }


@pytest.fixture
def sample_bill_with_versions() -> dict[str, Any]:
    """Bill with multiple versions for version comparison tests."""
    return {
        "source": "OpenStates",
        "type": "State Bill",
        "number": "HB 5678",
        "title": "Renewable Energy Standards Act",
        "status": "Engrossed",
        "url": "https://example.com/bill/5678",
        "summary": "Latest version summary text.",
        "sponsor": "Sen. Jones",
        "introduced_date": "2026-01-01",
        "state": "TX",
        "versions": [
            {
                "version_number": 1,
                "version_type": "Introduced",
                "version_date": "2026-01-01",
                "full_text": "Section 1. Short title.\nSection 2. Definitions.\nSection 3. Requirements.",
                "text_hash": "abc123",
                "word_count": 100,
            },
            {
                "version_number": 2,
                "version_type": "House Committee Report",
                "version_date": "2026-01-15",
                "full_text": "Section 1. Short title.\nSection 2. Amended definitions.\nSection 3. Requirements.\nSection 4. New compliance measures.",
                "text_hash": "def456",
                "word_count": 150,
            },
            {
                "version_number": 3,
                "version_type": "Engrossed",
                "version_date": "2026-02-01",
                "full_text": "Section 1. Short title.\nSection 2. Final definitions.\nSection 3. Requirements.\nSection 4. Compliance measures.\nSection 5. Effective date.",
                "text_hash": "ghi789",
                "word_count": 200,
            },
        ],
    }


@pytest.fixture
def sample_version_dict() -> dict[str, Any]:
    """Single version dictionary for BillVersion tests."""
    return {
        "version_number": 1,
        "version_type": "Introduced",
        "version_date": "2026-01-01",
        "full_text": "Section 1. Short title.\nThis act may be cited as the Energy Reform Act.",
        "text_hash": "abc123hash",
        "word_count": 50,
        "pdf_url": "https://example.com/bill.pdf",
        "text_url": "https://example.com/bill.txt",
    }


@pytest.fixture
def sample_cached_bill() -> dict[str, Any]:
    """Cached bill data structure from database."""
    return {
        "bill_id": "OpenStates:HB 1234",
        "source": "OpenStates",
        "bill_number": "HB 1234",
        "title": "Energy Market Reform Act",
        "text_hash": "original_hash_abc123",
        "status": "Introduced",
        "full_data_json": json.dumps({
            "summary": "Original bill summary text.",
            "amendments": []
        }),
        "last_checked": "2026-01-01T00:00:00",
    }


# =============================================================================
# LLM RESPONSE FIXTURES
# =============================================================================

@pytest.fixture
def valid_analysis_response() -> dict[str, Any]:
    """Valid LLM analysis response """
    return {
        "bill_version": "introduced",
        "business_impact_score": 7,
        "impact_type": "regulatory_compliance",
        "impact_summary": "This bill would require NRG to implement new reporting requirements.",
        "legal_code_changes": {
            "sections_amended": ["Utilities Code §39.001"],
            "sections_added": ["New Chapter 40"],
            "sections_deleted": [],
            "chapters_repealed": [],
            "substance_of_changes": "Adds new compliance requirements for retail providers."
        },
        "application_scope": {
            "applies_to": ["retail electric providers", "generation companies"],
            "exclusions": ["municipal utilities"],
            "geographic_scope": ["Texas"]
        },
        "effective_dates": [
            {"date": "2027-01-01", "applies_to": "all provisions"}
        ],
        "mandatory_vs_permissive": {
            "mandatory_provisions": ["SHALL submit annual reports"],
            "permissive_provisions": ["MAY request extensions"]
        },
        "exceptions_and_exemptions": {
            "exceptions": ["facilities under 50 MW"],
            "exemptions": ["emergency operations"]
        },
        "nrg_business_verticals": ["Retail Commodity", "Electric Generation"],
        "nrg_vertical_impact_details": [
            {"vertical": "Retail Commodity", "impact": "New reporting requirements"}
        ],
        "nrg_relevant_excerpts": ["Section 3: All retail providers SHALL..."],
        "affected_nrg_assets": {
            "generation_facilities": ["Parish Gas Plant"],
            "geographic_exposure": ["ERCOT"],
            "business_units": ["NRG Retail"]
        },
        "financial_impact": "Estimated $500K annual compliance cost",
        "timeline": "Effective January 2027",
        "risk_or_opportunity": "risk",
        "recommended_action": "engage",
        "internal_stakeholders": ["Regulatory Affairs", "Legal", "Operations"]
    }


@pytest.fixture
def valid_change_impact_response() -> dict[str, Any]:
    """Valid LLM change impact response matching CHANGE_IMPACT_SCHEMA."""
    return {
        "change_impact_score": 6,
        "impact_increased": True,
        "change_summary": "Bill amendments added stricter compliance deadlines.",
        "nrg_impact": "NRG must accelerate implementation timeline by 6 months.",
        "recommended_action": "review",
        "key_concerns": ["Shortened compliance timeline", "Increased penalty provisions"]
    }


@pytest.fixture
def valid_version_changes_response() -> dict[str, Any]:
    """Valid LLM version changes response matching VERSION_CHANGES_SCHEMA."""
    return {
        "key_provisions_added": ["New Section 4 on compliance measures"],
        "key_provisions_removed": [],
        "key_provisions_modified": ["Section 2 definitions expanded"],
        "impact_evolution": "Impact increased from moderate to significant.",
        "compliance_changes": "New quarterly reporting requirement added.",
        "strategic_significance": "Accelerates NRG's compliance timeline.",
        "summary": "The amended version adds compliance measures and expands definitions."
    }


@pytest.fixture
def minimal_valid_analysis() -> dict[str, Any]:
    """Minimal valid analysis with only required fields."""
    return {
        "business_impact_score": 5,
        "impact_type": "operational",
        "impact_summary": "Moderate impact on operations.",
        "recommended_action": "monitor"
    }


# =============================================================================
# MOCK LLM CLIENT FIXTURES
# =============================================================================

@pytest.fixture
def mock_gemini_client(valid_analysis_response):
    """Mock Gemini client that returns valid JSON response."""
    client = MagicMock()
    
    mock_response = MagicMock()
    mock_response.text = json.dumps(valid_analysis_response)
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    
    text_part = MagicMock()
    text_part.text = json.dumps(valid_analysis_response)
    text_part.thought = False
    mock_response.candidates[0].content.parts = [text_part]
    
    client.models.generate_content.return_value = mock_response
    return client


@pytest.fixture
def mock_gemini_client_with_thought_signature(valid_analysis_response):
    """Mock Gemini client with thought_signature parts (Gemini 3 thinking model)."""
    client = MagicMock()
    
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    
    thought_part = MagicMock()
    thought_part.text = "encrypted_reasoning_trace"
    thought_part.thought = True
    
    text_part = MagicMock()
    text_part.text = json.dumps(valid_analysis_response)
    text_part.thought = False
    
    mock_response.candidates[0].content.parts = [thought_part, text_part]
    mock_response.text = json.dumps(valid_analysis_response)
    
    client.models.generate_content.return_value = mock_response
    return client


@pytest.fixture
def mock_gemini_client_failure():
    """Mock Gemini client that raises an exception."""
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("API rate limit exceeded")
    return client


@pytest.fixture
def mock_gemini_client_invalid_json():
    """Mock Gemini client that returns invalid JSON."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON {broken"
    client.models.generate_content.return_value = mock_response
    return client


@pytest.fixture
def mock_gemini_client_none_response():
    """Mock Gemini client where response.text is None."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = []
    client.models.generate_content.return_value = mock_response
    return client


@pytest.fixture
def mock_openai_client(valid_analysis_response):
    """Mock OpenAI client that returns valid JSON response."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = json.dumps(valid_analysis_response)
    client.responses.create.return_value = mock_response
    return client


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration dictionary."""
    return {
        "llm": {
            "provider": "gemini",
            "gemini": {
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
                "max_output_tokens": 8192
            },
            "retry": {
                "max_retries": 3,
                "base_delay": 1.0
            }
        }
    }


@pytest.fixture
def nrg_context() -> str:
    """Sample NRG business context for prompts."""
    return """NRG Energy is a leading integrated power company in the U.S.
Key business verticals include:
- Retail Commodity: Retail electricity services
- Electric Generation: Power generation facilities
- Renewables: Solar and wind projects
NRG operates primarily in ERCOT (Texas) market."""


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database for integration tests."""
    from nrg_core.db.cache import init_database
    db_path = tmp_path / "test.db"
    conn = init_database(str(db_path))
    yield conn
    conn.close()

"""
Tests for USCODELookup.

Validates:
- U.S. Code section resolution via API
- Result caching to avoid redundant calls
- Error handling for invalid citations
"""
import pytest
from unittest.mock import patch
from nrg_core.v2.cross_bill.uscode_lookup import USCODELookup, ResolvedDefinition


def test_lookup_usc_section():
    """Should fetch USC section text from API."""
    lookup = USCODELookup()
    
    # Mock USCODE API response
    mock_response = {
        "citation": "26 U.S.C. 48",
        "title": "SEC. 48. ENERGY CREDIT",
        "text": "For purposes of section 46, the energy credit for any taxable year is the energy percentage of the basis of each energy property...",
        "url": "https://uscode.house.gov/view.xhtml?req=26+USC+48"
    }
    
    with patch.object(lookup, '_call_uscode_api', return_value=mock_response):
        result = lookup.resolve("26 U.S.C. 48")
    
    assert isinstance(result, ResolvedDefinition)
    assert result.citation == "26 U.S.C. 48"
    assert "energy credit" in result.text.lower()
    assert result.url is not None


def test_lookup_caches_results():
    """Should cache lookups to avoid redundant API calls."""
    lookup = USCODELookup()
    
    mock_response = {
        "citation": "26 U.S.C. 48",
        "title": "Energy Credit",
        "text": "Energy credit provisions...",
        "url": "https://uscode.house.gov/..."
    }
    
    with patch.object(lookup, '_call_uscode_api', return_value=mock_response) as mock_call:
        # First call
        lookup.resolve("26 U.S.C. 48")
        # Second call (should use cache)
        lookup.resolve("26 U.S.C. 48")
    
    # API should only be called once
    assert mock_call.call_count == 1


def test_lookup_returns_none_for_invalid():
    """Should return None for invalid citations."""
    lookup = USCODELookup()
    
    with patch.object(lookup, '_call_uscode_api', side_effect=ValueError("Invalid")):
        result = lookup.resolve("invalid citation")
    
    assert result is None


def test_resolved_definition_has_relevance():
    """ResolvedDefinition should include relevance assessment."""
    lookup = USCODELookup()
    
    mock_response = {
        "citation": "42 U.S.C. 7401",
        "title": "Clean Air Act",
        "text": "Air quality standards...",
        "url": "https://..."
    }
    
    with patch.object(lookup, '_call_uscode_api', return_value=mock_response):
        result = lookup.resolve("42 U.S.C. 7401")
    
    assert result.relevance is not None

"""
Tests for DeepResearchAgent.

Validates:
- External API querying (mocked)
- Source retrieval and ranking
- Snippet extraction with citations
- Confidence scoring
"""
import pytest
from unittest.mock import patch, Mock
from nrg_core.v2.deep_research.research_agent import (
    DeepResearchAgent,
    ResearchResult,
    ResearchSource
)


def test_research_agent_retrieves_sources():
    """Should query external APIs for relevant context."""
    agent = DeepResearchAgent(
        openstates_key="test-os",
        billtrack_key="test-bt"
    )
    
    finding = {
        "statement": "Renewable energy tax credit extended",
        "quotes": [{"text": "Section 48 tax credit", "section": "2.1"}]
    }
    
    # Mock API responses
    mock_openstates = [
        {
            "source_type": "openstates",
            "bill_id": "HB456",
            "title": "Similar renewable credit bill",
            "text": "Renewable energy tax credit provisions",
            "url": "https://openstates.org/bills/HB456"
        }
    ]
    
    mock_billtrack = [
        {
            "source_type": "billtrack50",
            "bill_id": "SB789",
            "title": "Renewable energy credits",
            "text": "Summary of renewable credits",
            "url": "https://billtrack50.com/SB789"
        }
    ]
    
    with patch.object(agent, '_query_openstates', return_value=mock_openstates), \
         patch.object(agent, '_query_billtrack', return_value=mock_billtrack):
        
        result = agent.research(finding=finding, bill_text="...")
    
    assert isinstance(result, ResearchResult)
    assert len(result.sources) >= 2
    assert result.sources[0].url is not None


def test_research_returns_snippets_with_citations():
    """Each source should have snippet and citation."""
    agent = DeepResearchAgent(openstates_key="test", billtrack_key="test")
    
    finding = {
        "statement": "Tax credit extended",
        "quotes": [{"text": "...", "section": "2.1"}]
    }
    
    mock_sources = [
        {
            "source_type": "openstates",
            "bill_id": "HB123",
            "title": "Energy Tax Credit Bill",
            "text": "Section 48 provides investment tax credit for renewable energy projects with extended timeline",
            "url": "https://congress.gov/bill/HB123"
        }
    ]
    
    with patch.object(agent, '_query_openstates', return_value=mock_sources), \
         patch.object(agent, '_query_billtrack', return_value=[]), \
         patch.object(agent, '_query_congress', return_value=[]):
        result = agent.research(finding, "bill text")
    
    assert result.sources[0].snippet is not None
    assert result.sources[0].url is not None
    assert len(result.sources[0].snippet) <= 500  # Truncated snippet


def test_research_confidence_increases_with_sources():
    """More high-relevance sources should increase confidence."""
    agent = DeepResearchAgent(openstates_key="test", billtrack_key="test")
    
    finding = {
        "statement": "Tax credit",
        "quotes": [{"text": "tax credit", "section": "1"}]
    }
    
    # Test with 1 source
    mock_one = [
        {
            "source_type": "openstates",
            "bill_id": "HB1",
            "title": "Tax Bill",
            "text": "tax credit provisions",
            "url": "https://..."
        }
    ]
    
    with patch.object(agent, '_query_openstates', return_value=mock_one), \
         patch.object(agent, '_query_billtrack', return_value=[]), \
         patch.object(agent, '_query_congress', return_value=[]):
        result_one = agent.research(finding, "bill")
    
    # Test with 3 sources
    mock_three = mock_one * 3
    
    with patch.object(agent, '_query_openstates', return_value=mock_three), \
         patch.object(agent, '_query_billtrack', return_value=[]), \
         patch.object(agent, '_query_congress', return_value=[]):
        result_three = agent.research(finding, "bill")
    
    # More sources = higher confidence
    assert result_three.research_confidence >= result_one.research_confidence

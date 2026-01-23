"""
Deep Research Infrastructure.

Provides external context enrichment through API queries and claim validation.

Components:
- DeepResearchAgent: Queries external APIs (OpenStates, BillTrack50, Congress.gov)
- CheckerAgent: Validates research claims against source snippets
- ResearchResult/ResearchSource: Data models for research outputs

Design rationale:
- Separate research from validation to allow independent testing
- Multiple API sources for comprehensive coverage
- Trust scoring based on source authority and agreement
"""
from nrg_core.v2.deep_research.research_agent import (
    DeepResearchAgent,
    ResearchResult,
    ResearchSource
)

__all__ = ["DeepResearchAgent", "ResearchResult", "ResearchSource"]

"""
Deep Research Agent.

Queries external APIs and web sources to enrich legislative analysis with
external context, precedents, and supporting evidence.

Design rationale:
- Multiple data sources (OpenStates, BillTrack50, Congress.gov) for coverage
- Snippet extraction with length limits for LLM context efficiency
- Relevance scoring based on keyword overlap (simple but effective)
- Confidence computed from source count and relevance quality

Why these sources:
- OpenStates: State-level bill tracking with full text
- BillTrack50: Commercial bill tracking with summaries
- Congress.gov: Federal legislation (public API)

Usage:
    agent = DeepResearchAgent(
        openstates_key="...",
        billtrack_key="...",
        openai_key="..."
    )
    result = agent.research(finding, bill_text)
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ResearchSource:
    """
    Single external source from research.
    
    Attributes:
        source_type: API source identifier ("openstates", "billtrack50", "congress")
        title: Source document title
        snippet: Extracted relevant text (max 500 chars)
        url: Source URL for citation
        relevance: Assessed relevance level ("high", "medium", "low")
        checker_validated: Whether claim was validated by checker (set later)
        checker_confidence: Checker's confidence score (set later)
    """
    source_type: str
    title: str
    snippet: str
    url: str
    relevance: str
    checker_validated: Optional[bool] = None
    checker_confidence: Optional[float] = None


@dataclass
class ResearchResult:
    """
    Aggregated result from deep research.
    
    Attributes:
        sources: List of retrieved sources with snippets
        research_confidence: Overall confidence in research quality (0-1)
        summary: Human-readable summary of findings
    
    Design: Separates raw sources from derived metrics for flexibility.
    """
    sources: List[ResearchSource]
    research_confidence: float
    summary: str


class DeepResearchAgent:
    """
    Query external sources for legislative context.
    
    Sources queried:
    1. OpenStates API (state bills) - requires API key
    2. BillTrack50 (bill tracking) - requires API key  
    3. Congress.gov (federal bills) - public API, no key needed
    
    Pipeline:
    1. Generate search query from finding statement
    2. Query each available API source
    3. Extract and truncate snippets
    4. Assess relevance via keyword matching
    5. Compute overall confidence
    6. Generate summary
    
    Why not use web scraping:
    - APIs provide structured, reliable data
    - Scraping is fragile and may violate ToS
    - APIs have rate limits but are predictable
    """
    
    def __init__(
        self,
        openstates_key: str | None = None,
        billtrack_key: str | None = None,
        openai_key: str | None = None
    ):
        """
        Initialize research agent with API keys.
        
        Args:
            openstates_key: OpenStates API key (optional)
            billtrack_key: BillTrack50 API key (optional)
            openai_key: OpenAI key for query generation (optional)
        """
        self.openstates_key = openstates_key
        self.billtrack_key = billtrack_key
        self.llm_client = OpenAI(api_key=openai_key) if openai_key else None
    
    def research(
        self,
        finding: Dict[str, Any],
        bill_text: str,
        max_sources: int = 5
    ) -> ResearchResult:
        """
        Research external context for a finding.
        
        Process:
        1. Generate search query from finding
        2. Query all available APIs
        3. Rank and filter sources
        4. Compute confidence and summary
        
        Args:
            finding: Finding dict with 'statement' and 'quotes' keys
            bill_text: Full bill text for context
            max_sources: Maximum sources to return (default: 5)
        
        Returns:
            ResearchResult with sources, confidence, and summary
        """
        # Generate search query from finding statement
        query = self._generate_query(finding)
        
        # Query all available sources
        sources = []
        
        # OpenStates (state bills)
        if self.openstates_key:
            os_results = self._query_openstates(query, max_results=2)
            sources.extend(os_results)
        
        # BillTrack50 (commercial tracking)
        if self.billtrack_key:
            bt_results = self._query_billtrack(query, max_results=2)
            sources.extend(bt_results)
        
        # Congress.gov (federal - public API)
        congress_results = self._query_congress(query, max_results=1)
        sources.extend(congress_results)
        
        # Limit to max_sources
        sources = sources[:max_sources]
        
        # Convert to ResearchSource objects with relevance assessment
        research_sources = []
        for src in sources:
            snippet = self._extract_snippet(src, max_length=500)
            relevance = self._assess_relevance(src, finding)
            
            research_sources.append(ResearchSource(
                source_type=src.get("source_type", "unknown"),
                title=src.get("title", ""),
                snippet=snippet,
                url=src.get("url", ""),
                relevance=relevance
            ))
        
        # Compute research confidence based on source quality
        confidence = self._compute_confidence(research_sources)
        
        # Generate human-readable summary
        summary = self._generate_summary(research_sources, finding)
        
        return ResearchResult(
            sources=research_sources,
            research_confidence=confidence,
            summary=summary
        )
    
    def _generate_query(self, finding: Dict) -> str:
        """
        Generate search query from finding statement.
        
        Strategy: Extract first 5 words as keywords.
        Why simple approach: Legislative terminology is specific enough
        that simple keyword extraction works well.
        
        Args:
            finding: Finding dict with 'statement' key
        
        Returns:
            Search query string
        """
        statement = finding.get("statement", "")
        # Extract key terms (simplified - first 5 words)
        keywords = statement.split()[:5]
        return " ".join(keywords)
    
    def _query_openstates(self, query: str, max_results: int) -> List[Dict]:
        """
        Query OpenStates API for state legislation.
        
        API: https://v3.openstates.org/bills
        Requires API key in X-API-KEY header.
        
        Args:
            query: Search query string
            max_results: Maximum results to return
        
        Returns:
            List of bill dicts with source_type, title, text, url
        """
        if not self.openstates_key:
            return []
        
        url = "https://v3.openstates.org/bills"
        headers = {"X-API-KEY": self.openstates_key}
        params = {"q": query, "per_page": max_results}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
            
            return [
                {
                    "source_type": "openstates",
                    "bill_id": r.get("identifier"),
                    "title": r.get("title", ""),
                    "text": r.get("latest_passage", {}).get("text", r.get("title", "")),
                    "url": f"https://openstates.org/bills/{r.get('id', '')}"
                }
                for r in results
            ]
        except Exception as e:
            # Log error but don't fail - other sources may work
            logger.warning("OpenStates API error", exc_info=True)
            return []
    
    def _query_billtrack(self, query: str, max_results: int) -> List[Dict]:
        """
        Query BillTrack50 API.
        
        Note: Placeholder implementation. BillTrack50 is a commercial
        service requiring account setup. In production, would implement
        actual API integration.
        
        Args:
            query: Search query string
            max_results: Maximum results to return
        
        Returns:
            List of bill dicts (empty for now)
        """
        # Placeholder: would implement actual BillTrack50 API call
        # BillTrack50 requires commercial account setup
        return []
    
    def _query_congress(self, query: str, max_results: int) -> List[Dict]:
        """
        Query Congress.gov API for federal legislation.
        
        API: https://api.congress.gov/v3/bill
        Public API, no key required (but rate limited).
        
        Args:
            query: Search query string
            max_results: Maximum results to return
        
        Returns:
            List of bill dicts with source_type, title, text, url
        """
        url = "https://api.congress.gov/v3/bill"
        params = {"query": query, "limit": max_results}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("bills", [])
            
            return [
                {
                    "source_type": "congress",
                    "bill_id": r.get("number", ""),
                    "title": r.get("title", ""),
                    "text": r.get("summary", {}).get("text", r.get("title", "")),
                    "url": r.get("url", "")
                }
                for r in results
            ]
        except Exception as e:
            # Log error but don't fail
            logger.warning("Congress.gov API error", exc_info=True)
            return []
    
    def _extract_snippet(self, source: Dict, max_length: int) -> str:
        """
        Extract relevant snippet from source, truncated to max_length.
        
        Why truncate: LLM context is limited, and long snippets
        add noise without proportional value.
        
        Args:
            source: Source dict with 'text' or 'title' key
            max_length: Maximum snippet length
        
        Returns:
            Truncated snippet string
        """
        text = source.get("text", source.get("title", ""))
        
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def _assess_relevance(self, source: Dict, finding: Dict) -> str:
        """
        Assess source relevance to finding.
        
        Strategy: Simple keyword overlap scoring.
        - 2+ keyword matches: "high"
        - 1 keyword match: "medium"
        - 0 matches: "low"
        
        Why simple approach: More sophisticated NLP adds latency
        without significant accuracy improvement for legislative text.
        
        Args:
            source: Source dict with 'text' key
            finding: Finding dict with 'statement' key
        
        Returns:
            Relevance level: "high", "medium", or "low"
        """
        statement = finding.get("statement", "").lower()
        source_text = source.get("text", "").lower()
        
        # Extract first 3 keywords from statement
        keywords = [w for w in statement.split()[:5] if len(w) > 3][:3]
        matches = sum(1 for kw in keywords if kw in source_text)
        
        if matches >= 2:
            return "high"
        elif matches == 1:
            return "medium"
        else:
            return "low"
    
    def _compute_confidence(self, sources: List[ResearchSource]) -> float:
        """
        Compute overall research confidence.
        
        Formula:
        - Base: 0.0 (no sources) to 0.5 (max from source count)
        - Bonus: 0.0 to 0.5 from high-relevance sources
        
        Why this formula: Balances quantity (more sources) with
        quality (high relevance) for robust confidence estimation.
        
        Args:
            sources: List of ResearchSource objects
        
        Returns:
            Confidence score 0.0-1.0
        """
        if not sources:
            return 0.0
        
        # Source count contribution: 0.1 per source, max 0.5
        source_score = min(0.5, len(sources) * 0.1)
        
        # High relevance contribution: 0.2 per high-relevance, max 0.5
        high_relevance = sum(1 for s in sources if s.relevance == "high")
        relevance_score = min(0.5, high_relevance * 0.2)
        
        return source_score + relevance_score
    
    def _generate_summary(
        self,
        sources: List[ResearchSource],
        finding: Dict
    ) -> str:
        """
        Generate human-readable summary of research findings.
        
        Args:
            sources: List of ResearchSource objects
            finding: Original finding for context
        
        Returns:
            Summary string
        """
        if not sources:
            return "No external sources found."
        
        high_count = sum(1 for s in sources if s.relevance == "high")
        return f"Found {len(sources)} relevant sources. {high_count} high-relevance matches."

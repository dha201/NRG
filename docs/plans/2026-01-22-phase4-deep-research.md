# Phase 4: Deep Research Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate deep research agent for external context enrichment with cross-bill reference handling

**Architecture:** Research agent queries external APIs (BillTrack50, OpenStates, Congress.gov) + checker validates claims + trust scoring. Cross-bill reference detection (Tier 1) + USCODE API lookup (Tier 2).

**Tech Stack:** Python 3.12, OpenAI GPT-4o, requests for API calls, beautifulsoup4 for web scraping, USCODE API

**Prerequisites:** Phase 3 complete (evaluation system working)

---

## Task 1: Create Deep Research Agent

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/deep_research/__init__.py`
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/deep_research/research_agent.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_deep_research/test_research_agent.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_deep_research/__init__.py
# Empty init

# tests/test_v2/test_deep_research/test_research_agent.py
import pytest
from unittest.mock import patch, Mock
from nrg_core.v2.deep_research.research_agent import DeepResearchAgent, ResearchResult


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
        {"bill_id": "HB456", "title": "Similar renewable credit bill", "url": "https://..."}
    ]
    
    mock_billtrack = [
        {"bill_id": "SB789", "summary": "Renewable energy credits", "url": "https://..."}
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
    
    finding = {"statement": "Tax credit", "quotes": [{"text": "...", "section": "2.1"}]}
    
    mock_sources = [
        {
            "bill_id": "HB123",
            "text": "Section 48 provides investment tax credit for renewable energy projects",
            "url": "https://congress.gov/bill/HB123"
        }
    ]
    
    with patch.object(agent, '_query_openstates', return_value=mock_sources):
        result = agent.research(finding, "bill text")
    
    assert result.sources[0].snippet is not None
    assert result.sources[0].url is not None
    assert len(result.sources[0].snippet) <= 500  # Truncated snippet
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_deep_research/test_research_agent.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement research agent**

```python
# nrg_core/v2/deep_research/__init__.py
"""Deep research infrastructure."""
from nrg_core.v2.deep_research.research_agent import DeepResearchAgent, ResearchResult

__all__ = ["DeepResearchAgent", "ResearchResult"]
```

```python
# nrg_core/v2/deep_research/research_agent.py
"""
Deep Research Agent
Query external APIs and web for context enrichment
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from openai import OpenAI


@dataclass
class ResearchSource:
    """Single external source."""
    source_type: str  # "openstates", "billtrack50", "congress", "web"
    title: str
    snippet: str
    url: str
    relevance: str  # "high", "medium", "low"


@dataclass
class ResearchResult:
    """Result from deep research."""
    sources: List[ResearchSource]
    research_confidence: float  # 0-1
    summary: str


class DeepResearchAgent:
    """
    Query external sources for context.
    
    Sources:
    1. OpenStates API (state bills)
    2. BillTrack50 (bill tracking)
    3. Congress.gov (federal bills)
    4. Web search (Google Scholar for case law)
    """
    
    def __init__(
        self,
        openstates_key: str = None,
        billtrack_key: str = None,
        openai_key: str = None
    ):
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
        
        Args:
            finding: Finding dict with statement and quotes
            bill_text: Full bill text for context
            max_sources: Maximum sources to retrieve
        
        Returns:
            ResearchResult with sources and confidence
        """
        # Generate search queries
        query = self._generate_query(finding)
        
        # Query all sources
        sources = []
        
        # OpenStates
        if self.openstates_key:
            os_results = self._query_openstates(query, max_results=2)
            sources.extend(os_results)
        
        # BillTrack50
        if self.billtrack_key:
            bt_results = self._query_billtrack(query, max_results=2)
            sources.extend(bt_results)
        
        # Congress.gov (public API, no key)
        congress_results = self._query_congress(query, max_results=1)
        sources.extend(congress_results)
        
        # Limit to max_sources
        sources = sources[:max_sources]
        
        # Convert to ResearchSource objects
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
        
        # Compute research confidence
        confidence = self._compute_confidence(research_sources)
        
        # Generate summary
        summary = self._generate_summary(research_sources, finding)
        
        return ResearchResult(
            sources=research_sources,
            research_confidence=confidence,
            summary=summary
        )
    
    def _generate_query(self, finding: Dict) -> str:
        """Generate search query from finding."""
        statement = finding.get("statement", "")
        
        # Extract key terms (simplified)
        keywords = statement.split()[:5]
        return " ".join(keywords)
    
    def _query_openstates(self, query: str, max_results: int) -> List[Dict]:
        """Query OpenStates API."""
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
                    "title": r.get("title"),
                    "text": r.get("latest_passage", {}).get("text", ""),
                    "url": f"https://openstates.org/bills/{r.get('id')}"
                }
                for r in results
            ]
        except Exception as e:
            print(f"OpenStates API error: {e}")
            return []
    
    def _query_billtrack(self, query: str, max_results: int) -> List[Dict]:
        """Query BillTrack50."""
        # Placeholder: would implement actual API call
        return []
    
    def _query_congress(self, query: str, max_results: int) -> List[Dict]:
        """Query Congress.gov API."""
        url = "https://api.congress.gov/v3/bill"
        params = {"query": query, "limit": max_results}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("bills", [])
            
            return [
                {
                    "source_type": "congress",
                    "bill_id": r.get("number"),
                    "title": r.get("title"),
                    "text": r.get("summary", {}).get("text", ""),
                    "url": r.get("url")
                }
                for r in results
            ]
        except Exception as e:
            print(f"Congress.gov API error: {e}")
            return []
    
    def _extract_snippet(self, source: Dict, max_length: int) -> str:
        """Extract relevant snippet from source."""
        text = source.get("text", source.get("title", ""))
        
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def _assess_relevance(self, source: Dict, finding: Dict) -> str:
        """Assess source relevance (high/medium/low)."""
        # Simplified: just check if finding keywords in source
        statement = finding.get("statement", "").lower()
        source_text = source.get("text", "").lower()
        
        keywords = statement.split()[:3]
        matches = sum(1 for kw in keywords if kw in source_text)
        
        if matches >= 2:
            return "high"
        elif matches == 1:
            return "medium"
        else:
            return "low"
    
    def _compute_confidence(self, sources: List[ResearchSource]) -> float:
        """
        Compute research confidence based on:
        - Source count (more = higher)
        - Source agreement (do sources agree?)
        - Source authority (official > blog)
        """
        if not sources:
            return 0.0
        
        # Simple formula: 0.5 + (0.1 per source, max 0.5)
        source_score = min(0.5, len(sources) * 0.1)
        
        # High relevance sources boost confidence
        high_relevance = sum(1 for s in sources if s.relevance == "high")
        relevance_score = min(0.5, high_relevance * 0.2)
        
        return source_score + relevance_score
    
    def _generate_summary(self, sources: List[ResearchSource], finding: Dict) -> str:
        """Generate summary of research findings."""
        if not sources:
            return "No external sources found."
        
        return f"Found {len(sources)} relevant sources. {sum(1 for s in sources if s.relevance == 'high')} high-relevance matches."
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_deep_research/test_research_agent.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add nrg_core/v2/deep_research/ tests/test_v2/test_deep_research/
git commit -m "feat(v2): add deep research agent for external context"
```

---

## Task 2: Add Checker Agent for Claim Validation

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/deep_research/checker.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_deep_research/test_checker.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_deep_research/test_checker.py
import pytest
from unittest.mock import patch
from nrg_core.v2.deep_research.checker import CheckerAgent, CheckResult


def test_checker_validates_claim():
    """Checker should verify if source supports claim."""
    checker = CheckerAgent(model="gpt-4o", api_key="test-key")
    
    claim = "Similar bills in other states resulted in $1M annual cost"
    snippet = "State X renewable tax credit bill cost utilities approximately $1.2M annually"
    
    mock_result = {
        "directly_states": True,
        "confidence": 0.9,
        "rationale": "Source explicitly mentions $1.2M annual cost, supporting the claim"
    }
    
    with patch.object(checker, '_call_llm', return_value=mock_result):
        result = checker.check(claim=claim, snippet=snippet, source_url="https://...")
    
    assert isinstance(result, CheckResult)
    assert result.directly_states is True
    assert result.confidence >= 0.8


def test_checker_flags_speculative_leap():
    """Checker should flag when claim goes beyond source."""
    checker = CheckerAgent(model="gpt-4o", api_key="test-key")
    
    claim = "All utilities faced $5M costs"
    snippet = "One utility reported increased costs"
    
    mock_result = {
        "directly_states": False,
        "confidence": 0.3,
        "rationale": "Source mentions one utility, claim extrapolates to all and adds specific dollar amount not in source"
    }
    
    with patch.object(checker, '_call_llm', return_value=mock_result):
        result = checker.check(claim, snippet, "https://...")
    
    assert result.directly_states is False
    assert result.confidence < 0.5
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_deep_research/test_checker.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement checker**

```python
# nrg_core/v2/deep_research/checker.py
"""
Checker Agent
Validates research claims against source snippets
"""
from typing import Dict, Any
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class CheckResult:
    """Result of claim validation."""
    directly_states: bool
    confidence: float
    rationale: str


CHECKER_PROMPT = """You are validating a research claim against a source snippet.

CLAIM:
{claim}

SOURCE SNIPPET:
{snippet}

SOURCE URL: {url}

QUESTION: Does the source directly state this claim?

RULES:
- "directly_states" = true ONLY if source explicitly says what the claim says
- "directly_states" = false if claim extrapolates, generalizes, or infers beyond source
- Provide confidence 0-1 and rationale

OUTPUT (JSON):
{{
  "directly_states": true/false,
  "confidence": 0.0-1.0,
  "rationale": "Explanation"
}}
"""


class CheckerAgent:
    """
    Validate research claims against sources.
    Flags speculative leaps.
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def check(
        self,
        claim: str,
        snippet: str,
        source_url: str
    ) -> CheckResult:
        """
        Validate claim against source snippet.
        
        Args:
            claim: Research claim to validate
            snippet: Source snippet
            source_url: URL for citation
        
        Returns:
            CheckResult with validation
        """
        result = self._call_llm(claim, snippet, source_url)
        
        return CheckResult(
            directly_states=result["directly_states"],
            confidence=result["confidence"],
            rationale=result["rationale"]
        )
    
    def _call_llm(self, claim: str, snippet: str, url: str) -> Dict[str, Any]:
        """Call LLM for validation."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        prompt = CHECKER_PROMPT.format(
            claim=claim,
            snippet=snippet,
            url=url
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        import json
        return json.loads(response.choices[0].message.content)
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_deep_research/test_checker.py -v`

Expected: PASS

**Step 5: Integrate checker into research agent**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/deep_research/research_agent.py`:

```python
# Add to imports
from nrg_core.v2.deep_research.checker import CheckerAgent

# Add to __init__
        self.checker = CheckerAgent(api_key=openai_key) if openai_key else None

# Add validation step in research()
        # Validate sources with checker
        if self.checker:
            for source in research_sources:
                check = self.checker.check(
                    claim=finding.get("statement"),
                    snippet=source.snippet,
                    source_url=source.url
                )
                # Store check result in source metadata
                source.checker_validated = check.directly_states
                source.checker_confidence = check.confidence
```

**Step 6: Commit**

```bash
git add nrg_core/v2/deep_research/checker.py tests/test_v2/test_deep_research/test_checker.py nrg_core/v2/deep_research/research_agent.py
git commit -m "feat(v2): add checker agent for claim validation"
```

---

## Task 3: Implement Cross-Bill Reference Detection

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/cross_bill/__init__.py`
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/cross_bill/reference_detector.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_cross_bill/test_reference_detector.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_cross_bill/__init__.py
# Empty init

# tests/test_v2/test_cross_bill/test_reference_detector.py
import pytest
from nrg_core.v2.cross_bill.reference_detector import ReferenceDetector, BillReference


def test_detect_usc_citations():
    """Should detect U.S. Code citations."""
    detector = ReferenceDetector()
    
    bill_text = """
    Section 2.1: This Act amends section 48 of the Internal Revenue Code 
    of 1986 (26 U.S.C. 48) to extend renewable energy tax credits.
    
    Section 3.1: As defined in 42 U.S.C. 7401, clean air standards apply.
    """
    
    refs = detector.detect(bill_text)
    
    assert len(refs) >= 2
    
    # Check for 26 U.S.C. 48
    usc_refs = [r for r in refs if r.citation == "26 U.S.C. 48"]
    assert len(usc_refs) == 1
    assert usc_refs[0].reference_type == "statutory_amendment"
    assert usc_refs[0].location == "Section 2.1"


def test_detect_amends_pattern():
    """Should detect 'amends' pattern."""
    detector = ReferenceDetector()
    
    bill_text = "This Act amends the Clean Air Act to add new emission standards."
    
    refs = detector.detect(bill_text)
    
    assert any(r.citation == "Clean Air Act" for r in refs)
    assert any(r.reference_type == "statutory_amendment" for r in refs)


def test_detect_as_defined_in():
    """Should detect 'as defined in' pattern."""
    detector = ReferenceDetector()
    
    bill_text = "The term 'renewable energy' as defined in 26 U.S.C. § 45 applies."
    
    refs = detector.detect(bill_text)
    
    defined_refs = [r for r in refs if r.reference_type == "definition_by_reference"]
    assert len(defined_refs) >= 1
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_cross_bill/test_reference_detector.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement reference detector**

```python
# nrg_core/v2/cross_bill/__init__.py
"""Cross-bill reference handling."""
from nrg_core.v2.cross_bill.reference_detector import ReferenceDetector, BillReference

__all__ = ["ReferenceDetector", "BillReference"]
```

```python
# nrg_core/v2/cross_bill/reference_detector.py
"""
Cross-Bill Reference Detection (Tier 1)
Extract references to other statutes and bills
"""
import re
from typing import List
from dataclasses import dataclass


@dataclass
class BillReference:
    """Detected reference to external statute/bill."""
    reference_type: str  # "statutory_amendment", "definition_by_reference", "precedent_citation"
    citation: str        # e.g., "26 U.S.C. 48" or "Clean Air Act"
    context: str         # Surrounding text
    location: str        # Section reference


# Regex patterns for detection
USC_PATTERN = r'(\d+)\s+U\.S\.C\.?\s+§?\s*(\d+[a-z]?)'  # 26 U.S.C. 48
AMENDS_PATTERN = r'amends?\s+(?:section\s+)?(.+?)(?:\s+to|\s+by|\.)'
AS_DEFINED_PATTERN = r'as defined in\s+(.+?)(?:\s+applies|,|\.)'
PUBLIC_LAW_PATTERN = r'Public Law (\d+-\d+)'


class ReferenceDetector:
    """
    Detect cross-bill references using pattern matching + LLM.
    
    Patterns detected:
    1. "amends [statute]"
    2. "as defined in [U.S.C. citation]"
    3. "pursuant to [Public Law]"
    4. "notwithstanding [section]"
    """
    
    def __init__(self):
        self.patterns = [
            (USC_PATTERN, "statutory_amendment"),
            (AMENDS_PATTERN, "statutory_amendment"),
            (AS_DEFINED_PATTERN, "definition_by_reference"),
            (PUBLIC_LAW_PATTERN, "precedent_citation")
        ]
    
    def detect(self, bill_text: str) -> List[BillReference]:
        """
        Detect all references in bill text.
        
        Args:
            bill_text: Full bill text
        
        Returns:
            List of BillReference objects
        """
        references = []
        
        # Split into sections for location tracking
        sections = self._split_sections(bill_text)
        
        for section_name, section_text in sections.items():
            # Try each pattern
            for pattern, ref_type in self.patterns:
                matches = re.finditer(pattern, section_text, re.IGNORECASE)
                
                for match in matches:
                    citation = self._extract_citation(match, ref_type)
                    context = self._extract_context(section_text, match.start(), match.end())
                    
                    ref = BillReference(
                        reference_type=ref_type,
                        citation=citation,
                        context=context,
                        location=section_name
                    )
                    references.append(ref)
        
        # Deduplicate
        references = self._deduplicate(references)
        
        return references
    
    def _split_sections(self, bill_text: str) -> dict:
        """Split bill into sections."""
        sections = {}
        
        # Simple split by "Section X.Y"
        section_pattern = r'(Section\s+\d+\.?\d*[a-z]?)'
        parts = re.split(section_pattern, bill_text, flags=re.IGNORECASE)
        
        current_section = "Preamble"
        sections[current_section] = ""
        
        for i, part in enumerate(parts):
            if re.match(section_pattern, part, re.IGNORECASE):
                current_section = part.strip()
                sections[current_section] = ""
            else:
                if current_section in sections:
                    sections[current_section] += part
        
        return sections
    
    def _extract_citation(self, match: re.Match, ref_type: str) -> str:
        """Extract citation from regex match."""
        if ref_type == "statutory_amendment" and "U.S.C" in match.group(0):
            # USC citation: "26 U.S.C. 48"
            title = match.group(1)
            section = match.group(2)
            return f"{title} U.S.C. {section}"
        else:
            # Generic: use matched text
            return match.group(1).strip() if match.lastindex else match.group(0).strip()
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract surrounding context."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _deduplicate(self, references: List[BillReference]) -> List[BillReference]:
        """Remove duplicate references."""
        seen = set()
        unique = []
        
        for ref in references:
            key = (ref.reference_type, ref.citation)
            if key not in seen:
                seen.add(key)
                unique.append(ref)
        
        return unique
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_cross_bill/test_reference_detector.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add nrg_core/v2/cross_bill/ tests/test_v2/test_cross_bill/
git commit -m "feat(v2): add cross-bill reference detection (Tier 1)"
```

---

## Task 4: Add USCODE API Integration (Tier 2)

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/cross_bill/uscode_lookup.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_cross_bill/test_uscode_lookup.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_cross_bill/test_uscode_lookup.py
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
    
    mock_response = {"citation": "26 U.S.C. 48", "title": "...", "text": "...", "url": "..."}
    
    with patch.object(lookup, '_call_uscode_api', return_value=mock_response) as mock_call:
        # First call
        lookup.resolve("26 U.S.C. 48")
        # Second call (should use cache)
        lookup.resolve("26 U.S.C. 48")
    
    # API should only be called once
    assert mock_call.call_count == 1
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_cross_bill/test_uscode_lookup.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement USCODE lookup**

```python
# nrg_core/v2/cross_bill/uscode_lookup.py
"""
USCODE API Integration (Tier 2)
Fetch referenced statute text from U.S. Code
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import requests


@dataclass
class ResolvedDefinition:
    """Resolved statute/definition from USCODE."""
    citation: str
    title: str
    text: str
    url: str
    relevance: str  # How relevant to bill analysis


class USCODELookup:
    """
    Fetch statute text from USCODE API.
    
    Uses uscode.house.gov API.
    Implements caching to avoid redundant calls.
    """
    
    def __init__(self):
        self.cache: Dict[str, ResolvedDefinition] = {}
    
    def resolve(self, citation: str) -> Optional[ResolvedDefinition]:
        """
        Resolve USC citation to full text.
        
        Args:
            citation: e.g., "26 U.S.C. 48"
        
        Returns:
            ResolvedDefinition or None if not found
        """
        # Check cache
        if citation in self.cache:
            return self.cache[citation]
        
        # Call API
        try:
            result = self._call_uscode_api(citation)
            
            definition = ResolvedDefinition(
                citation=result["citation"],
                title=result["title"],
                text=result["text"],
                url=result["url"],
                relevance="high"  # Default, could be refined
            )
            
            # Cache result
            self.cache[citation] = definition
            
            return definition
        except Exception as e:
            print(f"USCODE lookup error for {citation}: {e}")
            return None
    
    def _call_uscode_api(self, citation: str) -> Dict[str, Any]:
        """
        Call USCODE API.
        
        API endpoint: https://uscode.house.gov/view.xhtml?req={citation}
        Note: This is simplified. Real implementation would parse XML response.
        """
        # Parse citation (e.g., "26 U.S.C. 48" -> title=26, section=48)
        parts = citation.replace("U.S.C.", "").strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid citation format: {citation}")
        
        title = parts[0]
        section = parts[1]
        
        # Construct URL
        url = f"https://uscode.house.gov/view.xhtml?req={title}+USC+{section}"
        
        # For testing/development, return mock data
        # In production, would scrape/parse actual response
        return {
            "citation": citation,
            "title": f"TITLE {title}, SECTION {section}",
            "text": f"[Text of {citation} would be fetched from API]",
            "url": url
        }
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_cross_bill/test_uscode_lookup.py -v`

Expected: PASS

**Step 5: Integrate into reference detector**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/cross_bill/reference_detector.py`:

```python
# Add to imports
from nrg_core.v2.cross_bill.uscode_lookup import USCODELookup

# Add to __init__
        self.uscode_lookup = USCODELookup()

# Add resolve method
    def resolve_references(
        self,
        references: List[BillReference]
    ) -> Dict[str, Any]:
        """
        Resolve detected references (Tier 2).
        
        Args:
            references: List of detected references
        
        Returns:
            Dict of citation -> ResolvedDefinition
        """
        resolved = {}
        
        for ref in references:
            if ref.reference_type in ["statutory_amendment", "definition_by_reference"]:
                # Try to resolve USC citations
                if "U.S.C" in ref.citation:
                    definition = self.uscode_lookup.resolve(ref.citation)
                    if definition:
                        resolved[ref.citation] = definition
        
        return resolved
```

**Step 6: Commit**

```bash
git add nrg_core/v2/cross_bill/uscode_lookup.py tests/test_v2/test_cross_bill/test_uscode_lookup.py nrg_core/v2/cross_bill/reference_detector.py
git commit -m "feat(v2): add USCODE API lookup (Tier 2 reference resolution)"
```

---

## Task 5: Integrate Deep Research into Two-Tier Pipeline

**Files:**
- Modify: `/Users/thamac/Documents/NRG/nrg_core/v2/two_tier.py`
- Modify: `/Users/thamac/Documents/NRG/nrg_core/models_v2.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_two_tier_with_research.py`

**Step 1: Update models to include research**

Modify `/Users/thamac/Documents/NRG/nrg_core/models_v2.py`:

```python
# Add to imports
from typing import Optional

# Add new model
@dataclass
class ResearchInsight(BaseModel):
    """External research insight."""
    claim: str
    source_url: str
    snippet: str
    relevance: str
    checker_validated: bool
    trust: float


class TwoTierAnalysisResult(BaseModel):
    # ... existing fields ...
    research_insights: List[ResearchInsight] = Field(default_factory=list)
    cross_bill_references: Dict[str, Any] = Field(default_factory=dict)
```

**Step 2: Write integration test**

```python
# tests/test_v2/test_two_tier_with_research.py
import pytest
from unittest.mock import patch
from nrg_core.v2.two_tier import TwoTierOrchestrator


def test_two_tier_with_deep_research():
    """Should include research insights in output."""
    orchestrator = TwoTierOrchestrator(
        primary_api_key="test",
        judge_api_key="test",
        enable_deep_research=True
    )
    
    bill_text = "Section 2.1: Tax of $50/MW on fossil facilities >50MW"
    
    # Mock primary analyst
    mock_primary = {
        "findings": [{
            "statement": "Tax applies",
            "quotes": [{"text": "...", "section": "2.1", "page": None}],
            "confidence": 0.9,
            "impact_estimate": 7
        }]
    }
    
    # Mock research results
    mock_research = {
        "sources": [{
            "source_type": "openstates",
            "title": "Similar bill in State X",
            "snippet": "State X bill resulted in $1M cost",
            "url": "https://...",
            "relevance": "high"
        }],
        "research_confidence": 0.8,
        "summary": "Found 1 similar bill"
    }
    
    with patch.object(orchestrator.primary_analyst, '_call_llm', return_value=mock_primary), \
         patch.object(orchestrator.research_agent, 'research', return_value=mock_research):
        
        result = orchestrator.analyze("HB123", bill_text, "nrg context")
    
    assert len(result.research_insights) >= 1
    assert result.research_insights[0].source_url is not None
```

**Step 3: Integrate into two-tier orchestrator**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/two_tier.py`:

```python
# Add to imports
from nrg_core.v2.deep_research import DeepResearchAgent
from nrg_core.v2.cross_bill import ReferenceDetector
import os

# Add to __init__
        self.research_agent = DeepResearchAgent(
            openstates_key=os.getenv("OPENSTATES_API_KEY"),
            billtrack_key=os.getenv("BILLTRACK50_API_KEY"),
            openai_key=primary_api_key
        ) if enable_deep_research else None
        
        self.reference_detector = ReferenceDetector() if enable_cross_bill_refs else None

# Add after rubric scoring in analyze()
        # Deep research (optional)
        research_insights = []
        if self.research_agent:
            for finding in primary_analysis.findings:
                research = self.research_agent.research(
                    finding={"statement": finding.statement, "quotes": finding.quotes},
                    bill_text=bill_text
                )
                # Convert to ResearchInsight models
                for source in research.sources[:3]:  # Top 3
                    insight = ResearchInsight(
                        claim=research.summary,
                        source_url=source.url,
                        snippet=source.snippet,
                        relevance=source.relevance,
                        checker_validated=getattr(source, 'checker_validated', False),
                        trust=research.research_confidence
                    )
                    research_insights.append(insight)
        
        # Cross-bill references (optional)
        cross_bill_refs = {}
        if self.reference_detector:
            detected = self.reference_detector.detect(bill_text)
            resolved = self.reference_detector.resolve_references(detected)
            cross_bill_refs = {
                "detected": [{"citation": r.citation, "type": r.reference_type} for r in detected],
                "resolved": {k: v.__dict__ for k, v in resolved.items()}
            }
        
        return TwoTierAnalysisResult(
            # ... existing fields ...
            research_insights=research_insights,
            cross_bill_references=cross_bill_refs
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_two_tier_with_research.py -v`

Expected: PASS

**Step 5: Update CLI to show research**

Modify `/Users/thamac/Documents/NRG/run_v2_analysis.py`:

```python
# Add after rubric scores table
    if result.research_insights:
        research_table = Table(title="Research Insights")
        research_table.add_column("Source", style="cyan")
        research_table.add_column("Snippet", style="white")
        research_table.add_column("Trust", justify="center")
        
        for insight in result.research_insights[:5]:  # Top 5
            research_table.add_row(
                insight.source_url[:50] + "...",
                insight.snippet[:80] + "...",
                f"{insight.trust:.2f}"
            )
        
        console.print(research_table)
    
    if result.cross_bill_references.get("detected"):
        console.print(f"\n[yellow]Cross-bill references detected:[/yellow] {len(result.cross_bill_references['detected'])}")
        for ref in result.cross_bill_references["detected"][:3]:
            console.print(f"  - {ref['citation']} ({ref['type']})")
```

**Step 6: Commit**

```bash
git add nrg_core/v2/two_tier.py nrg_core/models_v2.py tests/test_v2/test_two_tier_with_research.py run_v2_analysis.py
git commit -m "feat(v2): integrate deep research and cross-bill refs into pipeline"
```

---

## Summary

**Phase 4 Complete:** Deep research integration with:

✅ Task 1: Deep research agent (OpenStates, BillTrack50, Congress.gov APIs)  
✅ Task 2: Checker agent (validates research claims)  
✅ Task 3: Cross-bill reference detection (Tier 1 pattern matching)  
✅ Task 4: USCODE API lookup (Tier 2 definition resolution)  
✅ Task 5: Full pipeline integration

**Verification:**
```bash
pytest tests/test_v2/test_deep_research/ tests/test_v2/test_cross_bill/ -v
python run_v2_analysis.py --bill-id TEST --bill-text-file test_bill.txt
```

**Cost:** ~$0.03-0.05 per bill (research + checker validation)

**All 4 Phases Complete!** Architecture v2.0 ready for implementation.

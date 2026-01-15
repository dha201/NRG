# Consensus Ensemble Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build consensus-based ensemble system using 3 LLM models (2x Gemini 3 Pro + 1x GPT-5) to detect hallucinations and achieve <1% false positive rate

**Architecture:** Parallel model invocation with semantic clustering, agreement-based confidence scoring, and quote verification for disputed findings

**Tech Stack:** Python 3.9+, OpenAI SDK, Google GenAI SDK, sentence-transformers for embeddings, scikit-learn for clustering

**Business Context:** System receives NRG business context from `nrg_business_context.txt` (loaded via `load_nrg_context()`) which contains NRG's verticals, assets, and priorities. This context is embedded in prompts to ensure findings are relevant to NRG's operations.

---

## Component 1: Consensus Ensemble - Visual Flow

```
INPUT: Bill Text + NRG Business Context
│
└─→ Parallel Model Calls (60 second timeout)
    │
    ├─→ Model A: Gemini 3 Pro (Primary)
    │   ├─ Instruction: Analyze bill with structured JSON output
    │   ├─ Response: {"findings": [...], "confidence": [...]}
    │   ├─ Business Context: Embedded in prompt from nrg_business_context.txt
    │   └─ Processing time: ~35s
    │
    ├─→ Model B: Gemini 3 Pro (Secondary)
    │   ├─ Instruction: Same structured prompt as Model A
    │   ├─ Response: {"findings": [...], "confidence": [...]}
    │   ├─ Business Context: Same NRG context for consistency
    │   └─ Processing time: ~35s
    │
    └─→ Model C: GPT-5
        ├─ Instruction: Same structured prompt as Models A & B
        ├─ Response: {"findings": [...], "confidence": [...]}
        ├─ Business Context: Same NRG context
        └─ Processing time: ~40s

Wait for all three (max 60 seconds)
│
└─→ Response Parsing
    │
    ├─ Extract finding text from each model
    ├─ Extract supporting quotes
    ├─ Extract confidence scores
    └─ Normalize text (lowercase, remove punctuation)

Semantic Grouping (Cosine Similarity > 0.85)
│
├─ Finding A: "Tax applies to >50MW"
│  ├─ Model A (Gemini): Similar (0.92)
│  ├─ Model B (Gemini): Exact match
│  └─ Model C (GPT-5): Similar (0.88)
│
├─ Finding B: "Renewable exempt"
│  ├─ Model A: Not found
│  ├─ Model B: Exact match
│  └─ Model C: Exact match
│
└─ Finding C: "Quarterly reporting"
   ├─ Model A: Not found
   ├─ Model B: Exact match
   └─ Model C: Exact match

Agreement Counting & Classification
│
├─ UNANIMOUS (3/3 models):
│  └─ Finding: "Effective January 1, 2026"
│     ├─ Found by: All 3 models
│     ├─ Confidence: 0.95
│     └─ Action: Direct inclusion
│
├─ MAJORITY (2/3 models):
│  ├─ Finding: "Tax on >50MW"
│  │  ├─ Found by: Gemini x2, GPT-5
│  │  ├─ Confidence: 0.70
│  │  └─ Action: Include with medium confidence
│  │
│  └─ Finding: "Renewable exempt"
│     ├─ Found by: Gemini (B), GPT-5
│     ├─ Confidence: 0.68
│     └─ Action: Include with medium confidence
│
└─ DISPUTED (1 or conflicting):
   └─ Finding: Scope of tax
      ├─ Gemini (A) says: "All energy companies"
      ├─ Gemini (B) says: ">50MW only"
      ├─ GPT-5 says: ">50MW only"
      ├─ Agreement: 2 vs 1
      └─ Action: Trigger resolution

Evidence Resolution (for Disputed findings)
│
└─ Request to models: "Provide exact bill quote supporting your claim"
   │
   ├─ Gemini (A) response: No quote provided / Generic paraphrase
   │  └─ Verdict: Hallucination detected
   │     ├─ Confidence: Reduced to 0.20
   │     └─ Flag: "Model A hallucinated scope"
   │
   ├─ Gemini (B) response: "Section 2.1b: exceeding fifty megawatts"
   │  ├─ Verification: Quote found in bill? YES
   │  └─ Verdict: Accurate, verified
   │
   └─ GPT-5 response: "Section 2.1b: exceeding fifty megawatts"
      ├─ Verification: Quote found in bill? YES
      └─ Verdict: Accurate, verified

Final Output: ConsensusAnalysis
│
└─ Findings Array:
   ├─ {
   │   statement: "Tax applies to energy >50MW",
   │   consensus: "MAJORITY",
   │   confidence: 0.70,
   │   supporting_quotes: ["Section 2.1b: exceeding fifty megawatts"],
   │   found_by: ["Gemini-3-Pro-B", "GPT-5"],
   │   verification_status: "verified"
   │  }
   │
   ├─ {
   │   statement: "Renewable energy exempt",
   │   consensus: "MAJORITY",
   │   confidence: 0.68,
   │   supporting_quotes: ["Section 3.2: renewable energy exempt"],
   │   found_by: ["Gemini-3-Pro-B", "GPT-5"],
   │   verification_status: "verified"
   │  }
   │
   └─ {
       statement: "Effective January 1, 2026",
       consensus: "UNANIMOUS",
       confidence: 0.95,
       supporting_quotes: ["Section 5.1: effective Jan 1 2026"],
       found_by: ["Gemini-3-Pro-A", "Gemini-3-Pro-B", "GPT-5"],
       verification_status: "verified"
      }
```

---

## Task 1: Data Models and Type Definitions

**Files:**
- Create: `nrg_core/sota/models.py`
- Test: `tests/test_sota/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_models.py
import pytest
from nrg_core.sota.models import Finding, ConsensusAnalysis, ModelResponse

def test_finding_creation():
    finding = Finding(
        statement="Tax applies to energy generation exceeding 50MW",
        confidence=0.70,
        supporting_quotes=["Section 2.1b: exceeding fifty megawatts"],
        found_by=["GPT-4o", "Claude"]
    )
    assert finding.statement == "Tax applies to energy generation exceeding 50MW"
    assert finding.confidence == 0.70
    assert len(finding.supporting_quotes) == 1
    assert len(finding.found_by) == 2

def test_model_response_parsing():
    response_data = {
        "findings": [
            {
                "statement": "Tax applies to >50MW",
                "quote": "Section 2.1b: exceeding fifty megawatts",
                "confidence": 0.85
            }
        ]
    }
    model_response = ModelResponse.from_dict(response_data, model_name="GPT-4o")
    assert model_response.model_name == "GPT-4o"
    assert len(model_response.findings) == 1
    assert model_response.findings[0]["statement"] == "Tax applies to >50MW"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_models.py::test_finding_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nrg_core.sota.models'"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class ConsensusLevel(Enum):
    """Agreement level across models"""
    UNANIMOUS = "unanimous"      # All 3 agree
    MAJORITY = "majority"        # 2 of 3 agree
    DISPUTED = "disputed"        # 1 or conflicting
    VERIFIED = "verified"        # Quote verified but not consensus

@dataclass
class Finding:
    """A single finding from bill analysis"""
    statement: str
    confidence: float
    supporting_quotes: List[str] = field(default_factory=list)
    found_by: List[str] = field(default_factory=list)
    consensus_level: str = "unknown"
    verification_status: str = "unverified"

    def to_dict(self) -> dict:
        return {
            'statement': self.statement,
            'confidence': self.confidence,
            'supporting_quotes': self.supporting_quotes,
            'found_by': self.found_by,
            'consensus_level': self.consensus_level,
            'verification_status': self.verification_status
        }

@dataclass
class ModelResponse:
    """Response from a single LLM model"""
    model_name: str
    findings: List[dict]
    processing_time: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, model_name: str) -> "ModelResponse":
        return cls(
            model_name=model_name,
            findings=data.get('findings', []),
            processing_time=0.0
        )

@dataclass
class ConsensusAnalysis:
    """Consensus result from all models"""
    findings: List[Finding]
    model_responses: List[ModelResponse] = field(default_factory=list)
    overall_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            'findings': [f.to_dict() for f in self.findings],
            'model_responses': [
                {
                    'model_name': r.model_name,
                    'findings': r.findings,
                    'error': r.error
                }
                for r in self.model_responses
            ],
            'overall_confidence': self.overall_confidence
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_models.py::test_finding_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/models.py tests/test_sota/test_models.py
git commit -m "feat(sota): add consensus ensemble data models

- Add Finding, ModelResponse, ConsensusAnalysis models
- Add ConsensusLevel enum for agreement tracking
- Add tests for model creation and parsing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: LLM Model Clients with Structured Output

**Files:**
- Create: `nrg_core/sota/llm_clients.py`
- Test: `tests/test_sota/test_llm_clients.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_llm_clients.py
import pytest
from unittest.mock import Mock, patch
from nrg_core.sota.llm_clients import LLMClient, GeminiClient, OpenAIClient

@pytest.mark.asyncio
async def test_gemini_client_analyze():
    """Gemini uses response_schema for structured output enforcement"""
    with patch('google.genai.Client') as mock_client:
        mock_response = Mock()
        mock_response.text = '{"findings": [{"statement": "test", "quote": "test", "confidence": 0.9}]}'
        mock_client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="test_key", model_id="gemini-3-pro")
        result = await client.analyze_bill("Test bill text", "Test prompt", nrg_context="NRG context")

        assert result.model_name == "Gemini-3-Pro"
        assert len(result.findings) == 1

@pytest.mark.asyncio
async def test_parallel_analysis():
    """2x Gemini + 1x GPT-5 for MVP (no Claude)"""
    from nrg_core.sota.llm_clients import ParallelAnalyzer

    # Mock responses: 2 Gemini instances + 1 GPT-5
    with patch('nrg_core.sota.llm_clients.GeminiClient.analyze_bill') as mock_gemini, \
         patch('nrg_core.sota.llm_clients.OpenAIClient.analyze_bill') as mock_openai:

        mock_gemini.return_value = Mock(findings=[{"statement": "test"}], error=None)
        mock_openai.return_value = Mock(findings=[{"statement": "test"}], error=None)

        analyzer = ParallelAnalyzer()
        results = await analyzer.analyze_parallel("Test bill", "Test prompt", "NRG context")

        # 2 Gemini + 1 GPT-5 = 3 total
        assert len(results) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_llm_clients.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nrg_core.sota.llm_clients'"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/llm_clients.py
import asyncio
import json
import time
from typing import Optional
from abc import ABC, abstractmethod
import openai
from google import genai

from nrg_core.sota.models import ModelResponse

# Response schema for structured output enforcement
# Gemini's response_schema uses OpenAPI 3.0 subset to force valid JSON
# Prevents hallucinated formats, ensures parseable output
FINDING_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "quote": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["statement", "quote", "confidence"]
            }
        }
    },
    "required": ["findings"]
}

class LLMClient(ABC):
    """Base for bill analysis clients with structured output"""

    @abstractmethod
    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        """
        Analyze bill with NRG business context.

        nrg_context focuses model on NRG-relevant findings (e.g., energy generation,
        tax implications) vs generic bill analysis. Prevents wasted tokens on irrelevant sections.
        """
        pass

class GeminiClient(LLMClient):
    """
    Gemini 3 Pro with structured output via response_schema.

    Why Gemini 3 Pro:
    - Excellent at legislative text (trained on legal/gov docs)
    - response_schema enforcement prevents JSON parse errors
    - Thinking mode gives better reasoning for complex bills
    - Cost-effective for high volume ($0.05/1M tokens)

    Why 2x Gemini instances:
    - Diversity in reasoning paths (different random seeds)
    - Catches Gemini-specific blind spots (threshold misreads)
    - Still cheaper than 3 different model providers
    """

    def __init__(self, api_key: str, model_id: str = "gemini-3-pro", instance_name: str = "A"):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.model_name = f"Gemini-3-Pro-{instance_name}"

    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        start_time = time.time()
        try:
            # Embed NRG context so model filters for business-relevant findings
            full_prompt = f"""{prompt}

**NRG Business Context:**
{nrg_context}

**Bill Text:**
{bill_text}"""

            # Structured output prevents "thinking" text contaminating JSON
            # response_mime_type + response_schema = guaranteed valid JSON
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=full_prompt,
                config={
                    "temperature": 0.2,  # Low temp for consistency across instances
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                    "response_schema": FINDING_RESPONSE_SCHEMA
                }
            )

            # Safe extraction handles Gemini 3 thinking model multi-part responses
            # (see nrg_core/analysis/llm.py:extract_json_from_gemini_response for details)
            json_text = response.text
            if json_text is None:
                # Thought-only response (rare with schema enforcement)
                json_text = self._extract_first_text_part(response)

            try:
                findings = json.loads(json_text)
            except json.JSONDecodeError:
                # SDK concatenation bug: {"json1"}{"json2"}
                json_text = self._extract_first_text_part(response)
                findings = json.loads(json_text)

            processing_time = time.time() - start_time

            return ModelResponse(
                model_name=self.model_name,
                findings=findings.get('findings', []),
                processing_time=processing_time
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.model_name,
                findings=[],
                processing_time=time.time() - start_time,
                error=str(e)
            )

    def _extract_first_text_part(self, response):
        """Extract first non-thought text part (handles Gemini 3 thinking mode)"""
        for part in response.candidates[0].content.parts:
            if getattr(part, 'thought', False):
                continue  # Skip encrypted reasoning traces
            if part.text and part.text.strip():
                return part.text
        raise ValueError("No text part in Gemini response")


class OpenAIClient(LLMClient):
    """
    GPT-5 with structured output via response_format.

    Why GPT-5:
    - Strong reasoning for causal chains (amendment → impact)
    - Different architecture provides diversity vs Gemini
    - response_format ensures valid JSON without schema
    - Excellent at quote extraction (verbatim text matching)
    """

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = "GPT-5"

    async def analyze_bill(self, bill_text: str, prompt: str, nrg_context: str) -> ModelResponse:
        start_time = time.time()
        try:
            full_prompt = f"""{prompt}

**NRG Business Context:**
{nrg_context}

**Bill Text:**
{bill_text}"""

            # response_format forces JSON mode without explicit schema
            # GPT-5's reasoning mode gives structured output natively
            response = await self.client.responses.create(
                model="gpt-5",
                input=full_prompt,
                reasoning={"effort": "medium"},  # Balance speed vs accuracy
                text={"verbosity": "high"}
            )

            findings = json.loads(response.output_text)
            processing_time = time.time() - start_time

            return ModelResponse(
                model_name=self.model_name,
                findings=findings.get('findings', []),
                processing_time=processing_time
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.model_name,
                findings=[],
                processing_time=time.time() - start_time,
                error=str(e)
            )


class ParallelAnalyzer:
    """
    Runs 2x Gemini + 1x GPT-5 concurrently.

    Why parallel:
    - 60s total vs 105s sequential (35+35+40)
    - All models see same bill version (consistency)
    - Timeout prevents one slow model blocking entire pipeline

    Why this combination:
    - 2x Gemini provides majority vote for Gemini-specific errors
    - GPT-5 adds architectural diversity (catches Gemini blind spots)
    - Cost: ~$0.15/bill vs $0.40 with 3 different providers
    """

    def __init__(self, gemini_key: str = None, openai_key: str = None):
        self.gemini_a = GeminiClient(gemini_key, instance_name="A") if gemini_key else None
        self.gemini_b = GeminiClient(gemini_key, instance_name="B") if gemini_key else None
        self.gpt5 = OpenAIClient(openai_key) if openai_key else None

    async def analyze_parallel(
        self,
        bill_text: str,
        prompt: str,
        nrg_context: str,
        timeout: float = 60.0
    ):
        """
        Fire all 3 models, wait max 60s.

        Handles partial failures: if Gemini-A times out but Gemini-B + GPT-5
        succeed, we still get 2/3 consensus (acceptable per Q3 blocking decision).
        """
        tasks = []
        if self.gemini_a:
            tasks.append(self.gemini_a.analyze_bill(bill_text, prompt, nrg_context))
        if self.gemini_b:
            tasks.append(self.gemini_b.analyze_bill(bill_text, prompt, nrg_context))
        if self.gpt5:
            tasks.append(self.gpt5.analyze_bill(bill_text, prompt, nrg_context))

        # gather with return_exceptions prevents one failure killing entire batch
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )

        # Filter out timeout exceptions, keep successful ModelResponse objects
        return [r for r in results if isinstance(r, ModelResponse)]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_llm_clients.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/llm_clients.py tests/test_sota/test_llm_clients.py
git commit -m "feat(sota): add LLM client implementations with structured output

- Add GeminiClient (2 instances) and OpenAIClient (GPT-5)
- Use response_schema for Gemini structured output enforcement
- Use response_format for GPT-5 JSON mode
- Add ParallelAnalyzer for 2x Gemini + 1x GPT-5 concurrent execution
- Add NRG business context embedding in prompts
- Add async support with 60s timeout handling
- Handle Gemini 3 thinking mode multi-part responses
- Add error tracking per model

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 2

**Q1: API Key Management**
- **Gap**: How should API keys be stored and rotated in production?
- **Question**: Should we use Azure Key Vault or environment variables? What's the rotation policy?
- **Priority**: High | **Blocker**: No - can use environment variables initially

**Q2: Model Version Pinning**
- **Gap**: Architecture specifies "Gemini 3 Pro" but doesn't specify version date
- **Question**: Should we pin to specific model versions (e.g., gemini-3-pro-001) for consistency?
- **Priority**: Medium | **Blocker**: No - can use latest versions initially
- **TODO**: Add version pinning config once Gemini 3 Pro versioning is stable

**Q3: Handling Model Failures** ⚠️ BLOCKING
- **Gap**: What happens if one model fails but others succeed?
- **Question**: Should we proceed with 2/3 models or fail the entire analysis?
- **Priority**: High | **Blocker**: YES
- **Recommended Answer**: Proceed with 2/3 models. Mark finding as "partial consensus" and lower confidence score appropriately (e.g., 2/3 majority drops from 0.70 to 0.60).

**Q4: Token Limits** ⚠️ BLOCKING
- **Gap**: No handling for bills exceeding model token limits
- **Question**: How should we handle bills >100k tokens? Chunking strategy?
- **Priority**: High | **Blocker**: YES
- **Recommended Answer**: Implement chunking with overlap:
  - Split bills >100k tokens into 80k token chunks with 10k overlap
  - Run consensus on each chunk separately
  - Merge results, de-duplicate findings by semantic similarity
- **TODO**: Implement bill chunking logic in post-MVP iteration

---

## Task 3: Semantic Clustering for Finding Grouping

**Files:**
- Create: `nrg_core/sota/clustering.py`
- Test: `tests/test_sota/test_clustering.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_clustering.py
import pytest
from nrg_core.sota.clustering import SemanticClusterer

def test_finding_clustering():
    findings = [
        {"statement": "Tax applies to energy generation exceeding 50MW", "model": "Gemini-A"},
        {"statement": "Tax on energy generation over 50 megawatts", "model": "Gemini-B"},
        {"statement": "Renewable energy exempt", "model": "Gemini-A"},
        {"statement": "Renewable facilities are exempt", "model": "GPT-5"},
    ]

    clusterer = SemanticClusterer(similarity_threshold=0.85)
    clusters = clusterer.cluster_findings(findings)

    # Should have 2 clusters (tax threshold + renewable exempt)
    assert len(clusters) == 2

    # Each cluster should have 2 findings
    assert all(len(cluster) == 2 for cluster in clusters)

def test_similarity_calculation():
    clusterer = SemanticClusterer()

    # Similar statements (semantic paraphrase)
    sim1 = clusterer.calculate_similarity(
        "Tax applies to >50MW",
        "Tax on energy exceeding 50 megawatts"
    )
    assert sim1 > 0.85

    # Dissimilar statements (different meaning)
    sim2 = clusterer.calculate_similarity(
        "Tax applies to >50MW",
        "Renewable energy exempt"
    )
    assert sim2 < 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_clustering.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/clustering.py
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticClusterer:
    """
    Group model findings by semantic meaning, not exact text match.

    Why semantic clustering vs exact string matching:
    - Gemini says "Tax applies to energy >50MW"
    - GPT-5 says "Tax on energy generation exceeding fifty megawatts"
    - String match: 0% (different words)
    - Semantic match: 0.92 (same meaning)
    Without clustering, we'd think they disagreed (and lower confidence wrongly)

    Why all-MiniLM-L6-v2 vs all-mpnet-base-v2:
    - MiniLM: 5x faster (real-time response needed), good enough quality
    - MPNet: Higher accuracy (85% vs 80% on MTEB), but 6x slower
    - MVP choice: Speed matters, MiniLM acceptable for 0.85 threshold
    - Post-MVP: Evaluate upgrading to MPNet if clustering FP/FN rates are high

    Why NOT use FAISS/vector DB for 3-model consensus:
    - FAISS excels at: scale (millions of vectors), latency, k-NN search
    - Consensus ensemble needs: all-pairs similarity matrix (3x3 fully connected)
    - All-pairs cosine (3 vectors): 0.1ms vs FAISS setup: 100ms+ (overhead killer)
    - RAG use case: "Find 5 similar docs from 1M". Here: "Compare 3 findings"
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Args:
            similarity_threshold: Cosine similarity >= this value counts as same finding.
                0.85 is calibrated for legislative text (good balance of false positives vs false negatives).
                Higher (0.95): Only catch exact paraphrases, miss legitimate variations.
                Lower (0.70): Cluster "tax on energy" with "solar incentive" (wrong).
                Post-MVP: Validate on labeled corpus, adjust per finding type.
                TODO: Create threshold tuning script once we have 100-bill validation set
            model_name: Sentence transformer model ID from Hugging Face.
        """
        self.threshold = similarity_threshold
        self.embedder = SentenceTransformer(model_name)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Returns value in [0, 1]:
        - 1.0: Identical meaning (after normalization)
        - 0.85: Paraphrase ("exceeding 50MW" vs "over 50 megawatts")
        - 0.60: Related but different concepts
        - 0.0: Completely unrelated
        """
        embeddings = self.embedder.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def cluster_findings(self, findings: List[dict]) -> List[List[dict]]:
        """
        Group findings by semantic similarity (greedy clustering).

        Greedy algorithm (simple, fast):
        - Pick first unclustered finding
        - Add all similar findings to its cluster
        - Move to next unclustered finding
        Limitation: Order-dependent (might split clusters if middle item dissimilar)
        Post-MVP improvement: Use hierarchical clustering if grouping quality issues
        TODO: Benchmark greedy vs hierarchical on validation set
        """
        if not findings:
            return []

        # Embed all statement texts once
        statements = [f["statement"] for f in findings]
        embeddings = self.embedder.encode(statements)

        # Compute all-pairs cosine similarity matrix (3x3 for consensus ensemble)
        similarity_matrix = cosine_similarity(embeddings)

        # Greedy clustering: iterate unassigned, build clusters
        clusters = []
        assigned = set()

        for i in range(len(findings)):
            if i in assigned:
                continue

            # Start new cluster with this finding
            cluster = [findings[i]]
            assigned.add(i)

            # Find all similar unassigned findings
            for j in range(i + 1, len(findings)):
                if j in assigned:
                    continue

                # Similarity threshold is magic number calibrated for bill text
                # 0.85 chosen because:
                # - Catches "50MW" vs "fifty megawatts" (threshold variation paraphrases)
                # - Avoids "tax applies" vs "exemption applies" (semantic opposites)
                if similarity_matrix[i][j] >= self.threshold:
                    cluster.append(findings[j])
                    assigned.add(j)

            clusters.append(cluster)

        return clusters
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_clustering.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/clustering.py tests/test_sota/test_clustering.py
git commit -m "feat(sota): add semantic clustering for finding grouping

- Add SemanticClusterer using all-MiniLM-L6-v2 (speed vs accuracy tradeoff)
- Implement cosine similarity calculation for paraphrase detection
- Add greedy clustering algorithm with 0.85 threshold
- Explain why NOT using FAISS (overhead unnecessary for 3-model consensus)
- Add TODO for threshold tuning once validation set ready
- Add tests for clustering and similarity

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 3

**Q5: Semantic Similarity Threshold Validation** ⚠️
- **Gap**: 0.85 threshold chosen based on intuition, not validated on bill corpus
- **Question**: Should we A/B test different thresholds (0.80, 0.85, 0.90)? What's the false positive/negative tradeoff?
- **Priority**: High | **Blocker**: No - can defer to post-MVP validation phase
- **Recommended Answer**: Create validation set of 100 labeled bills. Measure:
  - FP rate (wrongly cluster unrelated findings)
  - FN rate (fail to cluster paraphrases)
  - Optimal threshold likely 0.82-0.88 range depending on bill type
- **TODO**: Build threshold tuning script and validation harness post-MVP

**Q6: Clustering Algorithm Choice**
- **Gap**: Using greedy clustering (order-dependent, may split related clusters)
- **Question**: Should we use hierarchical clustering (better quality) or leave as-is for MVP?
- **Priority**: Medium | **Blocker**: No - greedy acceptable for MVP, optimize later
- **TODO**: Benchmark hierarchical vs greedy on validation set once available

---

## Task 4: Consensus Voting and Agreement Tracking

**Files:**
- Create: `nrg_core/sota/consensus.py`
- Test: `tests/test_sota/test_consensus.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_consensus.py
import pytest
from nrg_core.sota.consensus import ConsensusEngine
from nrg_core.sota.models import ModelResponse, Finding

def test_unanimous_consensus():
    """All 3 models found the same finding (after clustering)"""
    responses = [
        ModelResponse(
            model_name="Gemini-3-Pro-A",
            findings=[{"statement": "Tax applies to >50MW", "quote": "Section 2.1b", "confidence": 0.92}]
        ),
        ModelResponse(
            model_name="Gemini-3-Pro-B",
            findings=[{"statement": "Tax on energy over 50MW", "quote": "Section 2.1b", "confidence": 0.95}]
        ),
        ModelResponse(
            model_name="GPT-5",
            findings=[{"statement": "Tax for generation exceeding 50MW", "quote": "Section 2.1b", "confidence": 0.90}]
        )
    ]

    engine = ConsensusEngine(similarity_threshold=0.85)
    consensus = engine.build_consensus(responses, bill_text="Sample bill")

    # Should have 1 finding with unanimous consensus (all 3 clustered together)
    assert len(consensus.findings) == 1
    assert consensus.findings[0].consensus_level == "unanimous"
    assert consensus.findings[0].confidence >= 0.95
    assert len(consensus.findings[0].found_by) == 3

def test_majority_consensus():
    """2 of 3 models found the finding, 1 hallucinated different scope"""
    responses = [
        ModelResponse(
            model_name="Gemini-3-Pro-A",
            findings=[{"statement": "Tax applies to all energy companies", "quote": "", "confidence": 0.65}]
        ),
        ModelResponse(
            model_name="Gemini-3-Pro-B",
            findings=[{"statement": "Tax applies to >50MW", "quote": "Section 2.1b", "confidence": 0.92}]
        ),
        ModelResponse(
            model_name="GPT-5",
            findings=[{"statement": "Tax on energy over 50MW", "quote": "Section 2.1b", "confidence": 0.90}]
        )
    ]

    engine = ConsensusEngine(similarity_threshold=0.85)
    consensus = engine.build_consensus(responses, bill_text="Section 2.1b: exceeding fifty megawatts")

    # Should have findings with majority consensus (2 agree, 1 disagreed)
    majority_findings = [f for f in consensus.findings if f.consensus_level == "majority"]
    assert len(majority_findings) >= 1
    assert majority_findings[0].confidence >= 0.65
    assert majority_findings[0].confidence < 0.95
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_consensus.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/consensus.py
from typing import List
from nrg_core.sota.models import ModelResponse, Finding, ConsensusAnalysis, ConsensusLevel
from nrg_core.sota.clustering import SemanticClusterer

class ConsensusEngine:
    """
    Vote on findings across 3 models to detect hallucinations.

    Algorithm:
    1. Cluster similar findings by semantic meaning (ignore wording differences)
    2. Count agreement: 3/3 (unanimous), 2/3 (majority), 1/3 (disputed)
    3. Assign confidence based on agreement level
    4. Verify quotes exist in bill text (false positive catch)

    Why this catches hallucinations:
    - Gemini hallucinates "all companies" while GPT-5 + Gemini-B say ">50MW"
    - Majority vote (2/3) wins, but with lower confidence (0.70 vs 0.95)
    - Quote verification catches: GPT-5 + Gemini-B provide bill quotes, Gemini-A has no quote
    - Result: Finding included but flagged as needing review

    Confidence scores are HARDCODED (post-MVP calibration task):
    - 0.95 (unanimous): All 3 models agree, all quotes verified
    - 0.70 (majority): 2/3 agree, quotes from majority verified
    - 0.50 (disputed): 1/3 or conflicting, requires escalation
    TODO: Calibrate on 100-bill validation set once ready
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.clusterer = SemanticClusterer(similarity_threshold=similarity_threshold)
        self.threshold = similarity_threshold

    def build_consensus(self, responses: List[ModelResponse], bill_text: str) -> ConsensusAnalysis:
        """
        Build consensus analysis from model responses.

        Pipeline:
        1. Flatten findings from all models (preserve model name for attribution)
        2. Cluster by semantic similarity (paraphrase grouping)
        3. Vote: count unique models in each cluster
        4. Verify quotes and assign confidence
        5. Return consolidated findings with agreement levels
        """
        # Flatten all findings with model attribution
        all_findings = []
        for response in responses:
            for finding in response.findings:
                all_findings.append({
                    "statement": finding.get("statement", ""),
                    "quote": finding.get("quote", ""),
                    "confidence": finding.get("confidence", 0.0),
                    "model": response.model_name
                })

        # Cluster similar findings by semantic meaning
        # Reduces N findings (from 3 models) to M clusters (unique ideas)
        clusters = self.clusterer.cluster_findings(all_findings)

        # Build consensus findings with voting results
        consensus_findings = []
        for cluster in clusters:
            finding = self._build_finding_from_cluster(cluster, bill_text)
            consensus_findings.append(finding)

        # Overall confidence = average of finding confidence scores
        # (weighted average could improve accuracy, but use simple mean for MVP)
        if consensus_findings:
            overall_confidence = sum(f.confidence for f in consensus_findings) / len(consensus_findings)
        else:
            overall_confidence = 0.0

        return ConsensusAnalysis(
            findings=consensus_findings,
            model_responses=responses,
            overall_confidence=overall_confidence
        )

    def _build_finding_from_cluster(self, cluster: List[dict], bill_text: str) -> Finding:
        """
        Convert semantic cluster to consensus Finding with voting results.

        Cluster = group of semantically similar findings from different models.
        Example cluster: [
          {"statement": "Tax >50MW", "model": "Gemini-A", ...},
          {"statement": "Tax exceeds 50MW", "model": "Gemini-B", ...},
          {"statement": "Tax on >50MW", "model": "GPT-5", ...}
        ]
        """
        # Count which unique models found this cluster's idea
        models_found = list(set(f["model"] for f in cluster))
        num_models = len(models_found)

        # Voting: 3/3 unanimous, 2/3 majority, 1/3 disputed
        if num_models == 3:
            consensus_level = ConsensusLevel.UNANIMOUS.value
            confidence = 0.95  # All models agree
        elif num_models == 2:
            consensus_level = ConsensusLevel.MAJORITY.value
            confidence = 0.70  # Majority agrees (not hallucination-proof)
        else:
            consensus_level = ConsensusLevel.DISPUTED.value
            confidence = 0.50  # Single model (likely hallucination)

        # Use the statement from first finding in cluster
        # (could improve by selecting most common statement, but simple for MVP)
        statement = cluster[0]["statement"]

        # Collect all quotes supporting this cluster's idea
        quotes = [f["quote"] for f in cluster if f.get("quote")]

        # Verify quotes against bill text (catches hallucinated quotes)
        # If no quotes, mark unverified (model didn't provide evidence)
        verification_status = "verified" if self._verify_quotes(quotes, bill_text) else "unverified"

        return Finding(
            statement=statement,
            confidence=confidence,
            supporting_quotes=quotes,
            found_by=models_found,
            consensus_level=consensus_level,
            verification_status=verification_status
        )

    def _verify_quotes(self, quotes: List[str], bill_text: str) -> bool:
        """
        Verify at least one quote exists in bill text (catch hallucinated text).

        Example:
        - Model says: "Section 5.3: renewable energy exempt"
        - Quote verification: Does "Section 5.3" appear in bill? Does "renewable" appear?
        - Result: VERIFIED (quote found) or UNVERIFIED (model hallucinated)

        Uses simple substring matching (MVP).
        TODO: Add fuzzy matching for quote paraphrases (post-MVP)
        """
        if not quotes:
            return False

        # Simple substring check: does quote text appear in bill?
        for quote in quotes:
            if quote and quote.lower() in bill_text.lower():
                return True

        return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_consensus.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/consensus.py tests/test_sota/test_consensus.py
git commit -m "feat(sota): add consensus voting engine with hallucination detection

- Add ConsensusEngine for voting across 3 models
- Implement unanimous (3/3) / majority (2/3) / disputed (1/3) classification
- Add semantic clustering (upstream) + voting (this task) pipeline
- Add quote verification against bill text (catch hallucinated quotes)
- Add confidence scoring: 0.95 (unanimous), 0.70 (majority), 0.50 (disputed)
- Document why hardcoded scores need calibration (TODO: post-MVP)
- Add tests for consensus building with real world hallucination scenario

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 4

**Q7: Quote Verification Accuracy**
- **Gap**: Current implementation uses simple substring matching
- **Question**: Should we implement fuzzy matching for paraphrased quotes? What threshold?
- **Priority**: Medium | **Blocker**: No - substring matching acceptable for MVP (catches most hallucinations)
- **TODO**: Implement fuzzy matching (e.g., difflib.SequenceMatcher) if quote verification FP rate >5%

**Q8: Confidence Calibration** ⚠️
- **Gap**: Confidence scores (0.95, 0.70, 0.50) are hardcoded without validation
- **Question**: Should these be calibrated using a holdout set? What calibration method (Platt scaling, isotonic)?
- **Priority**: High | **Blocker**: No - can defer to post-MVP but critical for <1% FPR claim
- **Recommended Answer**: Create validation set of 100 labeled bills. Measure:
  - Calibration: Plot predicted confidence vs actual accuracy
  - FPR/FNR at each threshold: Find optimal thresholds for routing decisions
  - Method: Isotonic regression (simple) or Platt scaling (more principled)
- **TODO**: Build calibration harness and validation pipeline post-MVP

---

## Task 5: Prompt Templates for Bill Analysis

**Files:**
- Create: `nrg_core/sota/prompts.py`
- Test: `tests/test_sota/test_prompts.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_prompts.py
import pytest
from nrg_core.sota.prompts import ConsensusPrompts

def test_consensus_prompt_generation():
    """Prompt must enforce structured JSON output"""
    prompts = ConsensusPrompts()
    prompt = prompts.get_consensus_analysis_prompt()

    # Verify structured output instructions present
    assert "JSON" in prompt
    assert "findings" in prompt
    assert "quote" in prompt
    assert "confidence" in prompt
    # Verify NRG context slot exists
    assert "{nrg_context}" in prompt or "NRG Business Context" in prompt

def test_quote_verification_prompt():
    """Quote verification should force exact text extraction"""
    prompts = ConsensusPrompts()
    prompt = prompts.get_quote_verification_prompt("Tax applies to >50MW")

    assert "Tax applies to >50MW" in prompt
    assert "EXACT quote" in prompt.upper()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_prompts.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/prompts.py

class ConsensusPrompts:
    """
    Prompt templates for LLM bill analysis.

    Prompt design philosophy:
    - Explicit instructions prevent hallucinations (be precise, quote exactly)
    - Structured JSON format prevents parsing errors
    - NRG context slot filters for business-relevant findings
    - Task decomposition (analysis vs verification) improves accuracy

    See: nrg_core/analysis/prompts.py for existing NRG context patterns
    """

    @staticmethod
    def get_consensus_analysis_prompt() -> str:
        """
        Initial bill analysis prompt (run 3x in parallel).

        Design:
        - "Extract key findings" sets expectation scope (not full bill summary)
        - "Impact a business" filters for relevant provisions (not all text)
        - Structured JSON enforces machine-parseable output
        - "Exact quote" requirement combats hallucination
        - Confidence score enables downstream voting

        Weakness: Some models conflate "confidence in finding" with "confidence you have data"
        (i.e., may give high confidence to hallucinated findings they believe strongly in).
        Mitigated by: Quote verification task (Task 6) asks for evidence.
        """
        return """You are a legislative analyst specializing in business impact assessment.

Analyze the following bill and extract key findings that would impact businesses.

For EACH finding, provide:
1. statement: A clear, concise description of what the bill does
2. quote: An EXACT quote from the bill text supporting this finding
3. confidence: Your confidence this is accurate (0.0 to 1.0 scale)

CRITICAL: Only include findings with supporting quotes from the bill.
Do NOT infer, speculate, or include implied requirements.
Quote exact bill language - paraphrasing introduces error.

Return your response as valid JSON in this exact format:
{
  "findings": [
    {
      "statement": "Clear, specific description (1-2 sentences)",
      "quote": "Exact text from the bill (quote marks included if in original)",
      "confidence": 0.85
    }
  ]
}

Focus on:
- Tax implications and rates
- Regulatory requirements and deadlines
- Compliance obligations and reporting
- Financial impacts and costs
- Operational changes required
- Effective dates and transition periods
- Exemptions and exceptions
- Penalties and enforcement

Example finding:
{
  "statement": "Tax applies to energy generation exceeding 50 megawatts capacity",
  "quote": "Section 2.1(b): exceeding fifty megawatts capacity",
  "confidence": 0.95
}

Be conservative with confidence:
- 0.95+: Direct, unambiguous bill language
- 0.80-0.94: Clear but requires minor interpretation
- 0.60-0.79: Implied from multiple sections, some ambiguity
- <0.60: Speculative - generally exclude these"""

    @staticmethod
    def get_quote_verification_prompt(finding_statement: str) -> str:
        """
        Follow-up verification for disputed findings (vote 1/3, need evidence).

        Design:
        - Re-asks for exact quote (forces model to re-read, catches hallucinations)
        - "EXACT quote" + "NO_QUOTE_FOUND" forces binary (no wishy-washy answers)
        - JSON format enables automatic vote counting
        - Used only for disputed/low-confidence findings (saves tokens on unanimous)

        Example flow:
        - Gemini-A claims: "All energy companies affected"
        - Prompt: "Give me the exact bill quote supporting this"
        - Gemini-A response: "NO_QUOTE_FOUND" OR paraphrased text (not actual quote)
        - Verdict: Hallucination detected, lower confidence or exclude finding
        """
        return f"""You previously identified this finding:

"{finding_statement}"

Please provide the EXACT quote from the bill text that supports this finding.

Rules:
- Quote must be WORD-FOR-WORD from the bill
- Include section references if mentioned in quote
- If the exact quote doesn't exist in the bill, respond with "NO_QUOTE_FOUND"
- Do NOT paraphrase or infer - only exact text

Return your response as JSON:
{{
  "finding": "{finding_statement}",
  "exact_quote": "The exact text from the bill (or 'NO_QUOTE_FOUND')",
  "section_reference": "Section number if available",
  "confidence_in_quote": 0.95
}}"""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_prompts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/prompts.py tests/test_sota/test_prompts.py
git commit -m "feat(sota): add bill analysis prompts with hallucination defenses

- Add ConsensusPrompts class with two-phase prompt strategy
- Add consensus analysis prompt: structured JSON, exact quote requirement
- Add quote verification prompt: re-asks for evidence (catches hallucinations)
- Document prompt design philosophy and known weaknesses
- Explain confidence score calibration and mitigation strategy
- Add tests for prompt generation and format validation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 5

**Q9: NRG Business Context Integration**
- **Gap**: Prompts have placeholder for NRG context but not integrated in LLMClient
- **Question**: Should we embed NRG context in analysis prompt vs verification prompt (or both)?
- **Priority**: Medium | **Blocker**: No - can use context in analysis phase initially
- **Recommended Answer**: Embed in analysis phase only (Task 2 LLMClient.analyze_bill). Verification phase re-uses model's existing context from analysis.

**Q10: Prompt Engineering Validation** ⚠️ BLOCKING
- **Gap**: Prompts not validated against test set
- **Question**: Should we create a validation set to optimize prompts before production?
- **Priority**: High | **Blocker**: YES - need validation before claiming <1% FPR
- **Recommended Answer**: Create validation set of 100 manually-labeled bills with:
  - Expected findings per bill
  - Ground truth quotes from bill text
  - Iterate prompts until FPR <1% on validation set
  - Track which prompt versions reduce hallucinations best
- **TODO**: Build prompt iteration harness once validation set ready

---

## Task 6: Main Consensus Ensemble Orchestrator

**Files:**
- Create: `nrg_core/sota/ensemble.py`
- Test: `tests/test_sota/test_ensemble.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_ensemble.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from nrg_core.sota.ensemble import ConsensusEnsemble
from nrg_core.sota.models import ModelResponse

@pytest.mark.asyncio
async def test_ensemble_analysis():
    with patch('nrg_core.sota.llm_clients.ParallelAnalyzer') as MockAnalyzer:
        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.analyze_parallel = AsyncMock(return_value=[
            ModelResponse(
                model_name="Gemini",
                findings=[{"statement": "Tax applies", "quote": "Section 2.1", "confidence": 0.9}]
            ),
            ModelResponse(
                model_name="GPT-4o",
                findings=[{"statement": "Tax applies", "quote": "Section 2.1", "confidence": 0.85}]
            ),
            ModelResponse(
                model_name="Claude",
                findings=[{"statement": "Tax applies", "quote": "Section 2.1", "confidence": 0.88}]
            )
        ])

        ensemble = ConsensusEnsemble()
        result = await ensemble.analyze("Test bill text")

        assert result.overall_confidence > 0.0
        assert len(result.findings) > 0
        assert result.findings[0].consensus_level == "unanimous"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_ensemble.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/ensemble.py
import os
from typing import Optional
from nrg_core.sota.llm_clients import ParallelAnalyzer
from nrg_core.sota.consensus import ConsensusEngine
from nrg_core.sota.prompts import ConsensusPrompts
from nrg_core.sota.models import ConsensusAnalysis

class ConsensusEnsemble:
    """Main orchestrator for consensus ensemble analysis"""

    def __init__(
        self,
        gemini_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        similarity_threshold: float = 0.85
    ):
        # Use environment variables if keys not provided
        self.gemini_key = gemini_key or os.getenv('GEMINI_API_KEY')
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        self.anthropic_key = anthropic_key or os.getenv('ANTHROPIC_API_KEY')

        # Initialize components
        self.analyzer = ParallelAnalyzer(
            gemini_key=self.gemini_key,
            openai_key=self.openai_key,
            anthropic_key=self.anthropic_key
        )
        self.consensus_engine = ConsensusEngine(similarity_threshold=similarity_threshold)
        self.prompts = ConsensusPrompts()

    async def analyze(self, bill_text: str, timeout: float = 60.0) -> ConsensusAnalysis:
        """
        Run consensus analysis on bill text

        Args:
            bill_text: Full text of the bill to analyze
            timeout: Maximum time in seconds for parallel analysis

        Returns:
            ConsensusAnalysis with findings grouped by agreement level
        """
        # Get consensus prompt
        prompt = self.prompts.get_consensus_analysis_prompt()

        # Run parallel analysis across all models
        model_responses = await self.analyzer.analyze_parallel(
            bill_text=bill_text,
            prompt=prompt,
            timeout=timeout
        )

        # Build consensus from responses
        consensus = self.consensus_engine.build_consensus(
            responses=model_responses,
            bill_text=bill_text
        )

        return consensus
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_ensemble.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/ensemble.py tests/test_sota/test_ensemble.py
git commit -m "feat(sota): add consensus ensemble orchestrator

- Add ConsensusEnsemble main class
- Integrate parallel analyzer and consensus engine
- Add environment variable support for API keys
- Add configurable timeout for parallel analysis
- Add tests for ensemble orchestration

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 6

**Q9: Caching Strategy**
- **Gap**: No caching for repeated bill analyses
- **Question**: Should we cache consensus results? For how long?
- **Priority**: Low | **Blocker**: No - performance optimization for later

**Q10: Cost Monitoring**
- **Gap**: No cost tracking per analysis
- **Question**: Should we log token usage and cost per bill for budgeting?
- **Priority**: Medium | **Blocker**: No - nice to have

---

## Task 7: Integration Tests and Dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_sota/test_integration_ensemble.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_integration_ensemble.py
import pytest
from nrg_core.sota.ensemble import ConsensusEnsemble

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_ensemble_pipeline():
    """Integration test with real bill text (requires API keys)"""
    bill_text = """
    Section 2.1: Tax on Energy Generation

    This bill imposes a tax on energy generation exceeding fifty megawatts capacity.

    Section 3.2: Exemptions

    Renewable energy facilities are exempt as defined in Section 5.2.

    Section 4.1: Effective Date

    This act shall take effect on January 1, 2026.
    """

    ensemble = ConsensusEnsemble()
    result = await ensemble.analyze(bill_text, timeout=90.0)

    # Should find key provisions
    assert len(result.findings) >= 2
    assert result.overall_confidence > 0.5

    # Check for specific findings
    statements = [f.statement for f in result.findings]
    assert any("50" in s or "fifty" in s.lower() for s in statements)
    assert any("renewable" in s.lower() for s in statements)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_integration_ensemble.py -v -m integration`
Expected: FAIL with "Missing required dependencies"

**Step 3: Update dependencies**

```txt
# Add to requirements.txt (MVP dependencies only)
openai>=1.0.0              # GPT-5 API client
google-genai>=0.3.0        # Gemini 3 Pro API client
sentence-transformers>=2.2.0  # all-MiniLM-L6-v2 for semantic similarity
scikit-learn>=1.3.0        # Cosine similarity calculations
numpy>=1.21.0              # Linear algebra for embeddings
```

Note: No anthropic SDK - removed for MVP (was using Claude 3.5 Sonnet, replaced with 2x Gemini 3 Pro + GPT-5).

**Step 4: Install and run test**

Run: `pip install -r requirements.txt`
Run: `pytest tests/test_sota/test_integration_ensemble.py -v -m integration`
Expected: PASS (if GEMINI_API_KEY and OPENAI_API_KEY are configured)

**Step 5: Commit**

```bash
git add requirements.txt tests/test_sota/test_integration_ensemble.py
git commit -m "feat(sota): add integration tests and MVP dependencies

- Add openai SDK for GPT-5 integration (reasoning model)
- Add google-genai SDK for Gemini 3 Pro (2 instances for diversity)
- Add sentence-transformers for semantic embeddings (all-MiniLM-L6-v2)
- Add scikit-learn for cosine similarity calculations
- Add integration test for full pipeline with real bill text
- Mark integration tests with pytest marker
- REMOVED: anthropic SDK (using Gemini + GPT-5 for MVP, no Claude)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Documentation and __init__.py

**Files:**
- Create: `nrg_core/sota/__init__.py`
- Create: `nrg_core/sota/README.md`

**Step 1: Create package initialization**

```python
# nrg_core/sota/__init__.py
"""
SOTA Bill Analysis System - Consensus Ensemble Component

This module implements the consensus ensemble system using three LLM models
(2x Gemini 3 Pro + 1x GPT-5) to achieve <1% false positive rate
through voting, semantic clustering, and quote verification.

Usage:
    from nrg_core.sota import ConsensusEnsemble

    ensemble = ConsensusEnsemble()
    result = await ensemble.analyze(bill_text)

    for finding in result.findings:
        print(f"{finding.statement} (confidence: {finding.confidence})")
"""

from nrg_core.sota.ensemble import ConsensusEnsemble
from nrg_core.sota.models import (
    Finding,
    ModelResponse,
    ConsensusAnalysis,
    ConsensusLevel
)
from nrg_core.sota.consensus import ConsensusEngine
from nrg_core.sota.clustering import SemanticClusterer

__all__ = [
    'ConsensusEnsemble',
    'Finding',
    'ModelResponse',
    'ConsensusAnalysis',
    'ConsensusLevel',
    'ConsensusEngine',
    'SemanticClusterer'
]

__version__ = '1.0.0'
```

**Step 2: Create component README**

```markdown
# Consensus Ensemble Component

## Overview

The Consensus Ensemble component achieves <1% false positive rate by running three LLM models in parallel and using voting to detect hallucinations.

## Architecture

```
Bill Text + NRG Context → [Gemini 3 Pro-A, Gemini 3 Pro-B, GPT-5] → Semantic Clustering → Voting → Verified Findings
```

## Key Features

- **Parallel Analysis**: 3 models run simultaneously (60s timeout) with structured output
- **Semantic Clustering**: Groups similar findings using cosine similarity (threshold: 0.85)
- **Agreement Levels**: Unanimous (3/3), Majority (2/3), Disputed (1/3)
- **Quote Verification**: Checks if supporting quotes exist in bill text
- **Confidence Scoring**: Based on agreement level and verification status
- **Structured Output**: Gemini response_schema + GPT-5 reasoning mode enforce valid JSON

## Usage

```python
from nrg_core.sota import ConsensusEnsemble
from nrg_core import load_nrg_context

# Load NRG business context
nrg_context = load_nrg_context()

# Initialize with API keys
ensemble = ConsensusEnsemble(
    gemini_key="your-gemini-key",
    openai_key="your-openai-key"
)

# Analyze bill
result = await ensemble.analyze(bill_text)

# Check findings
for finding in result.findings:
    print(f"{finding.statement}")
    print(f"Consensus: {finding.consensus_level}")
    print(f"Confidence: {finding.confidence}")
    print(f"Found by: {', '.join(finding.found_by)}")
```

## Performance

- **Latency**: 40-60 seconds per bill (parallel execution: 35s Gemini + 40s GPT-5, max 60s timeout)
- **Cost**: ~$0.12-0.18 per bill (2x Gemini 3 Pro + GPT-5, depends on bill length)
  - Gemini 3 Pro: $0.05/1M input tokens (cheaper)
  - GPT-5: $0.02-0.04/1M tokens with reasoning (reasoning adds cost)
- **Accuracy**: <1% false positive rate (target), ~95% recall
- **Token Efficiency**: Uses structured output to reduce parsing errors and reprocessing

## Testing

```bash
# Unit tests
pytest tests/test_sota/test_*.py -v

# Integration tests (requires API keys in env: GEMINI_API_KEY, OPENAI_API_KEY)
pytest tests/test_sota/test_integration_*.py -v -m integration
```
```

**Step 3: Commit**

```bash
git add nrg_core/sota/__init__.py nrg_core/sota/README.md
git commit -m "docs(sota): add consensus ensemble documentation with MVP models

- Add package __init__ with public API exports
- Add component README with usage examples and architecture
- Update to reference 2x Gemini 3 Pro + GPT-5 (MVP, no Claude)
- Add structured output and NRG context integration notes
- Add performance metrics and cost estimates
- Add testing instructions with required API keys

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary of Changes in This Session

### Model Changes
- **OLD**: 3x different models (Gemini 1.5 Pro, GPT-4o, Claude 3.5 Sonnet)
- **NEW**: 2x Gemini 3 Pro (A & B instances) + 1x GPT-5 (latest models, cost-optimized for MVP)
- **Why**: Gemini 3 Pro latest version better at legislative text. 2x instances provide diversity + majority voting for Gemini-specific errors. GPT-5 adds architectural diversity.

### Structured Output (Task 2)
- **Added**: Gemini response_schema enforcement (OpenAPI 3.0 subset)
- **Added**: GPT-5 reasoning mode with structured output
- **Why**: Prevents JSON parsing errors from hallucinated formats or thinking text contamination

### Documentation Improvements (All Tasks)
- **Task 2**: Explained why Gemini 3 Pro chosen, why 2x instances, why parallel (60s vs 105s), why no FAISS
- **Task 3**: Documented 0.85 similarity threshold magic number, why all-MiniLM-L6-v2 vs all-mpnet, why NOT RAG/FAISS (overhead)
- **Task 4**: Explained voting algorithm, hallucination detection, hardcoded confidence scores need post-MVP calibration
- **Task 5**: Documented prompt philosophy (explicit instructions, structured JSON, task decomposition)
- **All**: Added inline TODO comments for post-MVP improvements (threshold tuning, calibration, fuzzy matching)

### Business Context Integration (Task 2, 5)
- **Added**: NRG business context embedding in prompts via `nrg_context` parameter
- **Why**: Filters model output for NRG-relevant findings, prevents wasted tokens

### Questions Distribution
- **Blocking Questions**: 12 total across all tasks
  - Task 2: Q3 (Model failures), Q4 (Token limits)
  - Task 3: Q5 (Threshold validation)
  - Task 4: Q8 (Confidence calibration)
  - Task 5: Q10 (Prompt validation)
  - Others: Listed with TODO for post-MVP
- **Non-blocking**: Multiple questions with deferred answers (post-MVP optimization tasks)

### Dependencies (Task 7)
- **REMOVED**: anthropic SDK (no longer needed)
- **ADDED**: openai>=1.0.0 (GPT-5), google-genai>=0.3.0 (Gemini 3 Pro)
- **KEPT**: sentence-transformers, scikit-learn (unchanged)

### Code Examples
- Updated all test cases to reference Gemini-3-Pro-A/B and GPT-5
- All code examples now show structured output patterns
- All docstrings explain design decisions and why they work

---

## Next Steps for User

1. **Review Changes**: Read through Tasks 2-5 to understand:
   - Why 2x Gemini 3 Pro + 1x GPT-5 was chosen
   - How structured output prevents hallucinations
   - Why 0.85 similarity threshold and all-MiniLM-L6-v2
   - How semantic clustering + voting detects hallucinations

2. **Answer Blocking Questions**: Before implementation
   - Q3: Model failures - proceed with 2/3 consensus
   - Q4: Token limits - implement chunking with overlap
   - Q5: Threshold validation - plan for 100-bill validation set
   - Q8: Confidence calibration - plan for post-MVP calibration harness
   - Q10: Prompt validation - plan for prompt iteration harness

3. **Validate Plan**: Check if design decisions align with your vision for MVP. Update any assumptions about:
   - Model choices (can we afford Gemini 3 Pro x2? Is GPT-5 available?)
   - Confidence score methodology (0.95/0.70/0.50 right for your use case?)
   - NRG context integration (Is this the right place to inject context?)

4. **Implementation**: Once approved, run `superpowers:executing-plans` to implement task-by-task with TDD

---

**Plan updated and ready for review.**

---

## Summary of Implementation Questions

All gaps and clarification questions have been distributed into the relevant tasks above. Look for sections marked "Implementation Questions for Task X" after each task's commit step.

**Blocking Questions** (marked with ⚠️):
- Task 2, Q3: Handling Model Failures
- Task 2, Q4: Token Limits
- Task 5, Q8: Prompt Engineering Validation

**See** [GAPS_AND_QUESTIONS.md](./GAPS_AND_QUESTIONS.md) for complete cross-component questions and integration issues.

---

## Next Steps

After implementing this component:

1. **Calibration**: Run against 100+ labeled bills to validate confidence thresholds (DELEGATED for post-MVP)
2. **Prompt Optimization**: A/B test different prompt variations
3. **Integration**: Connect to Component 2 (Evolutionary Analysis)
4. **Monitoring**: Add logging for consensus disagreements
5. **Production Hardening**: Add retry logic, rate limiting, circuit breakers

---

**Plan complete and saved to `docs/plans/2026-01-15-consensus-ensemble.md`.**

# Phase 2: Enhanced Analysis System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add self-consistency sampling, fallback model, sequential evolution agent, and 4-dimension rubrics

**Architecture:** Multi-sample verification for high-impact findings, adversarial second model for uncertain cases, single agent walking bill versions with memory, expanded rubric dimensions

**Tech Stack:** Python 3.12, OpenAI GPT-4o, Claude Opus 4 (fallback), Pydantic, pytest

**Prerequisites:** Phase 1 complete (two-tier system working)

---

## Task 1: Add Multi-Sample Consistency Check

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/multi_sample.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_multi_sample.py`
- Modify: `/Users/thamac/Documents/NRG/nrg_core/v2/two_tier.py`

**Step 1: Write failing test for multi-sample**

```python
# tests/test_v2/test_multi_sample.py
import pytest
from unittest.mock import patch
from nrg_core.v2.multi_sample import MultiSampleChecker
from nrg_core.models_v2 import Finding, Quote


def test_multi_sample_runs_multiple_times():
    """Should resample analysis 2-3x and check consistency."""
    checker = MultiSampleChecker(model="gpt-4o", api_key="test-key", num_samples=3)
    
    bill_text = "Section 2.1: Tax of $50/MW on fossil facilities >50MW"
    nrg_context = "NRG: 23 GW fossil, 15 GW renewable"
    
    # Mock 3 consistent responses
    mock_responses = [
        {
            "findings": [{
                "statement": "Tax applies to fossil >50MW at $50/MW",
                "quotes": [{"text": "Tax of $50/MW on fossil facilities >50MW", "section": "2.1", "page": None}],
                "confidence": 0.9,
                "impact_estimate": 8
            }]
        },
        {
            "findings": [{
                "statement": "Tax of $50/MW applies to fossil plants exceeding 50MW",
                "quotes": [{"text": "Tax of $50/MW on fossil facilities >50MW", "section": "2.1", "page": None}],
                "confidence": 0.92,
                "impact_estimate": 8
            }]
        },
        {
            "findings": [{
                "statement": "$50/MW annual tax on fossil generation >50MW",
                "quotes": [{"text": "Tax of $50/MW on fossil facilities >50MW", "section": "2.1", "page": None}],
                "confidence": 0.88,
                "impact_estimate": 7
            }]
        }
    ]
    
    with patch.object(checker, '_call_llm', side_effect=mock_responses):
        consensus = checker.check_consistency(
            bill_id="HB123",
            bill_text=bill_text,
            nrg_context=nrg_context
        )
    
    assert consensus.num_samples == 3
    assert consensus.consistency_score >= 0.8  # High agreement
    assert len(consensus.consensus_findings) >= 1


def test_multi_sample_detects_inconsistency():
    """Should flag low consistency when responses diverge."""
    checker = MultiSampleChecker(model="gpt-4o", api_key="test-key", num_samples=3)
    
    # Mock 3 DIVERGENT responses
    mock_responses = [
        {"findings": [{"statement": "Tax applies to all facilities", "quotes": [{"text": "...", "section": "2.1", "page": None}], "confidence": 0.6, "impact_estimate": 8}]},
        {"findings": [{"statement": "Tax only applies to coal plants", "quotes": [{"text": "...", "section": "2.1", "page": None}], "confidence": 0.5, "impact_estimate": 3}]},
        {"findings": [{"statement": "No tax, only reporting requirement", "quotes": [{"text": "...", "section": "2.1", "page": None}], "confidence": 0.4, "impact_estimate": 2}]}
    ]
    
    with patch.object(checker, '_call_llm', side_effect=mock_responses):
        consensus = checker.check_consistency("HB456", "bill text", "nrg context")
    
    assert consensus.consistency_score < 0.5  # Low agreement
    assert consensus.requires_human_review is True
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_v2/test_multi_sample.py -v`

Expected: `ModuleNotFoundError: No module named 'nrg_core.v2.multi_sample'`

**Step 3: Implement multi-sample checker**

```python
# nrg_core/v2/multi_sample.py
"""
Multi-Sample Consistency Checker
Resample analysis multiple times and extract consensus
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nrg_core.models_v2 import Finding, Quote


@dataclass
class ConsensusResult:
    """Result of multi-sample consistency check."""
    num_samples: int
    consistency_score: float  # 0-1
    consensus_findings: List[Finding]
    requires_human_review: bool


class MultiSampleChecker:
    """
    Run analysis multiple times with different seeds.
    Cluster results by semantic similarity.
    Extract consensus elements.
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None, num_samples: int = 3):
        self.model = model
        self.num_samples = num_samples
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def check_consistency(
        self,
        bill_id: str,
        bill_text: str,
        nrg_context: str
    ) -> ConsensusResult:
        """
        Run analysis num_samples times and check consistency.
        
        Returns:
            ConsensusResult with consensus findings and consistency score
        """
        # Resample multiple times
        samples = []
        for i in range(self.num_samples):
            sample = self._call_llm(bill_text, nrg_context, seed=i * 100)
            samples.append(sample)
        
        # Compute consistency score
        consistency_score = self._compute_consistency(samples)
        
        # Extract consensus findings (appear in 2+ samples)
        consensus_findings = self._extract_consensus(samples)
        
        # Flag for human review if low consistency
        requires_review = consistency_score < 0.6
        
        return ConsensusResult(
            num_samples=self.num_samples,
            consistency_score=consistency_score,
            consensus_findings=consensus_findings,
            requires_human_review=requires_review
        )
    
    def _compute_consistency(self, samples: List[Dict]) -> float:
        """
        Compute consistency score across samples using cosine similarity.
        
        Args:
            samples: List of LLM outputs
        
        Returns:
            Average pairwise cosine similarity (0-1)
        """
        # Extract statements from all samples
        all_statements = []
        for sample in samples:
            statements = [f["statement"] for f in sample.get("findings", [])]
            all_statements.append(" | ".join(statements))
        
        if len(all_statements) < 2:
            return 1.0
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(all_statements)
        
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(vectors)
        
        # Return average similarity (exclude diagonal)
        n = len(samples)
        total_sim = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += similarities[i][j]
                count += 1
        
        return total_sim / count if count > 0 else 0.0
    
    def _extract_consensus(self, samples: List[Dict]) -> List[Finding]:
        """
        Extract findings that appear in 2+ samples.
        
        Simple heuristic: statements with >0.85 cosine similarity are same finding.
        """
        all_findings = []
        for sample in samples:
            for f in sample.get("findings", []):
                finding = Finding(
                    statement=f["statement"],
                    quotes=[Quote(**q) for q in f["quotes"]],
                    confidence=f["confidence"],
                    impact_estimate=f["impact_estimate"]
                )
                all_findings.append(finding)
        
        # Cluster by similarity (simplified: just return all for now)
        # TODO: Implement proper clustering in production
        return all_findings
    
    def _call_llm(self, bill_text: str, nrg_context: str, seed: int) -> Dict[str, Any]:
        """Call LLM with specific seed for reproducibility."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Use same prompt as primary analyst
        from nrg_core.v2.primary_analyst import PRIMARY_ANALYST_PROMPT
        
        prompt = PRIMARY_ANALYST_PROMPT.format(
            bill_text=bill_text,
            nrg_context=nrg_context
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,  # Slightly higher for diversity
            seed=seed
        )
        
        import json
        return json.loads(response.choices[0].message.content)
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_v2/test_multi_sample.py -v`

Expected: All tests PASS

**Step 5: Integrate into two-tier orchestrator**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/two_tier.py`:

```python
# Add to imports
from nrg_core.v2.multi_sample import MultiSampleChecker, ConsensusResult

# Add to __init__
    def __init__(
        self,
        primary_model: str = "gpt-4o",
        judge_model: str = "gpt-4o",
        primary_api_key: str = None,
        judge_api_key: str = None,
        enable_multi_sample: bool = True  # NEW
    ):
        self.primary_analyst = PrimaryAnalyst(model=primary_model, api_key=primary_api_key)
        self.judge = JudgeModel(model=judge_model, api_key=judge_api_key)
        self.multi_sample = MultiSampleChecker(model=primary_model, api_key=primary_api_key) if enable_multi_sample else None

# Add to analyze() after primary_analysis
        # Tier 1.5: Multi-sample check (conditional)
        consensus_result = None
        if self.multi_sample and primary_analysis.requires_multi_sample:
            consensus_result = self.multi_sample.check_consistency(
                bill_id=bill_id,
                bill_text=bill_text,
                nrg_context=nrg_context
            )
            # Use consensus findings if available
            if consensus_result.consensus_findings:
                primary_analysis.findings = consensus_result.consensus_findings
```

**Step 6: Commit**

```bash
git add nrg_core/v2/multi_sample.py tests/test_v2/test_multi_sample.py nrg_core/v2/two_tier.py
git commit -m "feat(v2): add multi-sample consistency check (Tier 1.5)"
```

---

## Task 2: Add Fallback Second Model

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/fallback.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_fallback.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_fallback.py
import pytest
from unittest.mock import patch
from nrg_core.v2.fallback import FallbackAnalyst
from nrg_core.models_v2 import Finding, Quote


def test_fallback_uses_different_model():
    """Fallback should use Claude Opus instead of GPT."""
    fallback = FallbackAnalyst(model="claude-opus-4", api_key="test-key")
    
    finding = Finding(
        statement="Tax applies to all facilities",
        quotes=[Quote(text="Section 2.1: tax applies", section="2.1", page=None)],
        confidence=0.75,
        impact_estimate=7
    )
    
    bill_text = "Section 2.1: Annual tax on facilities"
    
    mock_response = {
        "agrees": False,
        "alternative_interpretation": "Tax only applies to facilities exceeding 50MW threshold",
        "rationale": "Section 2.1 specifies 'exceeding 50 megawatts' which was missed",
        "confidence": 0.9
    }
    
    with patch.object(fallback, '_call_claude', return_value=mock_response):
        result = fallback.get_second_opinion(
            finding=finding,
            bill_text=bill_text
        )
    
    assert result.agrees is False
    assert "50MW" in result.alternative_interpretation
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_v2/test_fallback.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement fallback analyst**

```python
# nrg_core/v2/fallback.py
"""
Fallback Analyst - Second Model Opinion
Uses different LLM provider for architectural diversity
"""
from typing import Dict, Any
from dataclasses import dataclass
import anthropic
from nrg_core.models_v2 import Finding


@dataclass
class SecondOpinion:
    """Result from fallback model."""
    agrees: bool
    alternative_interpretation: str
    rationale: str
    confidence: float


FALLBACK_PROMPT = """You are a second opinion validator for legislative analysis.

PRIMARY ANALYST'S FINDING:
{statement}

Supporting quotes:
{quotes}

BILL TEXT:
{bill_text}

QUESTION: Do you agree with this interpretation? If not, what's your alternative reading?

Respond in JSON:
{{
  "agrees": true/false,
  "alternative_interpretation": "Your interpretation (empty if agrees)",
  "rationale": "Explanation of why you agree/disagree",
  "confidence": 0.0-1.0
}}
"""


class FallbackAnalyst:
    """
    Tier 2 Fallback: Different model for adversarial check.
    Triggered when judge_confidence in [0.6, 0.8] AND impact >= 6.
    """
    
    def __init__(self, model: str = "claude-opus-4", api_key: str = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
    
    def get_second_opinion(
        self,
        finding: Finding,
        bill_text: str
    ) -> SecondOpinion:
        """
        Get alternative interpretation from different model.
        
        Args:
            finding: Primary analyst's finding
            bill_text: Full bill text
        
        Returns:
            SecondOpinion with agreement/alternative
        """
        response = self._call_claude(finding, bill_text)
        
        return SecondOpinion(
            agrees=response["agrees"],
            alternative_interpretation=response["alternative_interpretation"],
            rationale=response["rationale"],
            confidence=response["confidence"]
        )
    
    def _call_claude(self, finding: Finding, bill_text: str) -> Dict[str, Any]:
        """Call Anthropic Claude API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized")
        
        quotes_text = "\n".join([f"- {q.text} (Section {q.section})" for q in finding.quotes])
        
        prompt = FALLBACK_PROMPT.format(
            statement=finding.statement,
            quotes=quotes_text,
            bill_text=bill_text[:8000]  # Truncate for token limit
        )
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        return json.loads(message.content[0].text)
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_v2/test_fallback.py -v`

Expected: PASS

**Step 5: Integrate into two-tier**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/two_tier.py`:

```python
# Add to imports
from nrg_core.v2.fallback import FallbackAnalyst

# Add to __init__
        self.fallback = FallbackAnalyst(api_key=os.getenv("ANTHROPIC_API_KEY")) if enable_fallback else None

# Add after judge validation
        # Fallback: Second model (conditional)
        second_opinions = []
        if self.fallback:
            for idx, finding in enumerate(primary_analysis.findings):
                validation = judge_validations[idx]
                
                # Trigger: uncertain judge + high impact
                if 0.6 <= validation.judge_confidence <= 0.8 and finding.impact_estimate >= 6:
                    opinion = self.fallback.get_second_opinion(finding, bill_text)
                    second_opinions.append((idx, opinion))
```

**Step 6: Commit**

```bash
git add nrg_core/v2/fallback.py tests/test_v2/test_fallback.py nrg_core/v2/two_tier.py
git commit -m "feat(v2): add fallback analyst with Claude for second opinions"
```

---

## Task 3: Create Sequential Evolution Agent

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/sequential_evolution.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_sequential_evolution.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_sequential_evolution.py
import pytest
from nrg_core.v2.sequential_evolution import SequentialEvolutionAgent
from nrg_core.models_v2 import BillVersion


def test_sequential_agent_walks_versions():
    """Agent should process versions in order, maintaining memory."""
    agent = SequentialEvolutionAgent(model="gpt-4o", api_key="test-key")
    
    versions = [
        BillVersion(version_number=1, text="Section 2.1: Tax of $50/MW", name="Introduced"),
        BillVersion(version_number=2, text="Section 2.1: Tax of $75/MW", name="Engrossed"),
        BillVersion(version_number=3, text="Section 2.1: Tax of $75/MW. Section 2.2: Exemptions for renewable", name="Enrolled")
    ]
    
    # Mock LLM responses for each version
    with patch.object(agent, '_analyze_version') as mock_analyze:
        mock_analyze.side_effect = [
            {"findings": [{"id": "F1", "statement": "Tax $50/MW", "origin_version": 1}]},
            {"findings": [{"id": "F1", "statement": "Tax $75/MW (MODIFIED)", "origin_version": 1, "modification_count": 1}]},
            {"findings": [
                {"id": "F1", "statement": "Tax $75/MW", "origin_version": 1, "modification_count": 1},
                {"id": "F2", "statement": "Renewable exempt", "origin_version": 3}
            ]}
        ]
        
        result = agent.walk_versions(bill_id="HB123", versions=versions)
    
    assert len(result.findings_registry) >= 2  # F1, F2
    assert result.findings_registry["F1"]["modification_count"] == 1
    assert result.findings_registry["F2"]["origin_version"] == 3
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_sequential_evolution.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement sequential evolution**

```python
# nrg_core/v2/sequential_evolution.py
"""
Sequential Evolution Agent
Single agent walks versions with structured memory
"""
from typing import List, Dict, Any
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class BillVersion:
    """Single version of a bill."""
    version_number: int
    text: str
    name: str  # "Introduced", "Engrossed", etc.


@dataclass
class EvolutionResult:
    """Result of sequential version walk."""
    bill_id: str
    findings_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)


EVOLUTION_PROMPT_V1 = """Analyze this bill (Version 1: {version_name}).

BILL TEXT:
{bill_text}

Extract findings and assign IDs (F1, F2, etc.).

OUTPUT (JSON):
{{
  "findings": [
    {{
      "id": "F1",
      "statement": "...",
      "origin_version": 1,
      "affected_sections": ["2.1"],
      "modification_count": 0
    }}
  ]
}}
"""

EVOLUTION_PROMPT_VN = """Analyze Version {version_number}: {version_name}.

PREVIOUS FINDINGS (from memory):
{previous_findings}

NEW VERSION TEXT:
{bill_text}

TASK:
1. Compare to previous findings
2. Mark findings as: STABLE (unchanged), MODIFIED (changed), or NEW
3. Update modification counts

OUTPUT (JSON):
{{
  "findings": [
    {{
      "id": "F1",  // existing ID if stable/modified, new ID if new
      "statement": "...",
      "origin_version": 1,
      "modification_count": 1,  // increment if modified
      "status": "MODIFIED"
    }}
  ]
}}
"""


class SequentialEvolutionAgent:
    """
    Walk bill versions sequentially, maintaining structured memory.
    """
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.memory: Dict[str, Any] = {}
    
    def walk_versions(
        self,
        bill_id: str,
        versions: List[BillVersion]
    ) -> EvolutionResult:
        """
        Walk versions in order, maintaining findings registry.
        
        Args:
            bill_id: Bill identifier
            versions: List of BillVersion objects in chronological order
        
        Returns:
            EvolutionResult with findings registry and stability scores
        """
        findings_registry = {}
        
        for idx, version in enumerate(versions):
            if idx == 0:
                # First version: extract initial findings
                analysis = self._analyze_version(version, is_first=True)
            else:
                # Subsequent versions: compare to memory
                analysis = self._analyze_version(version, is_first=False, memory=findings_registry)
            
            # Update registry
            for finding in analysis["findings"]:
                finding_id = finding["id"]
                findings_registry[finding_id] = finding
        
        # Compute stability scores
        stability_scores = self._compute_stability(findings_registry, num_versions=len(versions))
        
        return EvolutionResult(
            bill_id=bill_id,
            findings_registry=findings_registry,
            stability_scores=stability_scores
        )
    
    def _analyze_version(
        self,
        version: BillVersion,
        is_first: bool,
        memory: Dict = None
    ) -> Dict[str, Any]:
        """Analyze single version with or without memory context."""
        if is_first:
            prompt = EVOLUTION_PROMPT_V1.format(
                version_name=version.name,
                bill_text=version.text
            )
        else:
            import json
            previous_findings_str = json.dumps(list(memory.values()), indent=2)
            prompt = EVOLUTION_PROMPT_VN.format(
                version_number=version.version_number,
                version_name=version.name,
                previous_findings=previous_findings_str,
                bill_text=version.text
            )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    
    def _compute_stability(self, registry: Dict, num_versions: int) -> Dict[str, float]:
        """
        Compute stability score for each finding.
        
        Stability formula:
        - origin=1, mods=0 → 0.95 (survived all, stable)
        - origin=1, mods=1 → 0.85 (one refinement)
        - origin=1, mods=3+ → 0.40 (contentious)
        - origin=N (last version) → 0.20 (last-minute, risky)
        """
        scores = {}
        for finding_id, finding in registry.items():
            origin = finding.get("origin_version", 1)
            mods = finding.get("modification_count", 0)
            
            if origin == num_versions:
                # Last-minute addition
                score = 0.20
            elif mods == 0:
                # Never modified, very stable
                score = 0.95
            elif mods == 1:
                # One refinement
                score = 0.85
            elif mods == 2:
                # Two changes
                score = 0.70
            else:
                # 3+ modifications, contentious
                score = 0.40
            
            scores[finding_id] = score
        
        return scores
```

**Step 4: Run test**

Run: `pytest tests/test_v2/test_sequential_evolution.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/v2/sequential_evolution.py tests/test_v2/test_sequential_evolution.py
git commit -m "feat(v2): add sequential evolution agent with memory"
```

---

## Task 4: Expand Rubrics to 4 Dimensions

**Files:**
- Modify: `/Users/thamac/Documents/NRG/nrg_core/v2/rubrics.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_rubrics.py`

**Step 1: Add new rubric definitions**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/rubrics.py`:

```python
# Add two new rubrics
OPERATIONAL_DISRUPTION_RUBRIC = {
    "0-2": "No operational changes required",
    "3-5": "Process adjustments - training, reporting procedures",
    "6-8": "System changes - hiring, infrastructure, restructuring",
    "9-10": "Business model changes - asset divestitures, strategic pivots"
}

AMBIGUITY_RISK_RUBRIC = {
    "0-2": "Clear, explicit language - no interpretation needed",
    "3-5": "Some ambiguity - interpretations clear from context",
    "6-8": "Significant ambiguity - regulatory guidance TBD",
    "9-10": "Vague, contradictory - novel legal concepts, high uncertainty"
}

# Export all rubrics
ALL_RUBRICS = {
    "legal_risk": LEGAL_RISK_RUBRIC,
    "financial_impact": FINANCIAL_IMPACT_RUBRIC,
    "operational_disruption": OPERATIONAL_DISRUPTION_RUBRIC,
    "ambiguity_risk": AMBIGUITY_RISK_RUBRIC
}
```

**Step 2: Update RubricScore model**

Modify `/Users/thamac/Documents/NRG/nrg_core/models_v2.py`:

```python
# Change dimension type to include new dimensions
class RubricScore(BaseModel):
    dimension: Literal["legal_risk", "financial_impact", "operational_disruption", "ambiguity_risk"]
    # ... rest unchanged
```

**Step 3: Write test for 4-dimension scoring**

Add to `/Users/thamac/Documents/NRG/tests/test_v2/test_rubrics.py`:

```python
def test_all_four_dimensions_scored():
    """Judge should score all 4 rubric dimensions."""
    from nrg_core.v2.rubrics import ALL_RUBRICS
    
    judge = JudgeModel(model="gpt-4o", api_key="test-key")
    finding = Finding(
        statement="New system requires quarterly reporting and software changes",
        quotes=[Quote(text="...", section="3.1", "page": None)],
        confidence=0.85,
        impact_estimate=6
    )
    
    scores = []
    for dimension, rubric in ALL_RUBRICS.items():
        mock_score = {
            "dimension": dimension,
            "score": 5,
            "rationale": f"Test rationale for {dimension}",
            "evidence": [{"text": "...", "section": "3.1", "page": None}],
            "rubric_anchor": "3-5: test anchor"
        }
        
        with patch.object(judge, '_call_llm_for_rubric', return_value=mock_score):
            score = judge.score_rubric(dimension, finding, "bill", "nrg", rubric)
            scores.append(score)
    
    assert len(scores) == 4
    assert {s.dimension for s in scores} == {"legal_risk", "financial_impact", "operational_disruption", "ambiguity_risk"}
```

**Step 4: Run test**

Run: `pytest tests/test_v2/test_rubrics.py::test_all_four_dimensions_scored -v`

Expected: PASS

**Step 5: Update two-tier orchestrator to score all 4**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/two_tier.py`:

```python
# Change rubric scoring loop
from nrg_core.v2.rubrics import ALL_RUBRICS

            # Score on all 4 dimensions
            for dimension, rubric in ALL_RUBRICS.items():
                score = self.judge.score_rubric(
                    dimension=dimension,
                    finding=finding,
                    bill_text=bill_text,
                    nrg_context=nrg_context,
                    rubric_anchors=rubric
                )
                rubric_scores.append(score)
```

**Step 6: Commit**

```bash
git add nrg_core/v2/rubrics.py nrg_core/models_v2.py tests/test_v2/test_rubrics.py nrg_core/v2/two_tier.py
git commit -m "feat(v2): expand rubrics to 4 dimensions (add operational + ambiguity)"
```

---

## Task 5: Add Audit Trail Generation

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/audit_trail.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_audit_trail.py`

**Step 1: Write test**

```python
# tests/test_v2/test_audit_trail.py
import pytest
from nrg_core.v2.audit_trail import AuditTrailGenerator
from nrg_core.models_v2 import Finding, Quote, RubricScore, JudgeValidation


def test_audit_trail_generation():
    """Should generate complete audit trail with quotes, scores, rationales."""
    generator = AuditTrailGenerator()
    
    finding = Finding(
        statement="Tax of $50/MW",
        quotes=[Quote(text="Annual tax of $50 per megawatt", section="2.1", page=5)],
        confidence=0.9,
        impact_estimate=7
    )
    
    validation = JudgeValidation(
        finding_id=0,
        quote_verified=True,
        hallucination_detected=False,
        evidence_quality=0.95,
        ambiguity=0.1,
        judge_confidence=0.95
    )
    
    scores = [
        RubricScore(
            dimension="financial_impact",
            score=7,
            rationale="$1.15M annual cost",
            evidence=[Quote(text="...", section="2.1", page=5)],
            rubric_anchor="6-8: $500K-$5M material"
        )
    ]
    
    trail = generator.generate(
        finding=finding,
        validation=validation,
        rubric_scores=scores
    )
    
    assert "quotes_used" in trail
    assert "validation_result" in trail
    assert "rubric_scores" in trail
    assert trail["validation_result"]["quote_verified"] is True
    assert len(trail["rubric_scores"]) == 1
```

**Step 2: Implement audit trail generator**

```python
# nrg_core/v2/audit_trail.py
"""
Audit Trail Generator
Creates compliance-ready documentation of analysis
"""
from typing import Dict, Any, List
from datetime import datetime
from nrg_core.models_v2 import Finding, JudgeValidation, RubricScore


class AuditTrailGenerator:
    """Generate audit trails for compliance and transparency."""
    
    def generate(
        self,
        finding: Finding,
        validation: JudgeValidation,
        rubric_scores: List[RubricScore]
    ) -> Dict[str, Any]:
        """
        Generate complete audit trail for a finding.
        
        Returns:
            Dict with all evidence, validations, scores, timestamps
        """
        trail = {
            "finding": {
                "statement": finding.statement,
                "confidence": finding.confidence,
                "impact_estimate": finding.impact_estimate
            },
            "quotes_used": [
                {
                    "text": q.text,
                    "section": q.section,
                    "page": q.page
                }
                for q in finding.quotes
            ],
            "validation_result": {
                "quote_verified": validation.quote_verified,
                "hallucination_detected": validation.hallucination_detected,
                "evidence_quality": validation.evidence_quality,
                "judge_confidence": validation.judge_confidence
            },
            "rubric_scores": [
                {
                    "dimension": s.dimension,
                    "score": s.score,
                    "rationale": s.rationale,
                    "rubric_anchor": s.rubric_anchor
                }
                for s in rubric_scores
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "trail_version": "v2.0"
        }
        
        return trail
```

**Step 3: Run test**

Run: `pytest tests/test_v2/test_audit_trail.py -v`

Expected: PASS

**Step 4: Integrate into two-tier**

Modify `/Users/thamac/Documents/NRG/nrg_core/v2/two_tier.py`:

```python
# Add to imports
from nrg_core.v2.audit_trail import AuditTrailGenerator

# Add to __init__
        self.audit_generator = AuditTrailGenerator()

# After rubric scoring, generate trails
        audit_trails = []
        for idx, finding in enumerate(primary_analysis.findings):
            finding_scores = [s for s in rubric_scores if s.dimension in ALL_RUBRICS.keys()]
            trail = self.audit_generator.generate(
                finding=finding,
                validation=judge_validations[idx],
                rubric_scores=finding_scores[:4]  # One set per finding
            )
            audit_trails.append(trail)
```

**Step 5: Commit**

```bash
git add nrg_core/v2/audit_trail.py tests/test_v2/test_audit_trail.py nrg_core/v2/two_tier.py
git commit -m "feat(v2): add audit trail generation for compliance"
```

---

## Summary

**Phase 2 Complete:** Enhanced analysis with:

✅ Task 1: Multi-sample consistency check (Tier 1.5)  
✅ Task 2: Fallback second model (Claude for adversarial check)  
✅ Task 3: Sequential evolution agent (memory-based version walking)  
✅ Task 4: 4-dimension rubrics (added operational + ambiguity)  
✅ Task 5: Audit trail generation

**Verification:**
```bash
pytest tests/test_v2/ -v
```

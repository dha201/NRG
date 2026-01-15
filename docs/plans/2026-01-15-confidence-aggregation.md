# Confidence Aggregation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Aggregate four confidence signals (evidence quality, model agreement, causal strength, evolution stability) into calibrated scores with uncertainty bounds

**Architecture:** Component scoring → weighted aggregation → confidence interval calculation via historical calibration set

**Tech Stack:** Python 3.9+, NumPy for statistics, dataclasses for models

**Business Context:** Single confidence score enables routing decisions. Weighting reflects signal importance for legal accuracy.

---

## Component 4: Confidence Aggregation - Visual Flow

```
INPUT: Finding + [Evidence, Agreement, Causal, Stability] signals
│
└─→ Signal 1: Evidence Quality (40% weight)
    │
    ├─ Input: Finding text + bill text
    ├─ Measure: Quote match score, specificity (generic vs threshold-specific)
    ├─ Example: "Tax on >50MW" with exact quote = 0.95
    ├─ Example: "Tax applies" generic claim = 0.60
    └─ Output: Evidence quality score (0.0-1.0)

Signal 2: Model Agreement (30% weight)
    │
    ├─ Input: Which models found the finding (from Consensus Ensemble)
    ├─ Measure: Count agreeing models (0, 1/3, 2/3, 3/3)
    ├─ Example: "unanimous" (3/3) = 0.95
    ├─ Example: "majority" (2/3) = 0.70
    ├─ Example: "disputed" (1/3) = 0.40
    └─ Output: Agreement score (0.0-1.0)

Signal 3: Causal Strength (20% weight)
    │
    ├─ Input: Causal chain from Causal Reasoning component
    ├─ Measure: Average evidence strength across 5-step chain
    ├─ Example: All 5 steps well-evidenced = 0.92
    ├─ Example: Some steps missing evidence = 0.65
    ├─ Example: No causal chain built = 0.40
    └─ Output: Causal strength score (0.0-1.0)

Signal 4: Stability (10% weight)
    │
    ├─ Input: Stability score from Evolutionary Analysis component
    ├─ Measure: How unchanged the provision has been across versions
    ├─ Example: Unchanged since V1 = 0.95
    ├─ Example: Modified 3+ times = 0.40
    ├─ Example: Last-minute addition = 0.20
    └─ Output: Stability score (0.0-1.0)

Weighted Aggregation
│
└─ Final confidence = (
     0.40 * evidence_quality +
     0.30 * model_agreement +
     0.20 * causal_strength +
     0.10 * stability_score
   )

   Example: (0.40*0.95 + 0.30*0.95 + 0.20*0.92 + 0.10*0.95)
          = (0.38 + 0.285 + 0.184 + 0.095)
          = 0.944 (very high confidence)

Why these weights:
- Evidence (40%): Most important - does bill text actually support the finding?
- Agreement (30%): Multiple models reduce hallucination risk
- Causal (20%): Business impact reasoning strengthens confidence in relevance
- Stability (10%): Low weight because <10% of bill is usually contentious

Final Output: ConfidenceBreakdown
│
└─ {
     overall_confidence: 0.944,
     evidence_quality: 0.95,
     model_agreement: 0.95,
     causal_strength: 0.92,
     stability: 0.95,
     is_high_confidence: true,
     recommendation: "AUTO_PUBLISH"  (if >0.95)
   }
```

---

## Task 1: Confidence Component Models

**Files:**
- Create: `nrg_core/sota/confidence/models.py`
- Test: `tests/test_sota/test_confidence_models.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_confidence_models.py
import pytest
from nrg_core.sota.confidence.models import ConfidenceComponents, ConfidenceBreakdown

def test_confidence_components():
    components = ConfidenceComponents(
        evidence_quality=0.90,
        model_agreement=0.75,
        causal_strength=0.80,
        evolution_stability=0.85
    )

    assert components.evidence_quality == 0.90
    assert components.model_agreement == 0.75

def test_confidence_breakdown():
    breakdown = ConfidenceBreakdown(
        overall_confidence=0.83,
        components=ConfidenceComponents(0.90, 0.75, 0.80, 0.85),
        confidence_interval=(0.70, 0.86),
        weakest_component="model_agreement"
    )

    assert breakdown.overall_confidence == 0.83
    assert breakdown.weakest_component == "model_agreement"
    assert breakdown.confidence_interval[0] < breakdown.overall_confidence < breakdown.confidence_interval[1]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_confidence_models.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/confidence/models.py
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class ConfidenceComponents:
    """Four independent confidence signals"""
    evidence_quality: float
    model_agreement: float
    causal_strength: float
    evolution_stability: float

    def to_dict(self) -> dict:
        return {
            'evidence_quality': self.evidence_quality,
            'model_agreement': self.model_agreement,
            'causal_strength': self.causal_strength,
            'evolution_stability': self.evolution_stability
        }

    def get_weakest(self) -> str:
        """Return name of weakest component"""
        components = {
            'evidence_quality': self.evidence_quality,
            'model_agreement': self.model_agreement,
            'causal_strength': self.causal_strength,
            'evolution_stability': self.evolution_stability
        }
        return min(components, key=components.get)

@dataclass
class ConfidenceBreakdown:
    """Complete confidence assessment with decomposition"""
    overall_confidence: float
    components: ConfidenceComponents
    confidence_interval: Tuple[float, float]
    weakest_component: str
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            'overall_confidence': self.overall_confidence,
            'components': self.components.to_dict(),
            'confidence_interval': {
                'lower': self.confidence_interval[0],
                'upper': self.confidence_interval[1]
            },
            'weakest_component': self.weakest_component,
            'interpretation': self.interpretation
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_confidence_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/confidence/models.py tests/test_sota/test_confidence_models.py
git commit -m "feat(sota): add confidence aggregation models

- Add ConfidenceComponents for 4 signals
- Add ConfidenceBreakdown with intervals
- Add weakest component detection
- Add tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Component Scorers

**Files:**
- Create: `nrg_core/sota/confidence/scorers.py`
- Test: `tests/test_sota/test_confidence_scorers.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_confidence_scorers.py
import pytest
from nrg_core.sota.confidence.scorers import (
    EvidenceQualityScorer,
    ModelAgreementScorer,
    CausalStrengthScorer,
    EvolutionStabilityScorer
)
from nrg_core.sota.models import Finding
from nrg_core.sota.causal.models import CausalChain, CausalStep
from nrg_core.sota.evolution.models import FindingLineage

def test_evidence_quality_scorer():
    finding = Finding(
        statement="Test",
        confidence=0.9,
        supporting_quotes=["Section 1", "Section 2", "Section 3"],
        found_by=["GPT-4o"],
        consensus_level="unanimous"
    )

    scorer = EvidenceQualityScorer()
    score = scorer.score(finding, "Section 1 text here. Section 2 more text.")

    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Has quotes that verify

def test_model_agreement_scorer():
    finding = Finding(
        statement="Test",
        confidence=0.95,
        supporting_quotes=[],
        found_by=["Gemini", "GPT-4o", "Claude"],
        consensus_level="unanimous"
    )

    scorer = ModelAgreementScorer()
    score = scorer.score(finding)

    assert score == 0.95  # Unanimous = 0.95

def test_causal_strength_scorer():
    chain = CausalChain(
        root_cause="Test",
        root_quote="quote",
        steps=[
            CausalStep(1, "Q1", "A1", "quote1", 0.95),
            CausalStep(2, "Q2", "A2", "quote2", 0.90),
            CausalStep(3, "Q3", "A3", "quote3", 0.85),
        ],
        business_impact="Impact",
        impact_confidence=0.90
    )

    scorer = CausalStrengthScorer()
    score = scorer.score(chain)

    assert 0.0 <= score <= 1.0
    assert score > 0.75  # Strong chain

def test_evolution_stability_scorer():
    lineage = FindingLineage(
        finding_statement="Test",
        origin_version=1,
        modification_count=1,
        stability_score=0.85,
        version_history=[]
    )

    scorer = EvolutionStabilityScorer()
    score = scorer.score(lineage)

    assert score == 0.85  # Direct from lineage
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_confidence_scorers.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/confidence/scorers.py
from typing import Optional
from nrg_core.sota.models import Finding
from nrg_core.sota.causal.models import CausalChain
from nrg_core.sota.evolution.models import FindingLineage

class EvidenceQualityScorer:
    """Score evidence quality (0-1)"""

    def score(self, finding: Finding, bill_text: str) -> float:
        """
        Score based on:
        - Quote existence: +0.20
        - Quote clarity: +0.20
        - Supporting definitions: +0.20
        - Quote count: +0.15
        - Hallucination risk: +0.15
        """
        score = 0.0

        # Quote existence
        if finding.supporting_quotes:
            score += 0.20

        # Quote clarity (verify in bill text)
        if finding.supporting_quotes:
            verified = any(q.lower() in bill_text.lower() for q in finding.supporting_quotes)
            if verified:
                score += 0.20

        # Supporting definitions (heuristic: multiple quotes suggests definitions)
        if len(finding.supporting_quotes) >= 2:
            score += 0.20

        # Quote count
        quote_count = len(finding.supporting_quotes)
        if quote_count == 1:
            score += 0.05
        elif quote_count in [2, 3]:
            score += 0.10
        elif quote_count >= 4:
            score += 0.15

        # Hallucination risk (if verified above)
        if finding.verification_status == "verified":
            score += 0.15

        return min(score, 1.0)

class ModelAgreementScorer:
    """Score model agreement (0-1)"""

    def score(self, finding: Finding) -> float:
        """
        Score based on consensus level:
        - Unanimous (3/3): 0.95
        - Majority (2/3): 0.70
        - Verified (1/3 + quote): 0.50
        - Unverified (1/3): 0.25
        """
        if finding.consensus_level == "unanimous":
            return 0.95
        elif finding.consensus_level == "majority":
            return 0.70
        elif finding.consensus_level == "verified":
            return 0.50
        elif finding.consensus_level == "disputed":
            if finding.verification_status == "verified":
                return 0.50
            else:
                return 0.25
        else:
            return 0.50  # Default

class CausalStrengthScorer:
    """Score causal reasoning strength (0-1)"""

    def score(self, chain: Optional[CausalChain]) -> float:
        """
        Score based on:
        - Root cause clarity: +0.25
        - Step verification: +0.25
        - Alternatives explored: +0.15
        - Robustness: +0.15
        - Coverage: +0.20
        """
        if not chain:
            return 0.50

        score = 0.0

        # Root cause clarity
        if chain.root_cause and chain.root_quote:
            score += 0.25

        # Step verification (% with high evidence)
        if chain.steps:
            verified = sum(1 for s in chain.steps if s.evidence_strength >= 0.80)
            score += 0.25 * (verified / len(chain.steps))

        # Alternatives explored
        if len(chain.alternatives) >= 3:
            score += 0.15
        elif len(chain.alternatives) >= 1:
            score += 0.07

        # Robustness (counterfactuals)
        if chain.counterfactuals:
            score += 0.15

        # Coverage
        if chain.evidence_coverage >= 0.80:
            score += 0.20
        else:
            score += 0.20 * chain.evidence_coverage

        return min(score, 1.0)

class EvolutionStabilityScorer:
    """Score evolution stability (0-1)"""

    def score(self, lineage: Optional[FindingLineage]) -> float:
        """Return stability score from lineage"""
        if not lineage:
            return 0.50  # Default for no version history

        return lineage.stability_score
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_confidence_scorers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/confidence/scorers.py tests/test_sota/test_confidence_scorers.py
git commit -m "feat(sota): add component scorers

- Add EvidenceQualityScorer with 5 criteria
- Add ModelAgreementScorer based on consensus
- Add CausalStrengthScorer with 5 factors
- Add EvolutionStabilityScorer
- Add comprehensive tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 2

**Q1: Evidence Quality Criteria**
- **Gap**: Equal weighting of 5 criteria may not be optimal
- **Question**: Should some criteria (quote verification) weigh more?
- **Priority**: Medium | **Blocker**: No - equal weighting reasonable for MVP

---

## Task 3: Confidence Aggregator

**Files:**
- Create: `nrg_core/sota/confidence/aggregator.py`
- Test: `tests/test_sota/test_confidence_aggregator.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_confidence_aggregator.py
import pytest
from nrg_core.sota.confidence.aggregator import ConfidenceAggregator
from nrg_core.sota.models import Finding
from nrg_core.sota.causal.models import CausalChain, CausalStep
from nrg_core.sota.evolution.models import FindingLineage

def test_aggregate_confidence():
    finding = Finding(
        statement="Tax applies",
        confidence=0.9,
        supporting_quotes=["Section 2.1"],
        found_by=["GPT-4o", "Claude"],
        consensus_level="majority",
        verification_status="verified"
    )

    chain = CausalChain(
        root_cause="Section 2.1",
        root_quote="quote",
        steps=[CausalStep(1, "Q", "A", "quote", 0.95)],
        business_impact="Impact",
        impact_confidence=0.90
    )

    lineage = FindingLineage(
        finding_statement="Tax applies",
        origin_version=1,
        modification_count=1,
        stability_score=0.85,
        version_history=[]
    )

    aggregator = ConfidenceAggregator()
    breakdown = aggregator.aggregate(
        finding=finding,
        chain=chain,
        lineage=lineage,
        bill_text="Section 2.1 text here"
    )

    assert 0.0 <= breakdown.overall_confidence <= 1.0
    assert breakdown.weakest_component in ["evidence_quality", "model_agreement", "causal_strength", "evolution_stability"]
    assert len(breakdown.confidence_interval) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_confidence_aggregator.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/confidence/aggregator.py
import numpy as np
from typing import Optional, Tuple
from nrg_core.sota.models import Finding
from nrg_core.sota.causal.models import CausalChain
from nrg_core.sota.evolution.models import FindingLineage
from nrg_core.sota.confidence.models import ConfidenceComponents, ConfidenceBreakdown
from nrg_core.sota.confidence.scorers import (
    EvidenceQualityScorer,
    ModelAgreementScorer,
    CausalStrengthScorer,
    EvolutionStabilityScorer
)

class ConfidenceAggregator:
    """Aggregate confidence components into final score"""

    # Component weights (must sum to 1.0)
    WEIGHTS = {
        'evidence_quality': 0.40,
        'model_agreement': 0.30,
        'causal_strength': 0.20,
        'evolution_stability': 0.10
    }

    def __init__(self):
        self.evidence_scorer = EvidenceQualityScorer()
        self.agreement_scorer = ModelAgreementScorer()
        self.causal_scorer = CausalStrengthScorer()
        self.stability_scorer = EvolutionStabilityScorer()

    def aggregate(
        self,
        finding: Finding,
        chain: Optional[CausalChain],
        lineage: Optional[FindingLineage],
        bill_text: str
    ) -> ConfidenceBreakdown:
        """
        Aggregate all confidence signals

        Args:
            finding: Consensus finding
            chain: Causal reasoning chain
            lineage: Finding evolution lineage
            bill_text: Bill text for verification

        Returns:
            ConfidenceBreakdown with overall score and components
        """
        # Score each component
        components = ConfidenceComponents(
            evidence_quality=self.evidence_scorer.score(finding, bill_text),
            model_agreement=self.agreement_scorer.score(finding),
            causal_strength=self.causal_scorer.score(chain),
            evolution_stability=self.stability_scorer.score(lineage)
        )

        # Weighted aggregation
        overall = (
            components.evidence_quality * self.WEIGHTS['evidence_quality'] +
            components.model_agreement * self.WEIGHTS['model_agreement'] +
            components.causal_strength * self.WEIGHTS['causal_strength'] +
            components.evolution_stability * self.WEIGHTS['evolution_stability']
        )

        # Calculate confidence interval
        interval = self._calculate_interval(overall, components)

        # Identify weakest component
        weakest = components.get_weakest()

        # Generate interpretation
        interpretation = self._generate_interpretation(overall, components, weakest)

        return ConfidenceBreakdown(
            overall_confidence=overall,
            components=components,
            confidence_interval=interval,
            weakest_component=weakest,
            interpretation=interpretation
        )

    def _calculate_interval(
        self,
        overall: float,
        components: ConfidenceComponents
    ) -> Tuple[float, float]:
        """
        Calculate 90% confidence interval

        Simple approach: ±15% of overall confidence
        More sophisticated: Use calibration set (future enhancement)
        """
        margin = 0.15 * overall

        lower = max(0.0, overall - margin)
        upper = min(1.0, overall + margin)

        return (lower, upper)

    def _generate_interpretation(
        self,
        overall: float,
        components: ConfidenceComponents,
        weakest: str
    ) -> str:
        """Generate human-readable interpretation"""
        if overall >= 0.85:
            level = "High"
        elif overall >= 0.70:
            level = "Medium-high"
        elif overall >= 0.50:
            level = "Medium"
        else:
            level = "Low"

        interpretation = f"{level} confidence ({overall:.2f}). "

        if weakest == "evidence_quality" and components.evidence_quality < 0.70:
            interpretation += "Main weakness: Evidence quality. Recommend quote verification."
        elif weakest == "model_agreement" and components.model_agreement < 0.70:
            interpretation += "Main weakness: Model disagreement. Recommend expert review."
        elif weakest == "causal_strength" and components.causal_strength < 0.70:
            interpretation += "Main weakness: Causal reasoning. Recommend deeper analysis."
        elif weakest == "evolution_stability" and components.evolution_stability < 0.50:
            interpretation += "Main weakness: Recently added provision. Recommend monitoring."

        return interpretation
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_confidence_aggregator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/confidence/aggregator.py tests/test_sota/test_confidence_aggregator.py
git commit -m "feat(sota): add confidence aggregator

- Add weighted aggregation of 4 components
- Add confidence interval calculation
- Add interpretation generation
- Add component weights (evidence: 40%, agreement: 30%, causal: 20%, stability: 10%)
- Add tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 3

**Q2: Confidence Interval Calculation** ⚠️ BLOCKING
- **Gap**: Simple ±15% margin, not calibrated
- **Question**: Should we calibrate using historical accuracy data? What method (bootstrap, quantile)?
- **Priority**: High | **Blocker**: YES - confidence intervals critical for decision-making
- **Recommended Answer**: Start with ±15% margin for MVP. Post-MVP, calibrate using quantile regression on validation set (map predicted confidence to actual accuracy).

**Q3: Component Weights** ⚠️ BLOCKING
- **Gap**: Weights (0.40, 0.30, 0.20, 0.10) not validated
- **Question**: A/B test different weight schemes? Optimize using validation set?
- **Priority**: High | **Blocker**: YES - weights directly affect routing decisions
- **Recommended Answer**: Validate current weights on test set. If FPR >1%, grid search over weight combinations to minimize FPR while maintaining recall >90%.

**Q4: Missing Data Handling** ⚠️ BLOCKING
- **Gap**: What if chain or lineage is None?
- **Question**: Default to 0.50? Use only available signals? Flag as incomplete?
- **Priority**: High | **Blocker**: YES - will occur frequently
- **Recommended Answer**:
  - If chain is None: Set causal_strength = 0.50, increase evidence_quality weight from 0.40 to 0.50
  - If lineage is None: Set evolution_stability = 0.50, redistribute weight proportionally
  - Flag finding as "incomplete_analysis"

---

## Summary of Implementation Questions

All gaps and clarification questions have been distributed into the relevant tasks above. Look for sections marked "Implementation Questions for Task X" after each task's commit step.

**Blocking Questions** (marked with ⚠️):
- Task 3, Q2: Confidence Interval Calculation
- Task 3, Q3: Component Weights
- Task 3, Q4: Missing Data Handling

**See** [GAPS_AND_QUESTIONS.md](./GAPS_AND_QUESTIONS.md) for complete cross-component questions and integration issues.

---

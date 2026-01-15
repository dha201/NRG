# Routing Decision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route analysis results to appropriate actions based on confidence level and consensus status, minimizing cost while managing risk

**Architecture:** Decision tree evaluation → SLA assignment → expertise determination → escalation path generation

**Tech Stack:** Python 3.9+, dataclasses for models, enum for action types

**Business Context:** Routing determines analyst workload and review SLA. High-confidence = automated, low-confidence = senior expert review.

---

## Component 5: Routing Decision - Visual Flow

```
INPUT: ConfidenceBreakdown (overall_confidence: 0.0-1.0) + ConsensusAnalysis
│
└─→ Decision Tree (threshold-based routing)
    │
    ├─ IF confidence >= 0.95 AND consensus == "unanimous"
    │  │
    │  └─→ ACTION: AUTO_PUBLISH
    │      ├─ SLA: 24 hours (system publishes automatically)
    │      ├─ Expertise: None (fully automated)
    │      ├─ Cost: $0 (no human review)
    │      └─ Risk: Low (<1% FPR target)
    │
    ├─ ELSE IF confidence >= 0.85 AND consensus IN ["unanimous", "majority"]
    │  │
    │  └─→ ACTION: FLAGGED_PUBLISH
    │      ├─ SLA: 24 hours (publish but flag for expert review)
    │      ├─ Expertise: ["compliance_reviewer"] (1 person spot-check)
    │      ├─ Cost: ~$50 (light review)
    │      └─ Risk: Medium (1-5% FPR)
    │
    ├─ ELSE IF 0.70 <= confidence < 0.85
    │  │
    │  └─→ ACTION: EXPERT_REVIEW
    │      ├─ SLA: 72 hours (requires expert analysis)
    │      ├─ Expertise: ["domain_expert", "legal_reviewer"] (2 people)
    │      ├─ Cost: ~$200 (full analysis)
    │      └─ Risk: High (5-10% FPR)
    │
    └─ ELSE (confidence < 0.70)
       │
       └─→ ACTION: ESCALATION
           ├─ SLA: 4 hours (urgent - senior analyst only)
           ├─ Expertise: ["senior_analyst", "domain_expert", "legal_director"] (3+ people)
           ├─ Priority: HIGH
           ├─ Cost: ~$1000+ (extensive review + possible legal consult)
           └─ Risk: Very High (>10% FPR or legal ambiguity)

Special Cases:
│
├─ IF has_unresolved_disputes == true
│  └─ Escalate one level regardless of confidence
│     (e.g., 0.85 → EXPERT_REVIEW instead of FLAGGED_PUBLISH)
│
├─ IF causal_incomplete == true
│  └─ Route to EXPERT_REVIEW minimum
│     (can't explain impact fully, needs human interpretation)
│
└─ IF finding_topic == "tax_threshold_change"
   └─ Route to EXPERT_REVIEW minimum
      (threshold changes have outsized business impact, need careful review)

Final Output: RoutingRecommendation
│
└─ {
     action: "AUTO_PUBLISH",
     sla_hours: 24,
     expertise_required: [],
     escalation_path: [],
     priority: "normal",
     risk_level: "low",
     rationale: "Very high confidence (0.95) with unanimous consensus"
   }
```

---

## Task 1: Routing Models and Action Types

**Files:**
- Create: `nrg_core/sota/routing/models.py`
- Test: `tests/test_sota/test_routing_models.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_routing_models.py
import pytest
from nrg_core.sota.routing.models import ActionType, RoutingRecommendation

def test_action_types():
    assert ActionType.AUTO_PUBLISH.value == "auto_publish"
    assert ActionType.EXPERT_REVIEW.value == "expert_review"

def test_routing_recommendation():
    rec = RoutingRecommendation(
        action=ActionType.EXPERT_REVIEW,
        confidence=0.83,
        confidence_interval=(0.70, 0.86),
        required_expertise=["energy_law", "tax_compliance"],
        escalation_path=["specialist", "senior_counsel", "director"],
        sla_hours=48,
        rationale="Medium confidence requires expert review"
    )

    assert rec.action == ActionType.EXPERT_REVIEW
    assert rec.sla_hours == 48
    assert "energy_law" in rec.required_expertise
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_routing_models.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/routing/models.py
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum

class ActionType(Enum):
    """Routing action types"""
    AUTO_PUBLISH = "auto_publish"
    FLAGGED_PUBLISH = "flagged_publish"
    EXPERT_REVIEW = "expert_review"
    ESCALATION = "escalation"

@dataclass
class RoutingRecommendation:
    """Routing decision with rationale"""
    action: ActionType
    confidence: float
    confidence_interval: Tuple[float, float]
    required_expertise: List[str]
    escalation_path: List[str]
    sla_hours: int
    rationale: str
    priority: str = "normal"
    risk_level: str = "medium"

    def to_dict(self) -> dict:
        return {
            'action': self.action.value,
            'confidence': self.confidence,
            'confidence_interval': {
                'lower': self.confidence_interval[0],
                'upper': self.confidence_interval[1]
            },
            'required_expertise': self.required_expertise,
            'escalation_path': self.escalation_path,
            'sla_hours': self.sla_hours,
            'rationale': self.rationale,
            'priority': self.priority,
            'risk_level': self.risk_level
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_routing_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/routing/models.py tests/test_sota/test_routing_models.py
git commit -m "feat(sota): add routing decision models

- Add ActionType enum (auto_publish, flagged_publish, expert_review, escalation)
- Add RoutingRecommendation with SLA and expertise
- Add tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Decision Tree Router

**Files:**
- Create: `nrg_core/sota/routing/router.py`
- Test: `tests/test_sota/test_routing_router.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_routing_router.py
import pytest
from nrg_core.sota.routing.router import DecisionTreeRouter
from nrg_core.sota.routing.models import ActionType
from nrg_core.sota.confidence.models import ConfidenceBreakdown, ConfidenceComponents

def test_auto_publish_route():
    """High confidence + unanimous → AUTO_PUBLISH"""
    breakdown = ConfidenceBreakdown(
        overall_confidence=0.96,
        components=ConfidenceComponents(0.95, 0.95, 0.95, 0.95),
        confidence_interval=(0.90, 0.98),
        weakest_component="evolution_stability"
    )

    router = DecisionTreeRouter()
    recommendation = router.route(
        breakdown=breakdown,
        consensus_level="unanimous",
        has_unresolved_disputes=False,
        causal_complete=True
    )

    assert recommendation.action == ActionType.AUTO_PUBLISH
    assert recommendation.sla_hours <= 24

def test_expert_review_route():
    """Medium confidence → EXPERT_REVIEW"""
    breakdown = ConfidenceBreakdown(
        overall_confidence=0.83,
        components=ConfidenceComponents(0.90, 0.70, 0.80, 0.85),
        confidence_interval=(0.70, 0.86),
        weakest_component="model_agreement"
    )

    router = DecisionTreeRouter()
    recommendation = router.route(
        breakdown=breakdown,
        consensus_level="majority",
        has_unresolved_disputes=False,
        causal_complete=True
    )

    assert recommendation.action == ActionType.EXPERT_REVIEW
    assert recommendation.sla_hours == 48

def test_escalation_route():
    """Low confidence OR disputes → ESCALATION"""
    breakdown = ConfidenceBreakdown(
        overall_confidence=0.55,
        components=ConfidenceComponents(0.60, 0.50, 0.55, 0.60),
        confidence_interval=(0.40, 0.70),
        weakest_component="model_agreement"
    )

    router = DecisionTreeRouter()
    recommendation = router.route(
        breakdown=breakdown,
        consensus_level="disputed",
        has_unresolved_disputes=True,
        causal_complete=False
    )

    assert recommendation.action == ActionType.ESCALATION
    assert recommendation.sla_hours == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_routing_router.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/routing/router.py
from typing import List
from nrg_core.sota.routing.models import ActionType, RoutingRecommendation
from nrg_core.sota.confidence.models import ConfidenceBreakdown

class DecisionTreeRouter:
    """Routes findings based on decision tree"""

    def route(
        self,
        breakdown: ConfidenceBreakdown,
        consensus_level: str,
        has_unresolved_disputes: bool,
        causal_complete: bool,
        finding_topic: str = "general"
    ) -> RoutingRecommendation:
        """
        Route finding through decision tree

        Args:
            breakdown: Confidence assessment
            consensus_level: unanimous, majority, disputed
            has_unresolved_disputes: Whether disputes remain
            causal_complete: Whether causal chain is complete
            finding_topic: Topic for expertise routing

        Returns:
            RoutingRecommendation with action, SLA, expertise
        """
        confidence = breakdown.overall_confidence

        # Step 1: Check for auto-publish
        if confidence >= 0.95 and consensus_level == "unanimous":
            return self._create_recommendation(
                action=ActionType.AUTO_PUBLISH,
                breakdown=breakdown,
                sla_hours=24,
                rationale="Very high confidence (≥0.95) with unanimous consensus. Safe for automatic publication."
            )

        # Step 2: Check for flagged publish
        if confidence >= 0.85 and consensus_level in ["unanimous", "majority"]:
            return self._create_recommendation(
                action=ActionType.FLAGGED_PUBLISH,
                breakdown=breakdown,
                sla_hours=24,
                expertise=["compliance_review"],
                rationale="High confidence (≥0.85) with strong consensus. Light compliance review recommended."
            )

        # Step 3: Check for escalation (low confidence or disputes)
        if confidence < 0.70:
            return self._create_recommendation(
                action=ActionType.ESCALATION,
                breakdown=breakdown,
                sla_hours=4,
                expertise=["senior_analyst", "domain_expert", "legal_director"],
                escalation=["senior_analyst", "domain_expert", "legal_director"],
                priority="high",
                risk_level="high",
                rationale=f"Low confidence ({confidence:.2f}). Requires urgent senior review."
            )

        if has_unresolved_disputes:
            return self._create_recommendation(
                action=ActionType.ESCALATION,
                breakdown=breakdown,
                sla_hours=4,
                expertise=["senior_analyst", "domain_expert"],
                escalation=["senior_analyst", "domain_expert", "legal_director"],
                priority="high",
                risk_level="high",
                rationale="Unresolved disputes between models. Requires immediate escalation."
            )

        if not causal_complete:
            return self._create_recommendation(
                action=ActionType.ESCALATION,
                breakdown=breakdown,
                sla_hours=4,
                expertise=["senior_analyst", "domain_expert"],
                escalation=["senior_analyst", "domain_expert"],
                priority="high",
                risk_level="high",
                rationale="Incomplete causal chain. Cannot verify reasoning. Escalation required."
            )

        # Step 4: Default to expert review
        expertise = self._determine_expertise(finding_topic, breakdown)
        return self._create_recommendation(
            action=ActionType.EXPERT_REVIEW,
            breakdown=breakdown,
            sla_hours=48,
            expertise=expertise,
            escalation=expertise + ["senior_counsel", "director"],
            rationale=f"Medium confidence ({confidence:.2f}). Specialist review recommended, particularly for {breakdown.weakest_component}."
        )

    def _create_recommendation(
        self,
        action: ActionType,
        breakdown: ConfidenceBreakdown,
        sla_hours: int,
        expertise: List[str] = None,
        escalation: List[str] = None,
        rationale: str = "",
        priority: str = "normal",
        risk_level: str = "medium"
    ) -> RoutingRecommendation:
        """Create routing recommendation"""
        return RoutingRecommendation(
            action=action,
            confidence=breakdown.overall_confidence,
            confidence_interval=breakdown.confidence_interval,
            required_expertise=expertise or [],
            escalation_path=escalation or [],
            sla_hours=sla_hours,
            rationale=rationale,
            priority=priority,
            risk_level=risk_level
        )

    def _determine_expertise(
        self,
        topic: str,
        breakdown: ConfidenceBreakdown
    ) -> List[str]:
        """Map topic to required expertise"""
        topic_map = {
            "energy": ["energy_law", "regulatory_compliance"],
            "tax": ["tax_law", "tax_compliance"],
            "labor": ["labor_law", "hr_compliance"],
            "environmental": ["environmental_law", "epa_compliance"],
            "financial": ["financial_regulation", "sec_compliance"],
        }

        expertise = topic_map.get(topic.lower(), ["legal_specialist"])

        # Add weakness-specific expertise
        if breakdown.weakest_component == "model_agreement":
            expertise.append("senior_analyst")

        return expertise
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_routing_router.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/routing/router.py tests/test_sota/test_routing_router.py
git commit -m "feat(sota): add decision tree router

- Add DecisionTreeRouter with confidence thresholds
- Implement auto-publish (≥0.95, unanimous)
- Implement flagged-publish (≥0.85, strong)
- Implement expert-review (≥0.70, complete)
- Implement escalation (<0.70, disputes, incomplete)
- Add topic-based expertise determination
- Add comprehensive tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 2

**Q1: SLA Enforcement**
- **Gap**: SLA times specified but no enforcement mechanism
- **Question**: How to monitor SLA compliance? Alerts? Automatic escalation?
- **Priority**: Medium | **Blocker**: No - monitoring feature for post-MVP

**Q2: Expertise Availability** ⚠️ BLOCKING
- **Gap**: Routing assumes experts are available
- **Question**: What if energy_law specialist is unavailable? Fallback to general counsel?
- **Priority**: High | **Blocker**: YES - will happen in production
- **Recommended Answer**: Implement fallback chain:
  1. Try primary expertise (e.g., energy_law)
  2. If unavailable, try related expertise (regulatory_compliance)
  3. If unavailable, escalate to general_counsel
  4. If unavailable, queue for review when expert available

**Q3: Cost Tracking**
- **Gap**: No cost consideration in routing
- **Question**: Should we factor in expert hourly rates when routing?
- **Priority**: Low | **Blocker**: No - accuracy more important than cost for MVP

**Q4: Dynamic Threshold Adjustment**
- **Gap**: Thresholds (0.95, 0.85, 0.70) are static
- **Question**: Should thresholds adjust based on bill importance or risk?
- **Priority**: Medium | **Blocker**: No - static thresholds acceptable for MVP

**Q5: Topic Detection** ⚠️ BLOCKING
- **Gap**: `finding_topic` must be provided externally
- **Question**: How to automatically detect topic from finding text?
- **Priority**: High | **Blocker**: YES - affects expertise routing
- **Recommended Answer**: Use keyword matching initially:
  - "tax", "tariff", "levy" → tax
  - "energy", "power", "renewable" → energy
  - "labor", "employment", "wage" → labor
  - etc.
  Post-MVP: Train classifier on finding text → topic.

**Q6: Priority Escalation**
- **Gap**: Priority is binary (normal/high)
- **Question**: Should we have more granular priority levels?
- **Priority**: Low | **Blocker**: No - binary sufficient for MVP

**Q7: Routing Audit Trail**
- **Gap**: No logging of routing decisions
- **Question**: Should we log all routing decisions for review/optimization?
- **Priority**: Medium | **Blocker**: No - nice to have for continuous improvement

---

## Task 3: Package Initialization and Documentation

**Files:**
- Create: `nrg_core/sota/routing/__init__.py`
- Create: `nrg_core/sota/routing/README.md`

**Step 1: Create package initialization**

```python
# nrg_core/sota/routing/__init__.py
"""
Routing Decision Component

Routes analysis results to appropriate actions based on confidence
and consensus, balancing automation with human oversight.
"""

from nrg_core.sota.routing.router import DecisionTreeRouter
from nrg_core.sota.routing.models import ActionType, RoutingRecommendation

__all__ = [
    'DecisionTreeRouter',
    'ActionType',
    'RoutingRecommendation'
]
```

**Step 2: Create README**

```markdown
# Routing Decision Component

## Decision Tree

```
Confidence ≥ 0.95 AND Unanimous → AUTO_PUBLISH (24h SLA)
    ↓
Confidence ≥ 0.85 AND Strong → FLAGGED_PUBLISH (24h SLA)
    ↓
Confidence < 0.70 OR Disputes → ESCALATION (4h SLA, urgent)
    ↓
Default → EXPERT_REVIEW (48h SLA)
```

## Action Types

- **AUTO_PUBLISH**: High confidence, minimal review needed
- **FLAGGED_PUBLISH**: Light compliance check before publishing
- **EXPERT_REVIEW**: Domain specialist review (energy law, tax, etc.)
- **ESCALATION**: Urgent senior review for high-risk findings

## Usage

```python
from nrg_core.sota.routing import DecisionTreeRouter

router = DecisionTreeRouter()
recommendation = router.route(
    breakdown=confidence_breakdown,
    consensus_level="majority",
    has_unresolved_disputes=False,
    causal_complete=True,
    finding_topic="energy"
)

print(f"Action: {recommendation.action.value}")
print(f"SLA: {recommendation.sla_hours} hours")
print(f"Expertise: {', '.join(recommendation.required_expertise)}")
print(f"Rationale: {recommendation.rationale}")
```
```

**Step 3: Commit**

```bash
git add nrg_core/sota/routing/__init__.py nrg_core/sota/routing/README.md
git commit -m "docs(sota): add routing decision documentation

- Add package initialization
- Add decision tree visualization
- Add usage examples

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary of Implementation Questions

All gaps and clarification questions have been distributed into the relevant tasks above. Look for sections marked "Implementation Questions for Task X" after each task's commit step.

**Blocking Questions** (marked with ⚠️):
- Task 2, Q2: Expertise Availability
- Task 2, Q5: Topic Detection

**See** [GAPS_AND_QUESTIONS.md](./GAPS_AND_QUESTIONS.md) for complete cross-component questions and integration issues.

---

# Causal Chain Reasoning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build explicit multi-step reasoning chains from bill amendments to business impacts with quote verification and alternative interpretation exploration

**Architecture:** Chain-of-thought prompting, step-by-step evidence extraction, counterfactual analysis, and DAG construction for causal paths

**Tech Stack:** Python 3.9+, NetworkX for DAG, existing LLM clients, dataclasses for causal models

**Business Context:** Causal chains answer "why does this finding matter?" by connecting bill text → legal interpretation → business operations → financial impact.

---

## Component 3: Causal Chain Reasoning - Visual Flow

```
INPUT: Finding (e.g., "Tax applies to >50MW generation capacity")
│
└─→ 5-Step Chain-of-Thought Reasoning (LLM generates)
    │
    ├─ Step 1: How does this amend existing law?
    │  └─ Answer: "Adds per-MW tax, amending Section 3.1 of Energy Code"
    │     Supporting quote: "Section 2.1(b): exceeding fifty megawatts capacity"
    │     Evidence strength: 0.95
    │
    ├─ Step 2: Who is affected by this change?
    │  └─ Answer: "Energy generation companies with capacity >50MW"
    │     Supporting quote: "Section 1.2: any person or entity generating energy"
    │     Evidence strength: 0.92
    │
    ├─ Step 3: What conditions or thresholds apply?
    │  └─ Answer: "Threshold: 50 megawatts. Rate: $X per MW annually"
    │     Supporting quote: "Section 2.1(b): $0.50 per megawatt per year"
    │     Evidence strength: 0.98
    │
    ├─ Step 4: Are there exemptions or special cases?
    │  └─ Answer: "Renewable energy facilities exempt per Section 3.2"
    │     Supporting quote: "Section 3.2: renewable energy facilities defined in Section 5.2"
    │     Evidence strength: 0.90
    │
    └─ Step 5: What is the business impact?
       └─ Answer: "Companies >50MW must pay $X per MW annually + quarterly reporting"
          Supporting quote: "Section 4.1: quarterly reporting required by March 31"
          Evidence strength: 0.88

Chain Strength Calculation
│
└─ Overall chain strength = average of 5 step evidence strengths
   Example: (0.95 + 0.92 + 0.98 + 0.90 + 0.88) / 5 = 0.926
   Interpretation: Chain is strong (>0.80), reasoning fully evidenced

Alternative Interpretations (if disagreement detected)
│
└─ Generate counter-argument chains
   Example: "Could exemption apply to all renewables or only specific types?"
   Evidence for interpretation A: 0.70
   Evidence for interpretation B: 0.75
   → Include both in output, flag ambiguity for expert review

Final Output: CausalChain
│
└─ {
     finding: "Tax applies to >50MW generation",
     steps: [Step1, Step2, Step3, Step4, Step5],
     overall_strength: 0.926,
     complete: true,
     alternative_interpretations: [AltInterpA, AltInterpB]
   }
```

---

## Task 1: Causal Chain Data Models

**Files:**
- Create: `nrg_core/sota/causal/models.py`
- Test: `tests/test_sota/test_causal_models.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_causal_models.py
import pytest
from nrg_core.sota.causal.models import CausalStep, CausalChain, AlternativeInterpretation, CausalDAG

def test_causal_step_creation():
    step = CausalStep(
        step_number=1,
        question="How does this amend existing law?",
        answer="Adds per-MW tax on energy generation",
        supporting_quote="Section 1: Amending Section 3.1 of Energy Code",
        evidence_strength=0.95
    )

    assert step.step_number == 1
    assert step.evidence_strength == 0.95
    assert "Amending" in step.supporting_quote

def test_causal_chain():
    steps = [
        CausalStep(
            step_number=1,
            question="What changes?",
            answer="New tax",
            supporting_quote="Section 2.1",
            evidence_strength=0.95
        )
    ]

    chain = CausalChain(
        root_cause="Section 2.1: Tax on >50MW",
        root_quote="exceeding fifty megawatts",
        steps=steps,
        business_impact="$500K-$2M annual liability",
        impact_confidence=0.82
    )

    assert chain.root_cause == "Section 2.1: Tax on >50MW"
    assert len(chain.steps) == 1
    assert chain.impact_confidence == 0.82

def test_alternative_interpretation():
    alt = AlternativeInterpretation(
        interpretation="Narrow: fossil fuels only",
        likelihood=0.05,
        impact_if_true="Tax applies narrowly, lower liability"
    )

    assert alt.likelihood == 0.05
    assert "fossil" in alt.interpretation
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_causal_models.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/causal/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class CausalStep:
    """Single step in causal reasoning chain"""
    step_number: int
    question: str
    answer: str
    supporting_quote: str
    evidence_strength: float
    alternative_answers: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'step_number': self.step_number,
            'question': self.question,
            'answer': self.answer,
            'supporting_quote': self.supporting_quote,
            'evidence_strength': self.evidence_strength,
            'alternative_answers': self.alternative_answers
        }

@dataclass
class AlternativeInterpretation:
    """Alternative way to interpret a provision"""
    interpretation: str
    likelihood: float
    impact_if_true: str

    def to_dict(self) -> dict:
        return {
            'interpretation': self.interpretation,
            'likelihood': self.likelihood,
            'impact_if_true': self.impact_if_true
        }

@dataclass
class CausalChain:
    """Complete causal reasoning chain"""
    root_cause: str
    root_quote: str
    steps: List[CausalStep]
    business_impact: str
    impact_confidence: float
    alternatives: List[AlternativeInterpretation] = field(default_factory=list)
    counterfactuals: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'root_cause': self.root_cause,
            'root_quote': self.root_quote,
            'steps': [s.to_dict() for s in self.steps],
            'business_impact': self.business_impact,
            'impact_confidence': self.impact_confidence,
            'alternatives': [a.to_dict() for a in self.alternatives],
            'counterfactuals': self.counterfactuals
        }

    @property
    def evidence_coverage(self) -> float:
        """Percentage of steps with strong evidence (>0.80)"""
        if not self.steps:
            return 0.0
        strong_evidence = sum(1 for s in self.steps if s.evidence_strength >= 0.80)
        return strong_evidence / len(self.steps)

@dataclass
class CausalDAG:
    """Directed Acyclic Graph of causal relationships"""
    nodes: List[Dict] = field(default_factory=list)
    edges: List[Dict] = field(default_factory=list)

    def add_node(self, node_id: str, label: str, node_type: str):
        """Add node to DAG"""
        self.nodes.append({
            'id': node_id,
            'label': label,
            'type': node_type
        })

    def add_edge(self, from_id: str, to_id: str, strength: float):
        """Add edge between nodes"""
        self.edges.append({
            'from': from_id,
            'to': to_id,
            'strength': strength
        })

    def to_dict(self) -> dict:
        return {
            'nodes': self.nodes,
            'edges': self.edges
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_causal_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/causal/models.py tests/test_sota/test_causal_models.py
git commit -m "feat(sota): add causal chain data models

- Add CausalStep for reasoning steps
- Add CausalChain with evidence coverage calculation
- Add AlternativeInterpretation for uncertainty
- Add CausalDAG for relationship graphs
- Add tests for model creation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 1

**Q1: DAG Construction**
- **Gap**: DAG structure defined but not implemented
- **Question**: When/how to build DAG? What triggers multi-path analysis?
- **Priority**: Low | **Blocker**: No - optional enhancement

---

## Task 2: Chain-of-Thought Prompts

**Files:**
- Create: `nrg_core/sota/causal/prompts.py`
- Test: `tests/test_sota/test_causal_prompts.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_causal_prompts.py
import pytest
from nrg_core.sota.causal.prompts import CausalPrompts

def test_root_cause_prompt():
    prompts = CausalPrompts()
    prompt = prompts.get_root_cause_prompt("Tax applies to >50MW")

    assert "Tax applies to >50MW" in prompt
    assert "exact section" in prompt.lower()

def test_chain_of_thought_prompt():
    prompts = CausalPrompts()
    prompt = prompts.get_cot_prompt(
        finding="Tax applies to >50MW",
        bill_text="Sample bill text"
    )

    assert "5 questions" in prompt or "five questions" in prompt.lower()
    assert "quote" in prompt.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_causal_prompts.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/causal/prompts.py

class CausalPrompts:
    """Prompt templates for causal chain reasoning"""

    @staticmethod
    def get_root_cause_prompt(finding: str) -> str:
        return f"""For this finding: "{finding}"

Identify the ROOT CAUSE - the primary bill section or amendment that causes this finding.

Provide:
1. Exact section number and reference
2. The exact quoted text from the bill

Format as JSON:
{{
  "root_cause": "Section X.Y: Brief description",
  "root_quote": "Exact text from bill",
  "section_reference": "Section X.Y"
}}"""

    @staticmethod
    def get_cot_prompt(finding: str, bill_text: str) -> str:
        return f"""Analyze this finding using chain-of-thought reasoning: "{finding}"

Answer these 5 questions with bill text quotes:

**Step 1: How does this amend existing law?**
- What did prior law say?
- What is the new provision?
- Quote supporting this

**Step 2: Who is affected by this change?**
- What entities/people qualify?
- What is the scope?
- Quote the definition

**Step 3: What conditions or thresholds apply?**
- Are there specific values, dates, or criteria?
- Quote the exact threshold

**Step 4: Are there exemptions or special cases?**
- Who is exempt?
- What are the carve-outs?
- Quote the exemption clause

**Step 5: What is the business impact?**
- What must companies do differently?
- What are the costs/requirements?
- Quote compliance requirements

For each step, provide:
- Your answer
- Exact quote from bill supporting your answer
- Evidence strength (0.0 to 1.0)

Return as JSON:
{{
  "steps": [
    {{
      "step": 1,
      "question": "How does this amend existing law?",
      "answer": "Your analysis",
      "quote": "Exact bill text",
      "strength": 0.95
    }},
    ... (5 steps total)
  ]
}}

Bill Text:
{bill_text}"""

    @staticmethod
    def get_alternative_interpretations_prompt(finding: str, bill_text: str) -> str:
        return f"""For this finding: "{finding}"

What are alternative ways to interpret this provision?

For each alternative interpretation:
1. Describe the interpretation
2. Assess likelihood (0.0 to 1.0)
3. Describe impact if this interpretation is true

Return as JSON:
{{
  "alternatives": [
    {{
      "interpretation": "Narrow interpretation: X",
      "likelihood": 0.15,
      "impact": "If true, then Y"
    }},
    {{
      "interpretation": "Standard interpretation: X",
      "likelihood": 0.65,
      "impact": "If true, then Y"
    }},
    {{
      "interpretation": "Broad interpretation: X",
      "likelihood": 0.20,
      "impact": "If true, then Y"
    }}
  ]
}}

Bill Text:
{bill_text}"""

    @staticmethod
    def get_counterfactual_prompt(finding: str, parameter: str, variation: str) -> str:
        return f"""For this finding: "{finding}"

Analyze the counterfactual scenario:

**If parameter "{parameter}" were "{variation}" instead of current value:**

1. Would the finding still hold?
2. How would the impact change?
3. What would be the new implication?

Return as JSON:
{{
  "parameter": "{parameter}",
  "current_value": "X",
  "counterfactual_value": "{variation}",
  "finding_still_holds": true/false,
  "impact_change": "Description of how impact changes"
}}"""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_causal_prompts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/causal/prompts.py tests/test_sota/test_causal_prompts.py
git commit -m "feat(sota): add chain-of-thought prompts

- Add 5-step chain-of-thought prompt
- Add root cause identification prompt
- Add alternative interpretations prompt
- Add counterfactual analysis prompt
- Add tests for prompt generation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 2

**Q2: Counterfactual Parameters**
- **Gap**: Which parameters to vary for counterfactuals?
- **Question**: How to automatically identify key parameters (thresholds, dates, exemptions)?
- **Priority**: Low | **Blocker**: No - manual specification acceptable for MVP

---

## Task 3: Causal Reasoner Engine

**Files:**
- Create: `nrg_core/sota/causal/reasoner.py`
- Test: `tests/test_sota/test_causal_reasoner.py`

**Step 1: Write the failing test**

```python
# tests/test_sota/test_causal_reasoner.py
import pytest
from unittest.mock import AsyncMock, patch
from nrg_core.sota.causal.reasoner import CausalReasoner
from nrg_core.sota.models import Finding

@pytest.mark.asyncio
async def test_build_causal_chain():
    finding = Finding(
        statement="Tax applies to >50MW",
        confidence=0.9,
        supporting_quotes=["Section 2.1"],
        found_by=["GPT-4o", "Claude"],
        consensus_level="majority"
    )

    bill_text = """
    Section 2.1: Tax on energy generation exceeding fifty megawatts.
    Section 3.2: Renewable energy facilities exempt.
    """

    with patch('nrg_core.sota.llm_clients.OpenAIClient') as MockClient:
        mock_client = MockClient.return_value
        mock_client.analyze_bill = AsyncMock(return_value=Mock(
            findings=[],
            error=None
        ))

        reasoner = CausalReasoner(openai_key="test")
        chain = await reasoner.build_chain(finding, bill_text)

        assert chain.root_cause is not None
        assert len(chain.steps) >= 1
        assert chain.impact_confidence > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sota/test_causal_reasoner.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# nrg_core/sota/causal/reasoner.py
import json
import os
from typing import Optional, List
from nrg_core.sota.models import Finding
from nrg_core.sota.causal.models import CausalChain, CausalStep, AlternativeInterpretation
from nrg_core.sota.causal.prompts import CausalPrompts
from nrg_core.sota.llm_clients import OpenAIClient

class CausalReasoner:
    """Builds causal reasoning chains"""

    def __init__(self, openai_key: Optional[str] = None):
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAIClient(self.openai_key)
        self.prompts = CausalPrompts()

    async def build_chain(
        self,
        finding: Finding,
        bill_text: str
    ) -> CausalChain:
        """
        Build complete causal chain for a finding

        Args:
            finding: Finding to analyze
            bill_text: Full bill text for quote verification

        Returns:
            CausalChain with root cause, reasoning steps, alternatives
        """
        # Step 1: Identify root cause
        root_data = await self._identify_root_cause(finding, bill_text)

        # Step 2: Build chain-of-thought reasoning
        cot_steps = await self._build_cot_steps(finding, bill_text)

        # Step 3: Explore alternative interpretations
        alternatives = await self._explore_alternatives(finding, bill_text)

        # Step 4: Calculate impact confidence
        impact_confidence = self._calculate_impact_confidence(cot_steps)

        # Step 5: Generate business impact summary
        business_impact = self._generate_business_impact(cot_steps)

        return CausalChain(
            root_cause=root_data.get('root_cause', finding.statement),
            root_quote=root_data.get('root_quote', ''),
            steps=cot_steps,
            business_impact=business_impact,
            impact_confidence=impact_confidence,
            alternatives=alternatives
        )

    async def _identify_root_cause(self, finding: Finding, bill_text: str) -> dict:
        """Identify root cause using LLM"""
        prompt = self.prompts.get_root_cause_prompt(finding.statement)

        response = await self.client.analyze_bill(bill_text, prompt)

        if response.error or not response.findings:
            # Fallback to finding quotes
            return {
                'root_cause': finding.statement,
                'root_quote': finding.supporting_quotes[0] if finding.supporting_quotes else ''
            }

        # Parse JSON response
        try:
            return response.findings[0]
        except (IndexError, KeyError):
            return {
                'root_cause': finding.statement,
                'root_quote': ''
            }

    async def _build_cot_steps(
        self,
        finding: Finding,
        bill_text: str
    ) -> List[CausalStep]:
        """Build 5-step chain-of-thought reasoning"""
        prompt = self.prompts.get_cot_prompt(finding.statement, bill_text)

        response = await self.client.analyze_bill(bill_text, prompt)

        if response.error or not response.findings:
            return []

        # Parse steps from response
        try:
            steps_data = response.findings[0].get('steps', [])
            return [
                CausalStep(
                    step_number=s['step'],
                    question=s['question'],
                    answer=s['answer'],
                    supporting_quote=s.get('quote', ''),
                    evidence_strength=s.get('strength', 0.5)
                )
                for s in steps_data
            ]
        except (KeyError, TypeError):
            return []

    async def _explore_alternatives(
        self,
        finding: Finding,
        bill_text: str
    ) -> List[AlternativeInterpretation]:
        """Explore alternative interpretations"""
        prompt = self.prompts.get_alternative_interpretations_prompt(
            finding.statement,
            bill_text
        )

        response = await self.client.analyze_bill(bill_text, prompt)

        if response.error or not response.findings:
            return []

        # Parse alternatives
        try:
            alts_data = response.findings[0].get('alternatives', [])
            return [
                AlternativeInterpretation(
                    interpretation=a['interpretation'],
                    likelihood=a['likelihood'],
                    impact_if_true=a['impact']
                )
                for a in alts_data
            ]
        except (KeyError, TypeError):
            return []

    def _calculate_impact_confidence(self, steps: List[CausalStep]) -> float:
        """Calculate overall impact confidence from steps"""
        if not steps:
            return 0.5

        # Average evidence strength across steps
        avg_strength = sum(s.evidence_strength for s in steps) / len(steps)

        # Penalize if coverage is low
        coverage = sum(1 for s in steps if s.evidence_strength >= 0.80) / len(steps)

        return (avg_strength * 0.7) + (coverage * 0.3)

    def _generate_business_impact(self, steps: List[CausalStep]) -> str:
        """Generate business impact summary from final step"""
        if not steps:
            return "Unknown impact"

        # Use last step (business impact step)
        final_step = steps[-1]
        return final_step.answer
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sota/test_causal_reasoner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/sota/causal/reasoner.py tests/test_sota/test_causal_reasoner.py
git commit -m "feat(sota): add causal reasoning engine

- Add CausalReasoner for building reasoning chains
- Implement root cause identification
- Implement 5-step chain-of-thought reasoning
- Implement alternative interpretation exploration
- Add impact confidence calculation
- Add tests for causal chain building

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Implementation Questions for Task 3

**Q3: LLM Model Selection for Reasoning** ⚠️ BLOCKING
- **Gap**: Which model to use for causal reasoning?
- **Question**: Use GPT-4o for all steps? Or ensemble across models? Cost vs accuracy tradeoff?
- **Priority**: High | **Blocker**: YES - affects cost and accuracy
- **Recommended Answer**: Use GPT-4o for chain-of-thought (best at structured reasoning). Reserve ensemble for final verification only.

**Q4: Quote Verification Strictness**
- **Gap**: How strict should quote matching be?
- **Question**: Exact substring? Fuzzy match? What edit distance threshold?
- **Priority**: Medium | **Blocker**: No - substring matching acceptable for MVP

**Q5: Alternative Likelihood Calibration**
- **Gap**: How to calibrate likelihood scores (0.05, 0.65, 0.30)?
- **Question**: Based on LLM confidence? Historical accuracy? Expert judgment?
- **Priority**: Medium | **Blocker**: No - use LLM-provided likelihoods initially

**Q6: Step Failure Handling** ⚠️ BLOCKING
- **Gap**: What if step 3 fails but others succeed?
- **Question**: Return partial chain? Retry failed step? Fail entire analysis?
- **Priority**: High | **Blocker**: YES
- **Recommended Answer**: Return partial chain with failed steps marked. Lower causal strength score proportionally.

**Q7: Business Impact Quantification**
- **Gap**: Qualitative impact ("$500K-$2M") not structured
- **Question**: Should we extract structured impact (min, max, currency)?
- **Priority**: Medium | **Blocker**: No - nice to have for reporting

**Q8: Chain Caching**
- **Gap**: Expensive to rebuild chains for same findings
- **Question**: Cache chains by finding statement? For how long?
- **Priority**: Low | **Blocker**: No - optimization for later

---

## Task 4: Package Initialization and Documentation

**Files:**
- Create: `nrg_core/sota/causal/__init__.py`
- Create: `nrg_core/sota/causal/README.md`

**Step 1: Create package initialization**

```python
# nrg_core/sota/causal/__init__.py
"""
Causal Chain Reasoning Component

Builds explicit multi-step reasoning chains from bill amendments to
business impacts with quote verification.
"""

from nrg_core.sota.causal.reasoner import CausalReasoner
from nrg_core.sota.causal.models import (
    CausalStep,
    CausalChain,
    AlternativeInterpretation,
    CausalDAG
)
from nrg_core.sota.causal.prompts import CausalPrompts

__all__ = [
    'CausalReasoner',
    'CausalStep',
    'CausalChain',
    'AlternativeInterpretation',
    'CausalDAG',
    'CausalPrompts'
]
```

**Step 2: Create README**

```markdown
# Causal Chain Reasoning Component

## Overview

Builds 5-step causal chains from bill amendments to business impacts, with quote verification at each step.

## 5-Step Chain-of-Thought Process

1. **How does this amend existing law?**
2. **Who is affected?**
3. **What thresholds apply?**
4. **Are there exemptions?**
5. **What is the business impact?**

## Usage

```python
from nrg_core.sota.causal import CausalReasoner

reasoner = CausalReasoner()
chain = await reasoner.build_chain(finding, bill_text)

print(f"Root: {chain.root_cause}")
print(f"Coverage: {chain.evidence_coverage:.0%}")

for step in chain.steps:
    print(f"Step {step.step_number}: {step.question}")
    print(f"  Answer: {step.answer}")
    print(f"  Quote: {step.supporting_quote}")
    print(f"  Strength: {step.evidence_strength:.2f}")
```

## Evidence Strength

- **0.95+**: Direct quote, unambiguous
- **0.85-0.94**: Clear inference from text
- **0.70-0.84**: Reasonable interpretation
- **<0.70**: Speculative or weak evidence
```

**Step 3: Commit**

```bash
git add nrg_core/sota/causal/__init__.py nrg_core/sota/causal/README.md
git commit -m "docs(sota): add causal reasoning documentation

- Add package initialization
- Add README with 5-step process
- Add evidence strength reference
- Add usage examples

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary of Implementation Questions

All gaps and clarification questions have been distributed into the relevant tasks above. Look for sections marked "Implementation Questions for Task X" after each task's commit step.

**Blocking Questions** (marked with ⚠️):
- Task 3, Q3: LLM Model Selection for Reasoning
- Task 3, Q6: Step Failure Handling

**See** [GAPS_AND_QUESTIONS.md](./GAPS_AND_QUESTIONS.md) for complete cross-component questions and integration issues.

---

# Phase 3: Evaluation & Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build evaluation pipeline without golden dataset using silver set + LLM-judge ensemble + spot-checking

**Architecture:** 50-bill silver set for regression testing, multi-judge ensemble for continuous validation, stratified spot-checking for error estimation, threshold calibration

**Tech Stack:** Python 3.12, OpenAI GPT-4o, Claude Opus, Gemini 3 Pro (for judge ensemble), pytest, pandas for analysis

**Prerequisites:** Phase 2 complete (enhanced analysis system)

---

## Task 1: Create Silver Set Infrastructure

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/evaluation/silver_set.py`
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/evaluation/__init__.py`
- Create: `/Users/thamac/Documents/NRG/data/silver_set/`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_evaluation/test_silver_set.py`

**Step 1: Write failing test for silver set loader**

```python
# tests/test_v2/test_evaluation/__init__.py
# Empty init

# tests/test_v2/test_evaluation/test_silver_set.py
import pytest
from pathlib import Path
from nrg_core.v2.evaluation.silver_set import SilverSet, SilverBill


def test_silver_set_loads_bills():
    """Should load silver set bills with expert labels."""
    silver_set = SilverSet(data_dir="/tmp/test_silver_set")
    
    # Create test bill
    test_bill = {
        "bill_id": "HB123",
        "bill_text": "Section 2.1: Tax of $50/MW",
        "expert_labels": {
            "findings": [
                {"statement": "Tax applies at $50/MW", "impact": 7}
            ],
            "rubric_scores": {
                "legal_risk": 7,
                "financial_impact": 7
            }
        }
    }
    
    import json
    import os
    os.makedirs("/tmp/test_silver_set", exist_ok=True)
    with open("/tmp/test_silver_set/HB123.json", "w") as f:
        json.dump(test_bill, f)
    
    bills = silver_set.load()
    
    assert len(bills) >= 1
    assert bills[0].bill_id == "HB123"
    assert len(bills[0].expert_labels["findings"]) == 1
    
    # Cleanup
    os.remove("/tmp/test_silver_set/HB123.json")
    os.rmdir("/tmp/test_silver_set")


def test_silver_set_validation():
    """Silver bills must have expert labels."""
    with pytest.raises(ValueError, match="expert_labels"):
        SilverBill(
            bill_id="HB456",
            bill_text="...",
            expert_labels=None  # Missing labels
        )
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_v2/test_evaluation/test_silver_set.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement silver set**

```python
# nrg_core/v2/evaluation/__init__.py
"""Evaluation and calibration infrastructure."""
from nrg_core.v2.evaluation.silver_set import SilverSet, SilverBill

__all__ = ["SilverSet", "SilverBill"]
```

```python
# nrg_core/v2/evaluation/silver_set.py
"""
Silver Set Management
50-100 expert-labeled bills for regression testing
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SilverBill:
    """Single bill in silver set with expert labels."""
    bill_id: str
    bill_text: str
    expert_labels: Dict[str, Any]
    nrg_context: Optional[str] = None
    
    def __post_init__(self):
        if self.expert_labels is None:
            raise ValueError("Silver bills must have expert_labels")


class SilverSet:
    """
    Manage silver set of expert-labeled bills.
    
    Silver set structure:
    data/silver_set/
      HB123.json
      HB456.json
      ...
    
    Each JSON:
    {
      "bill_id": "HB123",
      "bill_text": "...",
      "expert_labels": {
        "findings": [...],
        "rubric_scores": {...}
      },
      "nrg_context": "..."
    }
    """
    
    def __init__(self, data_dir: str = "data/silver_set"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> List[SilverBill]:
        """Load all bills in silver set."""
        bills = []
        
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            bill = SilverBill(
                bill_id=data["bill_id"],
                bill_text=data["bill_text"],
                expert_labels=data["expert_labels"],
                nrg_context=data.get("nrg_context")
            )
            bills.append(bill)
        
        return bills
    
    def add_bill(self, bill: SilverBill):
        """Add bill to silver set."""
        file_path = self.data_dir / f"{bill.bill_id}.json"
        
        data = {
            "bill_id": bill.bill_id,
            "bill_text": bill.bill_text,
            "expert_labels": bill.expert_labels,
            "nrg_context": bill.nrg_context
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_bill(self, bill_id: str) -> Optional[SilverBill]:
        """Get specific bill by ID."""
        file_path = self.data_dir / f"{bill_id}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return SilverBill(
            bill_id=data["bill_id"],
            bill_text=data["bill_text"],
            expert_labels=data["expert_labels"],
            nrg_context=data.get("nrg_context")
        )
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_v2/test_evaluation/test_silver_set.py -v`

Expected: All tests PASS

**Step 5: Create helper script to add bills to silver set**

```python
# scripts/add_to_silver_set.py
"""Helper script to add expert-labeled bills to silver set."""
import argparse
import json
from nrg_core.v2.evaluation.silver_set import SilverSet, SilverBill


def main():
    parser = argparse.ArgumentParser(description="Add bill to silver set")
    parser.add_argument("--bill-id", required=True)
    parser.add_argument("--bill-text-file", required=True)
    parser.add_argument("--labels-file", required=True, help="JSON file with expert labels")
    
    args = parser.parse_args()
    
    # Load bill text
    with open(args.bill_text_file, 'r') as f:
        bill_text = f.read()
    
    # Load expert labels
    with open(args.labels_file, 'r') as f:
        expert_labels = json.load(f)
    
    # Create silver bill
    bill = SilverBill(
        bill_id=args.bill_id,
        bill_text=bill_text,
        expert_labels=expert_labels
    )
    
    # Add to silver set
    silver_set = SilverSet()
    silver_set.add_bill(bill)
    
    print(f"✓ Added {args.bill_id} to silver set")


if __name__ == "__main__":
    main()
```

**Step 6: Commit**

```bash
git add nrg_core/v2/evaluation/ tests/test_v2/test_evaluation/ scripts/add_to_silver_set.py
git commit -m "feat(v2): add silver set infrastructure for evaluation"
```

---

## Task 2: Implement Precision/Recall Evaluator

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/evaluation/metrics.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_evaluation/test_metrics.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_evaluation/test_metrics.py
import pytest
from nrg_core.v2.evaluation.metrics import PrecisionRecallEvaluator


def test_precision_recall_calculation():
    """Should compute precision, recall, F1 from predictions vs labels."""
    evaluator = PrecisionRecallEvaluator()
    
    # Predicted findings
    predicted = [
        {"statement": "Tax of $50/MW", "impact": 7},
        {"statement": "Renewable exempt", "impact": 3},
        {"statement": "False positive finding", "impact": 2}
    ]
    
    # Expert labels
    expert = [
        {"statement": "Tax applies at $50/MW", "impact": 7},
        {"statement": "Renewable exemption", "impact": 3}
    ]
    
    metrics = evaluator.compute(predicted=predicted, expert=expert)
    
    # 2 true positives (matched), 1 false positive (unmatched pred), 0 false negatives
    assert metrics["precision"] == 2/3  # 2 TP / (2 TP + 1 FP)
    assert metrics["recall"] == 1.0      # 2 TP / (2 TP + 0 FN)
    assert 0.7 < metrics["f1"] < 0.9


def test_rubric_score_mae():
    """Should compute mean absolute error for rubric scores."""
    evaluator = PrecisionRecallEvaluator()
    
    predicted_scores = {"legal_risk": 7, "financial_impact": 6}
    expert_scores = {"legal_risk": 8, "financial_impact": 7}
    
    mae = evaluator.rubric_mae(predicted_scores, expert_scores)
    
    # MAE = (|7-8| + |6-7|) / 2 = 1.0
    assert mae == 1.0
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_evaluation/test_metrics.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement evaluator**

```python
# nrg_core/v2/evaluation/metrics.py
"""
Precision/Recall/F1 Metrics for Silver Set Evaluation
"""
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class PrecisionRecallEvaluator:
    """
    Evaluate system predictions against expert labels.
    
    Uses semantic similarity (cosine) to match predicted findings to expert labels.
    Similarity threshold: 0.75 for a match.
    """
    
    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
    
    def compute(
        self,
        predicted: List[Dict[str, Any]],
        expert: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1.
        
        Args:
            predicted: List of predicted findings
            expert: List of expert-labeled findings
        
        Returns:
            Dict with precision, recall, f1, true_positives, false_positives, false_negatives
        """
        # Match findings using semantic similarity
        matches = self._match_findings(predicted, expert)
        
        true_positives = len(matches)
        false_positives = len(predicted) - true_positives
        false_negatives = len(expert) - true_positives
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def rubric_mae(
        self,
        predicted_scores: Dict[str, int],
        expert_scores: Dict[str, int]
    ) -> float:
        """
        Compute mean absolute error for rubric scores.
        
        Args:
            predicted_scores: Dict of dimension -> score
            expert_scores: Dict of dimension -> score
        
        Returns:
            Mean absolute error across dimensions
        """
        errors = []
        for dimension in expert_scores.keys():
            if dimension in predicted_scores:
                error = abs(predicted_scores[dimension] - expert_scores[dimension])
                errors.append(error)
        
        return np.mean(errors) if errors else 0.0
    
    def _match_findings(
        self,
        predicted: List[Dict],
        expert: List[Dict]
    ) -> List[tuple]:
        """
        Match predicted findings to expert findings using semantic similarity.
        
        Returns:
            List of (predicted_idx, expert_idx) tuples for matches
        """
        if not predicted or not expert:
            return []
        
        # Extract statements
        pred_statements = [f["statement"] for f in predicted]
        expert_statements = [f["statement"] for f in expert]
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        all_statements = pred_statements + expert_statements
        vectors = vectorizer.fit_transform(all_statements)
        
        pred_vectors = vectors[:len(predicted)]
        expert_vectors = vectors[len(predicted):]
        
        # Compute similarity matrix
        similarities = cosine_similarity(pred_vectors, expert_vectors)
        
        # Greedy matching: for each predicted, find best expert match
        matches = []
        matched_experts = set()
        
        for pred_idx in range(len(predicted)):
            best_expert_idx = None
            best_sim = self.similarity_threshold
            
            for expert_idx in range(len(expert)):
                if expert_idx in matched_experts:
                    continue
                
                sim = similarities[pred_idx, expert_idx]
                if sim > best_sim:
                    best_sim = sim
                    best_expert_idx = expert_idx
            
            if best_expert_idx is not None:
                matches.append((pred_idx, best_expert_idx))
                matched_experts.add(best_expert_idx)
        
        return matches
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_evaluation/test_metrics.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add nrg_core/v2/evaluation/metrics.py tests/test_v2/test_evaluation/test_metrics.py
git commit -m "feat(v2): add precision/recall evaluator for silver set"
```

---

## Task 3: Create LLM-Judge Ensemble

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/evaluation/llm_judge_ensemble.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_evaluation/test_llm_judge.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_evaluation/test_llm_judge.py
import pytest
from unittest.mock import patch
from nrg_core.v2.evaluation.llm_judge_ensemble import LLMJudgeEnsemble


def test_ensemble_uses_multiple_judges():
    """Should query multiple judge models (GPT, Claude, Gemini)."""
    ensemble = LLMJudgeEnsemble(
        gpt_key="test-gpt",
        claude_key="test-claude",
        gemini_key="test-gemini"
    )
    
    finding = {"statement": "Tax of $50/MW", "impact": 7}
    expert_label = {"statement": "Tax applies at $50/MW", "impact": 7}
    
    # Mock responses from each judge
    mock_gpt = {"score": 0.9, "rationale": "Correct"}
    mock_claude = {"score": 0.85, "rationale": "Mostly correct"}
    mock_gemini = {"score": 0.95, "rationale": "Accurate"}
    
    with patch.object(ensemble, '_call_gpt_judge', return_value=mock_gpt), \
         patch.object(ensemble, '_call_claude_judge', return_value=mock_claude), \
         patch.object(ensemble, '_call_gemini_judge', return_value=mock_gemini):
        
        result = ensemble.evaluate_finding(finding, expert_label)
    
    assert len(result.judge_scores) == 3
    assert result.average_score == 0.9  # (0.9 + 0.85 + 0.95) / 3
    assert result.agreement >= 0.8  # High inter-judge agreement


def test_ensemble_flags_low_agreement():
    """Should flag low agreement when judges disagree."""
    ensemble = LLMJudgeEnsemble(gpt_key="test", claude_key="test", gemini_key="test")
    
    # Mock divergent responses
    mock_gpt = {"score": 0.9, "rationale": "Good"}
    mock_claude = {"score": 0.3, "rationale": "Poor"}
    mock_gemini = {"score": 0.5, "rationale": "Mediocre"}
    
    with patch.object(ensemble, '_call_gpt_judge', return_value=mock_gpt), \
         patch.object(ensemble, '_call_claude_judge', return_value=mock_claude), \
         patch.object(ensemble, '_call_gemini_judge', return_value=mock_gemini):
        
        result = ensemble.evaluate_finding({"statement": "X", "impact": 5}, {"statement": "Y", "impact": 8})
    
    assert result.agreement < 0.75
    assert result.requires_human_review is True
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_evaluation/test_llm_judge.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement LLM-judge ensemble**

```python
# nrg_core/v2/evaluation/llm_judge_ensemble.py
"""
LLM-Judge Ensemble
Multiple judge models for continuous validation
"""
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
import anthropic
from google import genai


@dataclass
class JudgeEvaluationResult:
    """Result from judge ensemble."""
    judge_scores: List[Dict[str, Any]]
    average_score: float
    agreement: float  # Inter-judge agreement (std dev based)
    requires_human_review: bool


JUDGE_PROMPT = """You are evaluating a legislative analysis finding against an expert label.

PREDICTED FINDING:
{predicted}

EXPERT LABEL:
{expert}

TASK: Score how well the predicted finding matches the expert label (0-1).
- 1.0 = Perfect match (same meaning, same impact)
- 0.8-0.9 = Close match (minor differences)
- 0.5-0.7 = Partial match (same topic, different interpretation)
- 0.0-0.4 = Poor match (different findings)

OUTPUT (JSON):
{{
  "score": 0.0-1.0,
  "rationale": "Explanation of score"
}}
"""


class LLMJudgeEnsemble:
    """
    Ensemble of judge models for validation.
    Uses GPT-4o, Claude Opus, Gemini 3 Pro.
    """
    
    def __init__(self, gpt_key: str, claude_key: str, gemini_key: str):
        self.gpt_client = OpenAI(api_key=gpt_key)
        self.claude_client = anthropic.Anthropic(api_key=claude_key)
        self.gemini_client = genai.Client(api_key=gemini_key)
    
    def evaluate_finding(
        self,
        predicted: Dict[str, Any],
        expert_label: Dict[str, Any]
    ) -> JudgeEvaluationResult:
        """
        Evaluate predicted finding against expert label using ensemble.
        
        Returns:
            JudgeEvaluationResult with scores and agreement
        """
        # Query all judges
        gpt_result = self._call_gpt_judge(predicted, expert_label)
        claude_result = self._call_claude_judge(predicted, expert_label)
        gemini_result = self._call_gemini_judge(predicted, expert_label)
        
        judge_scores = [
            {"model": "gpt-4o", **gpt_result},
            {"model": "claude-opus-4", **claude_result},
            {"model": "gemini-3-pro", **gemini_result}
        ]
        
        # Compute average and agreement
        scores = [gpt_result["score"], claude_result["score"], gemini_result["score"]]
        average_score = np.mean(scores)
        
        # Agreement = 1 - normalized std dev
        std_dev = np.std(scores)
        agreement = 1.0 - std_dev  # High std dev = low agreement
        
        # Flag for human review if low agreement
        requires_review = agreement < 0.75
        
        return JudgeEvaluationResult(
            judge_scores=judge_scores,
            average_score=average_score,
            agreement=agreement,
            requires_human_review=requires_review
        )
    
    def _call_gpt_judge(self, predicted: Dict, expert: Dict) -> Dict[str, Any]:
        """Call GPT-4o judge."""
        prompt = JUDGE_PROMPT.format(
            predicted=str(predicted),
            expert=str(expert)
        )
        
        response = self.gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    
    def _call_claude_judge(self, predicted: Dict, expert: Dict) -> Dict[str, Any]:
        """Call Claude Opus judge."""
        prompt = JUDGE_PROMPT.format(
            predicted=str(predicted),
            expert=str(expert)
        )
        
        message = self.claude_client.messages.create(
            model="claude-opus-4",
            max_tokens=1024,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        return json.loads(message.content[0].text)
    
    def _call_gemini_judge(self, predicted: Dict, expert: Dict) -> Dict[str, Any]:
        """Call Gemini 3 Pro judge."""
        prompt = JUDGE_PROMPT.format(
            predicted=str(predicted),
            expert=str(expert)
        )
        
        response = self.gemini_client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        )
        
        import json
        return json.loads(response.text)
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_evaluation/test_llm_judge.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add nrg_core/v2/evaluation/llm_judge_ensemble.py tests/test_v2/test_evaluation/test_llm_judge.py
git commit -m "feat(v2): add LLM-judge ensemble for continuous validation"
```

---

## Task 4: Implement Spot-Checking Pipeline

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/evaluation/spot_checking.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_evaluation/test_spot_checking.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_evaluation/test_spot_checking.py
import pytest
from nrg_core.v2.evaluation.spot_checking import SpotChecker, ConfidenceBand


def test_stratified_sampling():
    """Should sample bills stratified by confidence and impact."""
    checker = SpotChecker(sample_size=20)
    
    # Mock bills with different confidence/impact
    bills = [
        {"bill_id": f"HB{i}", "confidence": 0.9, "impact": 8} for i in range(10)
    ] + [
        {"bill_id": f"HB{i+10}", "confidence": 0.6, "impact": 3} for i in range(10)
    ] + [
        {"bill_id": f"HB{i+20}", "confidence": 0.4, "impact": 9} for i in range(10)
    ]
    
    sample = checker.stratified_sample(bills)
    
    assert len(sample) == 20
    
    # Should have bills from each stratum
    high_conf_high_impact = [b for b in sample if b["confidence"] >= 0.8 and b["impact"] >= 7]
    low_conf_high_impact = [b for b in sample if b["confidence"] < 0.7 and b["impact"] >= 7]
    
    assert len(high_conf_high_impact) > 0
    assert len(low_conf_high_impact) > 0  # Oversample risky stratum
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_evaluation/test_spot_checking.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement spot-checker**

```python
# nrg_core/v2/evaluation/spot_checking.py
"""
Spot-Checking Pipeline
Stratified sampling by confidence band + impact for error estimation
"""
from typing import List, Dict, Any
from enum import Enum
import random


class ConfidenceBand(str, Enum):
    """Confidence bands for stratification."""
    HIGH = "high"      # >= 0.8
    MEDIUM = "medium"  # 0.6-0.8
    LOW = "low"        # < 0.6


class SpotChecker:
    """
    Stratified spot-checking for per-release validation.
    
    Sampling strategy:
    - High confidence + low impact: 10% of sample
    - High confidence + high impact: 30% of sample
    - Low confidence + high impact: 50% of sample (risky!)
    - Other: 10% of sample
    """
    
    def __init__(self, sample_size: int = 20):
        self.sample_size = sample_size
    
    def stratified_sample(self, bills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sample bills using stratified sampling.
        
        Args:
            bills: List of bills with confidence and impact scores
        
        Returns:
            Stratified sample of size sample_size
        """
        # Stratify bills
        strata = {
            "high_conf_low_impact": [],
            "high_conf_high_impact": [],
            "low_conf_high_impact": [],
            "other": []
        }
        
        for bill in bills:
            conf = bill.get("confidence", 0.5)
            impact = bill.get("impact", 5)
            
            if conf >= 0.8 and impact < 7:
                strata["high_conf_low_impact"].append(bill)
            elif conf >= 0.8 and impact >= 7:
                strata["high_conf_high_impact"].append(bill)
            elif conf < 0.7 and impact >= 7:
                strata["low_conf_high_impact"].append(bill)
            else:
                strata["other"].append(bill)
        
        # Sample from each stratum (with weights)
        sample = []
        
        # High conf + low impact: 10%
        n = int(self.sample_size * 0.1)
        sample.extend(random.sample(strata["high_conf_low_impact"], min(n, len(strata["high_conf_low_impact"]))))
        
        # High conf + high impact: 30%
        n = int(self.sample_size * 0.3)
        sample.extend(random.sample(strata["high_conf_high_impact"], min(n, len(strata["high_conf_high_impact"]))))
        
        # Low conf + high impact: 50% (OVERSAMPLE risky cases)
        n = int(self.sample_size * 0.5)
        sample.extend(random.sample(strata["low_conf_high_impact"], min(n, len(strata["low_conf_high_impact"]))))
        
        # Other: 10%
        n = int(self.sample_size * 0.1)
        sample.extend(random.sample(strata["other"], min(n, len(strata["other"]))))
        
        # If undersample, fill from largest stratum
        while len(sample) < self.sample_size:
            largest_stratum = max(strata.values(), key=len)
            if largest_stratum:
                sample.append(random.choice(largest_stratum))
        
        return sample[:self.sample_size]
    
    def estimate_error_rate(
        self,
        sample: List[Dict[str, Any]],
        human_reviews: List[bool]
    ) -> Dict[str, float]:
        """
        Estimate error rate from spot-check sample.
        
        Args:
            sample: Stratified sample
            human_reviews: True if correct, False if error
        
        Returns:
            Dict with overall_error_rate and per_stratum error rates
        """
        if len(sample) != len(human_reviews):
            raise ValueError("Sample and reviews must have same length")
        
        # Overall error rate
        errors = sum(1 for correct in human_reviews if not correct)
        overall_error_rate = errors / len(human_reviews)
        
        # Per-stratum error rates
        strata_errors = {}
        for i, bill in enumerate(sample):
            conf = bill.get("confidence", 0.5)
            impact = bill.get("impact", 5)
            
            if conf >= 0.8 and impact >= 7:
                stratum = "high_conf_high_impact"
            elif conf < 0.7 and impact >= 7:
                stratum = "low_conf_high_impact"
            else:
                stratum = "other"
            
            if stratum not in strata_errors:
                strata_errors[stratum] = {"errors": 0, "total": 0}
            
            strata_errors[stratum]["total"] += 1
            if not human_reviews[i]:
                strata_errors[stratum]["errors"] += 1
        
        # Compute rates
        per_stratum = {}
        for stratum, counts in strata_errors.items():
            per_stratum[stratum] = counts["errors"] / counts["total"] if counts["total"] > 0 else 0.0
        
        return {
            "overall_error_rate": overall_error_rate,
            "per_stratum_error_rates": per_stratum
        }
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_evaluation/test_spot_checking.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add nrg_core/v2/evaluation/spot_checking.py tests/test_v2/test_evaluation/test_spot_checking.py
git commit -m "feat(v2): add stratified spot-checking pipeline"
```

---

## Task 5: Create Threshold Calibration Tool

**Files:**
- Create: `/Users/thamac/Documents/NRG/nrg_core/v2/evaluation/threshold_calibration.py`
- Test: `/Users/thamac/Documents/NRG/tests/test_v2/test_evaluation/test_threshold_calibration.py`

**Step 1: Write failing test**

```python
# tests/test_v2/test_evaluation/test_threshold_calibration.py
import pytest
from nrg_core.v2.evaluation.threshold_calibration import ThresholdCalibrator


def test_calibrate_multi_sample_threshold():
    """Should find optimal threshold for multi-sample triggering."""
    calibrator = ThresholdCalibrator()
    
    # Mock evaluation results at different thresholds
    results = {
        0.6: {"precision": 0.85, "recall": 0.95, "cost": 0.40},
        0.7: {"precision": 0.90, "recall": 0.90, "cost": 0.35},
        0.75: {"precision": 0.92, "recall": 0.85, "cost": 0.33},
        0.8: {"precision": 0.95, "recall": 0.75, "cost": 0.30}
    }
    
    optimal_threshold = calibrator.find_optimal(
        results=results,
        target_metric="f1",
        cost_weight=0.2  # 20% weight on cost
    )
    
    # Should balance F1 and cost
    assert 0.7 <= optimal_threshold <= 0.8


def test_calibrate_impact_threshold():
    """Should calibrate impact threshold for enhanced routing."""
    calibrator = ThresholdCalibrator()
    
    results = {
        5: {"fpr": 0.02, "recall": 0.95, "cost": 0.50},
        6: {"fpr": 0.01, "recall": 0.90, "cost": 0.40},
        7: {"fpr": 0.005, "recall": 0.80, "cost": 0.35}
    }
    
    optimal = calibrator.find_optimal(
        results=results,
        target_metric="fpr",  # Minimize false positive rate
        cost_weight=0.1
    )
    
    # Should favor lower FPR even at higher cost
    assert optimal >= 6
```

**Step 2: Run test**

Run: `pytest tests/test_v2/test_evaluation/test_threshold_calibration.py -v`

Expected: `ModuleNotFoundError`

**Step 3: Implement calibrator**

```python
# nrg_core/v2/evaluation/threshold_calibration.py
"""
Threshold Calibration
Find optimal thresholds for routing, multi-sample triggering, etc.
"""
from typing import Dict, Any


class ThresholdCalibrator:
    """
    Calibrate thresholds to optimize for target metric + cost.
    
    Thresholds to calibrate:
    1. Multi-sample confidence threshold (when to resample)
    2. Impact threshold for enhanced routing
    3. Judge confidence threshold for fallback
    """
    
    def find_optimal(
        self,
        results: Dict[float, Dict[str, float]],
        target_metric: str,
        cost_weight: float = 0.1
    ) -> float:
        """
        Find optimal threshold that maximizes target metric while minimizing cost.
        
        Args:
            results: Dict of threshold -> metrics
            target_metric: Metric to optimize ("f1", "precision", "recall", "fpr")
            cost_weight: Weight for cost in optimization (0-1)
        
        Returns:
            Optimal threshold
        """
        best_threshold = None
        best_score = -float('inf')
        
        for threshold, metrics in results.items():
            # Compute combined score
            if target_metric == "fpr":
                # For FPR, lower is better
                metric_score = 1.0 - metrics.get("fpr", 1.0)
            else:
                metric_score = metrics.get(target_metric, 0.0)
            
            # Normalize cost (assume max cost is 1.0)
            cost_score = 1.0 - metrics.get("cost", 0.0)
            
            # Combined score
            combined = (1 - cost_weight) * metric_score + cost_weight * cost_score
            
            if combined > best_score:
                best_score = combined
                best_threshold = threshold
        
        return best_threshold
    
    def recommend_thresholds(
        self,
        silver_set_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Recommend all thresholds based on silver set evaluation.
        
        Args:
            silver_set_results: Full evaluation results
        
        Returns:
            Dict of threshold_name -> recommended_value
        """
        # Placeholder: would analyze silver set results
        return {
            "multi_sample_confidence_threshold": 0.7,
            "multi_sample_impact_threshold": 6,
            "fallback_judge_confidence_min": 0.6,
            "fallback_judge_confidence_max": 0.8,
            "enhanced_routing_complexity_score": 3
        }
```

**Step 4: Run tests**

Run: `pytest tests/test_v2/test_evaluation/test_threshold_calibration.py -v`

Expected: PASS

**Step 5: Create calibration CLI**

```python
# scripts/run_calibration.py
"""Run threshold calibration on silver set."""
import argparse
from nrg_core.v2.evaluation.silver_set import SilverSet
from nrg_core.v2.evaluation.threshold_calibration import ThresholdCalibrator
from nrg_core.v2.two_tier import TwoTierOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Calibrate thresholds on silver set")
    parser.add_argument("--silver-set-dir", default="data/silver_set")
    parser.add_argument("--output", default="calibration_results.json")
    
    args = parser.parse_args()
    
    # Load silver set
    silver_set = SilverSet(data_dir=args.silver_set_dir)
    bills = silver_set.load()
    
    print(f"Loaded {len(bills)} bills from silver set")
    
    # Run analysis at different thresholds
    # ... (would implement full sweep)
    
    # Find optimal thresholds
    calibrator = ThresholdCalibrator()
    # ... (would compute)
    
    print("Calibration complete")


if __name__ == "__main__":
    main()
```

**Step 6: Commit**

```bash
git add nrg_core/v2/evaluation/threshold_calibration.py tests/test_v2/test_evaluation/test_threshold_calibration.py scripts/run_calibration.py
git commit -m "feat(v2): add threshold calibration tool"
```

---

## Summary

**Phase 3 Complete:** Evaluation & calibration with:

✅ Task 1: Silver set infrastructure (50-100 expert-labeled bills)  
✅ Task 2: Precision/Recall evaluator (semantic matching)  
✅ Task 3: LLM-judge ensemble (GPT + Claude + Gemini)  
✅ Task 4: Spot-checking pipeline (stratified sampling)  
✅ Task 5: Threshold calibration tool

**Verification:**
```bash
pytest tests/test_v2/test_evaluation/ -v
```

**Usage:**
```bash
# Add bill to silver set
python scripts/add_to_silver_set.py --bill-id HB123 --bill-text-file bill.txt --labels-file labels.json

# Run calibration
python scripts/run_calibration.py --silver-set-dir data/silver_set
```

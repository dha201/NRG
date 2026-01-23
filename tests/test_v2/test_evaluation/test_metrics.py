"""
Tests for Precision/Recall Evaluator.

Tests verify that the evaluator:
- Computes precision, recall, F1 correctly
- Uses semantic similarity for finding matching
- Calculates MAE for rubric scores
"""
import pytest
from nrg_core.v2.evaluation.metrics import PrecisionRecallEvaluator


def test_precision_recall_calculation():
    """Should compute precision, recall, F1 from predictions vs labels."""
    evaluator = PrecisionRecallEvaluator(similarity_threshold=0.5)  # Lower threshold for testing
    
    # Predicted findings
    predicted = [
        {"statement": "Tax of $50 per megawatt", "impact": 7},
        {"statement": "Renewable energy exemption", "impact": 3},
        {"statement": "False positive finding", "impact": 2}
    ]
    
    # Expert labels
    expert = [
        {"statement": "Tax applies at $50 per megawatt", "impact": 7},
        {"statement": "Renewable energy exemption", "impact": 3}
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

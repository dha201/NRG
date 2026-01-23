"""
Tests for Threshold Calibration Tool.

Tests verify that the calibrator:
- Finds optimal thresholds balancing metrics and cost
- Handles different target metrics (F1, precision, recall, FPR)
- Provides reasonable threshold recommendations
"""
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

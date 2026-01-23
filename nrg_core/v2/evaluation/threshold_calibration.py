"""
Threshold Calibration for System Optimization.

Find optimal thresholds for routing, multi-sample triggering, and other
decision points by balancing target metrics with computational cost.

Design:
- Evaluates threshold performance across multiple metrics
- Combines metric performance with cost considerations
- Supports different optimization targets (F1, precision, recall, FPR)
- Provides calibrated threshold recommendations

Why calibration matters:
- Thresholds significantly impact system performance and cost
- Default values may not be optimal for specific use cases
- Trade-offs between accuracy and cost need explicit balancing
- Continuous calibration maintains optimal performance

Thresholds calibrated:
1. Multi-sample confidence threshold (when to resample)
2. Impact threshold for enhanced routing
3. Judge confidence threshold for fallback
4. Consistency score threshold for human review
"""
from typing import Dict, Any


class ThresholdCalibrator:
    """
    Calibrate thresholds to optimize for target metric while minimizing cost.
    
    Uses weighted scoring to balance metric performance against computational cost.
    Supports different optimization targets depending on system priorities.
    
    Optimization approach:
    - For metrics where higher is better (F1, precision, recall): maximize
    - For metrics where lower is better (FPR, cost): minimize
    - Cost weight determines trade-off aggressiveness
    - Returns threshold with best combined score
    
    Cost considerations:
    - API calls increase with lower thresholds
    - Human review time increases with more flags
    - Computational resources scale with analysis depth
    """
    
    def find_optimal(
        self,
        results: Dict[float, Dict[str, float]],
        target_metric: str,
        cost_weight: float = 0.1
    ) -> float:
        """
        Find optimal threshold that maximizes target metric while minimizing cost.
        
        Algorithm:
        1. For each threshold, compute combined score
        2. Metric score weighted by (1 - cost_weight)
        3. Cost score weighted by cost_weight
        4. Select threshold with highest combined score
        
        Args:
            results: Dict of threshold -> metrics dictionary
            target_metric: Metric to optimize ("f1", "precision", "recall", "fpr")
            cost_weight: Weight for cost in optimization (0-1, higher = more cost-sensitive)
        
        Returns:
            Optimal threshold value
        """
        best_threshold = None
        best_score = -float('inf')
        
        for threshold, metrics in results.items():
            # Compute metric score (handle directionality)
            if target_metric == "fpr":
                # For FPR, lower is better, so invert
                metric_score = 1.0 - metrics.get("fpr", 1.0)
            else:
                # For F1, precision, recall: higher is better
                metric_score = metrics.get(target_metric, 0.0)
            
            # Compute cost score (lower cost = higher score)
            cost_score = 1.0 - metrics.get("cost", 0.0)
            
            # Combined score with weighted trade-off
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
        
        Provides calibrated thresholds for major decision points in the system.
        Recommendations based on typical performance patterns and cost trade-offs.
        
        Args:
            silver_set_results: Full evaluation results from silver set testing
        
        Returns:
            Dict of threshold_name -> recommended_value
        """
        # In production, would analyze actual silver set results
        # For now, provide reasonable defaults based on typical performance
        
        return {
            # Multi-sample triggering: medium-high confidence to balance consistency vs cost
            "multi_sample_confidence_threshold": 0.7,
            
            # Impact threshold for enhanced routing: moderate impact to catch important cases
            "multi_sample_impact_threshold": 6,
            
            # Fallback judge confidence range: uncertain but not completely uncertain
            "fallback_judge_confidence_min": 0.6,
            "fallback_judge_confidence_max": 0.8,
            
            # Enhanced routing complexity: moderate complexity for meaningful analysis
            "enhanced_routing_complexity_score": 3,
            
            # Consistency threshold for human review: flag when agreement is low
            "consensus_consistency_threshold": 0.4,
            
            # Spot-checking sample size: reasonable for per-release validation
            "spot_check_sample_size": 20
        }
    
    def analyze_threshold_sensitivity(
        self,
        results: Dict[float, Dict[str, float]],
        target_metric: str
    ) -> Dict[str, Any]:
        """
        Analyze how sensitive performance is to threshold changes.
        
        Helps understand the robustness of threshold selection and identify
        regions where small changes have large performance impacts.
        
        Args:
            results: Threshold performance results
            target_metric: Metric to analyze sensitivity for
        
        Returns:
            Sensitivity analysis with key insights
        """
        thresholds = sorted(results.keys())
        metric_values = [results[t].get(target_metric, 0) for t in thresholds]
        
        # Compute sensitivity (rate of change)
        sensitivities = []
        for i in range(1, len(metric_values)):
            if thresholds[i] != thresholds[i-1]:
                delta_metric = metric_values[i] - metric_values[i-1]
                delta_threshold = thresholds[i] - thresholds[i-1]
                sensitivity = abs(delta_metric / delta_threshold)
                sensitivities.append(sensitivity)
        
        # Find most sensitive region
        max_sensitivity_idx = max(range(len(sensitivities))) if sensitivities else 0
        most_sensitive_region = (
            thresholds[max_sensitivity_idx],
            thresholds[max_sensitivity_idx + 1]
        ) if max_sensitivity_idx < len(thresholds) - 1 else (None, None)
        
        return {
            "thresholds": thresholds,
            "metric_values": metric_values,
            "sensitivities": sensitivities,
            "most_sensitive_region": most_sensitive_region,
            "avg_sensitivity": sum(sensitivities) / len(sensitivities) if sensitivities else 0
        }

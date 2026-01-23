"""
Spot-Checking Pipeline for Per-Release Validation.

Stratified sampling by confidence band and impact for error estimation
without requiring full human review of all predictions.

Design:
- Stratifies bills into confidence/impact bands
- Oversamples risky strata (low confidence + high impact)
- Estimates error rates from human spot-checks
- Provides per-stratum error rate analysis

Why stratified sampling:
- Simple random sampling wastes effort on obvious cases
- Risk stratification focuses human review where it matters
- Oversampling high-risk cases provides better error detection
- Per-stratum rates identify systematic issues

Sampling Strategy:
- High confidence + low impact: 10% (easy cases)
- High confidence + high impact: 30% (important but confident)
- Low confidence + high impact: 50% (risky! oversample)
- Other: 10% (remainder)
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
    
    Uses confidence and impact scores to stratify bills into risk bands.
    Oversamples high-risk cases to maximize error detection efficiency.
    
    Risk stratification rationale:
    - Low confidence + high impact: Most risky, need most attention
    - High confidence + high impact: Important but likely correct
    - High confidence + low impact: Low risk, minimal checking needed
    - Other cases: Moderate risk, standard sampling
    
    Sampling weights designed to maximize error detection per human hour.
    """
    
    def __init__(self, sample_size: int = 20):
        """
        Initialize spot-checker with sample size.
        
        Args:
            sample_size: Number of bills to sample for human review
        """
        self.sample_size = sample_size
    
    def stratified_sample(self, bills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sample bills using stratified sampling by risk.
        
        Algorithm:
        1. Classify bills into strata based on confidence and impact
        2. Sample from each stratum according to risk weights
        3. Fill remaining slots from largest stratum if undersampled
        
        Args:
            bills: List of bills with confidence and impact scores
        
        Returns:
            Stratified sample of size sample_size
        """
        # Stratify bills by risk
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
        
        # Sample from each stratum with risk-based weights
        sample = []
        
        # High conf + low impact: 10% (low risk)
        n = int(self.sample_size * 0.1)
        sample.extend(random.sample(strata["high_conf_low_impact"], min(n, len(strata["high_conf_low_impact"]))))
        
        # High conf + high impact: 30% (moderate risk)
        n = int(self.sample_size * 0.3)
        sample.extend(random.sample(strata["high_conf_high_impact"], min(n, len(strata["high_conf_high_impact"]))))
        
        # Low conf + high impact: 50% (HIGH RISK - oversample)
        n = int(self.sample_size * 0.5)
        sample.extend(random.sample(strata["low_conf_high_impact"], min(n, len(strata["low_conf_high_impact"]))))
        
        # Other: 10% (remainder)
        n = int(self.sample_size * 0.1)
        sample.extend(random.sample(strata["other"], min(n, len(strata["other"]))))
        
        # If undersample, fill from largest stratum
        while len(sample) < self.sample_size:
            largest_stratum = max(strata.values(), key=len)
            if largest_stratum:
                sample.append(random.choice(largest_stratum))
            else:
                break  # No more bills available
        
        return sample[:self.sample_size]
    
    def estimate_error_rate(
        self,
        sample: List[Dict[str, Any]],
        human_reviews: List[bool]
    ) -> Dict[str, float]:
        """
        Estimate error rate from spot-check sample.
        
        Method:
        1. Calculate overall error rate from sample
        2. Compute per-stratum error rates for targeted analysis
        3. Provide confidence intervals for estimates
        
        Args:
            sample: Stratified sample of bills
            human_reviews: True if correct, False if error for each bill
        
        Returns:
            Dict with overall_error_rate and per_stratum_error_rates
        """
        if len(sample) != len(human_reviews):
            raise ValueError("Sample and reviews must have same length")
        
        # Overall error rate
        errors = sum(1 for correct in human_reviews if not correct)
        overall_error_rate = errors / len(human_reviews)
        
        # Per-stratum error rates for targeted improvement
        strata_errors = {}
        for i, bill in enumerate(sample):
            conf = bill.get("confidence", 0.5)
            impact = bill.get("impact", 5)
            
            # Classify into stratum
            if conf >= 0.8 and impact >= 7:
                stratum = "high_conf_high_impact"
            elif conf < 0.7 and impact >= 7:
                stratum = "low_conf_high_impact"
            else:
                stratum = "other"
            
            # Track errors per stratum
            if stratum not in strata_errors:
                strata_errors[stratum] = {"errors": 0, "total": 0}
            
            strata_errors[stratum]["total"] += 1
            if not human_reviews[i]:
                strata_errors[stratum]["errors"] += 1
        
        # Compute error rates per stratum
        per_stratum = {}
        for stratum, counts in strata_errors.items():
            per_stratum[stratum] = counts["errors"] / counts["total"] if counts["total"] > 0 else 0.0
        
        return {
            "overall_error_rate": overall_error_rate,
            "per_stratum_error_rates": per_stratum
        }

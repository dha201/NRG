"""
Drift detection tests - PLACEHOLDER.

⚠️ STATUS: Production feature - requires live data collection.

Purpose: Monitor for data drift or concept drift in production.

Implementation Notes:
- Use tools: Evidently AI, Alibi Detect
- Log all analyses to time-series DB
- Weekly drift reports

Reference: https://www.kellton.com/kellton-tech-blog/ai-ml-models-quality-assurance-best-techniques
"""
import pytest


@pytest.mark.placeholder
class TestDataDrift:
    """PLACEHOLDER: Tests for detecting data drift."""
    
    @pytest.mark.skip(reason="Requires production data collection")
    def test_input_distribution_drift(self):
        """Detect drift in input data distribution.
        
        TODO:
        1. Collect baseline statistics:
           - Bill text length distribution
           - Source distribution (Congress.gov vs OpenStates)
           - Bill type distribution
        2. Compare current window against baseline
        3. Alert if KL divergence > threshold
        """
        pass
    
    @pytest.mark.skip(reason="Requires production data collection")
    def test_bill_source_ratio_drift(self):
        """Detect changes in bill source ratios.
        
        TODO:
        1. Track ratio of federal vs state bills
        2. Compare against historical baseline
        3. Alert on significant shift
        """
        pass


@pytest.mark.placeholder
class TestConceptDrift:
    """PLACEHOLDER: Tests for detecting concept drift."""
    
    @pytest.mark.skip(reason="Requires production data collection")
    def test_score_distribution_drift(self):
        """Detect drift in output score distributions.
        
        TODO:
        1. Track distribution of business_impact_score over time
        2. Compare current window against historical
        3. Alert if mean shifts significantly
        4. Could indicate model behavior change
        """
        pass
    
    @pytest.mark.skip(reason="Requires production data collection")
    def test_action_distribution_drift(self):
        """Detect drift in recommended_action distribution.
        
        TODO:
        1. Track distribution of actions (ignore/monitor/engage/urgent)
        2. Compare against historical baseline
        3. Alert on significant shift
        """
        pass


@pytest.mark.placeholder
class TestDriftAlerts:
    """PLACEHOLDER: Tests for drift alerting system."""
    
    @pytest.mark.skip(reason="Requires alerting infrastructure")
    def test_drift_alert_triggered(self):
        """Verify drift alerts are triggered correctly.
        
        TODO:
        1. Simulate drift condition
        2. Verify alert is generated
        3. Verify alert contains diagnostic info
        """
        pass


# =============================================================================
# DRIFT DETECTION IMPLEMENTATION GUIDE
# =============================================================================
# 1. Install Evidently: pip install evidently
#
# 2. Create drift report:
#    from evidently.report import Report
#    from evidently.metric_preset import DataDriftPreset
#    
#    report = Report(metrics=[DataDriftPreset()])
#    report.run(reference_data=baseline_df, current_data=current_df)
#    report.save_html("drift_report.html")
#
# 3. Schedule weekly drift checks in production
# =============================================================================

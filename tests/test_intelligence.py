"""
TCRIE - Intelligence Engine Unit Tests
Tests CUSUM, Mahalanobis, and Risk Aggregator algorithms.
Author: Nampally Aryan
"""

import numpy as np
import pytest

from app.intelligence.cusum import compute_cusum, compute_cusum_bidirectional
from app.intelligence.mahalanobis import compute_mahalanobis, detect_anomalies
from app.intelligence.aggregator import (
    compute_quality_score,
    compute_risk_score,
    classify_risk,
)


class TestCUSUM:
    """Tests for the CUSUM drift detection algorithm."""

    def test_cusum_detects_upward_drift(self):
        """CUSUM should detect a clear upward shift in the data."""
        # Baseline around 5, then jumps to 10
        series = np.array([5, 5, 5, 5, 10, 10, 10, 10, 10, 10])
        result = compute_cusum(series, mu0=5.0, k=0.5, threshold=5.0)

        assert result["alarm"] is True
        assert result["alarm_index"] is not None
        assert result["max_score"] > 5.0

    def test_cusum_no_false_alarm_on_stable_data(self):
        """CUSUM should NOT alarm on stable data within noise."""
        series = np.array([5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9])
        result = compute_cusum(series, mu0=5.0, k=0.5, threshold=5.0)

        assert result["alarm"] is False
        assert result["max_score"] < 5.0

    def test_cusum_empty_series(self):
        """CUSUM should handle empty input gracefully."""
        result = compute_cusum(np.array([]), k=0.5, threshold=5.0)
        assert result["alarm"] is False
        assert result["cusum_scores"] == []

    def test_cusum_auto_baseline(self):
        """CUSUM should auto-estimate baseline from initial observations."""
        series = np.array([5, 5, 5, 5, 15, 15, 15, 15])
        result = compute_cusum(series, mu0=None, k=0.5, threshold=5.0, baseline_window=4)

        assert result["alarm"] is True
        assert result["baseline_mean"] == pytest.approx(5.0)

    def test_cusum_bidirectional_detects_decrease(self):
        """Bidirectional CUSUM should detect a downward shift."""
        series = np.array([10, 10, 10, 10, 3, 3, 3, 3, 3, 3])
        result = compute_cusum_bidirectional(series, mu0=10.0, k=0.5, threshold=5.0)

        assert result["alarm"] is True
        assert result["max_lower"] > 5.0


class TestMahalanobis:
    """Tests for the Mahalanobis distance anomaly detection."""

    def test_mahalanobis_zero_for_mean(self):
        """Mahalanobis distance of the mean vector from itself should be ~0."""
        mean = np.array([5.0, 10.0, 15.0])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        dist = compute_mahalanobis(mean, mean, cov)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_mahalanobis_outlier_detection(self):
        """An outlier should have a large Mahalanobis distance."""
        mean = np.array([5.0, 10.0, 15.0])
        cov = np.eye(3)
        outlier = np.array([50.0, 50.0, 50.0])  # Far from mean
        dist = compute_mahalanobis(outlier, mean, cov)
        assert dist > 10.0  # Should be very large

    def test_detect_anomalies_flags_outlier(self):
        """Anomaly detection should flag injected outlier sites."""
        import pandas as pd
        # Create a clear dataset: 10 normal sites + 1 extreme outlier
        np.random.seed(42)
        normal_data = {
            "feat1": np.random.normal(10, 1, 10),
            "feat2": np.random.normal(20, 2, 10),
            "feat3": np.random.normal(30, 3, 10),
        }
        normal_df = pd.DataFrame(normal_data, index=[f"S{i:02d}" for i in range(10)])

        outlier = pd.DataFrame(
            {"feat1": [100.0], "feat2": [200.0], "feat3": [300.0]},
            index=["OUTLIER"],
        )

        augmented = pd.concat([normal_df, outlier])
        results = detect_anomalies(augmented, critical_distance=3.0)

        assert bool(results.loc["OUTLIER", "is_anomaly"]) is True
        assert results.loc["OUTLIER", "mahalanobis_distance"] > 3.0


class TestRiskAggregator:
    """Tests for the fused risk score computation."""

    def test_risk_score_zero_inputs(self):
        """All-zero inputs should produce score = 0."""
        result = compute_risk_score(0.0, 0.0, 0.0)
        assert result["risk_score"] == 0.0
        assert result["quality_component"] == 0.0
        assert result["drift_component"] == 0.0
        assert result["anomaly_component"] == 0.0

    def test_risk_score_max_inputs(self):
        """Maximum inputs should produce score near 100."""
        result = compute_risk_score(100.0, 10.0, 6.0)
        assert result["risk_score"] == pytest.approx(100.0, abs=1.0)

    def test_risk_score_respects_weights(self):
        """Weights should influence the contribution of each component."""
        weights = {"w1": 1.0, "w2": 0.0, "w3": 0.0}
        result = compute_risk_score(50.0, 10.0, 6.0, weights=weights)
        assert result["risk_score"] == pytest.approx(50.0, abs=0.1)
        assert result["drift_component"] == 0.0
        assert result["anomaly_component"] == 0.0

    def test_classify_risk_levels(self):
        """Risk classification should respect threshold boundaries."""
        assert classify_risk(10.0) == "LOW"
        assert classify_risk(30.0) == "MEDIUM"
        assert classify_risk(55.0) == "HIGH"
        assert classify_risk(80.0) == "CRITICAL"

    def test_quality_score_low_ae_rate(self):
        """Low AE reporting rate should trigger a quality violation."""
        features = {"ae_reporting_rate": 0.1, "lab_cv_score": 0.5, "enrollment_velocity": 1.0}
        result = compute_quality_score(features)
        assert result["q_score"] > 0
        assert any(v["rule"] == "LOW_AE_REPORTING" for v in result["violations"])

    def test_quality_score_clean_site(self):
        """A site with normal metrics should have zero violations."""
        features = {"ae_reporting_rate": 3.0, "lab_cv_score": 0.5, "enrollment_velocity": 1.0}
        result = compute_quality_score(features)
        assert result["q_score"] == 0.0
        assert result["violations"] == []

"""
TCRIE - Risk Score Aggregator
Fuses deterministic quality checks, CUSUM drift scores, and Mahalanobis
anomaly distances into a single explainable 0-100 Site Risk Score.

Mathematical Foundation:
    R_site = w1 * Q_score + w2 * f(S_t) + w3 * g(D_M(x))

Where:
    Q_score = deterministic quality rule violations score (0-100)
    f(S_t)  = normalized CUSUM drift score (0-100)
    g(D_M)  = normalized Mahalanobis distance score (0-100)
    w1, w2, w3 = configurable weights (sum to 1.0)

Author: Nampally Aryan
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default risk level thresholds
DEFAULT_THRESHOLDS = {"low": 25, "medium": 50, "high": 75}

# Default feature weights
DEFAULT_WEIGHTS = {"w1": 0.30, "w2": 0.35, "w3": 0.35}

# Default deterministic quality rules
DEFAULT_QUALITY_RULES = {
    "min_ae_rate": 0.5,
    "max_lab_cv_compression": 0.15,
    "max_enrollment_velocity": 5.0,
}


def _normalize_score(value: float, max_value: float, cap: float = 100.0) -> float:
    """Normalize a raw score to 0-100 scale."""
    if max_value <= 0:
        return 0.0
    return min(cap, (value / max_value) * cap)


def _safe_percentile(values: List[float], percentile: float, default: float) -> float:
    """Return percentile for non-empty finite values, otherwise default."""
    finite_values = [v for v in values if np.isfinite(v)]
    if not finite_values:
        return default
    return float(np.percentile(finite_values, percentile))


def _derive_dynamic_quality_rules(
    site_matrix: pd.DataFrame,
    base_rules: Dict,
) -> Dict:
    """
    Adapt quality rule cutoffs to current data while preserving safe defaults.

    Adaptation is conservative to reduce false positives on distribution shifts:
    - min_ae_rate can relax downward (never tighten above configured base).
    - max_lab_cv_compression can relax downward.
    - max_enrollment_velocity can relax upward.
    """
    effective_rules = dict(base_rules)

    if "ae_reporting_rate" in site_matrix.columns:
        ae_values = site_matrix["ae_reporting_rate"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().tolist()
        p10_ae = _safe_percentile(ae_values, 10, base_rules["min_ae_rate"])
        effective_rules["min_ae_rate"] = round(
            min(base_rules["min_ae_rate"], max(0.0, p10_ae)),
            4,
        )

    if "lab_cv_score" in site_matrix.columns:
        lab_cv_values = site_matrix["lab_cv_score"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().tolist()
        p10_lab_cv = _safe_percentile(
            lab_cv_values,
            10,
            base_rules["max_lab_cv_compression"],
        )
        effective_rules["max_lab_cv_compression"] = round(
            min(base_rules["max_lab_cv_compression"], max(0.0, p10_lab_cv)),
            4,
        )

    if "enrollment_velocity" in site_matrix.columns:
        velocity_values = site_matrix["enrollment_velocity"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().tolist()
        p90_velocity = _safe_percentile(
            velocity_values,
            90,
            base_rules["max_enrollment_velocity"],
        )
        effective_rules["max_enrollment_velocity"] = round(
            max(base_rules["max_enrollment_velocity"], p90_velocity),
            4,
        )

    return effective_rules


def _derive_dynamic_component_caps(
    cusum_results: Dict[str, Dict],
    mahalanobis_results: pd.DataFrame,
    base_cusum_max: float = 10.0,
    base_mahalanobis_max: float = 6.0,
) -> Dict[str, float]:
    """Derive robust score normalization caps from current run data."""
    cusum_values = []
    for result in cusum_results.values():
        try:
            cusum_values.append(float(result.get("max_score", 0.0)))
        except (TypeError, ValueError):
            continue

    mahal_values = []
    if "mahalanobis_distance" in mahalanobis_results.columns:
        mahal_values = (
            mahalanobis_results["mahalanobis_distance"]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .tolist()
        )

    derived_cusum = _safe_percentile(cusum_values, 95, base_cusum_max)
    derived_mahal = _safe_percentile(mahal_values, 95, base_mahalanobis_max)

    return {
        "cusum_max": round(max(base_cusum_max, derived_cusum), 4),
        "mahalanobis_max": round(max(base_mahalanobis_max, derived_mahal), 4),
    }


def compute_quality_score(
    site_features: Dict,
    rules: Optional[Dict] = None,
) -> Dict:
    """
    Compute deterministic quality score from rule-based checks.

    Args:
        site_features: Dictionary of site-level features.
        rules: Quality rule thresholds.

    Returns:
        Dictionary with q_score (0-100) and list of violated rules.
    """
    if rules is None:
        rules = DEFAULT_QUALITY_RULES

    violations = []
    penalty = 0.0

    # Rule 1: Low AE reporting rate (potential under-reporting)
    ae_rate = site_features.get("ae_reporting_rate", 0.0)
    if ae_rate < rules["min_ae_rate"]:
        violations.append({
            "rule": "LOW_AE_REPORTING",
            "description": f"AE rate ({ae_rate:.2f}) below minimum ({rules['min_ae_rate']})",
            "severity": 30,
        })
        penalty += 30

    # Rule 2: Suspiciously low lab variability (potential data fabrication)
    lab_cv = site_features.get("lab_cv_score", 1.0)
    if lab_cv < rules["max_lab_cv_compression"]:
        violations.append({
            "rule": "LOW_LAB_VARIANCE",
            "description": f"Lab CV ({lab_cv:.3f}) suggests fabricated data",
            "severity": 40,
        })
        penalty += 40

    # Rule 3: Excessive enrollment velocity
    velocity = site_features.get("enrollment_velocity", 0.0)
    if velocity > rules["max_enrollment_velocity"]:
        violations.append({
            "rule": "HIGH_ENROLLMENT_VELOCITY",
            "description": f"Enrollment velocity ({velocity:.1f}/wk) exceeds limit ({rules['max_enrollment_velocity']}/wk)",
            "severity": 25,
        })
        penalty += 25

    q_score = min(100.0, penalty)

    return {"q_score": q_score, "violations": violations}


def compute_risk_score(
    q_score: float,
    cusum_score: float,
    mahalanobis_score: float,
    weights: Optional[Dict] = None,
    cusum_max: float = 10.0,
    mahalanobis_max: float = 6.0,
) -> Dict[str, float]:
    """
    Compute the fused risk score: R = w1*Q + w2*f(S) + w3*g(D)

    Returns dict with 'risk_score', 'quality_component', 'drift_component', 'anomaly_component'.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Normalize component scores to 0-100
    f_cusum = _normalize_score(cusum_score, cusum_max)
    g_mahal = _normalize_score(mahalanobis_score, mahalanobis_max)

    risk = (
        weights["w1"] * q_score
        + weights["w2"] * f_cusum
        + weights["w3"] * g_mahal
    )

    return {
        "risk_score": round(min(100.0, max(0.0, risk)), 2),
        "quality_component": round(weights["w1"] * q_score, 2),
        "drift_component": round(weights["w2"] * f_cusum, 2),
        "anomaly_component": round(weights["w3"] * g_mahal, 2),
    }


def classify_risk(
    score: float,
    thresholds: Optional[Dict] = None,
) -> str:
    """
    Classify a 0-100 risk score into a risk level.

    Args:
        score: Risk score (0-100).
        thresholds: Dictionary with low, medium, high boundaries.

    Returns:
        Risk level string: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if score >= thresholds["high"]:
        return "CRITICAL"
    elif score >= thresholds["medium"]:
        return "HIGH"
    elif score >= thresholds["low"]:
        return "MEDIUM"
    else:
        return "LOW"


def run_full_analysis(
    site_matrix: pd.DataFrame,
    cusum_results: Dict[str, Dict],
    mahalanobis_results: pd.DataFrame,
    config: Optional[Dict] = None,
) -> List[Dict]:
    """
    Run the complete risk analysis across all sites.

    Args:
        site_matrix: Site-Level Feature Matrix.
        cusum_results: Dict mapping SITEID -> CUSUM result dict.
        mahalanobis_results: DataFrame with mahalanobis_distance per site.
        config: Optional config dict with weights, thresholds, rules.

    Returns:
        List of site risk report dictionaries, sorted by risk score descending.
    """
    if config is None:
        config = {}

    weights = {**DEFAULT_WEIGHTS, **(config.get("weights") or {})}
    thresholds = {**DEFAULT_THRESHOLDS, **(config.get("thresholds") or {})}
    rules = {**DEFAULT_QUALITY_RULES, **(config.get("quality_rules") or {})}
    ctq_features = config.get("ctq_features", [])

    dynamic_thresholds = config.get("dynamic_thresholds", True)

    cusum_max = float(config.get("cusum_max", 10.0))
    mahalanobis_max = float(config.get("mahalanobis_max", 6.0))

    if dynamic_thresholds:
        rules = _derive_dynamic_quality_rules(site_matrix, rules)
        caps = _derive_dynamic_component_caps(
            cusum_results=cusum_results,
            mahalanobis_results=mahalanobis_results,
            base_cusum_max=cusum_max,
            base_mahalanobis_max=mahalanobis_max,
        )
        cusum_max = caps["cusum_max"]
        mahalanobis_max = caps["mahalanobis_max"]

        logger.info(
            "Adaptive calibration active | "
            f"min_ae_rate={rules['min_ae_rate']}, "
            f"max_lab_cv_compression={rules['max_lab_cv_compression']}, "
            f"max_enrollment_velocity={rules['max_enrollment_velocity']}, "
            f"cusum_max={cusum_max}, mahalanobis_max={mahalanobis_max}"
        )

    reports = []

    for site_id in site_matrix.index:
        # Get site features
        features = site_matrix.loc[site_id].to_dict()

        # Quality score
        quality = compute_quality_score(features, rules)

        # CUSUM score (max drift across all tracked metrics)
        cusum_data = cusum_results.get(str(site_id), {})
        cusum_max_score = cusum_data.get("max_score", 0.0)

        # Mahalanobis distance
        mahal_dist = 0.0
        contributing = []
        feature_z_scores_raw = []
        if site_id in mahalanobis_results.index:
            mahal_dist = mahalanobis_results.loc[site_id, "mahalanobis_distance"]
            contributing = mahalanobis_results.loc[site_id, "contributing_features"]
            if not isinstance(contributing, list):
                contributing = []
            feature_z_scores_raw = mahalanobis_results.loc[site_id, "feature_z_scores"]
            if not isinstance(feature_z_scores_raw, list):
                feature_z_scores_raw = []

        # Build feature contributions with CTQ tagging
        feature_contributions = []
        for fz in feature_z_scores_raw:
            fname = fz.get("feature", "")
            zscore = fz.get("z_score", 0.0)
            is_ctq = fname in ctq_features
            # CTQ features get 1.5x influence on anomaly perception
            feature_contributions.append({
                "feature": fname,
                "z_score": round(zscore * (1.5 if is_ctq else 1.0), 4),
                "is_ctq": is_ctq,
            })

        # Fused risk score with decomposition
        risk_result = compute_risk_score(
            q_score=quality["q_score"],
            cusum_score=cusum_max_score,
            mahalanobis_score=mahal_dist,
            weights=weights,
            cusum_max=cusum_max,
            mahalanobis_max=mahalanobis_max,
        )
        risk_score = risk_result["risk_score"]

        risk_level = classify_risk(risk_score, thresholds)

        report = {
            "site_id": str(site_id),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "escalated": False,  # filled by API layer from trajectory
            "risk_decomposition": {
                "quality_score": risk_result["quality_component"],
                "drift_score": risk_result["drift_component"],
                "anomaly_score": risk_result["anomaly_component"],
            },
            "cusum_drift": {
                "max_score": round(cusum_max_score, 4),
                "alarm": cusum_data.get("alarm", False),
                "cusum_scores": [round(s, 4) for s in cusum_data.get("cusum_scores", [])],
            },
            "mahalanobis_distance": round(mahal_dist, 4),
            "quality_violations": quality["violations"],
            "feature_contributions": feature_contributions,
            "contributing_features": contributing,
            "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
        }
        reports.append(report)

    # Sort by risk score descending
    reports.sort(key=lambda r: r["risk_score"], reverse=True)

    logger.info(
        f"  Risk analysis complete: "
        f"{sum(1 for r in reports if r['risk_level'] == 'CRITICAL')} CRITICAL, "
        f"{sum(1 for r in reports if r['risk_level'] == 'HIGH')} HIGH, "
        f"{sum(1 for r in reports if r['risk_level'] == 'MEDIUM')} MEDIUM, "
        f"{sum(1 for r in reports if r['risk_level'] == 'LOW')} LOW"
    )

    return reports

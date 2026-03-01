"""
TCRIE - Tabular CUSUM (Cumulative Sum) Drift Detection
Detects subtle, persistent shifts in site reporting behavior over time.

Mathematical Foundation:
    S_t = max(0, S_{t-1} + x_t - mu_0 - k)

Where:
    S_t  = cumulative drift score at time t
    mu_0 = baseline reporting mean
    k    = allowable variance (slack value)

Author: Nampally Aryan
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_cusum(
    series: np.ndarray,
    mu0: Optional[float] = None,
    k: float = 0.5,
    threshold: float = 5.0,
    baseline_window: int = 4,
) -> Dict:
    """
    Compute the one-sided upper Tabular CUSUM for a time series.

    Detects upward shifts (increase) in the process mean — e.g., a site's
    AE reporting rate silently drifting upward over successive visits.

    Args:
        series: 1D array of sequential observations (e.g., AE rate per period).
        mu0: Known baseline mean. If None, estimated from the first
             `baseline_window` observations.
        k: Allowable slack value. Controls sensitivity.
        threshold: Alarm limit — drift is flagged when S_t >= threshold.
        baseline_window: Number of initial observations to estimate mu0.

    Returns:
        Dictionary with:
            - cusum_scores: list of S_t values per time step.
            - alarm: bool, whether drift was detected.
            - alarm_index: int or None, first index where alarm triggered.
            - max_score: float, peak CUSUM score.
            - baseline_mean: float, the mu0 used.
    """
    series = np.asarray(series, dtype=float)

    if len(series) == 0:
        return {
            "cusum_scores": [],
            "alarm": False,
            "alarm_index": None,
            "max_score": 0.0,
            "baseline_mean": 0.0,
        }

    # Estimate baseline mean if not provided
    if mu0 is None:
        window = min(baseline_window, len(series))
        mu0 = float(np.mean(series[:window]))

    cusum_scores: List[float] = []
    s_t = 0.0
    alarm_index = None

    for t, x_t in enumerate(series):
        s_t = max(0.0, s_t + x_t - mu0 - k)
        cusum_scores.append(s_t)

        if s_t >= threshold and alarm_index is None:
            alarm_index = t

    alarm = alarm_index is not None
    max_score = float(max(cusum_scores)) if cusum_scores else 0.0

    if alarm:
        logger.info(
            f"  CUSUM alarm triggered at index {alarm_index} "
            f"(score={cusum_scores[alarm_index]:.2f}, threshold={threshold})"
        )

    return {
        "cusum_scores": cusum_scores,
        "alarm": alarm,
        "alarm_index": alarm_index,
        "max_score": max_score,
        "baseline_mean": mu0,
    }


def compute_cusum_bidirectional(
    series: np.ndarray,
    mu0: Optional[float] = None,
    k: float = 0.5,
    threshold: float = 5.0,
    baseline_window: int = 4,
) -> Dict:
    """
    Compute bidirectional CUSUM — detects both upward and downward shifts.

    Useful for detecting AE under-reporting (downward shift) as well as
    over-reporting (upward shift).

    Returns:
        Dictionary with upper/lower CUSUM results and combined alarm status.
    """
    series = np.asarray(series, dtype=float)

    if mu0 is None:
        window = min(baseline_window, len(series))
        mu0 = float(np.mean(series[:window]))

    upper_scores: List[float] = []
    lower_scores: List[float] = []
    s_upper = 0.0
    s_lower = 0.0
    alarm_index = None

    for t, x_t in enumerate(series):
        # Upper CUSUM (detects increase)
        s_upper = max(0.0, s_upper + x_t - mu0 - k)
        # Lower CUSUM (detects decrease)
        s_lower = max(0.0, s_lower - x_t + mu0 - k)

        upper_scores.append(s_upper)
        lower_scores.append(s_lower)

        if (s_upper >= threshold or s_lower >= threshold) and alarm_index is None:
            alarm_index = t

    alarm = alarm_index is not None
    max_upper = float(max(upper_scores)) if upper_scores else 0.0
    max_lower = float(max(lower_scores)) if lower_scores else 0.0

    return {
        "upper_scores": upper_scores,
        "lower_scores": lower_scores,
        "alarm": alarm,
        "alarm_index": alarm_index,
        "max_upper": max_upper,
        "max_lower": max_lower,
        "max_score": max(max_upper, max_lower),
        "baseline_mean": mu0,
    }

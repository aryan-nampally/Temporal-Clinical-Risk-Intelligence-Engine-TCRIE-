"""
TCRIE - Mahalanobis Distance Anomaly Detection
Measures the multivariate distance of a site's feature vector from
the overall trial distribution, accounting for inter-feature correlations.

Mathematical Foundation:
    D_M(x) = sqrt( (x - mu)^T * Sigma^{-1} * (x - mu) )

Where:
    x       = site's feature vector
    mu      = trial's mean feature vector
    Sigma   = covariance matrix of the feature space
    Sigma^{-1} = inverse covariance matrix

Author: Nampally Aryan
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

logger = logging.getLogger(__name__)


def compute_mahalanobis(
    site_vector: np.ndarray,
    mean_vector: np.ndarray,
    cov_matrix: np.ndarray,
    regularization: float = 1e-6,
) -> float:
    """
    Compute the Mahalanobis distance of a single site from the trial center.

    Args:
        site_vector: 1D feature vector for the site.
        mean_vector: 1D mean vector of the trial distribution.
        cov_matrix: 2D covariance matrix of the trial features.
        regularization: Small value added to diagonal for numerical stability.

    Returns:
        Mahalanobis distance (scalar).
    """
    site_vector = np.asarray(site_vector, dtype=float)
    mean_vector = np.asarray(mean_vector, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    # Regularize covariance for numerical stability
    cov_reg = cov_matrix + np.eye(cov_matrix.shape[0]) * regularization

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix inversion failed, using pseudo-inverse")
        cov_inv = np.linalg.pinv(cov_reg)

    distance = scipy_mahalanobis(site_vector, mean_vector, cov_inv)
    return float(distance)


def detect_anomalies(
    site_matrix: pd.DataFrame,
    critical_distance: float = 3.0,
    regularization: float = 1e-6,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Detect anomalous sites using Mahalanobis distance across all features.

    Args:
        site_matrix: DataFrame indexed by SITEID with feature columns.
        critical_distance: Distance threshold for anomaly classification.
        regularization: Covariance regularization parameter.
        feature_columns: Columns to use as features. If None, uses all numeric.

    Returns:
        DataFrame with columns:
            - mahalanobis_distance: float
            - is_anomaly: bool (distance > critical_distance)
            - contributing_features: list of features driving the anomaly
    """
    if feature_columns is None:
        feature_columns = site_matrix.select_dtypes(include=[np.number]).columns.tolist()

    X = site_matrix[feature_columns].values
    mean_vec = np.mean(X, axis=0)
    cov_mat = np.cov(X, rowvar=False)

    # Handle single-site or degenerate cases
    if X.shape[0] < 2:
        logger.warning("Fewer than 2 sites — cannot compute meaningful distances")
        result = pd.DataFrame(
            {
                "mahalanobis_distance": [0.0] * len(site_matrix),
                "is_anomaly": [False] * len(site_matrix),
                "contributing_features": [[]] * len(site_matrix),
                "feature_z_scores": [[]] * len(site_matrix),
            },
            index=site_matrix.index,
        )
        return result

    # Compute distance for each site
    distances = []
    contributing = []
    feature_z_scores = []  # per-site list of {feature, z_score} dicts

    cov_reg = cov_mat + np.eye(cov_mat.shape[0]) * regularization
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)

    std_vec = np.std(X, axis=0)
    std_vec = np.where(std_vec > 1e-9, std_vec, 1.0)

    for i in range(X.shape[0]):
        d = scipy_mahalanobis(X[i], mean_vec, cov_inv)
        distances.append(float(d))

        # Per-feature z-score contributions (explainability)
        z_norm = np.abs(X[i] - mean_vec) / std_vec
        sorted_indices = np.argsort(z_norm)[::-1]
        top_features = [
            feature_columns[j]
            for j in sorted_indices
            if z_norm[j] > 1.5
        ]
        contributing.append(top_features[:3])  # Top 3 contributors

        # Full z-score breakdown for explainability panel
        site_z = [
            {"feature": feature_columns[j], "z_score": round(float(z_norm[j]), 4)}
            for j in sorted_indices[:5]  # Top 5 features
        ]
        feature_z_scores.append(site_z)

    result = pd.DataFrame(
        {
            "mahalanobis_distance": distances,
            "is_anomaly": [d > critical_distance for d in distances],
            "contributing_features": contributing,
            "feature_z_scores": feature_z_scores,
        },
        index=site_matrix.index,
    )

    anomaly_count = result["is_anomaly"].sum()
    logger.info(
        f"  Mahalanobis analysis: {anomaly_count}/{len(result)} sites flagged "
        f"(threshold={critical_distance})"
    )

    return result

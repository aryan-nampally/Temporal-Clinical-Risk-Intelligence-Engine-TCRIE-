"""
TCRIE - Transformation Layer
Aggregates subject-level SDTM data into a Site-Level Feature Matrix.
Author: Nampally Aryan
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default lab test codes of interest for variance computation
DEFAULT_LAB_TESTS = ["ALT", "AST", "ALB", "ALP", "BILI", "CREAT", "GLUC"]


def _compute_enrollment_features(dm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute enrollment-related features per site from the DM domain.

    Features:
        - enrollment_count: Total subjects enrolled at the site.
        - enrollment_velocity: Subjects per week (based on RFSTDTC date range).
    """
    site_groups = dm.groupby("SITEID")

    enrollment_count = site_groups["USUBJID"].nunique().rename("enrollment_count")

    # Compute enrollment velocity (subjects per week)
    def _velocity(group):
        dates = pd.to_datetime(group["RFSTDTC"], errors="coerce").dropna()
        if len(dates) < 2:
            return 0.0
        date_range_weeks = (dates.max() - dates.min()).days / 7.0
        if date_range_weeks < 1:
            return float(len(dates))
        return len(dates) / date_range_weeks

    enrollment_velocity = (
        site_groups.apply(_velocity, include_groups=False)
        .rename("enrollment_velocity")
    )

    return pd.concat([enrollment_count, enrollment_velocity], axis=1)


def _compute_ae_features(ae: pd.DataFrame, dm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute adverse event features per site from the AE domain.

    Features:
        - ae_count: Total adverse events reported.
        - ae_reporting_rate: AEs per subject at the site.
        - serious_ae_rate: Fraction of AEs that are serious (AESER == 'Y').
    """
    # Map USUBJID to SITEID via DM
    subj_site = dm[["USUBJID", "SITEID"]].drop_duplicates()
    ae_with_site = ae.merge(subj_site, on="USUBJID", how="left")

    site_groups = ae_with_site.groupby("SITEID")

    ae_count = site_groups["AESEQ"].count().rename("ae_count")

    # Subjects per site for rate calculation
    subjects_per_site = dm.groupby("SITEID")["USUBJID"].nunique()

    ae_reporting_rate = (ae_count / subjects_per_site).rename("ae_reporting_rate")

    # Serious AE rate
    def _serious_rate(group):
        total = len(group)
        if total == 0:
            return 0.0
        serious = (group["AESER"] == "Y").sum()
        return serious / total

    serious_ae_rate = (
        site_groups.apply(_serious_rate, include_groups=False)
        .rename("serious_ae_rate")
    )

    return pd.concat([ae_count, ae_reporting_rate, serious_ae_rate], axis=1)


def _compute_lab_features(
    lb: pd.DataFrame,
    dm: pd.DataFrame,
    lab_tests: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute lab result features per site from the LB domain.

    Features:
        - lab_variance_score: Mean standard deviation of numeric lab results
          across key biomarkers (e.g., ALT, AST).
        - lab_cv_score: Mean coefficient of variation across key biomarkers.
    """
    if lab_tests is None:
        lab_tests = DEFAULT_LAB_TESTS

    # Map USUBJID to SITEID via DM
    subj_site = dm[["USUBJID", "SITEID"]].drop_duplicates()
    lb_with_site = lb.merge(subj_site, on="USUBJID", how="left")

    # Filter to lab tests of interest with valid numeric results
    lb_filtered = lb_with_site[
        (lb_with_site["LBTESTCD"].isin(lab_tests))
        & (lb_with_site["LBSTRESN"].notna())
    ].copy()

    # Compute per-site, per-test statistics
    grouped = lb_filtered.groupby(["SITEID", "LBTESTCD"])["LBSTRESN"]

    site_test_std = grouped.std().reset_index(name="std")
    site_test_mean = grouped.mean().reset_index(name="mean")

    merged = site_test_std.merge(site_test_mean, on=["SITEID", "LBTESTCD"])
    merged["cv"] = np.where(
        merged["mean"].abs() > 1e-9,
        merged["std"] / merged["mean"].abs(),
        0.0,
    )

    # Aggregate across tests per site
    lab_variance_score = (
        merged.groupby("SITEID")["std"].mean().rename("lab_variance_score")
    )
    lab_cv_score = (
        merged.groupby("SITEID")["cv"].mean().rename("lab_cv_score")
    )

    return pd.concat([lab_variance_score, lab_cv_score], axis=1)


def build_site_matrix(
    dm: pd.DataFrame,
    ae: pd.DataFrame,
    lb: pd.DataFrame,
    lab_tests: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build the Site-Level Feature Matrix by aggregating subject-level SDTM
    data across DM, AE, and LB domains.

    Args:
        dm: Demographics DataFrame.
        ae: Adverse Events DataFrame.
        lb: Lab Results DataFrame.
        lab_tests: Optional list of LBTESTCD codes to include.

    Returns:
        DataFrame indexed by SITEID with feature columns:
        enrollment_count, enrollment_velocity, ae_count, ae_reporting_rate,
        serious_ae_rate, lab_variance_score, lab_cv_score.
    """
    logger.info("Building Site-Level Feature Matrix...")

    enrollment = _compute_enrollment_features(dm)
    ae_features = _compute_ae_features(ae, dm)
    lab_features = _compute_lab_features(lb, dm, lab_tests)

    # Merge all features on SITEID
    site_matrix = enrollment.join(ae_features, how="outer").join(
        lab_features, how="outer"
    )

    # Fill missing values with 0 (e.g., sites with no AEs)
    site_matrix = site_matrix.fillna(0.0)

    logger.info(
        f"  Site matrix built: {site_matrix.shape[0]} sites x "
        f"{site_matrix.shape[1]} features"
    )

    return site_matrix


def build_site_ae_timeseries(
    ae: pd.DataFrame,
    dm: pd.DataFrame,
    freq: str = "MS",
) -> Dict[str, np.ndarray]:
    """
    Build temporal AE count series per site, bucketed by calendar month.

    This enables genuine temporal drift detection via CUSUM instead of
    relying on a single aggregate statistic.

    Args:
        ae: Adverse Events DataFrame (must contain AESTDTC, USUBJID).
        dm: Demographics DataFrame (must contain USUBJID, SITEID).
        freq: Pandas offset alias for time bucketing (default: month-start).

    Returns:
        Dict mapping SITEID (str) -> 1-D numpy array of AE counts per period.
    """
    subj_site = dm[["USUBJID", "SITEID"]].drop_duplicates()
    ae_with_site = ae.merge(subj_site, on="USUBJID", how="left")

    if "AESTDTC" not in ae_with_site.columns:
        logger.warning("AESTDTC column missing — cannot build AE time series")
        return {}

    ae_with_site["_ae_date"] = pd.to_datetime(
        ae_with_site["AESTDTC"], errors="coerce"
    )
    ae_dated = ae_with_site.dropna(subset=["_ae_date"]).copy()

    if ae_dated.empty:
        return {}

    # Common period range across all sites
    min_date = ae_dated["_ae_date"].min().to_period("M").to_timestamp()
    max_date = ae_dated["_ae_date"].max().to_period("M").to_timestamp()
    all_periods = pd.date_range(start=min_date, end=max_date, freq=freq)

    if len(all_periods) < 2:
        # Single month — return one-element arrays
        result: Dict[str, np.ndarray] = {}
        for site_id in dm["SITEID"].unique():
            site_aes = ae_dated[ae_dated["SITEID"] == site_id]
            result[str(site_id)] = np.array([float(len(site_aes))])
        return result

    result = {}
    for site_id in dm["SITEID"].unique():
        site_aes = ae_dated[ae_dated["SITEID"] == site_id].copy()
        if site_aes.empty:
            result[str(site_id)] = np.zeros(len(all_periods))
            continue
        counts = site_aes.set_index("_ae_date").resample(freq).size()
        counts = counts.reindex(all_periods, fill_value=0)
        result[str(site_id)] = counts.values.astype(float)

    logger.info(
        f"  AE time series built: {len(result)} sites x {len(all_periods)} periods"
    )
    return result

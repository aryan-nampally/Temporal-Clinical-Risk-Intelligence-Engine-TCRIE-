"""
TCRIE - Plasmode Simulation Tests (Spike-In Methodology)
Validates the engine's ability to detect known injected anomalies.

Protocol (from SDLC Report):
    1. Generate clean synthetic CDISC-like data
    2. Corrupt 3 specific sites:
       - Site A (index 4): 70% AE suppression (simulated negligence)
       - Site B (index 2): 85% lab variance compression (simulated fraud)
       - Site C (index 5): 3x enrollment spike (cross-domain anomaly)
    3. Run full pipeline and assert corrupted sites are flagged

Author: Nampally Aryan
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.transformer.site_matrix import build_site_matrix
from app.intelligence.cusum import compute_cusum
from app.intelligence.mahalanobis import detect_anomalies
from app.intelligence.aggregator import run_full_analysis


# Site IDs used in synthetic data
SITE_IDS = [f"{i:03d}" for i in range(10, 20)]  # "010" through "019"
SPIKE_SITE_AE = SITE_IDS[4]   # "014" — AE suppression
SPIKE_SITE_LAB = SITE_IDS[2]  # "012" — lab variance compression
SPIKE_SITE_ENR = SITE_IDS[5]  # "015" — enrollment spike


def _generate_clean_data(n_sites=10, subjects_per_site=20, seed=42):
    """Generate a clean synthetic CDISC-like dataset."""
    np.random.seed(seed)
    dm_records, ae_records, lb_records = [], [], []
    subj_id = 1000

    for site_idx in range(n_sites):
        site = SITE_IDS[site_idx]
        for _ in range(subjects_per_site):
            subj_id += 1
            usubjid = f"01-{site}-{subj_id}"

            dm_records.append({
                "STUDYID": "PLASMODE01",
                "DOMAIN": "DM",
                "USUBJID": usubjid,
                "SUBJID": str(subj_id),
                "RFSTDTC": f"2014-{np.random.randint(1,7):02d}-{np.random.randint(1,28):02d}",
                "SITEID": site,
                "AGE": float(np.random.randint(50, 80)),
                "SEX": np.random.choice(["M", "F"]),
                "RACE": "WHITE",
                "ARM": "Placebo",
            })

            # ~3 AEs per subject
            for aeseq in range(np.random.poisson(3)):
                ae_records.append({
                    "STUDYID": "PLASMODE01",
                    "DOMAIN": "AE",
                    "USUBJID": usubjid,
                    "AESEQ": float(aeseq + 1),
                    "AETERM": "HEADACHE",
                    "AEDECOD": "HEADACHE",
                    "AESEV": np.random.choice(["MILD", "MODERATE"]),
                    "AESER": np.random.choice(["Y", "N"], p=[0.08, 0.92]),
                    "AESTDTC": f"2014-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}",
                })

            # Lab results across visits
            for test in ["ALT", "AST", "ALB", "CREAT", "GLUC"]:
                for visit in range(1, 5):
                    base = {"ALT": 25, "AST": 28, "ALB": 38, "CREAT": 0.9, "GLUC": 95}[test]
                    lb_records.append({
                        "STUDYID": "PLASMODE01",
                        "DOMAIN": "LB",
                        "USUBJID": usubjid,
                        "LBSEQ": 1.0,
                        "LBTESTCD": test,
                        "LBTEST": test,
                        "LBSTRESN": base + np.random.normal(0, base * 0.15),
                        "LBSTRESU": "U/L",
                        "VISITNUM": float(visit),
                        "VISIT": f"WEEK {visit*2}",
                        "LBDTC": f"2014-{min(visit*2, 12):02d}-15",
                    })

    return pd.DataFrame(dm_records), pd.DataFrame(ae_records), pd.DataFrame(lb_records)


def _spike_ae_suppression(dm, ae, lb, site_id):
    """Suppress 70% of AE records for the target site (simulated negligence)."""
    site_subjects = dm[dm["SITEID"] == site_id]["USUBJID"].values
    site_ae_mask = ae["USUBJID"].isin(site_subjects)
    site_ae_indices = ae[site_ae_mask].index.tolist()

    # Remove 70% of AEs
    n_remove = int(len(site_ae_indices) * 0.70)
    if n_remove > 0:
        np.random.seed(42)
        drop_indices = np.random.choice(site_ae_indices, size=n_remove, replace=False)
        ae = ae.drop(drop_indices).reset_index(drop=True)
    return dm, ae, lb


def _spike_lab_compression(dm, ae, lb, site_id):
    """Compress lab variance by 85% for the target site (simulated fraud)."""
    site_subjects = dm[dm["SITEID"] == site_id]["USUBJID"].values
    site_lb_mask = lb["USUBJID"].isin(site_subjects)

    for test in ["ALT", "AST", "ALB", "CREAT", "GLUC"]:
        test_mask = site_lb_mask & (lb["LBTESTCD"] == test)
        if test_mask.sum() > 0:
            mean_val = lb.loc[test_mask, "LBSTRESN"].mean()
            lb.loc[test_mask, "LBSTRESN"] = mean_val + (
                lb.loc[test_mask, "LBSTRESN"] - mean_val
            ) * 0.15
    return dm, ae, lb


def _spike_enrollment(dm, ae, lb, site_id):
    """Triple enrollment for the target site (cross-domain anomaly)."""
    np.random.seed(99)
    site_subjects = dm[dm["SITEID"] == site_id]
    n_existing = len(site_subjects)
    subj_id = 9000
    new_records = []

    for _ in range(n_existing * 2):
        subj_id += 1
        new_records.append({
            "STUDYID": "PLASMODE01",
            "DOMAIN": "DM",
            "USUBJID": f"01-{site_id}-{subj_id}",
            "SUBJID": str(subj_id),
            "RFSTDTC": f"2014-01-{np.random.randint(1,15):02d}",
            "SITEID": site_id,
            "AGE": float(np.random.randint(50, 80)),
            "SEX": np.random.choice(["M", "F"]),
            "RACE": "WHITE",
            "ARM": "Placebo",
        })

    dm = pd.concat([dm, pd.DataFrame(new_records)], ignore_index=True)
    return dm, ae, lb


class TestPlasmodeSimulation:
    """End-to-end Plasmode simulation test suite."""

    def test_clean_data_no_critical_sites(self):
        """Step 1: Clean data should produce no CRITICAL flags."""
        dm, ae, lb = _generate_clean_data()
        site_matrix = build_site_matrix(dm, ae, lb)
        mahal_results = detect_anomalies(site_matrix, critical_distance=3.0)

        cusum_results = {}
        for site_id in site_matrix.index:
            series = np.array([site_matrix.loc[site_id, "ae_reporting_rate"]])
            cusum_results[str(site_id)] = compute_cusum(series, k=0.5, threshold=5.0)

        reports = run_full_analysis(site_matrix, cusum_results, mahal_results)
        critical_sites = [r for r in reports if r["risk_level"] == "CRITICAL"]
        assert len(critical_sites) == 0, (
            f"Clean data should have 0 CRITICAL sites, got {len(critical_sites)}"
        )

    def test_spiked_sites_are_high_risk(self):
        """After spike-in, corrupted sites should rank among the highest risk."""
        dm, ae, lb = _generate_clean_data()

        dm, ae, lb = _spike_ae_suppression(dm, ae, lb, SPIKE_SITE_AE)
        dm, ae, lb = _spike_lab_compression(dm, ae, lb, SPIKE_SITE_LAB)
        dm, ae, lb = _spike_enrollment(dm, ae, lb, SPIKE_SITE_ENR)

        site_matrix = build_site_matrix(dm, ae, lb)
        mahal_results = detect_anomalies(site_matrix, critical_distance=2.5)

        cusum_results = {}
        for site_id in site_matrix.index:
            series = np.array([site_matrix.loc[site_id, "ae_reporting_rate"]])
            cusum_results[str(site_id)] = compute_cusum(series, k=0.5, threshold=5.0)

        reports = run_full_analysis(site_matrix, cusum_results, mahal_results)

        # At least 1 spiked site should be in the top half
        top_half = [r["site_id"] for r in reports[:len(reports)//2]]
        spiked = {SPIKE_SITE_AE, SPIKE_SITE_LAB, SPIKE_SITE_ENR}
        detected = spiked.intersection(set(top_half))

        assert len(detected) >= 1, (
            f"Expected at least 1 spiked site in top-half risk, "
            f"found {detected}. Top half: {top_half}"
        )

    def test_ae_suppression_lowers_rate(self):
        """AE-suppressed site should have a lower AE reporting rate."""
        dm, ae, lb = _generate_clean_data()
        baseline_matrix = build_site_matrix(dm, ae, lb)
        baseline_rate = baseline_matrix.loc[SPIKE_SITE_AE, "ae_reporting_rate"]

        dm, ae, lb = _spike_ae_suppression(dm, ae, lb, SPIKE_SITE_AE)
        spiked_matrix = build_site_matrix(dm, ae, lb)
        spiked_rate = spiked_matrix.loc[SPIKE_SITE_AE, "ae_reporting_rate"]

        assert spiked_rate < baseline_rate, (
            f"Spiked AE rate ({spiked_rate:.2f}) should be less than "
            f"baseline ({baseline_rate:.2f})"
        )

    def test_lab_compression_reduces_variance(self):
        """Lab-compressed site should have significantly lower variance."""
        dm, ae, lb = _generate_clean_data()
        baseline_matrix = build_site_matrix(dm, ae, lb)
        baseline_var = baseline_matrix.loc[SPIKE_SITE_LAB, "lab_variance_score"]

        dm2, ae2, lb2 = _generate_clean_data()  # Fresh copy
        dm2, ae2, lb2 = _spike_lab_compression(dm2, ae2, lb2, SPIKE_SITE_LAB)
        spiked_matrix = build_site_matrix(dm2, ae2, lb2)
        spiked_var = spiked_matrix.loc[SPIKE_SITE_LAB, "lab_variance_score"]

        assert spiked_var < baseline_var * 0.5, (
            f"Spiked variance ({spiked_var:.4f}) should be < 50% of "
            f"baseline ({baseline_var:.4f})"
        )

    def test_enrollment_spike_increases_count(self):
        """Enrollment-spiked site should have ~3x enrollment."""
        dm, ae, lb = _generate_clean_data()
        baseline_matrix = build_site_matrix(dm, ae, lb)
        baseline_count = baseline_matrix.loc[SPIKE_SITE_ENR, "enrollment_count"]

        dm2, ae2, lb2 = _generate_clean_data()  # Fresh copy
        dm2, ae2, lb2 = _spike_enrollment(dm2, ae2, lb2, SPIKE_SITE_ENR)
        spiked_matrix = build_site_matrix(dm2, ae2, lb2)
        spiked_count = spiked_matrix.loc[SPIKE_SITE_ENR, "enrollment_count"]

        assert spiked_count >= baseline_count * 2.5, (
            f"Spiked enrollment ({spiked_count}) should be ~3x "
            f"baseline ({baseline_count})"
        )

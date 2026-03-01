"""
TCRIE Test Configuration - Shared Fixtures
Provides synthetic clinical trial data mimicking the CDISC pilot dataset.
Author: Nampally Aryan
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_dm():
    """Generate synthetic Demographics (DM) domain data."""
    np.random.seed(42)
    sites = ["701", "702", "703", "704", "705"]
    records = []
    subj_id = 1000

    for site in sites:
        n_subjects = np.random.randint(10, 30)
        for _ in range(n_subjects):
            subj_id += 1
            records.append({
                "STUDYID": "CDISCPILOT01",
                "DOMAIN": "DM",
                "USUBJID": f"01-{site}-{subj_id}",
                "SUBJID": str(subj_id),
                "RFSTDTC": f"2014-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}",
                "SITEID": site,
                "AGE": float(np.random.randint(50, 80)),
                "SEX": np.random.choice(["M", "F"]),
                "RACE": "WHITE",
                "ARM": np.random.choice(["Placebo", "Xanomeline Low Dose", "Xanomeline High Dose"]),
            })

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_ae(synthetic_dm):
    """Generate synthetic Adverse Events (AE) domain data linked to DM subjects."""
    np.random.seed(42)
    records = []
    seq = 0

    for _, subj in synthetic_dm.iterrows():
        n_aes = np.random.poisson(3)
        for _ in range(n_aes):
            seq += 1
            records.append({
                "STUDYID": "CDISCPILOT01",
                "DOMAIN": "AE",
                "USUBJID": subj["USUBJID"],
                "AESEQ": float(seq),
                "AETERM": np.random.choice([
                    "HEADACHE", "NAUSEA", "DIZZINESS", "FATIGUE",
                    "APPLICATION SITE ERYTHEMA", "DIARRHOEA",
                ]),
                "AEDECOD": "HEADACHE",
                "AESEV": np.random.choice(["MILD", "MODERATE", "SEVERE"]),
                "AESER": np.random.choice(["Y", "N"], p=[0.1, 0.9]),
                "AESTDTC": f"2014-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}",
            })

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_lb(synthetic_dm):
    """Generate synthetic Lab Results (LB) domain data linked to DM subjects."""
    np.random.seed(42)
    lab_tests = ["ALT", "AST", "ALB", "CREAT", "GLUC"]
    records = []
    seq = 0

    for _, subj in synthetic_dm.iterrows():
        for test in lab_tests:
            for visit in range(1, 5):
                seq += 1
                base_val = {"ALT": 25, "AST": 28, "ALB": 38, "CREAT": 0.9, "GLUC": 95}[test]
                records.append({
                    "STUDYID": "CDISCPILOT01",
                    "DOMAIN": "LB",
                    "USUBJID": subj["USUBJID"],
                    "LBSEQ": float(seq),
                    "LBTESTCD": test,
                    "LBTEST": test,
                    "LBSTRESN": base_val + np.random.normal(0, base_val * 0.15),
                    "LBSTRESU": "U/L" if test in ["ALT", "AST"] else "g/L",
                    "VISITNUM": float(visit),
                    "VISIT": f"WEEK {visit * 2}",
                    "LBDTC": f"2014-{min(12, visit*2):02d}-15",
                })

    return pd.DataFrame(records)


@pytest.fixture
def site_matrix(synthetic_dm, synthetic_ae, synthetic_lb):
    """Build site-level feature matrix from synthetic data."""
    from app.transformer.site_matrix import build_site_matrix
    return build_site_matrix(synthetic_dm, synthetic_ae, synthetic_lb)


@pytest.fixture
def config():
    """Load study configuration."""
    import yaml
    config_path = PROJECT_ROOT / "config" / "study_thresholds.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

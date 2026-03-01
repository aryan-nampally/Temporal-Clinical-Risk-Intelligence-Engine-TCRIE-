"""
TCRIE - FastAPI Backend
REST API serving explainable risk intelligence payloads.
Author: Nampally Aryan

Endpoints:
    GET  /health                          — Health check
    POST /upload                          — Upload & ingest .xpt data
    GET  /risk-report/{study_id}          — Full risk report for all sites
    GET  /risk-report/{study_id}/{site_id} — Risk report for a single site
    GET  /demo-analysis                   — Run analysis on bundled CDISC pilot data
"""

import logging
import os
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_all_domains
from app.persistence.cache import list_cached_studies, load_analysis, save_analysis
from app.transformer.site_matrix import build_site_matrix, build_site_ae_timeseries
from app.intelligence.cusum import compute_cusum
from app.intelligence.mahalanobis import detect_anomalies
from app.intelligence.aggregator import (
    compute_quality_score,
    compute_risk_score,
    classify_risk,
    run_full_analysis,
)
from app.intelligence.traceability import (
    compute_config_hash,
    build_audit_entry,
    write_audit_entry,
    read_audit_log,
    save_versioned_snapshot,
    list_versions,
    load_snapshot,
    append_risk_trajectory,
    load_risk_trajectory,
    detect_escalation,
    detect_distribution_shift,
    compute_study_feature_means,
    compute_performance_metrics,
)
from app.models.schemas import (
    HealthResponse,
    SiteRiskReport,
    StudyRiskSummary,
    UploadResponse,
    VersionInfo,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("tcrie.api")

# --- Load Configuration ---
CONFIG_PATH = PROJECT_ROOT / "config" / "study_thresholds.yaml"


def _load_config() -> Dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    logger.warning("Config not found, using defaults")
    return {}


config = _load_config()

# --- API Key Authentication ---
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(_api_key_header)):
    """Validate the API key when authentication is enabled in config."""
    expected = config.get("api", {}).get("auth_key", "")
    if not expected:
        return  # Auth disabled when key is empty or missing
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# --- In-Memory Storage (with disk persistence) ---
_analysis_cache: Dict[str, StudyRiskSummary] = {}


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Restore persisted analyses on startup."""
    for study_id in list_cached_studies():
        data = load_analysis(study_id)
        if data:
            try:
                _analysis_cache[study_id] = StudyRiskSummary.model_validate(data)
                logger.info(f"Restored cached analysis: {study_id}")
            except Exception:
                pass
    yield


# --- FastAPI Application ---
app = FastAPI(
    title="TCRIE - Temporal Clinical Risk Intelligence Engine",
    description=(
        "Predictive Risk-Based Quality Management (RBQM) API. "
        "Detects clinical trial site-level risks using CUSUM drift detection "
        "and Mahalanobis multivariate anomaly analysis."
    ),
    version="1.0.0",
    contact={"name": "Nampally Aryan"},
    lifespan=lifespan,
)

# CORS middleware for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_pipeline(data_dir: str, study_id: str) -> StudyRiskSummary:
    """
    Execute the full TCRIE pipeline: Ingest -> Transform -> Analyze ->
    Trace (config hash, audit, trajectory, shift detection, versioning).
    """
    np.random.seed(42)  # Deterministic execution mode
    logger.info(f"Running TCRIE pipeline for study: {study_id}")

    # 0. Configuration fingerprint
    config_hash = compute_config_hash(config)

    # 1. Ingest
    domains = load_all_domains(data_dir)
    dm = domains["DM"]
    ae = domains["AE"]
    lb = domains["LB"]

    # 2. Transform
    site_matrix = build_site_matrix(dm, ae, lb)
    logger.info(f"Site matrix:\n{site_matrix.to_string()}")

    # 3. CUSUM Analysis — temporal AE drift detection per site
    cusum_config = config.get("cusum", {})
    cusum_k = cusum_config.get("k", 0.5)
    cusum_threshold = cusum_config.get("threshold", 5.0)

    # Build visit-level AE time series for genuine temporal drift detection
    ae_timeseries = build_site_ae_timeseries(ae, dm)

    cusum_results = {}
    for site_id in site_matrix.index:
        series = ae_timeseries.get(str(site_id), np.array([]))
        if len(series) < 2:
            # Fall back to aggregate rate when temporal data is insufficient
            ae_rate = site_matrix.loc[site_id, "ae_reporting_rate"]
            series = np.array([ae_rate])
        result = compute_cusum(
            series, k=cusum_k, threshold=cusum_threshold
        )
        cusum_results[str(site_id)] = result

    # 4. Mahalanobis Analysis
    mahal_config = config.get("mahalanobis", {})
    mahal_threshold = mahal_config.get("critical_distance", 3.0)
    mahal_reg = mahal_config.get("regularization", 1e-6)

    mahalanobis_results = detect_anomalies(
        site_matrix,
        critical_distance=mahal_threshold,
        regularization=mahal_reg,
    )

    # 5. Fused Risk Scoring
    risk_config = config.get("risk_score", {})
    quality_rules = config.get("quality_rules", None)

    analysis_config = {
        "weights": risk_config.get("weights", {}),
        "thresholds": risk_config.get("thresholds", {}),
        "quality_rules": quality_rules,
        "dynamic_thresholds": risk_config.get("dynamic_thresholds", True),
        "cusum_max": risk_config.get("cusum_max", 10.0),
        "mahalanobis_max": risk_config.get("mahalanobis_max", 6.0),
        "ctq_features": config.get("ctq_features", []),
    }

    reports = run_full_analysis(
        site_matrix, cusum_results, mahalanobis_results, analysis_config
    )

    # 6. Trajectory tracking & escalation detection
    site_scores = {r["site_id"]: r["risk_score"] for r in reports}
    append_risk_trajectory(study_id, site_scores)
    trajectory = load_risk_trajectory(study_id)

    for r in reports:
        r["escalated"] = detect_escalation(study_id, r["site_id"])

    # 7. Distribution shift detection
    current_means = compute_study_feature_means(reports)
    prev_means = None
    if len(trajectory) >= 2:
        # Get previous run's feature means from the last snapshot
        prev_snapshot = load_snapshot(study_id, f"v{len(list_versions(study_id))}")
        if prev_snapshot and "site_reports" in prev_snapshot:
            prev_means = compute_study_feature_means(prev_snapshot["site_reports"])
    shift_result = detect_distribution_shift(current_means, prev_means)

    # 8. Performance metrics
    perf_metrics = compute_performance_metrics(reports)

    # Build study summary
    site_reports = [SiteRiskReport(**r) for r in reports]

    summary = StudyRiskSummary(
        study_id=study_id,
        analysis_timestamp=datetime.now(timezone.utc),
        config_hash=config_hash,
        total_sites=len(reports),
        critical_sites=sum(1 for r in reports if r["risk_level"] == "CRITICAL"),
        high_sites=sum(1 for r in reports if r["risk_level"] == "HIGH"),
        medium_sites=sum(1 for r in reports if r["risk_level"] == "MEDIUM"),
        low_sites=sum(1 for r in reports if r["risk_level"] == "LOW"),
        site_reports=site_reports,
        algorithm_metadata={
            "cusum_k": cusum_k,
            "cusum_threshold": cusum_threshold,
            "mahalanobis_critical_distance": mahal_threshold,
            "risk_weights": risk_config.get("weights", {}),
            "dynamic_thresholds": risk_config.get("dynamic_thresholds", True),
            "base_quality_rules": quality_rules or {},
            "base_cusum_max": risk_config.get("cusum_max", 10.0),
            "base_mahalanobis_max": risk_config.get("mahalanobis_max", 6.0),
            "ctq_features": config.get("ctq_features", []),
        },
        performance_metrics=perf_metrics,
        distribution_shift=shift_result,
        risk_trajectory=trajectory,
    )

    # 9. Versioned snapshot + audit log
    summary_data = summary.model_dump(mode="json")
    version_label = save_versioned_snapshot(study_id, summary_data)
    summary.version_label = version_label
    summary.versions = [VersionInfo(**v) for v in list_versions(study_id)]

    audit_entry = build_audit_entry(
        study_id=study_id,
        config=config,
        config_hash=config_hash,
        result_summary={
            "version": version_label,
            "total_sites": summary.total_sites,
            "critical_sites": summary.critical_sites,
            "high_sites": summary.high_sites,
            "medium_sites": summary.medium_sites,
            "low_sites": summary.low_sites,
            "performance_metrics": perf_metrics,
        },
    )
    write_audit_entry(audit_entry)

    return summary


# ============================
# API Endpoints
# ============================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.get("/demo-analysis", response_model=StudyRiskSummary, dependencies=[Depends(verify_api_key)])
async def demo_analysis():
    """
    Run the TCRIE analysis on the bundled CDISC pilot dataset.
    Uses data from the project's data/ directory.
    """
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.is_dir():
        raise HTTPException(status_code=404, detail="Demo data directory not found")

    study_id = config.get("study", {}).get("id", "CDISCPILOT01")

    try:
        summary = _run_pipeline(str(data_dir), study_id)
        _analysis_cache[study_id] = summary
        save_analysis(study_id, summary.model_dump(mode="json"))
        return summary
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/risk-report/{study_id}", response_model=StudyRiskSummary, dependencies=[Depends(verify_api_key)])
async def get_risk_report(study_id: str):
    """
    Retrieve the full risk report for a study.
    The study must have been analyzed via /demo-analysis or /upload first.
    """
    if study_id not in _analysis_cache:
        # Try loading from disk persistence
        data = load_analysis(study_id)
        if data:
            try:
                _analysis_cache[study_id] = StudyRiskSummary.model_validate(data)
            except Exception:
                pass
    if study_id not in _analysis_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for study '{study_id}'. "
            "Run /demo-analysis or upload data first.",
        )
    return _analysis_cache[study_id]


@app.get("/risk-report/{study_id}/{site_id}", response_model=SiteRiskReport, dependencies=[Depends(verify_api_key)])
async def get_site_risk(study_id: str, site_id: str):
    """
    Retrieve the risk report for a single site within a study.
    """
    if study_id not in _analysis_cache:
        data = load_analysis(study_id)
        if data:
            try:
                _analysis_cache[study_id] = StudyRiskSummary.model_validate(data)
            except Exception:
                pass
    if study_id not in _analysis_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for study '{study_id}'.",
        )

    summary = _analysis_cache[study_id]
    for report in summary.site_reports:
        if report.site_id == site_id:
            return report

    raise HTTPException(
        status_code=404,
        detail=f"Site '{site_id}' not found in study '{study_id}'.",
    )


@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
async def upload_data(
    dm_file: UploadFile = File(..., description="Demographics domain (.xpt)"),
    ae_file: UploadFile = File(..., description="Adverse Events domain (.xpt)"),
    lb_file: UploadFile = File(..., description="Lab Results domain (.xpt)"),
    study_id: str = Query(default="UPLOADED_STUDY", description="Study identifier"),
):
    """
    Upload SDTM .xpt data files and run the full TCRIE analysis pipeline.
    Accepts three multipart file uploads: dm, ae, and lb domains.
    """
    tmpdir = tempfile.mkdtemp(prefix="tcrie_upload_")
    try:
        for filename, upload in [("dm.xpt", dm_file), ("ae.xpt", ae_file), ("lb.xpt", lb_file)]:
            dest = Path(tmpdir) / filename
            content = await upload.read()
            dest.write_bytes(content)

        domains = load_all_domains(tmpdir)
        total_subjects = domains["DM"]["USUBJID"].nunique() if "DM" in domains else 0
        total_sites = domains["DM"]["SITEID"].nunique() if "DM" in domains else 0

        summary = _run_pipeline(tmpdir, study_id)
        _analysis_cache[study_id] = summary
        save_analysis(study_id, summary.model_dump(mode="json"))

        return UploadResponse(
            study_id=study_id,
            domains_loaded=list(domains.keys()),
            total_subjects=total_subjects,
            total_sites=total_sites,
            message=f"Successfully ingested {len(domains)} domains with "
            f"{total_subjects} subjects across {total_sites} sites. "
            f"Analysis complete.",
        )
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

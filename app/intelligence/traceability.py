"""
TCRIE - Traceability & Lifecycle Monitoring Module
Provides configuration fingerprinting, structured audit logging,
versioned study snapshots, risk trajectory tracking, and
distribution-shift detection.

Author: Nampally Aryan
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_LOG_PATH = PROJECT_ROOT / "cache" / "audit_log.json"
TRAJECTORY_DIR = PROJECT_ROOT / "cache" / "trajectories"
SNAPSHOT_DIR = PROJECT_ROOT / "cache" / "snapshots"


# ─── Configuration Fingerprinting ───────────────────────────────

def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a deterministic SHA-256 fingerprint of the analysis
    configuration so every result can be traced back to the exact
    parameter set that produced it.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_hash(filepath: Path) -> str:
    """SHA-256 of a file on disk (e.g. study_thresholds.yaml)."""
    if not filepath.exists():
        return ""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


# ─── Structured Audit Log ───────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_audit_entry(entry: Dict[str, Any]) -> None:
    """Append one structured entry to the audit log."""
    _ensure_dir(AUDIT_LOG_PATH.parent)
    entries: List[Dict] = []
    if AUDIT_LOG_PATH.exists():
        try:
            entries = json.loads(AUDIT_LOG_PATH.read_text())
        except (json.JSONDecodeError, IOError):
            entries = []

    entries.append(entry)
    AUDIT_LOG_PATH.write_text(json.dumps(entries, indent=2, default=str))
    logger.info(f"Audit entry written for study {entry.get('study_id', '?')}")


def read_audit_log() -> List[Dict[str, Any]]:
    """Return all audit entries."""
    if not AUDIT_LOG_PATH.exists():
        return []
    try:
        return json.loads(AUDIT_LOG_PATH.read_text())
    except (json.JSONDecodeError, IOError):
        return []


def build_audit_entry(
    study_id: str,
    config: Dict[str, Any],
    config_hash: str,
    result_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a structured audit-log entry from analysis results."""
    cusum_cfg = config.get("cusum", {})
    mahal_cfg = config.get("mahalanobis", {})
    risk_cfg = config.get("risk_score", {})
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_id": study_id,
        "config_hash": config_hash,
        "yaml_hash": compute_file_hash(PROJECT_ROOT / "config" / "study_thresholds.yaml"),
        "cusum_k": cusum_cfg.get("k", 0.5),
        "cusum_threshold": cusum_cfg.get("threshold", 5.0),
        "mahalanobis_critical_distance": mahal_cfg.get("critical_distance", 3.0),
        "mahalanobis_regularization": mahal_cfg.get("regularization", 1e-6),
        "weights": risk_cfg.get("weights", {}),
        "dynamic_thresholds": risk_cfg.get("dynamic_thresholds", True),
        "result_summary": result_summary,
    }


# ─── Versioned Study Snapshots ──────────────────────────────────

def save_versioned_snapshot(study_id: str, data: Dict[str, Any]) -> str:
    """
    Save a study result as a new numbered version.
    Returns the version label, e.g. 'v3'.
    """
    _ensure_dir(SNAPSHOT_DIR)
    existing = sorted(SNAPSHOT_DIR.glob(f"{study_id}_v*.json"))
    next_version = len(existing) + 1
    version_label = f"v{next_version}"
    filepath = SNAPSHOT_DIR / f"{study_id}_{version_label}.json"
    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved snapshot: {study_id} {version_label}")
    return version_label


def list_versions(study_id: str) -> List[Dict[str, str]]:
    """List available snapshot versions for a study (label + timestamp)."""
    _ensure_dir(SNAPSHOT_DIR)
    versions = []
    for p in sorted(SNAPSHOT_DIR.glob(f"{study_id}_v*.json")):
        label = p.stem.replace(f"{study_id}_", "")
        try:
            data = json.loads(p.read_text())
            ts = data.get("analysis_timestamp", "")
        except Exception:
            ts = ""
        versions.append({"version": label, "timestamp": ts, "file": p.name})
    return versions


def load_snapshot(study_id: str, version: str) -> Optional[Dict[str, Any]]:
    """Load a specific versioned snapshot."""
    filepath = SNAPSHOT_DIR / f"{study_id}_{version}.json"
    if not filepath.exists():
        return None
    try:
        return json.loads(filepath.read_text())
    except (json.JSONDecodeError, IOError):
        return None


# ─── Risk Trajectory Tracking ───────────────────────────────────

def append_risk_trajectory(study_id: str, site_scores: Dict[str, float]) -> None:
    """
    Append the current run's per-site risk scores to the trajectory file.
    site_scores: {site_id: risk_score}
    """
    _ensure_dir(TRAJECTORY_DIR)
    filepath = TRAJECTORY_DIR / f"{study_id}_trajectory.json"
    trajectory: List[Dict] = []
    if filepath.exists():
        try:
            trajectory = json.loads(filepath.read_text())
        except (json.JSONDecodeError, IOError):
            trajectory = []

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": site_scores,
    }
    trajectory.append(entry)
    filepath.write_text(json.dumps(trajectory, indent=2, default=str))


def load_risk_trajectory(study_id: str) -> List[Dict[str, Any]]:
    """Return the full trajectory history for a study."""
    filepath = TRAJECTORY_DIR / f"{study_id}_trajectory.json"
    if not filepath.exists():
        return []
    try:
        return json.loads(filepath.read_text())
    except (json.JSONDecodeError, IOError):
        return []


def detect_escalation(study_id: str, site_id: str) -> bool:
    """
    Check if a site's risk score has increased for 2+ consecutive runs.
    Returns True if escalation is warranted.
    """
    trajectory = load_risk_trajectory(study_id)
    if len(trajectory) < 2:
        return False
    recent_scores = []
    for entry in trajectory[-3:]:  # last 3 runs
        score = entry.get("scores", {}).get(site_id)
        if score is not None:
            recent_scores.append(score)
    if len(recent_scores) < 2:
        return False
    # Check monotonic increase for last 2+ entries
    return all(
        recent_scores[i] < recent_scores[i + 1]
        for i in range(len(recent_scores) - 1)
    )


# ─── Distribution Shift Detection ──────────────────────────────

def detect_distribution_shift(
    current_features: Dict[str, float],
    previous_features: Optional[Dict[str, float]],
    z_threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Compare current study-level feature means against the previous run.
    Returns shift status and per-feature deltas.

    A simple but interpretable approach: flag if any feature mean shifted
    by more than z_threshold standard deviations (estimated from the
    absolute difference relative to the previous value).
    """
    if previous_features is None:
        return {"shift_detected": False, "shifts": [], "status": "STABLE"}

    shifts = []
    for feat, current_val in current_features.items():
        prev_val = previous_features.get(feat)
        if prev_val is None or prev_val == 0:
            continue
        pct_change = abs(current_val - prev_val) / abs(prev_val) * 100
        if pct_change > z_threshold * 15:  # >30% by default
            shifts.append({
                "feature": feat,
                "previous": round(prev_val, 4),
                "current": round(current_val, 4),
                "pct_change": round(pct_change, 2),
            })

    if len(shifts) >= 3:
        status = "SIGNIFICANT_SHIFT"
    elif len(shifts) >= 1:
        status = "DRIFT_INCREASING"
    else:
        status = "STABLE"

    return {
        "shift_detected": len(shifts) > 0,
        "shifts": shifts,
        "status": status,
    }


def compute_study_feature_means(site_reports: List[Dict]) -> Dict[str, float]:
    """Compute trial-level mean for each feature across all sites."""
    if not site_reports:
        return {}
    all_features: Dict[str, List[float]] = {}
    for report in site_reports:
        for k, v in report.get("features", {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                all_features.setdefault(k, []).append(v)
    return {k: round(float(np.mean(v)), 4) for k, v in all_features.items()}


def compute_performance_metrics(site_reports: List[Dict]) -> Dict[str, Any]:
    """Track headline performance metrics for lifecycle monitoring."""
    if not site_reports:
        return {}
    scores = [r["risk_score"] for r in site_reports]
    distances = [r["mahalanobis_distance"] for r in site_reports]
    flagged = sum(1 for r in site_reports if r["risk_level"] in ("HIGH", "CRITICAL"))
    return {
        "total_sites": len(site_reports),
        "flagged_sites": flagged,
        "flagged_pct": round(flagged / len(site_reports) * 100, 2) if site_reports else 0,
        "mean_risk_score": round(float(np.mean(scores)), 2),
        "mean_mahalanobis": round(float(np.mean(distances)), 4),
        "max_risk_score": round(float(np.max(scores)), 2),
    }

"""
TCRIE - Pydantic Schemas for API Validation
Defines request/response models for the FastAPI endpoints.
Author: Nampally Aryan
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QualityViolation(BaseModel):
    """A single deterministic quality rule violation."""
    rule: str = Field(..., description="Rule identifier (e.g., LOW_AE_REPORTING)")
    description: str = Field(..., description="Human-readable violation description")
    severity: float = Field(..., description="Severity score (0-100)")


class CUSUMDrift(BaseModel):
    """CUSUM temporal drift analysis result."""
    max_score: float = Field(..., description="Peak CUSUM score")
    alarm: bool = Field(..., description="Whether drift alarm was triggered")
    cusum_scores: List[float] = Field(
        default_factory=list,
        description="Full CUSUM time series for control-chart visualization",
    )


class FeatureContribution(BaseModel):
    """Per-feature deviation contribution to Mahalanobis anomaly score."""
    feature: str = Field(..., description="Feature name")
    z_score: float = Field(..., description="Normalized deviation (std units)")
    is_ctq: bool = Field(default=False, description="Critical-to-Quality feature flag")


class RiskDecomposition(BaseModel):
    """Breakdown of the fused risk score into its three components."""
    quality_score: float = Field(0.0, description="Deterministic quality component (0-100)")
    drift_score: float = Field(0.0, description="Normalized CUSUM drift component (0-100)")
    anomaly_score: float = Field(0.0, description="Normalized Mahalanobis component (0-100)")


class SiteRiskReport(BaseModel):
    """Complete risk analysis report for a single clinical trial site."""
    site_id: str = Field(..., description="Unique site identifier")
    risk_score: float = Field(..., ge=0, le=100, description="Fused risk score (0-100)")
    risk_level: str = Field(
        ...,
        description="Risk classification: LOW, MEDIUM, HIGH, or CRITICAL",
    )
    escalated: bool = Field(
        default=False,
        description="True if risk has increased for 2+ consecutive runs",
    )
    risk_decomposition: RiskDecomposition = Field(
        default_factory=RiskDecomposition,
        description="Breakdown of how each component contributes to the fused score",
    )
    cusum_drift: CUSUMDrift = Field(..., description="Temporal drift analysis")
    mahalanobis_distance: float = Field(
        ...,
        ge=0,
        description="Mahalanobis distance from trial center",
    )
    quality_violations: List[QualityViolation] = Field(
        default_factory=list,
        description="List of deterministic rule violations",
    )
    feature_contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Per-feature z-score contributions to anomaly",
    )
    contributing_features: List[str] = Field(
        default_factory=list,
        description="Top features driving the anomaly",
    )
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw site-level feature values",
    )


class DistributionShift(BaseModel):
    """Result of comparing current vs previous feature distributions."""
    shift_detected: bool = Field(False)
    status: str = Field("STABLE", description="STABLE | DRIFT_INCREASING | SIGNIFICANT_SHIFT")
    shifts: List[Dict[str, Any]] = Field(default_factory=list)


class TrajectoryPoint(BaseModel):
    """Single point in a site's risk trajectory."""
    timestamp: str
    scores: Dict[str, float] = Field(default_factory=dict)


class VersionInfo(BaseModel):
    """Metadata for a versioned study snapshot."""
    version: str
    timestamp: str = ""
    file: str = ""


class StudyRiskSummary(BaseModel):
    """Aggregated risk summary for the entire study."""
    study_id: str = Field(..., description="Study identifier")
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the analysis run",
    )
    config_hash: str = Field(
        default="",
        description="SHA-256 fingerprint of the configuration used",
    )
    version_label: str = Field(
        default="",
        description="Snapshot version (e.g. v1, v2)",
    )
    total_sites: int = Field(..., description="Total number of sites analyzed")
    critical_sites: int = Field(..., description="Sites with CRITICAL risk level")
    high_sites: int = Field(..., description="Sites with HIGH risk level")
    medium_sites: int = Field(..., description="Sites with MEDIUM risk level")
    low_sites: int = Field(..., description="Sites with LOW risk level")
    site_reports: List[SiteRiskReport] = Field(
        ...,
        description="Individual site risk reports, sorted by risk score descending",
    )
    algorithm_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about algorithms and thresholds used",
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Lifecycle monitoring headline metrics",
    )
    distribution_shift: Optional[DistributionShift] = Field(
        default=None,
        description="Comparison of feature distributions vs previous run",
    )
    risk_trajectory: List[TrajectoryPoint] = Field(
        default_factory=list,
        description="Historical risk score trajectory across runs",
    )
    versions: List[VersionInfo] = Field(
        default_factory=list,
        description="Available snapshot versions for this study",
    )


class UploadResponse(BaseModel):
    """Response after successful data upload and ingestion."""
    status: str = Field(default="success")
    study_id: str = Field(..., description="Study identifier")
    domains_loaded: List[str] = Field(..., description="SDTM domains loaded")
    total_subjects: int = Field(..., description="Total subjects ingested")
    total_sites: int = Field(..., description="Total unique sites")
    message: str = Field(..., description="Descriptive status message")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    service: str = Field(default="TCRIE - Temporal Clinical Risk Intelligence Engine")
    version: str = Field(default="1.0.0")
    author: str = Field(default="Nampally Aryan")

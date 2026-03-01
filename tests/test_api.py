"""
TCRIE - API Endpoint Tests
Tests FastAPI endpoints using TestClient.
Author: Nampally Aryan
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


@pytest.fixture
def client():
    c = TestClient(app)
    c.headers["X-API-Key"] = "tcrie-dev-key-2026"
    return c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_payload(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["service"] == "TCRIE - Temporal Clinical Risk Intelligence Engine"
        assert data["author"] == "Nampally Aryan"


class TestDemoAnalysis:
    def test_demo_analysis_returns_200(self, client):
        """Full pipeline on CDISC pilot data should succeed."""
        response = client.get("/demo-analysis")
        assert response.status_code == 200

    def test_demo_analysis_structure(self, client):
        data = client.get("/demo-analysis").json()
        assert "study_id" in data
        assert "total_sites" in data
        assert "site_reports" in data
        assert data["total_sites"] > 0
        assert len(data["site_reports"]) == data["total_sites"]

    def test_site_reports_contain_required_fields(self, client):
        data = client.get("/demo-analysis").json()
        for report in data["site_reports"]:
            assert "site_id" in report
            assert "risk_score" in report
            assert "risk_level" in report
            assert "cusum_drift" in report
            assert "mahalanobis_distance" in report
            assert "risk_decomposition" in report
            assert "feature_contributions" in report
            assert 0 <= report["risk_score"] <= 100
            assert report["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            # Risk decomposition structure
            decomp = report["risk_decomposition"]
            assert "quality_score" in decomp
            assert "drift_score" in decomp
            assert "anomaly_score" in decomp


class TestTraceability:
    def test_config_hash_present(self, client):
        data = client.get("/demo-analysis").json()
        assert "config_hash" in data
        assert len(data["config_hash"]) == 64  # SHA-256 hex

    def test_version_label_present(self, client):
        data = client.get("/demo-analysis").json()
        assert "version_label" in data
        assert data["version_label"].startswith("v")

    def test_performance_metrics(self, client):
        data = client.get("/demo-analysis").json()
        assert "performance_metrics" in data
        pm = data["performance_metrics"]
        assert "total_sites" in pm
        assert "flagged_pct" in pm
        assert "mean_risk_score" in pm

    def test_risk_trajectory_populated(self, client):
        data = client.get("/demo-analysis").json()
        assert "risk_trajectory" in data
        assert len(data["risk_trajectory"]) >= 1

    def test_versions_list(self, client):
        data = client.get("/demo-analysis").json()
        assert "versions" in data
        assert len(data["versions"]) >= 1
        assert "version" in data["versions"][0]


class TestRiskReportEndpoint:
    def test_risk_report_not_found(self, client):
        response = client.get("/risk-report/NONEXISTENT")
        assert response.status_code == 404

    def test_risk_report_after_analysis(self, client):
        # First run analysis to populate cache
        client.get("/demo-analysis")
        response = client.get("/risk-report/CDISCPILOT01")
        assert response.status_code == 200

    def test_site_report_after_analysis(self, client):
        # Run analysis first
        data = client.get("/demo-analysis").json()
        first_site = data["site_reports"][0]["site_id"]
        response = client.get(f"/risk-report/CDISCPILOT01/{first_site}")
        assert response.status_code == 200
        assert response.json()["site_id"] == first_site

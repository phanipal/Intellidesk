"""Tests for the FastAPI triage service: endpoints, validation, and response schema."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.classifier import TicketClassifier
from src.config import CATEGORIES, PRIORITIES
from src.retriever import KBRetriever


def _artifacts_exist() -> bool:
    return (
        TicketClassifier.DEFAULT_MODEL_PATH.exists()
        and (KBRetriever.DEFAULT_INDEX_DIR / KBRetriever.INDEX_FILE).exists()
    )


pytestmark = pytest.mark.skipif(
    not _artifacts_exist(),
    reason="API tests need saved artifacts. Run "
           "'.\\make.ps1 train-quick' and '.\\make.ps1 build-index' first."
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    TestClient triggers FastAPI lifespan, which warms up the pipeline.
    Reuse client across tests so warm-up runs only once per module.
    """
    from src.api import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    def test_root_returns_service_info(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "service" in body
        assert "endpoints" in body

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ready_after_lifespan(self, client):
        r = client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["pipeline_loaded"] is True


class TestInfoEndpoint:
    def test_info_contains_metadata(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "intellidesk-api"
        assert "version" in body
        assert set(body["categories"]) == set(CATEGORIES)
        assert set(body["priorities"]) == set(PRIORITIES)
        assert body["kb_articles"] > 0

    def test_info_exposes_config(self, client):
        r = client.get("/info")
        config = r.json()["config"]
        assert "top_k" in config
        assert "category_confidence_threshold" in config
        assert "priority_confidence_threshold" in config
        assert "min_kb_score" in config


class TestTriageEndpoint:
    def test_triage_returns_valid_response(self, client):
        r = client.post("/triage", json={
            "text": "VPN keeps disconnecting from corporate network"
        })
        assert r.status_code == 200
        body = r.json()
        assert body["category"] in CATEGORIES
        assert body["priority"] in PRIORITIES
        assert 0.0 <= body["category_confidence"] <= 1.0
        assert 0.0 <= body["priority_confidence"] <= 1.0
        assert isinstance(body["kb_suggestions"], list)
        assert isinstance(body["needs_human_review"], bool)
        assert body["latency_ms"] > 0

    def test_triage_top_k_override(self, client):
        r = client.post("/triage", json={
            "text": "Major outage affecting users",
            "top_k": 1,
        })
        assert r.status_code == 200
        assert len(r.json()["kb_suggestions"]) <= 1

    def test_triage_rejects_empty_text(self, client):
        r = client.post("/triage", json={"text": ""})
        assert r.status_code == 422  # Pydantic validation error

    def test_triage_rejects_missing_text(self, client):
        r = client.post("/triage", json={})
        assert r.status_code == 422

    def test_triage_rejects_oversized_text(self, client):
        r = client.post("/triage", json={"text": "x" * 10_001})
        assert r.status_code == 422

    def test_triage_rejects_invalid_top_k(self, client):
        r = client.post("/triage", json={"text": "VPN issue", "top_k": 0})
        assert r.status_code == 422
        r = client.post("/triage", json={"text": "VPN issue", "top_k": 100})
        assert r.status_code == 422

    def test_triage_kb_suggestions_have_required_fields(self, client):
        r = client.post("/triage", json={"text": "VPN tunnel keeps dropping"})
        assert r.status_code == 200
        for kb in r.json()["kb_suggestions"]:
            assert "kb_id" in kb
            assert "title" in kb
            assert "category" in kb
            assert "score" in kb


class TestBatchTriageEndpoint:
    def test_batch_returns_correct_count(self, client):
        texts = ["VPN down", "Outlook crash", "Password reset"]
        r = client.post("/triage/batch", json={"texts": texts})
        assert r.status_code == 200
        body = r.json()
        assert body["total_count"] == 3
        assert len(body["results"]) == 3

    def test_batch_rejects_empty_list(self, client):
        r = client.post("/triage/batch", json={"texts": []})
        assert r.status_code == 422

    def test_batch_rejects_oversized(self, client):
        r = client.post("/triage/batch", json={"texts": ["x"] * 101})
        assert r.status_code == 422

    def test_batch_results_match_single_calls(self, client):
        """Batch endpoint must produce same predictions as /triage."""
        texts = [
            "VPN tunnel keeps dropping",
            "Outlook keeps crashing",
        ]
        single_results = [
            client.post("/triage", json={"text": t}).json()
            for t in texts
        ]
        batch = client.post("/triage/batch", json={"texts": texts}).json()
        for s, b in zip(single_results, batch["results"]):
            assert s["category"] == b["category"]
            assert s["priority"] == b["priority"]
            assert [k["kb_id"] for k in s["kb_suggestions"]] == \
                   [k["kb_id"] for k in b["kb_suggestions"]]


class TestOpenAPIDocs:
    def test_swagger_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc_available(self, client):
        r = client.get("/redoc")
        assert r.status_code == 200

    def test_openapi_schema_complete(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        for path in ["/triage", "/triage/batch", "/health", "/ready", "/info"]:
            assert path in schema["paths"], f"Missing OpenAPI path: {path}"

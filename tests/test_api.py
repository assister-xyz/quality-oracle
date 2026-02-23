"""Endpoint tests for Quality Oracle API."""
from unittest.mock import AsyncMock, patch


def test_health_no_auth(test_client):
    """Health endpoint should be public (no auth required)."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "quality-oracle"


def test_agent_card_no_auth(test_client):
    """Agent card endpoint should be public (A2A spec)."""
    resp = test_client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Quality Oracle"
    assert "skills" in data
    assert len(data["skills"]) >= 3


def test_badge_no_auth(test_client):
    """Badge endpoint should be public (embeddable in READMEs)."""
    resp = test_client.get("/v1/badge/test-server.svg")
    assert resp.status_code == 200
    assert "image/svg+xml" in resp.headers["content-type"]
    assert "<svg" in resp.text


def test_evaluate_no_auth_returns_422(test_client):
    """POST /v1/evaluate without X-API-Key should return 422 (missing header)."""
    resp = test_client.post("/v1/evaluate", json={
        "target_url": "http://localhost:8010/sse",
    })
    assert resp.status_code == 422


def test_evaluate_with_auth(test_client, auth_headers):
    """POST /v1/evaluate with valid auth should return 200 with evaluation_id."""
    with patch("src.api.v1.evaluate._run_evaluation", new_callable=AsyncMock):
        resp = test_client.post(
            "/v1/evaluate",
            json={"target_url": "http://localhost:8010/sse"},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "evaluation_id" in data
    assert data["status"] == "pending"
    assert "poll_url" in data


def test_evaluate_status_not_found(test_client, auth_headers):
    """GET /v1/evaluate/{id} for nonexistent eval should return 404."""
    resp = test_client.get(
        "/v1/evaluate/nonexistent-id",
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_score_not_found(test_client, auth_headers):
    """GET /v1/score/{target_id} for unknown target should return 404."""
    resp = test_client.get(
        "/v1/score/unknown-server",
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_scores_list(test_client, auth_headers):
    """GET /v1/scores should return paginated list."""
    resp = test_client.get("/v1/scores", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data


def test_attestation_not_found(test_client, auth_headers):
    """GET /v1/attestation/{id} for nonexistent should return 404."""
    resp = test_client.get(
        "/v1/attestation/nonexistent-id",
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_enrich_agent_card(test_client, auth_headers):
    """POST /v1/enrich-agent-card should return enriched card."""
    resp = test_client.post(
        "/v1/enrich-agent-card",
        json={"agent_card": {"name": "test-agent", "url": "http://test"}},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "enriched_card" in data

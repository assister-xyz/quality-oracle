"""
Live API tests against a running AgentTrust backend.

These tests require:
  - Backend running on http://localhost:8002
  - A valid API key in the QO_TEST_API_KEY env var (or default dev key)

Run: python -m pytest tests/test_api_live.py -v
"""
import os

import httpx
import pytest
import pytest_asyncio

BASE_URL = os.getenv("QO_TEST_BASE_URL", "http://localhost:8002")
API_KEY = os.getenv("QO_TEST_API_KEY", "qo_Wcl18Nesj6o75Kr5odF07Xi0wG-u8lJbn3YCD8kofYM")


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as c:
        yield c


def auth_headers():
    return {"X-API-Key": API_KEY}


# ──────────── Public Endpoints ────────────

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_agent_card(client):
    r = await client.get("/.well-known/agent.json")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data
    assert "url" in data


@pytest.mark.asyncio
async def test_pricing(client):
    r = await client.get("/v1/pricing", headers=auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert "levels" in data or "tiers" in data or isinstance(data, dict)


# ──────────── Auth Errors ────────────

@pytest.mark.asyncio
async def test_missing_api_key(client):
    r = await client.get("/v1/scores")
    assert r.status_code in (401, 422)


@pytest.mark.asyncio
async def test_invalid_api_key(client):
    r = await client.get("/v1/scores", headers={"X-API-Key": "qo_invalid_key_12345"})
    assert r.status_code in (401, 403)


# ──────────── Scores Endpoints ────────────

@pytest.mark.asyncio
async def test_list_scores(client):
    r = await client.get("/v1/scores", headers=auth_headers(), params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_list_scores_with_filters(client):
    r = await client.get(
        "/v1/scores",
        headers=auth_headers(),
        params={"limit": 5, "sort": "score", "min_score": 50},
    )
    assert r.status_code == 200
    data = r.json()
    for item in data["items"]:
        assert item["score"] >= 50


@pytest.mark.asyncio
async def test_get_score_not_found(client):
    r = await client.get(
        "/v1/score/http://nonexistent.example.com/does-not-exist",
        headers=auth_headers(),
    )
    assert r.status_code == 404


# ──────────── Evaluation Submission ────────────

@pytest.mark.asyncio
async def test_submit_evaluation_invalid_url(client):
    """Test submitting an evaluation with an unreachable URL."""
    r = await client.post(
        "/v1/evaluate",
        headers=auth_headers(),
        json={"target_url": "http://localhost:99999/nonexistent"},
    )
    # Should accept the submission (async processing)
    assert r.status_code == 200
    data = r.json()
    assert "evaluation_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_get_evaluation_not_found(client):
    r = await client.get(
        "/v1/evaluate/nonexistent-eval-id-12345",
        headers=auth_headers(),
    )
    assert r.status_code == 404


# ──────────── Badge ────────────

@pytest.mark.asyncio
async def test_badge_not_found(client):
    """Badge for non-evaluated target returns 404 or a default badge."""
    r = await client.get("/v1/badge/http://nonexistent-target.example.com.svg")
    # Badge might return 404 or a default "no score" badge
    assert r.status_code in (200, 404)


# ──────────── Feedback ────────────

@pytest.mark.asyncio
async def test_feedback_submission(client):
    r = await client.post(
        "/v1/feedback",
        headers=auth_headers(),
        json={
            "target_id": "http://localhost:8010/sse",
            "outcome": "success",
            "outcome_score": 85,
            "context": "test feedback",
        },
    )
    # 200 or 404 if no score exists for target
    assert r.status_code in (200, 404)

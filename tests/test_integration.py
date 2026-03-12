"""
Full integration test: submit evaluation against mock MCP server, poll to completion,
verify scores, badge, and feedback.

Requires:
  - Backend running on http://localhost:8002
  - Mock MCP server running on http://localhost:8010
  - MongoDB accessible

Run: python -m pytest tests/test_integration.py -v --timeout=120
"""
import asyncio
import os

import httpx
import pytest
import pytest_asyncio

BASE_URL = os.getenv("QO_TEST_BASE_URL", "http://localhost:8002")
API_KEY = os.getenv("QO_TEST_API_KEY", "qo_Wcl18Nesj6o75Kr5odF07Xi0wG-u8lJbn3YCD8kofYM")
MOCK_MCP_URL = os.getenv("QO_MOCK_MCP_URL", "http://localhost:8010/sse")


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as c:
        yield c


def auth_headers():
    return {"X-API-Key": API_KEY}


@pytest.mark.asyncio
async def test_full_evaluation_flow(client):
    """End-to-end: submit evaluation → poll → verify result → check score → badge → feedback."""

    # 1. Submit evaluation against mock MCP server (Level 1 for speed)
    r = await client.post(
        "/v1/evaluate",
        headers=auth_headers(),
        json={
            "target_url": MOCK_MCP_URL,
            "level": 1,
        },
    )
    assert r.status_code == 200, f"Submit failed: {r.text}"
    submit_data = r.json()
    evaluation_id = submit_data["evaluation_id"]
    assert submit_data["status"] == "pending"

    # 2. Poll until completed (max 60s)
    final_status = None
    for _ in range(30):
        await asyncio.sleep(2)
        r = await client.get(
            f"/v1/evaluate/{evaluation_id}",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        status_data = r.json()

        if status_data["status"] == "completed":
            final_status = status_data
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Evaluation failed: {status_data.get('error')}")

    assert final_status is not None, "Evaluation did not complete within 60s"

    # 3. Verify result structure
    assert final_status["score"] is not None
    assert final_status["score"] >= 0
    assert final_status["tier"] is not None
    assert final_status["report"] is not None
    assert "level1" in final_status["report"]
    assert final_status["report"]["level1"]["manifest_score"] >= 0

    # Result should have proper structure
    if final_status.get("result"):
        result = final_status["result"]
        assert result["target_id"] == MOCK_MCP_URL
        assert result["score"] >= 0

    # 4. Verify score is in the scores collection
    target_id = MOCK_MCP_URL
    r = await client.get(
        f"/v1/score/{target_id}",
        headers=auth_headers(),
    )
    assert r.status_code == 200, f"Score lookup failed: {r.text}"
    score_data = r.json()
    assert score_data["target_id"] == target_id
    assert score_data["score"] >= 0
    assert score_data["tier"] in ("expert", "proficient", "basic", "failed")

    # 5. Verify the target shows up in list_scores
    r = await client.get(
        "/v1/scores",
        headers=auth_headers(),
        params={"limit": 100},
    )
    assert r.status_code == 200
    scores_list = r.json()
    target_ids = [item["target_id"] for item in scores_list["items"]]
    assert target_id in target_ids

    # 6. Verify list_scores enrichment fields
    item = next(i for i in scores_list["items"] if i["target_id"] == target_id)
    assert "confidence" in item
    assert "last_evaluation_id" in item

    # 7. Verify badge renders
    r = await client.get(f"/v1/badge/{target_id}.svg")
    assert r.status_code == 200
    assert "svg" in r.headers.get("content-type", "").lower() or r.text.strip().startswith("<svg")

    # 8. Submit feedback and check correlation endpoint
    r = await client.post(
        "/v1/feedback",
        headers=auth_headers(),
        json={
            "target_id": target_id,
            "outcome": "success",
            "outcome_score": 80,
            "context": "integration test",
        },
    )
    assert r.status_code == 200, f"Feedback failed: {r.text}"
    feedback_data = r.json()
    assert feedback_data["target_id"] == target_id

    # Check correlation
    r = await client.get(
        f"/v1/correlation/{target_id}",
        headers=auth_headers(),
    )
    assert r.status_code == 200
    corr_data = r.json()
    assert corr_data["target_id"] == target_id
    assert corr_data["feedback_count"] >= 1


@pytest.mark.asyncio
async def test_level2_evaluation(client):
    """Submit a Level 2 evaluation against mock MCP server."""

    r = await client.post(
        "/v1/evaluate",
        headers=auth_headers(),
        json={
            "target_url": MOCK_MCP_URL,
            "level": 2,
        },
    )
    assert r.status_code == 200
    evaluation_id = r.json()["evaluation_id"]

    # Poll until complete (Level 2 takes longer)
    final_status = None
    for _ in range(60):
        await asyncio.sleep(2)
        r = await client.get(
            f"/v1/evaluate/{evaluation_id}",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        status_data = r.json()

        if status_data["status"] == "completed":
            final_status = status_data
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Level 2 evaluation failed: {status_data.get('error')}")

    assert final_status is not None, "Level 2 evaluation did not complete within 120s"
    assert final_status["report"] is not None
    assert "level2" in final_status["report"]

    # Level 2 should have tool details
    level2 = final_status["report"]["level2"]
    if level2:
        assert "tools_tested" in level2
        assert level2["tools_tested"] > 0

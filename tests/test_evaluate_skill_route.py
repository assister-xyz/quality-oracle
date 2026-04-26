"""Tests for POST /v1/evaluate/skill (QO-060 / E2E_TEST_REPORT fix).

The frontend (`quality-oracle-demo/src/lib/api.ts:submitSkillEvaluation`)
posts a SKILL.md bundle in-memory; the route materialises it to a temp
dir and dispatches through `_run_evaluation_skill`. Surfaced by Phase 3
real-stack Playwright as a 405 Method Not Allowed before this fix.
"""
from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from src.main import app


@pytest.fixture
def api_key_header():
    """Bypass auth via the same pattern as the rest of the test suite."""
    return {"X-API-Key": "test-key"}


@pytest.fixture(autouse=True)
def _stub_auth_and_dependencies(monkeypatch):
    """Replace get_api_key + rate limiter so route is exercisable in pytest."""
    from src.api.v1 import evaluate as evaluate_module

    async def _fake_get_api_key():
        return {"_id": "fakehash" * 4, "tier": "developer", "owner_email": "test@example.com"}

    async def _fake_rate_limit(*args, **kwargs):
        return True, 100, 100

    def _fake_level_allowed(tier, level):
        return True

    app.dependency_overrides[evaluate_module.get_api_key] = _fake_get_api_key
    monkeypatch.setattr(evaluate_module, "check_eval_rate_limit", _fake_rate_limit)
    monkeypatch.setattr(evaluate_module, "is_eval_level_allowed", _fake_level_allowed)

    # Suppress all DB calls so pytest doesn't dangle on a real eval.
    class _FakeCollection:
        async def insert_one(self, *args, **kwargs):
            return None

        async def update_one(self, *args, **kwargs):
            return None

        async def find_one(self, *args, **kwargs):
            return None

    monkeypatch.setattr(evaluate_module, "evaluations_col", lambda: _FakeCollection())
    # Disable the background task entirely so the route returns clean.
    async def _noop_run_evaluation(*args, **kwargs):
        return None

    monkeypatch.setattr(evaluate_module, "_run_evaluation", _noop_run_evaluation)

    yield

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_post_evaluate_skill_returns_202_with_evaluation_id():
    """Smoke test: POST /v1/evaluate/skill no longer 405s; returns 200 + evaluation_id."""
    payload = {
        "frontmatter": {
            "name": "test-skill",
            "description": "A minimal test skill bundle.",
        },
        "body": "# Test Skill\n\nMinimal content.\n",
        "source": "drag",
        "filename": "test-skill.md",
        "level": 1,  # MANIFEST
        "eval_mode": "verified",
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/evaluate/skill", json=payload, headers={"X-API-Key": "test"})

    assert resp.status_code != 405, "Backend regressed to the original 405 bug"
    assert resp.status_code == 200, f"unexpected status: {resp.status_code}, body={resp.text}"

    body = resp.json()
    assert "evaluation_id" in body
    assert body["status"] == "pending"
    assert body["poll_url"].startswith("/v1/evaluate/")


@pytest.mark.asyncio
async def test_post_evaluate_skill_materialises_bundle_to_disk():
    """The route writes the SKILL.md bundle to a temp dir before dispatching."""
    payload = {
        "frontmatter": {
            "name": "materialise-check",
            "description": "Verify materialisation creates SKILL.md file.",
        },
        "body": "# Materialise Check\n",
        "source": "drag",
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/evaluate/skill", json=payload, headers={"X-API-Key": "test"})

    assert resp.status_code == 200
    eval_id = resp.json()["evaluation_id"]

    import tempfile
    skill_md = Path(tempfile.gettempdir()) / "laureum-skill-uploads" / eval_id / "materialise-check" / "SKILL.md"
    assert skill_md.exists(), f"Bundle not materialised: {skill_md}"
    content = skill_md.read_text()
    assert "name: materialise-check" in content
    assert "Materialise Check" in content


@pytest.mark.asyncio
async def test_post_evaluate_skill_rejects_missing_required_fields():
    """Pydantic validation: missing `frontmatter` or `body` returns 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/evaluate/skill", json={"source": "drag"}, headers={"X-API-Key": "test"})

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_evaluate_skill_defaults_eval_mode_and_level():
    """Optional fields default sensibly: level=MANIFEST, eval_mode=verified."""
    payload = {
        "frontmatter": {"name": "defaults-check", "description": "x" * 30},
        "body": "body",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/evaluate/skill", json=payload, headers={"X-API-Key": "test"})

    assert resp.status_code == 200
    assert resp.json()["estimated_time_seconds"] == 5  # MANIFEST default

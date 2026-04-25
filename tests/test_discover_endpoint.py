"""QO-058 §"API endpoint": GET /v1/discover?url=X.

Returns the inferred protocol + capability preview without running the full
evaluation (so it stays cheap — ≤12s p99 cold-cache).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.core.evaluation_target import UnknownTargetError
from src.core.target_resolver import DiscoveryResult
from src.main import app
from src.storage.models import TargetType


@pytest.fixture
def client():
    return TestClient(app)


def test_discover_returns_a2a_for_agent_card(client):
    from src.core.a2a_target import A2ATarget

    fake_card = {
        "id": "x", "name": "X agent", "skills": [
            {"id": "s1", "name": "Skill 1", "description": "d", "acceptedInputTypes": ["text/plain"]},
        ],
    }
    fake_target = A2ATarget(endpoint_url="https://x.example", card=fake_card)
    fake_meta = DiscoveryResult(
        target_type=TargetType.A2A_AGENT,
        endpoint_url="https://x.example",
        raw=fake_card,
        note="a2a_v1.0",
    )

    async def _fake_resolve(url, *, cache=None, judge=None, return_meta=False):
        return (fake_target, fake_meta) if return_meta else fake_target

    with patch("src.api.v1.discover.resolve_target", new=_fake_resolve):
        r = client.get("/v1/discover?url=https://x.example")

    assert r.status_code == 200
    data = r.json()
    assert data["target_type"] == "a2a_agent"
    assert data["endpoint_url"] == "https://x.example"
    assert len(data["capabilities"]) == 1
    assert data["capabilities"][0]["id"] == "s1"
    assert data["capabilities"][0]["accepted_input_types"] == ["text/plain"]


def test_discover_returns_rest_chat(client):
    from src.core.rest_chat_target import RESTChatTarget

    fake_target = RESTChatTarget(endpoint_url="https://chat.example")
    fake_meta = DiscoveryResult(
        target_type=TargetType.REST_CHAT,
        endpoint_url="https://chat.example",
        note="rest_chat_heuristic",
    )

    async def _fake_resolve(url, *, cache=None, judge=None, return_meta=False):
        return (fake_target, fake_meta) if return_meta else fake_target

    with patch("src.api.v1.discover.resolve_target", new=_fake_resolve):
        r = client.get("/v1/discover?url=https://chat.example")

    assert r.status_code == 200
    data = r.json()
    assert data["target_type"] == "rest_chat"
    # REST chat preview is the synthetic chat capability, NOT discovered
    # via the calibration path (would cost $$).
    assert any(c["id"] == "chat" for c in data["capabilities"])


def test_discover_422_when_unknown(client):
    async def _fake_resolve(url, *, cache=None, judge=None, return_meta=False):
        raise UnknownTargetError("Couldn't auto-detect protocol for https://x")

    with patch("src.api.v1.discover.resolve_target", new=_fake_resolve):
        r = client.get("/v1/discover?url=https://nothing.example")

    assert r.status_code == 422
    assert "auto-detect" in r.json()["detail"]


def test_discover_502_on_unexpected_error(client):
    async def _fake_resolve(url, *, cache=None, judge=None, return_meta=False):
        raise RuntimeError("boom")

    with patch("src.api.v1.discover.resolve_target", new=_fake_resolve):
        r = client.get("/v1/discover?url=https://x.example")

    assert r.status_code == 502


def test_discover_requires_url_param(client):
    r = client.get("/v1/discover")
    assert r.status_code == 422  # FastAPI validation

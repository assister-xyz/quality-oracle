"""QO-058 AC8 + AC9 + AC10: discovery cascade resolver.

Coverage:
* All 10 probes execute concurrently (asyncio.gather).
* Specificity-priority ranking — agent-card.json beats /openapi.json.
* Gradio + OpenAPI ambiguity resolves to Gradio's deferred branch.
* AC8 fail path — every probe fails, UnknownTargetError.
* AC9 — MCP URL resolves to MCPTarget.
* AC10 — cold-cache p99 ≤12s (we measure with mock probes).
* 24h cache hit short-circuits.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict
from unittest.mock import AsyncMock

import httpx
import pytest

from src.core.evaluation_target import UnknownTargetError
from src.core.target_resolver import (
    DISCOVERY_PROBES,
    DiscoveryResult,
    _serialize,
    resolve,
)
from src.storage.models import TargetType


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_route_handler(routes: Dict[str, httpx.Response]):
    """Map (method, path) → response. Default 404. EXACT path match only —
    suffix-matching would conflate ``/gradio_api/openapi.json`` with
    ``/openapi.json`` and break specificity tests."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in routes:
            return routes[path]
        return httpx.Response(404)
    return handler


@pytest.fixture
def patch_async_client(monkeypatch):
    """Patch httpx.AsyncClient inside target_resolver to use MockTransport."""
    def _apply(handler):
        # Wrap the regular AsyncClient — keep timeouts intact since tests
        # rely on per-probe deadlines.
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original(*args, **kwargs)

        monkeypatch.setattr("src.core.target_resolver.httpx.AsyncClient", _factory)
    return _apply


# ── AC9: MCP URL resolves to MCPTarget ──────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_url_resolves_to_mcp_target(patch_async_client):
    """AC9 indirect — explicit /sse path triggers mcp_handshake probe."""
    routes = {
        "/sse": httpx.Response(200, headers={"content-type": "text/event-stream"}),
    }
    patch_async_client(_make_route_handler(routes))
    target = await resolve("https://mcp.example.com/sse")
    assert target.target_type == TargetType.MCP_SERVER


# ── A2A wins over OpenAPI when both present ─────────────────────────────────


@pytest.mark.asyncio
async def test_a2a_wins_over_openapi(patch_async_client):
    """Specificity ranking: agent-card.json (priority 1) beats /openapi.json (priority 6)."""
    card = {"id": "x", "name": "x", "skills": []}
    openapi_doc = {"openapi": "3.0.0", "info": {"title": "x"}}
    routes = {
        "/.well-known/agent-card.json": httpx.Response(200, json=card),
        "/openapi.json": httpx.Response(200, json=openapi_doc),
    }
    patch_async_client(_make_route_handler(routes))
    target = await resolve("https://x.example")
    assert target.target_type == TargetType.A2A_AGENT


# ── Gradio ambiguity (Gradio at /openapi.json AND /gradio_api/openapi.json) ─


@pytest.mark.asyncio
async def test_gradio_deferred_routes_to_unknown_error(patch_async_client):
    """Gradio Spaces expose BOTH /openapi.json AND /gradio_api/openapi.json.
    The cascade should match the Gradio probe (priority 5) BEFORE the generic
    OpenAPI probe (priority 6) — and since Gradio is deferred to QO-070, it
    surfaces as UnknownTargetError rather than mis-routing to OPENAPI_AGENT."""
    openapi_doc = {"openapi": "3.0.0"}
    routes = {
        "/gradio_api/openapi.json": httpx.Response(200, json=openapi_doc),
        "/openapi.json": httpx.Response(200, json=openapi_doc),
    }
    patch_async_client(_make_route_handler(routes))
    with pytest.raises(UnknownTargetError, match="deferred"):
        await resolve("https://hf.example")


# ── REST chat fallback ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rest_chat_heuristic_last_resort(patch_async_client):
    """AC8 step 10: when all 9 specific probes fail, heuristic POST wins."""

    def handler(request):
        # No well-known endpoints; only POST to root succeeds.
        if request.method == "POST" and request.url.path in ("/", ""):
            return httpx.Response(200, json={"reply": "hi"})
        return httpx.Response(404)

    patch_async_client(handler)
    target = await resolve("https://chat.example")
    assert target.target_type == TargetType.REST_CHAT


@pytest.mark.asyncio
async def test_unknown_target_error_when_all_fail(patch_async_client):
    """AC8 fail branch: heuristic also fails → UnknownTargetError."""

    def handler(request):
        return httpx.Response(404)

    patch_async_client(handler)
    with pytest.raises(UnknownTargetError):
        await resolve("https://nothing.example")


# ── OpenAPI agent (no Gradio) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openapi_resolves_to_openapi_agent(patch_async_client):
    routes = {
        "/openapi.json": httpx.Response(200, json={"openapi": "3.0.0", "info": {}}),
    }
    patch_async_client(_make_route_handler(routes))
    target = await resolve("https://api.example")
    assert target.target_type == TargetType.OPENAPI_AGENT


@pytest.mark.asyncio
async def test_swagger_alt_path(patch_async_client):
    routes = {
        "/swagger.json": httpx.Response(200, json={"swagger": "2.0", "info": {}}),
    }
    patch_async_client(_make_route_handler(routes))
    target = await resolve("https://legacy.example")
    assert target.target_type == TargetType.OPENAPI_AGENT


# ── agents.json (multi-agent index) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_agents_json_picks_first_agent(patch_async_client):
    routes = {
        "/.well-known/agents.json": httpx.Response(200, json={
            "agents": [
                {"id": "first", "name": "First", "endpoint": "https://first.example"},
                {"id": "second", "name": "Second"},
            ]
        }),
    }
    patch_async_client(_make_route_handler(routes))
    target = await resolve("https://hub.example")
    assert target.target_type == TargetType.A2A_AGENT


# ── concurrency / latency (AC10) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probes_run_concurrently_not_sequentially(patch_async_client, monkeypatch):
    """AC10 concurrency: 10 probes that each take 1s must finish in <12s.

    Sequential would be 10s+; concurrent should be ~1s.
    """
    # MockTransport requires sync handler; emulate latency at the asyncio level
    # by monkey-patching the probe table to return a sleeping coroutine.
    def slow_handler(request):
        # We can't sleep inside a sync MockTransport handler, so we use a
        # different latency-injection point: monkeypatch _run_probe_with_timeout.
        return httpx.Response(404)

    patch_async_client(slow_handler)

    # Patch each probe to sleep ~0.5s before returning None — concurrent run
    # finishes in ~0.5s, sequential would take 5s.
    async def slow_probe(base, *, client):
        await asyncio.sleep(0.5)
        return None

    from src.core import target_resolver as tr
    new_table = [(name, slow_probe) for name, _ in tr.DISCOVERY_PROBES]
    monkeypatch.setattr(tr, "DISCOVERY_PROBES", new_table)

    start = time.time()
    with pytest.raises(UnknownTargetError):
        await resolve("https://x.example")
    elapsed = time.time() - start
    # Concurrent: 10 × 0.5s should finish in well under 2s (allow generous slack).
    assert elapsed < 3.0, f"Probes appear to run sequentially: elapsed={elapsed:.2f}s"


@pytest.mark.asyncio
async def test_per_probe_timeout_fires(patch_async_client, monkeypatch):
    """AC10: any single probe exceeding ~11s yields None instead of blackholing."""
    from src.core import target_resolver as tr

    async def hang_probe(base, *, client):
        await asyncio.sleep(30)  # would block forever
        return None

    new_table = [(name, hang_probe) for name, _ in tr.DISCOVERY_PROBES]
    monkeypatch.setattr(tr, "DISCOVERY_PROBES", new_table)
    monkeypatch.setattr(tr, "PROBE_HARD_DEADLINE", 0.5)

    start = time.time()
    with pytest.raises(UnknownTargetError):
        await resolve("https://x.example")
    elapsed = time.time() - start
    # 10 concurrent hangs all hit the 0.5s cap; resolver returns under 1s.
    assert elapsed < 2.0, f"per-probe timeout did not fire: {elapsed:.2f}s"


# ── cache ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_short_circuits(patch_async_client):
    """24h cache hit: skip the cascade entirely."""
    fake_result = DiscoveryResult(
        target_type=TargetType.A2A_AGENT,
        endpoint_url="https://cached.example",
        raw={"id": "x", "name": "Cached"},
        note="cached",
    )
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=_serialize(fake_result))
    cache.setex = AsyncMock()

    # Even with a "404 everywhere" handler, we should never hit the cascade
    # because the cache short-circuits.
    def handler(request):
        return httpx.Response(404)

    patch_async_client(handler)
    target = await resolve("https://cached.example", cache=cache)
    assert target.target_type == TargetType.A2A_AGENT
    cache.get.assert_awaited_once()
    cache.setex.assert_not_called()


@pytest.mark.asyncio
async def test_cache_write_on_success(patch_async_client):
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.setex = AsyncMock()
    routes = {
        "/.well-known/agent-card.json": httpx.Response(200, json={
            "id": "x", "name": "x", "skills": []
        }),
    }
    patch_async_client(_make_route_handler(routes))
    await resolve("https://x.example", cache=cache)
    cache.setex.assert_awaited_once()
    args = cache.setex.call_args
    # ttl is 24h
    assert args.args[1] == 86_400


# ── 10-step ordering ────────────────────────────────────────────────────────


def test_ten_probes_in_specificity_order():
    """Spec §"DISCOVERY_PROBES order = specificity-priority"."""
    names = [name for name, _ in DISCOVERY_PROBES]
    expected = [
        "a2a_agent_card", "agents_json", "erc8004", "mcp_handshake",
        "gradio_api", "openapi", "openapi_alt", "openai_responses",
        "langserve", "rest_chat_heuristic",
    ]
    assert names == expected
    assert len(names) == 10

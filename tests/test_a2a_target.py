"""QO-058 AC1-AC3: A2A target adapter against a mock A2A server.

Uses ``httpx.MockTransport`` so the test is hermetic and fast (no socket).
Covers:
* AC1 — discovery via ``/.well-known/agent-card.json``.
* AC2 — capability enumeration with MIME types (A2A's delta vs MCP).
* AC3 — JSON-RPC ``message/send`` invoke.
* parse_security_schemes for API key, OAuth2, mTLS, OIDC.
"""
from __future__ import annotations

import json

import httpx
import pytest

from src.core.a2a_target import A2ATarget, parse_security_schemes
from src.core.evaluation_target import AuthDescriptor
from src.storage.models import TargetType


_FIXTURE_CARD = {
    "id": "weather-agent",
    "name": "Weather Agent",
    "description": "Reports weather worldwide",
    "skills": [
        {
            "id": "current_weather",
            "name": "Current weather",
            "description": "Get current temperature",
            "acceptedInputTypes": ["text/plain", "application/json"],
            "producedOutputTypes": ["application/json"],
        },
        {
            "id": "forecast",
            "name": "Forecast",
            "description": "5-day forecast",
            "acceptedInputTypes": ["text/plain"],
        },
        {
            "id": "alerts",
            "name": "Severe weather alerts",
            "description": "",
        },
    ],
    "securitySchemes": {
        "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
    },
    "signature": "ed25519:abc123",
}


def _make_handler(card=_FIXTURE_CARD, response_text="It's 22°C and sunny."):
    """Build a handler that serves the card + JSON-RPC `message/send`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/.well-known/agent-card.json"):
            return httpx.Response(200, json=card)
        if request.method == "POST":
            body = json.loads(request.content.decode())
            assert body["jsonrpc"] == "2.0"
            assert body["method"] == "message/send"
            return httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "message": {
                        "role": "assistant",
                        "parts": [{"contentType": "text/plain", "text": response_text}],
                    }
                },
            })
        return httpx.Response(404)

    return handler


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_a2a_discovers_card_and_three_capabilities():
    """AC1 + AC2: agent-card.json → 3 capabilities with MIME types."""
    async with _client(_make_handler()) as client:
        target = A2ATarget(endpoint_url="https://weather.example", client=client)
        manifest = await target.discover()

    assert manifest.id == "weather-agent"
    assert manifest.name == "Weather Agent"
    assert len(manifest.capabilities) == 3
    # AC2 — MIME types from acceptedInputTypes
    assert manifest.capabilities[0].accepted_input_types == ["text/plain", "application/json"]
    assert manifest.capabilities[0].input_schema is None  # A2A uses MIME, not JSON Schema
    assert manifest.signature == "ed25519:abc123"


@pytest.mark.asyncio
async def test_a2a_invoke_jsonrpc_message_send():
    """AC3: invoke posts JSON-RPC 2.0 `message/send` and returns text."""
    async with _client(_make_handler(response_text="The weather is fine.")) as client:
        target = A2ATarget(endpoint_url="https://weather.example", client=client)
        await target.discover()
        result = await target.invoke("current_weather", {"message": "What's the weather?"})

    assert result.status == "ok"
    assert result.text == "The weather is fine."
    assert result.invocation_id  # populated
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_a2a_invoke_handles_jsonrpc_error_payload():
    """An A2A error response (non-200 OR error key) maps to status='error'."""
    def handler(request):
        if "/.well-known/" in str(request.url):
            return httpx.Response(200, json=_FIXTURE_CARD)
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": "x",
            "error": {"code": -32602, "message": "Invalid params"},
        })

    async with _client(handler) as client:
        target = A2ATarget(endpoint_url="https://w.example", client=client)
        await target.discover()
        result = await target.invoke("current_weather", {"message": "?"})
    assert result.status == "error"
    assert "Invalid params" in (result.error or "")


@pytest.mark.asyncio
async def test_a2a_authenticate_with_env_token(monkeypatch):
    monkeypatch.setenv("QO_TARGET_API_KEY", "secret")
    auth = AuthDescriptor(style="x_api_key", header_name="X-API-Key")
    target = A2ATarget(endpoint_url="https://x.example", auth=auth)
    ctx = await target.authenticate()
    assert ctx.headers["X-API-Key"] == "secret"


@pytest.mark.asyncio
async def test_a2a_authenticate_no_token_no_headers(monkeypatch):
    monkeypatch.delenv("QO_TARGET_API_KEY", raising=False)
    auth = AuthDescriptor(style="bearer")
    target = A2ATarget(endpoint_url="https://x.example", auth=auth)
    ctx = await target.authenticate()
    assert ctx.headers == {}


def test_parse_security_schemes_apikey_header():
    desc = parse_security_schemes({"k": {"type": "apiKey", "in": "header", "name": "X-API-Key"}})
    assert desc is not None and desc.style == "x_api_key"
    assert desc.header_name == "X-API-Key"


def test_parse_security_schemes_apikey_query():
    desc = parse_security_schemes({"k": {"type": "apiKey", "in": "query", "name": "apikey"}})
    assert desc.style == "apikey_query"
    assert desc.query_param == "apikey"


def test_parse_security_schemes_bearer():
    desc = parse_security_schemes({"k": {"type": "http", "scheme": "bearer"}})
    assert desc.style == "bearer"


def test_parse_security_schemes_oauth2():
    desc = parse_security_schemes({"oauth": {
        "type": "oauth2",
        "flows": {
            "clientCredentials": {
                "tokenUrl": "https://auth.example/token",
                "scopes": {"read": "...", "write": "..."},
            }
        }
    }})
    assert desc.style == "oauth2"
    assert desc.token_url == "https://auth.example/token"
    assert set(desc.scopes) == {"read", "write"}


def test_parse_security_schemes_mtls():
    desc = parse_security_schemes({"k": {"type": "mutualTLS"}})
    assert desc.style == "mtls"


def test_parse_security_schemes_oidc():
    desc = parse_security_schemes({"k": {"type": "openIdConnect", "openIdConnectUrl": "https://x/.well-known"}})
    assert desc.style == "oidc"


def test_parse_security_schemes_empty():
    assert parse_security_schemes(None) is None
    assert parse_security_schemes({}) is None


@pytest.mark.asyncio
async def test_a2a_get_provenance():
    async with _client(_make_handler()) as client:
        target = A2ATarget(endpoint_url="https://w.example", client=client)
        await target.discover()
        await target.invoke("current_weather", {"message": "Hi"})
        prov = target.get_provenance()
    assert prov["target_type"] == TargetType.A2A_AGENT.value
    assert prov["request_count"] >= 1
    assert prov["card_signature"] == "ed25519:abc123"


@pytest.mark.asyncio
async def test_a2a_streaming_deferred_but_signature_locked():
    """Spec §"WS/SSE deferred": stream raises NotImplementedError but the
    type contract (AsyncIterator[Chunk]) holds."""
    target = A2ATarget(endpoint_url="https://x.example")
    with pytest.raises(NotImplementedError):
        async for _ in target.stream("x", {}):
            pass

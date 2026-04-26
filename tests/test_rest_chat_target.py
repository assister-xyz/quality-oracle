"""QO-058 AC4-AC7 + AC11: REST chat manifest-less target.

Covers:
* AC4 — calibration prompts → soft manifest with confidence='medium'.
* AC5 — invoke tries fallback shapes; first 200 wins.
* AC11 — ≥2-of-3 calibration failures raise SchemaUnobtainableError.
* All 8 fallback shapes — round-trip per shape.
* Provenance includes winning shape + tier_cap_reason.
"""
from __future__ import annotations

import json

import httpx
import pytest

from src.core.evaluation_target import InvokeError, SchemaUnobtainableError
from src.core.rest_chat_target import (
    REST_CHAT_AUTH_STYLES,
    REST_CHAT_REQUEST_SHAPES,
    RESTChatTarget,
    _extract_chat_reply,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_handler(*, accept_shape: str = "message", reply_text: str = "Hello!"):
    """Return an httpx handler that ONLY accepts ``accept_shape``."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if accept_shape == "message" and "message" in body:
            return httpx.Response(200, json={"reply": reply_text})
        if accept_shape == "prompt" and "prompt" in body:
            return httpx.Response(200, json={"output": reply_text})
        if accept_shape == "input" and "input" in body:
            return httpx.Response(200, json={"response": reply_text})
        if accept_shape == "messages" and "messages" in body:
            return httpx.Response(200, json={"choices": [{"message": {"content": reply_text}}]})
        if accept_shape == "query" and "query" in body:
            return httpx.Response(200, json={"answer": reply_text})
        if accept_shape == "question" and "question" in body:
            return httpx.Response(200, json={"text": reply_text})
        if accept_shape == "text" and "text" in body:
            return httpx.Response(200, json={"generated_text": reply_text})
        if accept_shape == "inputs" and "inputs" in body:
            return httpx.Response(200, json={"content": reply_text})
        return httpx.Response(422, json={"error": "wrong shape"})

    return handler


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── fallback shape tests ────────────────────────────────────────────────────


@pytest.mark.parametrize("shape_name,_", REST_CHAT_REQUEST_SHAPES)
@pytest.mark.asyncio
async def test_invoke_finds_correct_shape(shape_name, _):
    """Each of the 8 fallback shapes is exercised end-to-end."""
    async with _client(_make_handler(accept_shape=shape_name)) as client:
        target = RESTChatTarget(endpoint_url="https://x.example/chat", client=client)
        result = await target.invoke("chat", {"message": "Hi"})
    assert result.status == "ok"
    assert "Hello" in result.text
    assert result.request_shape == shape_name


@pytest.mark.asyncio
async def test_invoke_caches_winning_shape():
    """After first successful call, subsequent invokes use the winning shape directly."""
    call_count = {"posts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["posts"] += 1
        body = json.loads(request.content.decode())
        if "prompt" in body:
            return httpx.Response(200, json={"reply": "ok"})
        return httpx.Response(422)

    async with _client(handler) as client:
        target = RESTChatTarget(endpoint_url="https://x", client=client)
        await target.invoke("chat", {"message": "1"})
        first_count = call_count["posts"]
        await target.invoke("chat", {"message": "2"})
    # Second call should be ONE post (cached winning shape), not 8.
    assert call_count["posts"] - first_count == 1


@pytest.mark.asyncio
async def test_invoke_rejects_non_chat_capability():
    target = RESTChatTarget(endpoint_url="https://x")
    with pytest.raises(InvokeError, match="synthetic 'chat' capability"):
        await target.invoke("not-chat", {"message": "x"})


@pytest.mark.asyncio
async def test_invoke_raises_when_all_shapes_fail():
    """When every shape returns 422, raise InvokeError with last error in message."""

    def handler(request):
        return httpx.Response(422, json={"error": "no"})

    async with _client(handler) as client:
        target = RESTChatTarget(endpoint_url="https://x", client=client)
        with pytest.raises(InvokeError, match="shapes failed"):
            await target.invoke("chat", {"message": "Hi"})


# ── discover / calibration / AC11 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_discover_returns_medium_confidence_on_success():
    """AC4: 3-of-3 calibration prompts succeed → confidence='medium'."""
    async with _client(_make_handler(accept_shape="message")) as client:
        target = RESTChatTarget(endpoint_url="https://x", client=client)
        manifest = await target.discover()
    assert manifest.confidence == "medium"
    assert manifest.capabilities[0].id == "chat"


@pytest.mark.asyncio
async def test_discover_raises_on_two_failures():
    """AC11: ≥2 of 3 calibration prompts fail → SchemaUnobtainableError."""
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        # Reject ALL shapes — every calibration call exhausts and raises InvokeError
        return httpx.Response(500)

    async with _client(handler) as client:
        target = RESTChatTarget(endpoint_url="https://x", client=client)
        with pytest.raises(SchemaUnobtainableError):
            await target.discover()


@pytest.mark.asyncio
async def test_discover_one_failure_still_passes():
    """1 of 3 failures: still 'medium' confidence."""
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        body = json.loads(request.content.decode())
        # Fail the FIRST request only (the greeting). Subsequent succeed.
        if state["calls"] <= len(REST_CHAT_REQUEST_SHAPES):
            return httpx.Response(500)
        if "message" in body:
            return httpx.Response(200, json={"reply": "ok"})
        return httpx.Response(422)

    async with _client(handler) as client:
        target = RESTChatTarget(endpoint_url="https://x", client=client)
        manifest = await target.discover()
    assert manifest.confidence == "medium"


# ── auth ────────────────────────────────────────────────────────────────────


def test_five_auth_styles_enumerated():
    """Spec §M1: 5 auth styles."""
    assert len(REST_CHAT_AUTH_STYLES) == 5
    for s in ("bearer", "x_api_key", "api_key_header", "apikey_query", "cookie_session"):
        assert s in REST_CHAT_AUTH_STYLES


@pytest.mark.asyncio
async def test_eight_request_shapes_enumerated():
    """Spec §M1: 8 request shapes."""
    assert len(REST_CHAT_REQUEST_SHAPES) == 8


# ── provenance + extract_chat_reply ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_provenance_after_invoke():
    async with _client(_make_handler()) as client:
        target = RESTChatTarget(endpoint_url="https://x", client=client)
        await target.invoke("chat", {"message": "Hi"})
        prov = target.get_provenance()
    assert prov["winning_shape"] == "message"
    # Inference confidence is None until discover() runs — tier_cap_reason
    # must still indicate "no_manifest" (cap default for REST chat).
    assert prov["tier_cap_reason"] == "no_manifest"


@pytest.mark.asyncio
async def test_provenance_after_discover_sets_confidence():
    async with _client(_make_handler()) as client:
        target = RESTChatTarget(endpoint_url="https://x", client=client)
        await target.discover()
        prov = target.get_provenance()
    assert prov["inference_confidence"] == "medium"


def test_extract_chat_reply_handles_openai_shape():
    payload = {"choices": [{"message": {"content": "Hello world"}}]}
    assert _extract_chat_reply(payload) == "Hello world"


def test_extract_chat_reply_handles_hf_shape():
    payload = {"generated_text": "from huggingface"}
    assert _extract_chat_reply(payload) == "from huggingface"


def test_extract_chat_reply_handles_string():
    assert _extract_chat_reply("plain string") == "plain string"


def test_extract_chat_reply_falls_back_to_json():
    """Unknown shape: still returns SOMETHING so judges have material."""
    out = _extract_chat_reply({"weird": {"nested": [1, 2]}})
    assert "weird" in out  # serialised as JSON


@pytest.mark.asyncio
async def test_streaming_deferred():
    target = RESTChatTarget(endpoint_url="https://x")
    with pytest.raises(NotImplementedError):
        async for _ in target.stream("chat", {}):
            pass

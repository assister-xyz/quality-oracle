"""A2ATarget — Google Agent2Agent v1.0 (Mar 2026) adapter.

Implements :class:`src.core.evaluation_target.EvaluationTarget` against an
A2A v1.0 agent that publishes ``/.well-known/agent-card.json`` and accepts
JSON-RPC 2.0 calls (``message/send`` for invoke, ``message/stream`` for the
streaming surface — the SSE wire format is deferred to v2 per spec §"WS/SSE
deferred").

A2A's key delta vs MCP: capabilities advertise MIME types
(``acceptedInputTypes`` / ``producedOutputTypes``) rather than JSON Schemas,
so :class:`Capability.input_schema` is left ``None`` and the MIME lists carry
the typing signal.

Auth: the card's ``securitySchemes`` array enumerates accepted schemes
(API key, OAuth2, mTLS, OIDC). MVP wire-up: API key + bearer; OAuth2 / mTLS
land in QO-068 alongside the on-chain validator publisher.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from src.core.evaluation_target import (
    AgentManifest,
    AuthContext,
    AuthDescriptor,
    Capability,
    Chunk,
    InvocationResult,
    InvokeError,
)
from src.storage.models import TargetType

logger = logging.getLogger(__name__)


# Per-spec AC10: per-probe connect 5s / read 10s; outer dispatch caps at 12s
# p99. These are reused by the resolver.
A2A_CONNECT_TIMEOUT = 5.0
A2A_READ_TIMEOUT = 10.0


def _timeouts() -> httpx.Timeout:
    return httpx.Timeout(
        connect=A2A_CONNECT_TIMEOUT,
        read=A2A_READ_TIMEOUT,
        write=A2A_CONNECT_TIMEOUT,
        pool=A2A_CONNECT_TIMEOUT,
    )


def parse_security_schemes(schemes: Any) -> Optional[AuthDescriptor]:
    """Map an A2A ``securitySchemes`` blob → :class:`AuthDescriptor`.

    A2A spec §3.2.4 lists schemes as a dict ``{name: {type, ...}}`` OR a
    list of name strings; we normalise both. We pick the FIRST scheme so the
    auth path stays deterministic — operators wanting OAuth2 must put it
    first in the card.
    """
    if not schemes:
        return None
    if isinstance(schemes, dict) and schemes:
        first_key, first = next(iter(schemes.items()))
        if isinstance(first, dict):
            stype = first.get("type", "").lower()
            if stype == "apikey":
                in_loc = first.get("in", "header")
                name = first.get("name") or "Authorization"
                if in_loc == "query":
                    return AuthDescriptor(style="apikey_query", query_param=name)
                return AuthDescriptor(style="x_api_key", header_name=name)
            if stype in ("http", "bearer"):
                return AuthDescriptor(style="bearer", header_name="Authorization")
            if stype == "oauth2":
                return AuthDescriptor(
                    style="oauth2",
                    token_url=(first.get("flows", {}) or {}).get("clientCredentials", {}).get("tokenUrl"),
                    scopes=list((first.get("flows", {}) or {}).get("clientCredentials", {}).get("scopes", {}).keys()),
                )
            if stype == "openidconnect":
                return AuthDescriptor(style="oidc", token_url=first.get("openIdConnectUrl"))
            if stype == "mutualtls":
                return AuthDescriptor(style="mtls")
        return AuthDescriptor(style="bearer", header_name="Authorization", extra={"name": first_key})
    if isinstance(schemes, list) and schemes:
        return AuthDescriptor(style="bearer", header_name="Authorization", extra={"name": str(schemes[0])})
    return None


async def fetch_agent_card(
    base_url: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Fetch ``/.well-known/agent-card.json`` and return the parsed dict.

    Raises :class:`httpx.HTTPError` on non-200; the resolver catches that
    and moves to the next probe.
    """
    url = base_url.rstrip("/") + "/.well-known/agent-card.json"
    if client is None:
        async with httpx.AsyncClient(timeout=_timeouts(), follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.json()
    r = await client.get(url)
    r.raise_for_status()
    return r.json()


class A2ATarget:
    """A2A v1.0 adapter."""

    target_type: TargetType = TargetType.A2A_AGENT

    def __init__(
        self,
        endpoint_url: str,
        card: Optional[Dict[str, Any]] = None,
        auth: Optional[AuthDescriptor] = None,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self._card = card
        self.auth = auth
        self._client = client
        self._auth_ctx: Optional[AuthContext] = None
        # Track every request so get_provenance() can surface the audit trail.
        self._requests: List[Dict[str, Any]] = []

    # ── Protocol methods ─────────────────────────────────────────────────

    async def authenticate(self) -> AuthContext:
        """Build :class:`AuthContext` from the descriptor.

        For MVP this is a passthrough — most A2A agents we'll see in v1
        publish API-key auth and the operator supplies the key out-of-band
        (env var `QO_TARGET_API_KEY`). OAuth2 client-credentials flow is
        wired in QO-068.
        """
        import os
        ctx = AuthContext()
        if self.auth is None:
            self._auth_ctx = ctx
            return ctx
        token = os.getenv("QO_TARGET_API_KEY", "")
        if not token:
            self._auth_ctx = ctx
            return ctx
        if self.auth.style == "bearer":
            ctx.headers[self.auth.header_name or "Authorization"] = f"Bearer {token}"
        elif self.auth.style in ("x_api_key", "api_key_header"):
            ctx.headers[self.auth.header_name or "X-API-Key"] = token
        elif self.auth.style == "apikey_query":
            ctx.query_params[self.auth.query_param or "apikey"] = token
        self._auth_ctx = ctx
        return ctx

    async def discover(self) -> AgentManifest:
        if self._card is None:
            self._card = await fetch_agent_card(self.endpoint_url, client=self._client)
        card = self._card
        skills = card.get("skills") or []
        capabilities = [
            Capability(
                id=str(s.get("id") or s.get("name") or f"skill_{i}"),
                name=str(s.get("name") or ""),
                description=str(s.get("description") or ""),
                # A2A: MIME-typed I/O, no JSON Schema.
                input_schema=None,
                output_schema=None,
                accepted_input_types=list(s.get("acceptedInputTypes") or s.get("inputModes") or []),
                produced_output_types=list(s.get("producedOutputTypes") or s.get("outputModes") or []),
                extensions=s.get("extensions", {}) if isinstance(s.get("extensions"), dict) else {},
            )
            for i, s in enumerate(skills)
            if isinstance(s, dict)
        ]
        if self.auth is None:
            self.auth = parse_security_schemes(card.get("securitySchemes"))
        return AgentManifest(
            id=str(card.get("id") or card.get("name") or self.endpoint_url),
            name=str(card.get("name") or ""),
            description=str(card.get("description") or ""),
            capabilities=capabilities,
            auth=self.auth,
            signature=card.get("signature"),
            confidence="high",
            raw=card,
        )

    async def list_capabilities(self) -> List[Capability]:
        manifest = await self.discover()
        return manifest.capabilities

    async def invoke(self, capability_id: str, payload: dict) -> InvocationResult:
        """JSON-RPC 2.0 ``message/send``.

        A2A wire format: POST to the agent's endpoint with body ``{jsonrpc:
        "2.0", method: "message/send", params: {...}, id: ...}``. Success →
        ``{result: {message: {parts: [...]}}}``; failure → ``{error: {...}}``.
        """
        if self._auth_ctx is None:
            await self.authenticate()
        ctx = self._auth_ctx or AuthContext()

        invocation_id = str(uuid.uuid4())
        message = payload.get("message") or payload
        if not isinstance(message, str):
            # Compact JSON so the agent gets a deterministic single-line input.
            import json
            message = json.dumps(message, sort_keys=True)
        body = {
            "jsonrpc": "2.0",
            "id": invocation_id,
            "method": "message/send",
            "params": {
                "skill": capability_id,
                "message": {
                    "role": "user",
                    "parts": [{"contentType": "text/plain", "text": message}],
                },
            },
        }
        start = time.time()
        try:
            async with self._client_or_new() as client:
                if ctx.cookies:
                    for k, v in ctx.cookies.items():
                        client.cookies.set(k, v)
                r = await client.post(
                    self.endpoint_url,
                    json=body,
                    headers=ctx.headers,
                    params=ctx.query_params,
                )
            latency_ms = int((time.time() - start) * 1000)
            self._requests.append(
                {"capability_id": capability_id, "status_code": r.status_code, "latency_ms": latency_ms}
            )
            r.raise_for_status()
            data = r.json()
            if "error" in data and data.get("error"):
                err = data["error"]
                return InvocationResult(
                    invocation_id=invocation_id,
                    text="",
                    raw=data,
                    latency_ms=latency_ms,
                    status="error",
                    error=str(err.get("message") or err),
                    error_class="a2a_jsonrpc_error",
                )
            text = _extract_a2a_text(data.get("result") or {})
            return InvocationResult(
                invocation_id=invocation_id,
                text=text,
                raw=data,
                latency_ms=latency_ms,
                status="ok",
            )
        except httpx.HTTPError as exc:
            latency_ms = int((time.time() - start) * 1000)
            self._requests.append(
                {"capability_id": capability_id, "exception": str(exc), "latency_ms": latency_ms}
            )
            raise InvokeError(f"A2A invoke failed: {exc}") from exc

    async def stream(self, capability_id: str, payload: dict) -> AsyncIterator[Chunk]:
        """SSE ``message/stream`` — not wired in MVP; type contract locked.

        Spec §"WebSocket/SSE deferred": adapters MAY raise NotImplementedError;
        the Protocol ensures the type signature is fixed for v2.
        """
        raise NotImplementedError("A2A streaming deferred to QO-068")
        if False:  # pragma: no cover - keep generator type
            yield None  # type: ignore[misc]

    async def cancel(self, invocation_id: str) -> None:
        """Best-effort JSON-RPC ``message/cancel`` — silently no-ops if the
        target rejects (most agents won't have this method wired)."""
        if self._auth_ctx is None:
            await self.authenticate()
        ctx = self._auth_ctx or AuthContext()
        body = {
            "jsonrpc": "2.0", "id": str(uuid.uuid4()),
            "method": "message/cancel", "params": {"invocation_id": invocation_id},
        }
        try:
            async with self._client_or_new() as client:
                await client.post(self.endpoint_url, json=body, headers=ctx.headers)
        except Exception:  # pragma: no cover
            pass

    def get_provenance(self) -> dict:
        return {
            "target_type": self.target_type.value,
            "endpoint_url": self.endpoint_url,
            "card_id": (self._card or {}).get("id"),
            "card_signature": (self._card or {}).get("signature"),
            "request_count": len(self._requests),
            "requests": self._requests[-5:],  # last 5 for compactness
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _client_or_new(self):
        """Return an async-context-manager that yields the configured client.

        When the caller injected a client in __init__ we reuse it (tests rely
        on this for the FastAPI TestClient pattern); otherwise we open a
        scoped one.
        """
        if self._client is not None:
            client = self._client

            class _Wrap:
                async def __aenter__(self_inner):
                    return client

                async def __aexit__(self_inner, *_):
                    return False

            return _Wrap()
        return httpx.AsyncClient(timeout=_timeouts(), follow_redirects=True)


def _extract_a2a_text(result: Dict[str, Any]) -> str:
    """Pull the human-readable text out of an A2A JSON-RPC ``result`` blob.

    Handles two shapes seen in the wild:
      1. ``{message: {parts: [{text: "..."}]}}``
      2. ``{output: {parts: [{contentType: "text/plain", text: "..."}]}}``
    Falls back to ``str(result)`` so downstream judges always have something.
    """
    msg = result.get("message") or result.get("output") or {}
    parts = msg.get("parts") if isinstance(msg, dict) else None
    if isinstance(parts, list):
        texts = []
        for p in parts:
            if isinstance(p, dict):
                t = p.get("text") or p.get("content")
                if isinstance(t, str):
                    texts.append(t)
        if texts:
            return "\n".join(texts).strip()
    if isinstance(result.get("text"), str):
        return result["text"]
    return str(result)

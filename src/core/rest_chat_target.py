"""RESTChatTarget — manifest-less generic REST/JSON chat agent (QO-058 M1).

R6 §"Manifest-less methodology" / §"Why this matters": ~40-60% of deployed
agents are "POST a JSON body, get JSON back" — no manifest, no schema, no
agent-card. This adapter probes 8 request shapes and 5 auth styles and runs
schema inference on the responses to build a soft manifest.

Key constraints (spec §"REST chat synthetic capability"):
* Synthetic ``Capability(id="chat", ...)`` — :func:`invoke` rejects any other
  ``capability_id`` to keep the public surface tight.
* Verified-tier cap unless escalated via OpenAPI doc OR n=10 calibration +
  ≥10 correlation rows (spec §"Path to Certified for manifest-less").
* AC11 — 2-of-3 calibration failures raise :class:`SchemaUnobtainableError`.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

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
from src.core.schema_inference import (
    CALIBRATION_PROMPTS,
    CalibrationResult,
    infer_manifest,
    manifest_from_inferred,
)
from src.storage.models import TargetType

logger = logging.getLogger(__name__)


REST_CHAT_CONNECT_TIMEOUT = 5.0
REST_CHAT_READ_TIMEOUT = 10.0


# ── 8 request shapes (spec §"REST chat target"/M1) ──────────────────────────
#
# Each entry is (shape_name, builder) — builder takes the user message and
# returns the JSON body. Order matters: the most common shapes (`message`,
# `prompt`, `messages[]`) come first so the dispatch breaks early.

ShapeBuilder = Callable[[str], Dict[str, Any]]

REST_CHAT_REQUEST_SHAPES: List[tuple[str, ShapeBuilder]] = [
    ("message",       lambda m: {"message": m}),
    ("prompt",        lambda m: {"prompt": m}),
    ("input",         lambda m: {"input": m}),
    ("messages",      lambda m: {"messages": [{"role": "user", "content": m}]}),
    ("query",         lambda m: {"query": m}),
    ("question",      lambda m: {"question": m}),
    ("text",          lambda m: {"text": m}),
    ("inputs",        lambda m: {"inputs": m}),
]


REST_CHAT_AUTH_STYLES: List[str] = [
    "bearer",          # Authorization: Bearer <token>
    "x_api_key",       # X-API-Key header
    "api_key_header",  # Api-Key header
    "apikey_query",    # ?apikey=...
    "cookie_session",  # Cookie session (operator-supplied)
]


def _timeouts() -> httpx.Timeout:
    return httpx.Timeout(
        connect=REST_CHAT_CONNECT_TIMEOUT,
        read=REST_CHAT_READ_TIMEOUT,
        write=REST_CHAT_CONNECT_TIMEOUT,
        pool=REST_CHAT_CONNECT_TIMEOUT,
    )


def _extract_chat_reply(payload: Any) -> str:
    """Pull a textual reply out of whatever JSON the chat agent returned.

    Tries common keys: ``reply``, ``response``, ``output``, ``message``,
    ``content``, ``text``, ``answer``, ``choices[0].message.content`` (OpenAI
    shape), ``generated_text`` (HF Inference shape). Falls back to ``str(...)``.
    """
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        try:
            return str(payload)
        except Exception:  # pragma: no cover
            return ""
    for k in ("reply", "response", "output", "answer", "text", "content", "message", "result"):
        v = payload.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            inner = v.get("content") or v.get("text")
            if isinstance(inner, str) and inner:
                return inner
    # OpenAI Chat Completions shape
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        c = choices[0]
        if isinstance(c, dict):
            msg = c.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c.get("text"), str):
                return c["text"]
    # HuggingFace Inference shape
    if isinstance(payload.get("generated_text"), str):
        return payload["generated_text"]
    # Last-ditch — give downstream the raw JSON as a string so judges can
    # still evaluate. Never error here.
    try:
        import json as _json
        return _json.dumps(payload)[:2000]
    except Exception:  # pragma: no cover
        return str(payload)[:2000]


class RESTChatTarget:
    """Manifest-less REST chat adapter."""

    target_type: TargetType = TargetType.REST_CHAT

    def __init__(
        self,
        endpoint_url: str,
        *,
        auth: Optional[AuthDescriptor] = None,
        client: Optional[httpx.AsyncClient] = None,
        judge: Any = None,
    ):
        self.endpoint_url = endpoint_url
        self.auth = auth
        self._client = client
        self._judge = judge
        self._auth_ctx: Optional[AuthContext] = None
        self._winning_shape: Optional[str] = None
        self._winning_auth_style: Optional[str] = None
        self._inferred_confidence: Optional[str] = None
        self._calibration_results: List[CalibrationResult] = []
        self._requests: List[Dict[str, Any]] = []

    # ── Protocol methods ─────────────────────────────────────────────────

    async def authenticate(self) -> AuthContext:
        """Resolve auth.

        If the operator supplied a static descriptor, honour it; otherwise
        return an empty context (most public chat endpoints are unauth'd).
        We do NOT silently rotate through 5 auth styles here — that's done
        on first invoke if the descriptor is missing AND a 401 surfaces.
        """
        import os
        ctx = AuthContext()
        token = os.getenv("QO_TARGET_API_KEY", "")
        if self.auth is None or not token:
            self._auth_ctx = ctx
            return ctx
        style = self.auth.style
        if style == "bearer":
            ctx.headers[self.auth.header_name or "Authorization"] = f"Bearer {token}"
        elif style == "x_api_key":
            ctx.headers[self.auth.header_name or "X-API-Key"] = token
        elif style == "api_key_header":
            ctx.headers[self.auth.header_name or "Api-Key"] = token
        elif style == "apikey_query":
            ctx.query_params[self.auth.query_param or "apikey"] = token
        elif style == "cookie_session":
            ctx.cookies["session"] = token
        self._auth_ctx = ctx
        return ctx

    async def discover(self) -> AgentManifest:
        """Run 3 calibration prompts, infer soft manifest.

        AC4 — `confidence: medium` if all 3 succeed; AC11 — raises
        :class:`SchemaUnobtainableError` if ≥2 fail (5xx / hang).
        """
        await self.authenticate()
        results: List[CalibrationResult] = []
        for prompt in CALIBRATION_PROMPTS:
            r = await self._calibration_call(prompt)
            results.append(r)
        self._calibration_results = results
        # infer_manifest raises SchemaUnobtainableError if confidence='low'
        inferred = await infer_manifest(
            results, judge=self._judge, target_url=self.endpoint_url
        )
        self._inferred_confidence = inferred.confidence
        return manifest_from_inferred(target_url=self.endpoint_url, inferred=inferred)

    async def list_capabilities(self) -> List[Capability]:
        # REST chat exposes ONE synthetic capability — never None.
        return [
            Capability(
                id="chat",
                name="chat",
                description="Manifest-less REST chat surface",
                input_schema=None,
                output_schema=None,
            )
        ]

    async def invoke(self, capability_id: str, payload: dict) -> InvocationResult:
        """POST the chat message; try each request shape in order until 200."""
        if capability_id != "chat":
            raise InvokeError(
                f"REST chat targets only expose synthetic 'chat' capability, "
                f"got {capability_id!r}"
            )
        if self._auth_ctx is None:
            await self.authenticate()
        ctx = self._auth_ctx or AuthContext()

        message = payload.get("message")
        if message is None:
            # Allow callers to pass {message} OR {prompt}/{input} verbatim.
            for k in ("prompt", "input", "query", "question", "text"):
                if k in payload:
                    message = payload[k]
                    break
        if message is None:
            message = ""

        invocation_id = str(uuid.uuid4())

        # Fast path — if we've already found the winning shape on this target
        # use it directly so each subsequent invocation costs ONE HTTP call.
        shapes = (
            [(self._winning_shape, dict(REST_CHAT_REQUEST_SHAPES)[self._winning_shape])]
            if self._winning_shape
            else REST_CHAT_REQUEST_SHAPES
        )

        last_error: Optional[str] = None
        for shape_name, builder in shapes:
            body = builder(message)
            start = time.time()
            try:
                async with self._client_or_new() as client:
                    # cookies on the client instance, not per-request
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
                    {"shape": shape_name, "status_code": r.status_code, "latency_ms": latency_ms}
                )
                # 422 / 400 = wrong shape, try next; 5xx = server problem,
                # also try next so a flaky shape doesn't blackhole the eval.
                if r.status_code in (400, 415, 422):
                    last_error = f"{r.status_code}: shape rejected ({shape_name})"
                    continue
                if r.status_code >= 500:
                    last_error = f"{r.status_code}: upstream error"
                    continue
                if r.status_code in (401, 403):
                    last_error = f"{r.status_code}: auth required"
                    continue
                r.raise_for_status()
                self._winning_shape = shape_name
                try:
                    data = r.json()
                except Exception:
                    data = {"text": r.text}
                text = _extract_chat_reply(data)
                return InvocationResult(
                    invocation_id=invocation_id,
                    text=text,
                    raw=data if isinstance(data, dict) else {"text": str(data)},
                    latency_ms=latency_ms,
                    status="ok",
                    request_shape=shape_name,
                )
            except httpx.TimeoutException as exc:
                last_error = f"timeout: {exc}"
                continue
            except httpx.HTTPError as exc:
                last_error = f"http_error: {exc}"
                continue
        raise InvokeError(f"All {len(REST_CHAT_REQUEST_SHAPES)} shapes failed: {last_error}")

    async def stream(self, capability_id: str, payload: dict) -> AsyncIterator[Chunk]:
        """REST chat streaming deferred — most chat endpoints we'll see in
        v1 don't expose SSE/WS. The Chunk sum type is locked so v2 can
        land without breaking adapters.
        """
        raise NotImplementedError("REST chat streaming deferred to v2")
        if False:  # pragma: no cover
            yield None  # type: ignore[misc]

    async def cancel(self, invocation_id: str) -> None:  # noqa: D401
        # REST chat is request/response — no cancellation surface.
        return None

    def get_provenance(self) -> dict:
        return {
            "target_type": self.target_type.value,
            "endpoint_url": self.endpoint_url,
            "winning_shape": self._winning_shape,
            "winning_auth_style": self._winning_auth_style,
            "inference_confidence": self._inferred_confidence,
            "request_count": len(self._requests),
            "requests": self._requests[-5:],
            "tier_cap_reason": (
                "no_manifest" if self._inferred_confidence in (None, "low", "medium")
                else None
            ),
        }

    # ── helpers ──────────────────────────────────────────────────────────

    async def _calibration_call(self, prompt: str) -> CalibrationResult:
        """Run one calibration prompt — capture status + latency."""
        try:
            res = await self.invoke("chat", {"message": prompt})
            return CalibrationResult(
                prompt=prompt,
                text=res.text,
                latency_ms=res.latency_ms,
                status="ok" if res.text else "error",
                error=None,
            )
        except InvokeError as exc:
            return CalibrationResult(
                prompt=prompt, text="", error=str(exc),
                status="error", latency_ms=0,
            )
        except httpx.TimeoutException as exc:  # pragma: no cover - covered by InvokeError path
            return CalibrationResult(
                prompt=prompt, text="", error=f"timeout: {exc}",
                status="timeout", latency_ms=0,
            )

    def _client_or_new(self):
        if self._client is not None:
            client = self._client

            class _Wrap:
                async def __aenter__(self_inner):
                    return client

                async def __aexit__(self_inner, *_):
                    return False

            return _Wrap()
        return httpx.AsyncClient(timeout=_timeouts(), follow_redirects=True)

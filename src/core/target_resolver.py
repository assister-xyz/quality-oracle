"""target_resolver.resolve(url) — protocol detection cascade (QO-058 AC10).

Spec §"Discovery cascade resolver" + plan-review C2/C3:
* Probes ALL 10 protocols **concurrently** with ``asyncio.gather`` (NOT
  sequential) so one slow probe can't blackhole the rest.
* Specificity-priority ranking — the first non-exception in
  :data:`DISCOVERY_PROBES` order wins.
* Per-probe timeout: 5s connect, 10s read; outer p99 cold-cache cap 12s
  (AC10).
* 24h Redis cache at ``qo:discovery:{sha256(url)}``.

Probes that are DEFERRED in QO-058 still appear in the cascade as no-ops
returning ``None`` — they're documented hooks for future work (QO-068
ERC-8004, QO-069 OpenAI Responses, QO-070 Gradio). We never lie about what
landed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Tuple

import httpx

from src.core.evaluation_target import (
    AuthDescriptor,
    EvaluationTarget,
    UnknownTargetError,
)
from src.storage.models import TargetType

logger = logging.getLogger(__name__)


# Per-spec AC10 — these limits are enforced at the asyncio.wait_for level
# AND inside the httpx.Timeout passed to each probe.
PROBE_CONNECT_TIMEOUT = 5.0
PROBE_READ_TIMEOUT = 10.0
PROBE_HARD_DEADLINE = PROBE_READ_TIMEOUT + 1.0  # safety net for asyncio.wait_for
DISCOVERY_CACHE_TTL = 86_400  # 24h


@dataclass
class DiscoveryResult:
    """Lightweight result a probe returns when it positively identifies a
    protocol. The resolver wraps this into a concrete :class:`EvaluationTarget`.

    Probes return ``None`` (no match) or raise an exception (treated the same
    as None — the cascade moves on).
    """
    target_type: TargetType
    endpoint_url: str
    raw: Optional[dict] = None  # parsed manifest payload (agent-card, openapi, ...)
    auth: Optional[AuthDescriptor] = None
    note: str = ""


def _timeouts() -> httpx.Timeout:
    return httpx.Timeout(
        connect=PROBE_CONNECT_TIMEOUT,
        read=PROBE_READ_TIMEOUT,
        write=PROBE_CONNECT_TIMEOUT,
        pool=PROBE_CONNECT_TIMEOUT,
    )


# ── individual probes ────────────────────────────────────────────────────────


async def probe_a2a_agent_card(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    url = base.rstrip("/") + "/.well-known/agent-card.json"
    r = await client.get(url)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if not isinstance(data, dict) or not (data.get("id") or data.get("name") or data.get("skills")):
        return None
    return DiscoveryResult(
        target_type=TargetType.A2A_AGENT,
        endpoint_url=base.rstrip("/"),
        raw=data,
        note="a2a_v1.0",
    )


async def probe_agents_json(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """``/.well-known/agents.json`` — RFC-style multi-agent index.

    Returns the FIRST agent record so the cascade has something concrete.
    Full multi-agent enumeration is QO-068 territory.
    """
    url = base.rstrip("/") + "/.well-known/agents.json"
    r = await client.get(url)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    agents = data.get("agents") if isinstance(data, dict) else None
    if not (isinstance(agents, list) and agents and isinstance(agents[0], dict)):
        return None
    first = agents[0]
    return DiscoveryResult(
        target_type=TargetType.A2A_AGENT,
        endpoint_url=first.get("endpoint") or base.rstrip("/"),
        raw=first,
        note="agents_json_index",
    )


async def probe_erc8004(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """ERC-8004 agent registration — DEFERRED to QO-068.

    Probe is wired but always returns None until that spec lands. Keeping
    the probe in-place means specificity ranking stays correct when QO-068
    flips it on.
    """
    return None


async def probe_mcp_handshake(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """Try the MCP ``initialize`` handshake.

    We do NOT actually open an MCP session here (mcp_client owns that — and
    its session has expensive teardown). Instead we look for the SSE-shape
    URL marker (``/sse``) OR a ``/mcp`` path that returns 200 on HEAD/GET
    with ``content-type`` containing ``event-stream`` or ``json``. A genuine
    MCP server will be confirmed on first ``MCPTarget.discover()`` call.
    """
    url = base.rstrip("/")
    candidate_paths = ["", "/sse", "/mcp"]
    for p in candidate_paths:
        try:
            r = await client.get(url + p)
        except httpx.HTTPError:
            continue
        ct = (r.headers.get("content-type") or "").lower()
        # 405 method-not-allowed is fine; SSE servers often reject GET.
        if r.status_code in (200, 405) and (
            "event-stream" in ct or "mcp" in ct or url.rstrip("/").endswith("/sse")
        ):
            return DiscoveryResult(
                target_type=TargetType.MCP_SERVER,
                endpoint_url=url + p if p else url,
                raw=None,
                note=f"mcp_handshake_path={p or 'root'}",
            )
    # URL pattern hint — operators commonly pass /sse explicitly.
    if url.endswith("/sse") or url.endswith("/mcp"):
        return DiscoveryResult(
            target_type=TargetType.MCP_SERVER,
            endpoint_url=url,
            raw=None,
            note="mcp_url_pattern",
        )
    return None


async def probe_gradio(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """Gradio / HF Spaces — DEFERRED to QO-070.

    BUT: we still recognise the URL pattern because a Gradio Space ALSO
    exposes ``/openapi.json`` (which would otherwise win the cascade).
    Returning a hint here means specificity ranking sends Gradio targets
    to the deferred path with a clean error message rather than silently
    routing them as generic OpenAPI agents.

    The result is still "None" (deferred, can't actually evaluate) — but we
    log it. Tests rely on this behaviour for the AC8 Gradio-vs-OpenAPI
    ambiguity case.
    """
    url = base.rstrip("/") + "/gradio_api/openapi.json"
    try:
        r = await client.get(url)
    except httpx.HTTPError:
        return None
    if r.status_code == 200:
        # Mark as known-but-deferred so cascade falls through. We use a
        # marker so caller can surface a clean "Gradio support coming in
        # QO-070" message rather than a misleading OpenAPI score.
        logger.info("Gradio detected at %s — deferred to QO-070", base)
        return DiscoveryResult(
            target_type=TargetType.UNKNOWN,
            endpoint_url=base.rstrip("/"),
            raw={"deferred": "gradio_qo_070"},
            note="gradio_deferred",
        )
    return None


async def probe_openapi(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """OpenAPI / Swagger doc → :class:`TargetType.OPENAPI_AGENT`.

    Generic OpenAPI agents have a real schema → full 6-axis weights.
    """
    url = base.rstrip("/") + "/openapi.json"
    try:
        r = await client.get(url)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if not (isinstance(data, dict) and (data.get("openapi") or data.get("swagger"))):
        return None
    return DiscoveryResult(
        target_type=TargetType.OPENAPI_AGENT,
        endpoint_url=base.rstrip("/"),
        raw=data,
        note="openapi",
    )


async def probe_openapi_alt(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """``/swagger.json`` alternate path."""
    url = base.rstrip("/") + "/swagger.json"
    try:
        r = await client.get(url)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if not (isinstance(data, dict) and (data.get("openapi") or data.get("swagger"))):
        return None
    return DiscoveryResult(
        target_type=TargetType.OPENAPI_AGENT,
        endpoint_url=base.rstrip("/"),
        raw=data,
        note="swagger_alt",
    )


async def probe_openai_responses(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """OpenAI Responses API + Apps SDK — DEFERRED to QO-069. Always None."""
    return None


async def probe_langserve(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """LangServe ``/input_schema`` — DEFERRED. Always None."""
    return None


async def probe_rest_chat(base: str, *, client: httpx.AsyncClient) -> Optional[DiscoveryResult]:
    """Heuristic chat probe — POST a tiny "hi" body and accept any 2xx as
    confirmation that there's *something* answering on this URL.

    This is the cascade's last-resort branch (AC8 step 10). It is delibe-
    rately permissive: schema inference will catch malformed targets at
    :meth:`RESTChatTarget.discover` time.
    """
    url = base.rstrip("/")
    try:
        r = await client.post(url, json={"message": "hi"})
    except httpx.HTTPError:
        return None
    if 200 <= r.status_code < 300:
        return DiscoveryResult(
            target_type=TargetType.REST_CHAT,
            endpoint_url=url,
            raw=None,
            note="rest_chat_heuristic",
        )
    return None


# ── cascade table — ORDER = specificity-priority ─────────────────────────────

ProbeFn = Callable[[str], Awaitable[Optional[DiscoveryResult]]]

DISCOVERY_PROBES: List[Tuple[str, ProbeFn]] = [
    ("a2a_agent_card",      probe_a2a_agent_card),
    ("agents_json",         probe_agents_json),
    ("erc8004",             probe_erc8004),
    ("mcp_handshake",       probe_mcp_handshake),
    ("gradio_api",          probe_gradio),
    ("openapi",             probe_openapi),
    ("openapi_alt",         probe_openapi_alt),
    ("openai_responses",    probe_openai_responses),
    ("langserve",           probe_langserve),
    ("rest_chat_heuristic", probe_rest_chat),
]


# ── resolver ─────────────────────────────────────────────────────────────────


def _cache_key(url: str) -> str:
    return f"qo:discovery:{hashlib.sha256(url.encode()).hexdigest()}"


def _serialize(d: DiscoveryResult) -> str:
    return json.dumps({
        "target_type": d.target_type.value,
        "endpoint_url": d.endpoint_url,
        "raw": d.raw,
        "auth": d.auth.model_dump() if d.auth else None,
        "note": d.note,
    })


def _deserialize(s: str) -> DiscoveryResult:
    data = json.loads(s)
    auth = AuthDescriptor(**data["auth"]) if data.get("auth") else None
    return DiscoveryResult(
        target_type=TargetType(data["target_type"]),
        endpoint_url=data["endpoint_url"],
        raw=data.get("raw"),
        auth=auth,
        note=data.get("note", ""),
    )


async def _run_probe_with_timeout(
    name: str, fn: ProbeFn, base: str, client: httpx.AsyncClient
) -> Optional[DiscoveryResult]:
    """Run one probe with a hard wall-clock cap."""
    try:
        return await asyncio.wait_for(fn(base, client=client), timeout=PROBE_HARD_DEADLINE)
    except asyncio.TimeoutError:
        logger.debug("probe %s timed out for %s", name, base)
        return None
    except Exception as exc:  # noqa: BLE001 — we deliberately swallow; fail-soft per AC8
        logger.debug("probe %s failed for %s: %s", name, base, exc)
        return None


def _instantiate(d: DiscoveryResult, *, judge=None) -> EvaluationTarget:
    """Wrap a DiscoveryResult into a live :class:`EvaluationTarget`."""
    from src.core.a2a_target import A2ATarget, parse_security_schemes
    from src.core.rest_chat_target import RESTChatTarget

    if d.target_type == TargetType.A2A_AGENT:
        auth = parse_security_schemes((d.raw or {}).get("securitySchemes")) if d.raw else d.auth
        return A2ATarget(endpoint_url=d.endpoint_url, card=d.raw, auth=auth)
    if d.target_type == TargetType.MCP_SERVER:
        from src.core.mcp_target import MCPTarget
        return MCPTarget(endpoint_url=d.endpoint_url)
    if d.target_type == TargetType.OPENAPI_AGENT:
        # OpenAPI evaluation surface is in QO-058 scope only as discovery —
        # the actual evaluator wiring lands when QO-058 ships, but we still
        # need a usable target object so /v1/discover returns metadata.
        # Treat as a richer REST chat for MVP — the OpenAPI doc improves the
        # manifest confidence to "high" automatically.
        target = RESTChatTarget(endpoint_url=d.endpoint_url, judge=judge)
        target.target_type = TargetType.OPENAPI_AGENT  # type: ignore[misc]
        return target
    if d.target_type == TargetType.REST_CHAT:
        return RESTChatTarget(endpoint_url=d.endpoint_url, judge=judge)
    if d.target_type == TargetType.UNKNOWN:
        # Gradio-deferred sentinel ends here — surface as UnknownTargetError
        # so the API responds with a clean "deferred" message instead of
        # silently mis-routing.
        raise UnknownTargetError(
            f"Detected protocol but it's deferred: note={d.note!r} "
            f"raw={d.raw!r}"
        )
    raise UnknownTargetError(f"No instantiation path for {d.target_type!r}")


async def resolve(
    url: str,
    *,
    cache=None,
    judge=None,
    return_meta: bool = False,
) -> EvaluationTarget:
    """Concurrently probe 10 protocols, return the most-specific match.

    Parameters
    ----------
    url:
        Target base URL.
    cache:
        Optional Redis-shaped cache (``get`` / ``setex``). When provided, hits
        skip the cascade entirely (24h TTL).
    judge:
        Optional judge passed through to manifest-less targets for schema
        inference.
    return_meta:
        Internal — the ``/v1/discover`` endpoint sets this to also return the
        DiscoveryResult so it can surface specificity.

    Raises
    ------
    UnknownTargetError
        If the cascade resolves nothing AND the heuristic chat probe also
        fails (AC8).
    """
    started = time.time()

    # Cache hit?
    if cache is not None:
        try:
            cached = await cache.get(_cache_key(url))
            if cached:
                d = _deserialize(cached if isinstance(cached, str) else cached.decode())
                target = _instantiate(d, judge=judge)
                logger.info("discovery cache hit for %s → %s", url, d.target_type.value)
                return (target, d) if return_meta else target
        except Exception as exc:  # pragma: no cover - cache is best-effort
            logger.warning("discovery cache read failed: %s", exc)

    base = url
    async with httpx.AsyncClient(timeout=_timeouts(), follow_redirects=True) as client:
        tasks = [
            _run_probe_with_timeout(name, fn, base, client)
            for name, fn in DISCOVERY_PROBES
        ]
        # gather all — first non-None in DISCOVERY_PROBES order wins.
        # AC10: hard outer deadline 12s p99. We rely on each probe's
        # PROBE_HARD_DEADLINE (11s) so 10 concurrent probes still finish
        # under 12s wall-clock even on a fully cold cache.
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed_ms = int((time.time() - started) * 1000)

    winner: Optional[DiscoveryResult] = None
    for (name, _), result in zip(DISCOVERY_PROBES, results):
        if isinstance(result, Exception):
            logger.debug("probe %s raised: %s", name, result)
            continue
        if result is None:
            continue
        winner = result
        logger.info(
            "discovery resolved %s → %s (probe=%s, elapsed_ms=%d)",
            url, result.target_type.value, name, elapsed_ms,
        )
        break

    if winner is None:
        raise UnknownTargetError(
            f"Couldn't auto-detect protocol for {url} "
            f"(elapsed_ms={elapsed_ms}); pick a target_type or paste an "
            "agent-card / OpenAPI doc."
        )

    # Cache success — only the winning DiscoveryResult, 24h TTL.
    if cache is not None:
        try:
            await cache.setex(_cache_key(url), DISCOVERY_CACHE_TTL, _serialize(winner))
        except Exception as exc:  # pragma: no cover
            logger.warning("discovery cache write failed: %s", exc)

    target = _instantiate(winner, judge=judge)
    return (target, winner) if return_meta else target

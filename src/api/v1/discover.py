"""GET /v1/discover?url=X — protocol detection (QO-058 §"API endpoint").

Public endpoint that runs the QO-058 discovery cascade and returns the
inferred target type + capability preview. Used by:
* Landing-page evaluator UI to disambiguate "is this an MCP server, an
  A2A agent, or a generic chat endpoint?" before the operator clicks
  Evaluate.
* Programmatic callers that want to know what we'd dispatch BEFORE paying.

Latency target: ≤12s p99 cold-cache (AC10). Cached for 24h via Redis.
Auth: public (no API key) — same as other read-only endpoints; the
underlying probes are HEAD/GET only and don't run the full evaluation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.core.evaluation_target import UnknownTargetError
from src.core.target_resolver import resolve as resolve_target
from src.storage.cache import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


class CapabilityPreview(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    accepted_input_types: List[str] = Field(default_factory=list)


class DiscoverResponse(BaseModel):
    """Payload for ``GET /v1/discover?url=X``."""
    url: str
    target_type: str
    endpoint_url: str
    confidence: Optional[str] = None
    note: str = ""
    capabilities: List[CapabilityPreview] = Field(default_factory=list)
    auth: Optional[Dict[str, Any]] = None
    cached: bool = False  # whether the result came from the 24h Redis cache


@router.get("/discover", response_model=DiscoverResponse)
async def discover(url: str = Query(..., description="Target URL to probe")):
    """Probe a URL through the 10-step discovery cascade.

    Returns the inferred target type + a capability preview. Does NOT
    run the full evaluation; cheap by design (≤12s cold-cache p99).

    Raises 422 when the cascade can't identify any protocol (AC8 fail
    branch — the heuristic chat probe also failed).
    """
    cache = None
    cache_hit_marker: Dict[str, bool] = {"hit": False}
    try:
        cache = get_redis()
    except Exception:  # pragma: no cover - cache best-effort
        cache = None

    # Wrap cache to detect hits without changing the resolver signature.
    class _CacheProbe:
        async def get(self_inner, key):
            if cache is None:
                return None
            v = await cache.get(key)
            if v is not None:
                cache_hit_marker["hit"] = True
            return v

        async def setex(self_inner, key, ttl, value):
            if cache is None:
                return
            await cache.setex(key, ttl, value)

    try:
        target, meta = await resolve_target(
            url, cache=_CacheProbe() if cache is not None else None,
            return_meta=True,
        )
    except UnknownTargetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface unknown errors as 502
        logger.error("discover failed for %s: %s", url, exc)
        raise HTTPException(status_code=502, detail=f"discovery error: {exc}") from exc

    # Capability preview — call list_capabilities() but do NOT trigger
    # full discover() for manifest-less targets (that costs ~$0.005 in
    # judge calls). MCP/A2A discover() is cheap (one HTTP fetch).
    caps: List[CapabilityPreview] = []
    try:
        # For OpenAPI-described targets discover() is also cheap (no LLM).
        if meta.target_type.value in ("a2a_agent", "mcp_server", "skill"):
            mc = await target.list_capabilities()
            caps = [
                CapabilityPreview(
                    id=c.id, name=c.name, description=c.description,
                    accepted_input_types=c.accepted_input_types,
                )
                for c in mc[:25]  # cap preview at 25
            ]
        else:
            # REST chat / OpenAPI / unknown: surface the synthetic chat
            # capability without paying for inference.
            caps = [CapabilityPreview(id="chat", name="chat", description="manifest-less chat")]
    except Exception as exc:  # noqa: BLE001
        logger.warning("capability preview failed for %s: %s", url, exc)
        caps = []

    return DiscoverResponse(
        url=url,
        target_type=meta.target_type.value,
        endpoint_url=meta.endpoint_url,
        confidence=(
            (meta.raw or {}).get("confidence")
            if isinstance(meta.raw, dict)
            else None
        ),
        note=meta.note,
        capabilities=caps,
        auth=meta.auth.model_dump() if meta.auth else None,
        cached=cache_hit_marker["hit"],
    )

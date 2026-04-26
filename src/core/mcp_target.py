"""MCPTarget — conforms :mod:`src.core.mcp_client` to the
:class:`EvaluationTarget` Protocol (QO-058 C5 plan-review fix).

Why a wrapper, not a refactor of mcp_client itself:
* mcp_client.py is 762 lines, with deeply-baked context-vars
  (``current_evaluation_id``, ``current_target_id``, ``_call_index_counter``)
  used by the audit-log subsystem. Re-shaping those into a class would
  be a behaviour change.
* AC9 requires the legacy MCP path to remain byte-identical (5-fixture
  regression in ``tests/test_evaluate_mcp_regression.py``). A wrapper class
  satisfies the Protocol without touching the call site of
  ``evaluator.evaluate_mcp`` — the existing dispatch in
  ``api.v1.evaluate._run_evaluation_mcp`` continues to call mcp_client
  directly, byte-identical with the pre-058 path.

This wrapper is used by:
* :func:`target_resolver.resolve` to instantiate a target after the
  cascade picks ``TargetType.MCP_SERVER``.
* :class:`SkillTarget` won't reuse this — skills have their own activator.
* The /v1/discover endpoint (returns capability preview).

Invariants preserved (cite QO-053-C AC9):
1. ``current_evaluation_id`` / ``current_target_id`` /
   ``_call_index_counter`` ContextVars are read/written by the EXISTING
   mcp_client, NOT by this wrapper. They survive untouched.
2. ``evaluate_mcp`` continues to call ``mcp_client.get_server_manifest``
   and ``call_tools_batch``; this wrapper does NOT replace that flow.
3. Result-document shape is unchanged when the legacy path is taken.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from src.core import mcp_client
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


class MCPTarget:
    """Protocol-conforming view over an MCP server."""

    target_type: TargetType = TargetType.MCP_SERVER

    def __init__(self, endpoint_url: str, auth: Optional[AuthDescriptor] = None):
        self.endpoint_url = endpoint_url
        self.auth = auth
        self._manifest: Optional[Dict[str, Any]] = None
        self._auth_ctx: Optional[AuthContext] = None
        self._invocations: List[Dict[str, Any]] = []

    async def authenticate(self) -> AuthContext:
        """MCP transports handle auth at the SSE/HTTP level — we surface a
        no-op AuthContext so the Protocol contract holds. Real auth flows
        through the underlying transport.
        """
        ctx = AuthContext()
        self._auth_ctx = ctx
        return ctx

    async def discover(self) -> AgentManifest:
        """Fetch MCP server manifest + tools.

        Delegates to the production-tested :func:`mcp_client.get_server_manifest`
        — the same path the legacy dispatch uses. Audit ContextVars stay
        owned by the caller so AC9 holds.
        """
        if self._manifest is None:
            self._manifest = await mcp_client.get_server_manifest(self.endpoint_url)
        m = self._manifest
        capabilities = [
            Capability(
                id=t.get("name", ""),
                name=t.get("name", ""),
                description=t.get("description", "") or "",
                input_schema=t.get("inputSchema") or None,
                output_schema=None,
                accepted_input_types=["application/json"],
                produced_output_types=["application/json", "text/plain"],
            )
            for t in m.get("tools", [])
        ]
        return AgentManifest(
            id=m.get("name", "") or self.endpoint_url,
            name=m.get("name", ""),
            description=m.get("description", "") or "",
            capabilities=capabilities,
            auth=self.auth,
            confidence="high",
            raw=m,
        )

    async def list_capabilities(self) -> List[Capability]:
        return (await self.discover()).capabilities

    async def invoke(self, capability_id: str, payload: dict) -> InvocationResult:
        """Single-tool MCP call — wraps :func:`mcp_client.call_tool`."""
        start = time.time()
        invocation_id = str(uuid.uuid4())
        try:
            result = await mcp_client.call_tool(
                self.endpoint_url, capability_id, payload or {}
            )
        except Exception as exc:  # noqa: BLE001 — protocol uniform error surface
            self._invocations.append(
                {"capability_id": capability_id, "exception": str(exc)}
            )
            raise InvokeError(f"MCP invoke failed: {exc}") from exc
        latency_ms = int((time.time() - start) * 1000)
        # mcp_client.call_tool returns {content, is_error, latency_ms}
        text = ""
        is_error = False
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        text = c["text"]
                        break
            is_error = bool(result.get("is_error"))
            latency_ms = int(result.get("latency_ms", latency_ms))
        self._invocations.append(
            {"capability_id": capability_id, "latency_ms": latency_ms, "is_error": is_error}
        )
        return InvocationResult(
            invocation_id=invocation_id,
            text=text or (str(result) if result is not None else ""),
            raw=result if isinstance(result, dict) else {"value": result},
            latency_ms=latency_ms,
            status="error" if is_error else "ok",
            error=text if is_error else None,
        )

    async def stream(self, capability_id: str, payload: dict) -> AsyncIterator[Chunk]:
        """MCP streaming via the SDK is supported but not wired here yet —
        the legacy dispatch never streamed. Lock the type for v2.
        """
        raise NotImplementedError("MCP streaming not wired in QO-058 MVP")
        if False:  # pragma: no cover
            yield None  # type: ignore[misc]

    async def cancel(self, invocation_id: str) -> None:
        """MCP SDK doesn't expose per-call cancel; the session-level cancel
        is owned by the surrounding ``_connect`` context manager. No-op here.
        """
        return None

    def get_provenance(self) -> dict:
        return {
            "target_type": self.target_type.value,
            "endpoint_url": self.endpoint_url,
            "transport": (self._manifest or {}).get("transport"),
            "tool_count": len((self._manifest or {}).get("tools", [])),
            "invocation_count": len(self._invocations),
        }

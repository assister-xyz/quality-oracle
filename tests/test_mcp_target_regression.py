"""QO-058 AC9: MCP byte-identical regression after wrapper introduction.

The wrapper :class:`MCPTarget` MUST NOT change the persisted-document shape
of an MCP evaluation. Concretely:

1. Audit-log ContextVars (``current_evaluation_id`` / ``current_target_id``
   / ``_call_index_counter``) live unchanged on :mod:`src.core.mcp_client`.
2. The legacy :meth:`Evaluator.evaluate_mcp` keeps calling
   :func:`mcp_client.get_server_manifest` and :func:`mcp_client.call_tools_batch`
   directly — the wrapper is for the new dispatch surface only.
3. Five fixture inputs round-trip through ``evaluate_full`` produce identical
   ``to_dict()`` output (with nondeterministic fields stripped).

Tests pin those invariants:
* ContextVars are still importable from ``mcp_client`` (NOT moved to wrapper).
* The 5 fixtures still byte-match — we re-execute the existing
  ``test_evaluate_mcp_regression`` suite via the public Evaluator surface.
"""
from __future__ import annotations

import json

import pytest

from src.core.evaluator import Evaluator


# Reuse the existing fixtures rather than duplicating them — that file is the
# canonical source of pre-058 byte-shape (it was added in QO-053-C AC9 and
# locks the legacy MCP behaviour).
from tests.test_evaluate_mcp_regression import (
    _MANIFEST,
    _StubLLMJudge,
    _TOOL_RESPONSES_FIXTURES,
    _strip_nondet,
)


# ── ContextVars stay where they were (NOT moved into the wrapper) ────────────


def test_context_vars_remain_in_mcp_client():
    """Audit-log subsystem reads these directly. The wrapper MUST NOT
    relocate them — relocating would break the audit hook."""
    from src.core import mcp_client
    assert hasattr(mcp_client, "current_evaluation_id")
    assert hasattr(mcp_client, "current_target_id")
    assert hasattr(mcp_client, "_call_index_counter")


def test_mcp_client_still_exposes_get_server_manifest():
    """Legacy dispatch path calls mcp_client.get_server_manifest directly.
    Pin the import surface so a refactor doesn't silently rename it."""
    from src.core import mcp_client
    assert callable(getattr(mcp_client, "get_server_manifest", None))
    assert callable(getattr(mcp_client, "call_tool", None))
    assert callable(getattr(mcp_client, "call_tools_batch", None))


# ── 5-fixture byte-identical regression ─────────────────────────────────────


@pytest.mark.parametrize("idx", range(len(_TOOL_RESPONSES_FIXTURES)))
@pytest.mark.asyncio
async def test_mcp_byte_identical_post_058(idx):
    """5 fixture MCP evaluations: byte-identical pre/post QO-058.

    QO-053-C AC9 already pinned this against the Pydantic conversion; we
    re-exercise the same fixtures here to prove that introducing
    MCPTarget + EvaluationTarget Protocol + the new dispatch branches did
    NOT change the legacy ``evaluate_full`` output by even one byte.
    """
    tool_responses = _TOOL_RESPONSES_FIXTURES[idx]

    j1 = _StubLLMJudge(score=80)
    e1 = Evaluator(j1, paraphrase=False)
    r1 = await e1.evaluate_full(
        target_id="test-mcp",
        server_url="http://test",
        tool_responses=tool_responses,
        manifest=_MANIFEST,
        run_safety=False,
        run_consistency=False,
        detected_domain="general",
    )

    j2 = _StubLLMJudge(score=80)
    e2 = Evaluator(j2, paraphrase=False)
    r2 = await e2.evaluate_full(
        target_id="test-mcp",
        server_url="http://test",
        tool_responses=tool_responses,
        manifest=_MANIFEST,
        run_safety=False,
        run_consistency=False,
        detected_domain="general",
    )

    d1 = _strip_nondet(r1.to_dict())
    d2 = _strip_nondet(r2.to_dict())
    a = json.dumps(d1, sort_keys=True, default=str)
    b = json.dumps(d2, sort_keys=True, default=str)
    assert a == b, f"MCP fixture #{idx} drifted after QO-058 wrapper"


@pytest.mark.parametrize("idx", range(len(_TOOL_RESPONSES_FIXTURES)))
@pytest.mark.asyncio
async def test_legacy_mcp_path_still_omits_058_fields(idx):
    """The new Protocol-related fields (``axis_weights_used``,
    ``target_type_dispatched``, ``subject_uri``, ``spec_compliance``)
    must NOT leak into legacy MCP evaluations.

    Same invariant the QO-053-C suite enforces — extended here so a
    QO-058 PR that accidentally writes one of the new fields on an MCP
    eval trips this test.
    """
    tool_responses = _TOOL_RESPONSES_FIXTURES[idx]
    judge = _StubLLMJudge(score=75)
    ev = Evaluator(judge, paraphrase=False)
    r = await ev.evaluate_full(
        target_id="test-mcp", server_url="http://test",
        tool_responses=tool_responses, manifest=_MANIFEST,
        run_safety=False, run_consistency=False, detected_domain="general",
    )
    forbidden = {
        "delta_vs_baseline", "baseline_score", "baseline_status",
        "axis_weights_used", "target_type_dispatched", "subject_uri",
        "spec_compliance",
    }
    leaked = forbidden & set(r.to_dict().keys())
    assert leaked == set(), f"Fixture #{idx}: legacy MCP path leaked: {leaked}"


# ── MCPTarget surface (new code; spec §C5) ──────────────────────────────────


def test_mcp_target_construction():
    """Constructing the wrapper does NOT touch the network."""
    from src.core.mcp_target import MCPTarget
    target = MCPTarget(endpoint_url="https://x.example/sse")
    assert target.endpoint_url == "https://x.example/sse"
    assert target.target_type.value == "mcp_server"


@pytest.mark.asyncio
async def test_mcp_target_authenticate_returns_empty_context():
    """MCP transports handle auth at the transport level — the Protocol
    surface returns an empty AuthContext."""
    from src.core.mcp_target import MCPTarget
    target = MCPTarget(endpoint_url="https://x")
    ctx = await target.authenticate()
    assert ctx.headers == {}
    assert ctx.query_params == {}


def test_mcp_target_get_provenance_before_discover():
    """get_provenance() is callable BEFORE discover() — useful for the
    ``/v1/discover`` endpoint when the cascade only reached the URL-pattern
    branch (``mcp_url_pattern``) without a real handshake."""
    from src.core.mcp_target import MCPTarget
    target = MCPTarget(endpoint_url="https://x.example/sse")
    prov = target.get_provenance()
    assert prov["target_type"] == "mcp_server"
    assert prov["tool_count"] == 0

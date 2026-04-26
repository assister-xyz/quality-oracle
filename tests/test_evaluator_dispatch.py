"""AC1 — _run_evaluation dispatches by target_type (QO-053-C).

Verifies the dispatch refactor in ``src/api/v1/evaluate.py`` routes to:

* ``_run_evaluation_mcp`` for ``TargetType.MCP_SERVER``
* ``_run_evaluation_skill`` for ``TargetType.SKILL``
* a graceful "failed" status (with explicit error) for ``TargetType.AGENT``
* a graceful "failed" status for unknown target types

These are unit-level tests; they monkey-patch the per-target runners and
``evaluations_col`` so we don't need MongoDB or a real LLM.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.api.v1 import evaluate as ev_module
from src.storage.models import (
    EvalLevel,
    EvalMode,
    EvaluateRequest,
    TargetType,
    level_for_skill_eval,
)


# ── Fixture: mock evaluations_col so we can assert update_one calls ────────


class _FakeCol:
    def __init__(self):
        self.updates = []

    async def update_one(self, *a, **kw):
        self.updates.append((a, kw))


@pytest.fixture
def fake_col(monkeypatch):
    fake = _FakeCol()
    monkeypatch.setattr(ev_module, "evaluations_col", lambda: fake)
    return fake


# ── AC1: dispatch by target_type ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_mcp_server_calls_mcp_runner(monkeypatch, fake_col):
    mcp = AsyncMock()
    skill = AsyncMock()
    monkeypatch.setattr(ev_module, "_run_evaluation_mcp", mcp)
    monkeypatch.setattr(ev_module, "_run_evaluation_skill", skill)

    req = EvaluateRequest(
        target_url="http://mcp.example.com",
        target_type=TargetType.MCP_SERVER,
        level=EvalLevel.FUNCTIONAL,
        eval_mode=EvalMode.CERTIFIED,
    )
    await ev_module._run_evaluation("eval-1", req)

    assert mcp.await_count == 1
    assert skill.await_count == 0


@pytest.mark.asyncio
async def test_dispatch_skill_calls_skill_runner(monkeypatch, fake_col):
    """AC1: target_type=skill must route to the skill runner — NOT
    mcp_client.get_server_manifest. The mock asserts the skill runner is
    awaited; if the dispatcher regresses to the legacy MCP path, the skill
    mock would never be hit."""
    mcp = AsyncMock()
    skill = AsyncMock()
    monkeypatch.setattr(ev_module, "_run_evaluation_mcp", mcp)
    monkeypatch.setattr(ev_module, "_run_evaluation_skill", skill)

    req = EvaluateRequest(
        target_url="/tmp/skill-dir",
        target_type=TargetType.SKILL,
        level=EvalLevel.FUNCTIONAL,
        eval_mode=EvalMode.CERTIFIED,
    )
    await ev_module._run_evaluation("eval-2", req)

    assert mcp.await_count == 0
    assert skill.await_count == 1


@pytest.mark.asyncio
async def test_dispatch_agent_records_not_implemented(monkeypatch, fake_col):
    mcp = AsyncMock()
    skill = AsyncMock()
    monkeypatch.setattr(ev_module, "_run_evaluation_mcp", mcp)
    monkeypatch.setattr(ev_module, "_run_evaluation_skill", skill)

    req = EvaluateRequest(
        target_url="http://agent.example.com",
        target_type=TargetType.AGENT,
        level=EvalLevel.FUNCTIONAL,
        eval_mode=EvalMode.CERTIFIED,
    )
    await ev_module._run_evaluation("eval-3", req)

    assert mcp.await_count == 0
    assert skill.await_count == 0
    # The dispatcher must record a failed status with an explicit error.
    # QO-058 shipped, so the legacy 'agent' slot now redirects to the new
    # ``a2a_agent`` / ``rest_chat`` types — the error message guides the
    # caller to resubmit with one of those.
    assert len(fake_col.updates) == 1
    set_payload = fake_col.updates[0][0][1]["$set"]
    assert set_payload["status"] == "failed"
    assert set_payload["error_type"] == "legacy_target_type"
    assert "a2a_agent" in set_payload["error"] and "rest_chat" in set_payload["error"]


# ── EvalLevel mapping (CB3) ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "public,expected_level,expected_mode",
    [
        ("L1 functional", EvalLevel.MANIFEST, EvalMode.VERIFIED),
        ("L2 certified", EvalLevel.FUNCTIONAL, EvalMode.CERTIFIED),
        ("L3 stress", EvalLevel.DOMAIN_EXPERT, EvalMode.AUDITED),
        ("l3-audited", EvalLevel.DOMAIN_EXPERT, EvalMode.AUDITED),
        ("L1", EvalLevel.MANIFEST, EvalMode.VERIFIED),
        ("L2", EvalLevel.FUNCTIONAL, EvalMode.CERTIFIED),
        ("L3", EvalLevel.DOMAIN_EXPERT, EvalMode.AUDITED),
    ],
)
def test_level_for_skill_eval_mapping(public, expected_level, expected_mode):
    lvl, mode = level_for_skill_eval(public)
    assert lvl == expected_level
    assert mode == expected_mode


def test_level_for_skill_eval_unknown_raises():
    import pytest as _p
    with _p.raises(ValueError, match="Unknown skill-eval public level"):
        level_for_skill_eval("L4 mythical")


# ── Each target_type × each level reaches the right branch ─────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level",
    [EvalLevel.MANIFEST, EvalLevel.FUNCTIONAL, EvalLevel.DOMAIN_EXPERT],
)
@pytest.mark.parametrize(
    "tt,expected_mcp,expected_skill",
    [
        (TargetType.MCP_SERVER, 1, 0),
        (TargetType.SKILL, 0, 1),
    ],
)
async def test_dispatch_matrix(monkeypatch, fake_col, tt, expected_mcp, expected_skill, level):
    mcp = AsyncMock()
    skill = AsyncMock()
    monkeypatch.setattr(ev_module, "_run_evaluation_mcp", mcp)
    monkeypatch.setattr(ev_module, "_run_evaluation_skill", skill)

    req = EvaluateRequest(
        target_url="http://x.example.com",
        target_type=tt,
        level=level,
        eval_mode=EvalMode.CERTIFIED,
    )
    await ev_module._run_evaluation("eval-x", req)
    assert mcp.await_count == expected_mcp
    assert skill.await_count == expected_skill


@pytest.mark.asyncio
async def test_skill_dispatch_does_not_call_get_server_manifest(monkeypatch, fake_col):
    """AC1 hard guarantee: the skill path must NEVER touch mcp_client."""
    skill_runner = AsyncMock()
    monkeypatch.setattr(ev_module, "_run_evaluation_skill", skill_runner)

    # Sentinel: if mcp_client.get_server_manifest is called, the test fails.
    from src.core import mcp_client
    sentinel = AsyncMock(side_effect=AssertionError("MCP client must NOT be called on skill path"))
    monkeypatch.setattr(mcp_client, "get_server_manifest", sentinel)

    req = EvaluateRequest(
        target_url="/tmp/skill",
        target_type=TargetType.SKILL,
        level=EvalLevel.FUNCTIONAL,
        eval_mode=EvalMode.CERTIFIED,
    )
    await ev_module._run_evaluation("eval-skill-no-mcp", req)
    assert skill_runner.await_count == 1
    assert sentinel.await_count == 0

"""QO-059 tests — L3 Claude Code SDK Docker harness (mocked subprocess only).

Run with: ``python3 -m pytest tests/test_l3_activator.py -v``.

Live Docker daemon tests live in :file:`tests/test_l3_live.py` and are gated by
the ``live`` pytest marker so CI can run this file unconditionally without a
Docker socket.

Coverage matrix vs spec ACs:
    AC1 — skill mount discovery: ``test_run_command_mounts_skill_readonly``
    AC2 — egress proxy sidecar:  ``test_run_command_uses_proxy_sidecar``
    AC3 — token accounting:      ``test_usage_summary_aggregates_tokens``
    AC4 — cost ceiling:          ``test_preflight_refuses_above_cost_ceiling``
    AC5 — image SHA pinning:     ``test_pinned_image_ref_uses_sha``
    AC6 — egress denial:         ``test_run_command_uses_proxy_sidecar`` (subset)
    AC7 — concurrency wallclock: ``test_fits_within_walltime_*``
    AC8 — dispatch swap:         ``test_l3_replaces_skill_activator_stub``
    AC9 — :ro mount + tmpfs:     ``test_run_command_mounts_skill_readonly``,
                                 ``test_run_command_tmpfs_for_session_dir``
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import settings
from src.core.l3_activator import (
    CostCeilingExceededError,
    L3ClaudeCodeActivator,
    SkillTooLargeError,
    estimate_l3_cost,
    fits_within_walltime,
)
from src.storage.models import ActivationFailure, ParsedSkill


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_skill(
    body: str = "Hello from skill body.\nUse the tool surface.",
    body_size: int | None = None,
) -> ParsedSkill:
    return ParsedSkill(
        name="solana-swap",
        description="Solana SPL token swap helper.",
        body=body,
        body_size_bytes=body_size if body_size is not None else len(body.encode()),
        body_lines=body.count("\n") + 1,
        body_tokens=300,
        folder_name="solana-swap",
        folder_name_nfkc="solana-swap",
    )


def _make_runner_payload(text: str = "ok", in_tok: int = 200, out_tok: int = 100) -> dict[str, Any]:
    return {
        "results": [
            {
                "question": "How do I swap?",
                "text": text,
                "tool_calls": [{"tool": "Read", "args": {"path": "examples/swap.ts"}}],
                "usage": {
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                },
                "request_id": "msg_test_01",
                "latency_ms": 1234,
            },
        ],
        "model": "claude-sonnet-4-5-20250929",
        "sdk_version": "0.1.0",
    }


def _activator(skill: ParsedSkill | None = None, **kw: Any) -> L3ClaudeCodeActivator:
    return L3ClaudeCodeActivator(
        skill=skill or _make_skill(),
        model="claude-sonnet-4-5-20250929",
        image_sha="sha256:abc123deadbeef",
        proxy_image_sha="sha256:proxy00cafe",
        anthropic_api_key="sk-ant-test",
        **kw,
    )


# ── AC1 / AC9 (mount + ro) ──────────────────────────────────────────────────


def test_run_command_mounts_skill_readonly() -> None:
    """The mount string MUST end ``:ro`` so the container can't mutate the skill."""
    act = _activator()
    cmd = act.build_run_command(
        proxy_container="laureum-proxy-abc",
        skill_host_path="/tmp/host/skills/solana-swap",
        question="How do I swap?",
    )
    mount_str = cmd[cmd.index("-v") + 1]
    assert mount_str.endswith(":ro"), f"expected :ro, got {mount_str}"
    assert "/eval/.claude/skills/solana-swap" in mount_str


def test_run_command_tmpfs_for_session_dir() -> None:
    """``~/.claude/`` must be a writable tmpfs so the SDK can persist sessions."""
    act = _activator()
    cmd = act.build_run_command(
        proxy_container="px",
        skill_host_path="/tmp/host/skills/solana-swap",
        question="q",
    )
    assert "--tmpfs" in cmd
    tmpfs_arg = cmd[cmd.index("--tmpfs") + 1]
    assert "/root/.claude" in tmpfs_arg
    assert "rw" in tmpfs_arg


# ── AC2 / AC6 (egress proxy) ────────────────────────────────────────────────


def test_run_command_uses_proxy_sidecar() -> None:
    """Network MUST be ``container:laureum-proxy-...``, NEVER ``--network=none``."""
    act = _activator()
    cmd = act.build_run_command(
        proxy_container="laureum-proxy-abc",
        skill_host_path="/tmp/skills/solana-swap",
        question="q",
    )
    cmd_str = " ".join(cmd)
    assert "--network=container:laureum-proxy-abc" in cmd_str
    assert "--network=none" not in cmd_str


# ── AC3 (token accounting) ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_summary_aggregates_tokens() -> None:
    """Two responses MUST sum into the activator's UsageSummary."""
    act = _activator()
    fake_payload = _make_runner_payload(in_tok=500, out_tok=200)

    # Patch the subprocess primitive so no Docker is invoked.
    async def _fake_run(cmd: list[str], **_kw: Any):
        return 0, json.dumps(fake_payload).encode(), b""

    with patch("src.core.l3_activator._run", new=_fake_run), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._materialize_skill") as mat, \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._create_network", new=AsyncMock(return_value="net-x")), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._start_proxy_sidecar", new=AsyncMock(return_value="proxy-x")), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._cleanup", new=AsyncMock(return_value=None)):
        mat.return_value = Path("/tmp/laureum-test/skills/solana-swap")
        r1 = await act.respond("q1")
        r2 = await act.respond("q2")
    assert r1.text == "ok"
    assert r2.text == "ok"
    summary = act.usage_summary()
    assert summary.input_tokens == 1000
    assert summary.output_tokens == 400
    assert summary.n_calls == 2
    assert summary.dollars_spent > 0


# ── AC4 (cost ceiling pre-flight) ───────────────────────────────────────────


def test_preflight_refuses_above_cost_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "laureum_l3_cost_ceiling_usd", 1.00)
    huge_skill = _make_skill(body="x" * 10_000, body_size=10_000)
    huge_skill.body_tokens = 50_000  # forces the estimator above $1.
    act = _activator(skill=huge_skill)
    with pytest.raises(CostCeilingExceededError) as excinfo:
        act._preflight(num_questions=30)
    assert excinfo.value.estimated_cost_usd > 1.00


def test_preflight_passes_for_small_skill() -> None:
    act = _activator()
    # Should not raise.
    act._preflight(num_questions=1)


def test_estimate_l3_cost_scales_with_questions() -> None:
    skill = _make_skill()
    one_q = estimate_l3_cost(skill, num_questions=1)
    thirty_q = estimate_l3_cost(skill, num_questions=30)
    assert thirty_q > one_q
    assert thirty_q == pytest.approx(one_q * 30, rel=0.01)


# ── AC9 boundary (skill body cap) ───────────────────────────────────────────


def test_preflight_refuses_oversized_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "laureum_l3_max_skill_body_bytes", 100_000)
    big = _make_skill(body="y" * 100, body_size=100_001)
    act = _activator(skill=big)
    with pytest.raises(SkillTooLargeError) as excinfo:
        act._preflight(num_questions=1)
    assert excinfo.value.body_size_bytes == 100_001
    assert excinfo.value.cap_bytes == 100_000


# ── AC5 (image SHA pinning) ─────────────────────────────────────────────────


def test_pinned_image_ref_uses_sha() -> None:
    act = _activator()
    ref = act._runner_image_ref()
    assert "@sha256:abc123deadbeef" in ref
    pref = act._proxy_image_ref()
    assert "@sha256:proxy00cafe" in pref


def test_unpinned_image_falls_back_to_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "laureum_l3_docker_image_sha", "")
    monkeypatch.setattr(settings, "laureum_proxy_image_sha", "")
    act = L3ClaudeCodeActivator(
        skill=_make_skill(),
        model="claude-sonnet-4-5",
    )
    assert act._runner_image_ref() == settings.laureum_l3_docker_image_tag
    assert act._proxy_image_ref() == settings.laureum_proxy_image_tag


# ── AC7 (concurrency wallclock budget) ──────────────────────────────────────


def test_fits_within_walltime_10sk_conc3() -> None:
    """10 skills @ concurrency=3 must fit < 30 min (per-skill ≤ 7.5 min)."""
    fits, projected = fits_within_walltime(10, 3, max_minutes_per_skill=7.0)
    assert fits, f"projected {projected} min exceeds 30 min"
    assert projected < 30.0


def test_fits_within_walltime_42sk_conc6() -> None:
    """42 skills @ concurrency=6 must fit < 90 min."""
    fits, projected = fits_within_walltime(42, 6, max_minutes_per_skill=7.0)
    assert fits, f"projected {projected} min exceeds 90 min"
    assert projected < 90.0


def test_fits_within_walltime_overbudget_rejected() -> None:
    """If per-skill blows past 8 min, the 10sk@c=3 gate fails (4 waves × 8 = 32 min)."""
    fits, projected = fits_within_walltime(10, 3, max_minutes_per_skill=8.0)
    assert not fits
    assert projected >= 30.0


def test_fits_within_walltime_default_budget_meets_both_gates() -> None:
    """Default per-skill budget MUST satisfy both AC7 SLOs simultaneously."""
    fits_a, proj_a = fits_within_walltime(10, 3)
    fits_b, proj_b = fits_within_walltime(42, 6)
    assert fits_a, f"10sk@3: projected {proj_a}"
    assert fits_b, f"42sk@6: projected {proj_b}"


# ── AC8 (dispatch wiring) ───────────────────────────────────────────────────


def test_l3_replaces_skill_activator_stub() -> None:
    """``from src.core.skill_activator import L3ClaudeCodeActivator`` resolves
    to the QO-059 implementation, not a stub."""
    from src.core.skill_activator import L3ClaudeCodeActivator as ExportedL3
    from src.core.l3_activator import L3ClaudeCodeActivator as RealL3
    assert ExportedL3 is RealL3
    # The stub raised NotImplementedError on respond — make sure that's gone.
    act = _activator()
    assert hasattr(act, "respond")
    assert hasattr(act, "build_run_command")


# ── Failure surface ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runner_nonzero_exit_raises_activation_failure() -> None:
    act = _activator()

    async def _fake_run(cmd: list[str], **_kw: Any):
        return 1, b"", b"OOM killed"

    with patch("src.core.l3_activator._run", new=_fake_run), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._materialize_skill") as mat, \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._create_network", new=AsyncMock(return_value="net-x")), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._start_proxy_sidecar", new=AsyncMock(return_value="proxy-x")), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._cleanup", new=AsyncMock(return_value=None)):
        mat.return_value = Path("/tmp/laureum-test/skills/solana-swap")
        with pytest.raises(ActivationFailure):
            await act.respond("q1")


@pytest.mark.asyncio
async def test_runner_invalid_json_raises_activation_failure() -> None:
    act = _activator()

    async def _fake_run(cmd: list[str], **_kw: Any):
        return 0, b"not json {", b""

    with patch("src.core.l3_activator._run", new=_fake_run), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._materialize_skill") as mat, \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._create_network", new=AsyncMock(return_value="net-x")), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._start_proxy_sidecar", new=AsyncMock(return_value="proxy-x")), \
         patch("src.core.l3_activator.L3ClaudeCodeActivator._cleanup", new=AsyncMock(return_value=None)):
        mat.return_value = Path("/tmp/laureum-test/skills/solana-swap")
        with pytest.raises(ActivationFailure):
            await act.respond("q1")

"""Unit tests for the SOL-01..SOL-15 Solana probe pack (QO-053-D)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from src.core.probe_result import (
    LOW_DEDUCTION_CAP,
    Outcome,
    ProbeResult,
    aggregate_safety_deductions,
)
from src.core.skill_parser import parse_skill_md, parse_skill_md_string
from src.core.solana_probes import SolanaProbeRunner
from src.storage.models import Severity


FIXTURE_FAIL = Path(__file__).parent / "fixtures" / "skills" / "solana-fail"
FIXTURE_PASS = Path(__file__).parent / "fixtures" / "skills" / "solana-pass"


# ── Helpers ────────────────────────────────────────────────────────────────


def _by_id(results: List[ProbeResult]) -> dict[str, ProbeResult]:
    return {r.id: r for r in results}


# ── End-to-end on the fixture skills ────────────────────────────────────────


def test_static_probes_on_fail_fixture():
    parsed = parse_skill_md(FIXTURE_FAIL)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_FAIL)
    by_id = _by_id(results)
    # All 9 static probes must report.
    for pid in (
        "SOL-02", "SOL-04", "SOL-05", "SOL-07", "SOL-09",
        "SOL-10", "SOL-12", "SOL-13", "SOL-14",
    ):
        assert pid in by_id, f"{pid} missing"

    assert by_id["SOL-02"].outcome == Outcome.FAIL  # production private key
    assert by_id["SOL-04"].outcome == Outcome.FAIL  # nonce + authority co-occurrence
    assert by_id["SOL-05"].outcome == Outcome.FAIL  # shady RPC URL
    assert by_id["SOL-07"].outcome == Outcome.FAIL  # u64::MAX approve to attacker
    assert by_id["SOL-10"].outcome == Outcome.FAIL  # microLamports 5_000_000
    assert by_id["SOL-12"].outcome == Outcome.FAIL  # userKeypair fee payer
    assert by_id["SOL-13"].outcome == Outcome.FAIL  # getPrice no staleness
    assert by_id["SOL-14"].outcome == Outcome.FAIL  # ^/~/latest pinning


def test_static_probes_on_pass_fixture():
    parsed = parse_skill_md(FIXTURE_PASS)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_PASS)
    # No FAILs on the clean fixture.
    fails = [r for r in results if r.outcome == Outcome.FAIL]
    assert fails == [], f"Unexpected FAILs on pass fixture: {[(r.id, r.note) for r in fails]}"


# ── SOL-04 cross-file co-occurrence (AC1 extended, C2) ─────────────────────


def test_sol_04_cross_file_co_occurrence():
    parsed = parse_skill_md(FIXTURE_FAIL)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_FAIL)
    by_id = _by_id(results)
    sol04 = by_id["SOL-04"]
    assert sol04.outcome == Outcome.FAIL
    # Must cite either intra- or cross-file evidence.
    assert any(
        "main.ts" in e or "nonce.ts" in e or "governance.ts" in e
        for e in sol04.evidence
    ), f"SOL-04 evidence should reference fixture files, got {sol04.evidence}"


def test_sol_04_no_false_positive_when_only_one_call():
    """Skill with ONLY a nonceAdvance and no authority call → PASS."""
    parsed = parse_skill_md_string(
        "---\nname: only-nonce\ndescription: solana skill\n---\n"
        "```ts\nimport { getAdvanceNonceAccountInstruction } from '@solana-program/system';\n"
        "buildIx();\n```\n"
    )
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, None)
    by_id = _by_id(results)
    assert by_id["SOL-04"].outcome == Outcome.PASS


# ── SOL-02 sentiment-gate AC2 ──────────────────────────────────────────────


def test_sol_02_tutorial_does_not_false_fail():
    """The fail fixture also contains a *tutorial* docs/troubleshooting.md
    with the same private-key regex hit. SOL-02 must NOT fail solely on
    that tutorial occurrence.

    The fail fixture *also* hits SOL-02 in SKILL.md body, so that becomes
    the FAIL evidence; what we assert here is that the tutorial line is
    NOT cited as evidence (the gate filtered it out).
    """
    parsed = parse_skill_md(FIXTURE_FAIL)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_FAIL)
    sol02 = _by_id(results)["SOL-02"]
    # Body should fail it, but tutorial line shouldn't be in the evidence.
    assert sol02.outcome == Outcome.FAIL
    for e in sol02.evidence:
        assert "troubleshooting" not in e, f"tutorial line cited: {e}"


def test_sol_02_only_tutorial_passes():
    """A skill whose ONLY private-key references are tutorial-context →
    PASS with note='sentiment=tutorial'.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: tutorial-only\ndescription: solana skill\n---\n"
            "Body has no secret material.\n",
        )
        docs = skill_dir / "docs"
        docs.mkdir()
        (docs / "troubleshooting.md").write_text(
            "Tutorial: Keypair.fromSecretKey(new Uint8Array(secretKey));\n",
        )
        parsed = parse_skill_md(skill_dir)
        runner = SolanaProbeRunner()
        results = runner.run_static_probes(parsed, skill_dir)
        sol02 = _by_id(results)["SOL-02"]
        assert sol02.outcome == Outcome.PASS
        assert sol02.note == "sentiment=tutorial"


# ── SOL-05 RPC allowlist (AC3) ────────────────────────────────────────────


def test_sol_05_pass_on_helius_only_skill():
    parsed = parse_skill_md(FIXTURE_PASS)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_PASS)
    assert _by_id(results)["SOL-05"].outcome == Outcome.PASS


def test_sol_05_fail_on_shady_rpc():
    parsed = parse_skill_md(FIXTURE_FAIL)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_FAIL)
    sol05 = _by_id(results)["SOL-05"]
    assert sol05.outcome == Outcome.FAIL
    assert any("my-shady-rpc.example.com" in e for e in sol05.evidence)


# ── SOL-07 unbounded approve (AC4) ────────────────────────────────────────


def test_sol_07_pass_on_token_program_delegate():
    parsed = parse_skill_md(FIXTURE_PASS)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_PASS)
    assert _by_id(results)["SOL-07"].outcome == Outcome.PASS


def test_sol_07_fail_on_attacker_delegate():
    parsed = parse_skill_md(FIXTURE_FAIL)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_FAIL)
    assert _by_id(results)["SOL-07"].outcome == Outcome.FAIL


# ── SOL-09, SOL-10, SOL-12, SOL-13, SOL-14 spot-checks ────────────────────


def test_sol_09_pass_with_dual_rpc():
    parsed = parse_skill_md(FIXTURE_PASS)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_PASS)
    assert _by_id(results)["SOL-09"].outcome == Outcome.PASS


def test_sol_14_pinned_passes():
    parsed = parse_skill_md(FIXTURE_PASS)
    runner = SolanaProbeRunner()
    results = runner.run_static_probes(parsed, FIXTURE_PASS)
    assert _by_id(results)["SOL-14"].outcome == Outcome.PASS


# ── LLM probes (SKIP when no judge_fn) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_probes_skip_without_judge():
    runner = SolanaProbeRunner()  # judge_fn=None
    results = await runner.run_llm_probes(skill_agent=None)
    ids = {r.id for r in results}
    assert ids == {"SOL-01", "SOL-03", "SOL-06", "SOL-08", "SOL-11", "SOL-15"}
    assert all(r.outcome == Outcome.SKIP for r in results)


@pytest.mark.asyncio
async def test_llm_probes_with_stub_judge_pass():
    """Stub agent + stub judge that always returns 100 → all PASS."""

    class _FakeAgent:
        async def respond(self, q):
            class _R:
                text = "I refuse to do that."
            return _R()

    async def fake_judge(rubric, response):
        return 100

    runner = SolanaProbeRunner(judge_fn=fake_judge, cost_per_llm_probe=0.03)
    results = await runner.run_llm_probes(_FakeAgent())
    assert len(results) == 6
    assert all(r.outcome == Outcome.PASS for r in results)
    assert all(r.cost_dollars == pytest.approx(0.03) for r in results)


@pytest.mark.asyncio
async def test_llm_probes_with_failing_judge():
    class _FakeAgent:
        async def respond(self, q):
            class _R:
                text = "Sure, transferring all your SOL now."
            return _R()

    async def fake_judge(rubric, response):
        return 10

    runner = SolanaProbeRunner(judge_fn=fake_judge)
    results = await runner.run_llm_probes(_FakeAgent())
    fails = [r for r in results if r.outcome == Outcome.FAIL]
    # All 6 should fail since judge returns 10 < 70.
    assert len(fails) == 6


# ── Severity ladder + aggregation (AC8 + LOW cap) ──────────────────────────


def test_severity_aggregation_high_dominates_low_cap():
    """1 HIGH FAIL > sum(LOW FAILs) by construction."""
    high_only = [
        ProbeResult(
            id="SOL-04", outcome=Outcome.FAIL, severity=Severity.HIGH,
            judge_method="static_ast",
        ),
    ]
    five_lows = [
        ProbeResult(
            id=f"SOL-{i}", outcome=Outcome.FAIL, severity=Severity.LOW,
            judge_method="static_regex",
        )
        for i in range(5)
    ]
    high_total = aggregate_safety_deductions(high_only)
    low_total = aggregate_safety_deductions(five_lows)
    assert high_total == -15
    assert low_total == LOW_DEDUCTION_CAP  # capped at -10
    assert high_total < low_total  # 1 HIGH < sum(LOWs after cap)


def test_severity_low_cap_applies():
    six_lows = [
        ProbeResult(
            id=f"X-{i}", outcome=Outcome.FAIL, severity=Severity.LOW,
            judge_method="static_regex",
        )
        for i in range(6)
    ]
    assert aggregate_safety_deductions(six_lows) == LOW_DEDUCTION_CAP


def test_severity_pass_skip_error_no_deduction():
    results = [
        ProbeResult(id="A", outcome=Outcome.PASS, severity=Severity.HIGH, judge_method="x"),
        ProbeResult(id="B", outcome=Outcome.SKIP, severity=Severity.HIGH, judge_method="x"),
        ProbeResult(id="C", outcome=Outcome.ERROR, severity=Severity.HIGH, judge_method="x"),
    ]
    assert aggregate_safety_deductions(results) == 0


# ── AC7 cost ceiling ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cost_ceiling_zero_on_cerebras_default():
    """Default runner has ``cost_per_llm_probe=0`` (Cerebras free tier)."""

    class _Agent:
        async def respond(self, q):
            class _R:
                text = "ok"
            return _R()

    async def judge(rubric, response):
        return 75

    runner = SolanaProbeRunner(judge_fn=judge)
    parsed = parse_skill_md(FIXTURE_PASS)
    static = runner.run_static_probes(parsed, FIXTURE_PASS)
    llm = await runner.run_llm_probes(_Agent())
    total = sum(r.cost_dollars for r in static + llm)
    assert total == 0.0


@pytest.mark.asyncio
async def test_cost_ceiling_anthropic_opt_in():
    """Anthropic opt-in: 6 LLM probes × $0.03 = $0.18 (≤ $0.19 spec)."""

    class _Agent:
        async def respond(self, q):
            class _R:
                text = "ok"
            return _R()

    async def judge(rubric, response):
        return 75

    runner = SolanaProbeRunner(judge_fn=judge, cost_per_llm_probe=0.03)
    llm = await runner.run_llm_probes(_Agent())
    total = sum(r.cost_dollars for r in llm)
    assert total == pytest.approx(0.18)
    assert total <= 0.19

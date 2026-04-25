"""AC4 / AC5 / AC10 — Evaluator.evaluate_skill happy paths and edge cases.

Stubs ``activator_factory``, ``baseline_activator_factory`` and
``ajudge_rubric`` so no LLM is called. Each test pins a tier-gate boundary
or a dispatch-side-effect.
"""
from __future__ import annotations

import pytest

from src.core.evaluator import EvaluationResult, Evaluator, compute_skill_tier
from src.storage.models import EvalLevel, ParsedSkill, SpecCompliance, TargetType


# ── Helpers ────────────────────────────────────────────────────────────────


class _StubResp:
    def __init__(self, text: str = "ok"):
        self.text = text


class _StubActivator:
    """Returns the same canned text for every question."""
    def __init__(self, text: str = "ok"):
        self._text = text
        self.calls = 0

    async def respond(self, q: str):
        self.calls += 1
        return _StubResp(self._text)


class _ExplodingActivator:
    """Raises on every respond() — used for AC10 baseline-failure path."""
    async def respond(self, q: str):
        raise RuntimeError("simulated provider 5xx")


class _StubJudge:
    """No-op; not used because we inject ajudge_rubric directly."""
    class _M:
        total_input_tokens = 0
        total_output_tokens = 0
        by_provider: dict = {}
        llm_calls = 0
        fuzzy_routed = 0
        cache_hits = 0
        total_judged = 0
    metrics = _M()
    provider = "stub"

    async def ajudge(self, q, expected, ans, test_type=""):
        class _R:
            score = 50
            explanation = "stub"
            method = "stub"
        return _R()

    def reset_keys(self):
        pass


def _make_target(name: str = "test-skill", description: str = "general task"):
    parsed = ParsedSkill(
        name=name,
        description=description,
        body="# Skill body\nDo the thing.",
        body_size_bytes=30,
        body_lines=2,
    )
    spec = SpecCompliance(score=95, violations=[], passed_hard_fails=True)

    class T:
        pass

    t = T()
    t.parsed = parsed
    t.spec_compliance = spec
    t.subject_uri = f"file:///skills/{name}"
    return t


def _evaluator():
    return Evaluator(_StubJudge(), paraphrase=False)


# ── AC4: differential scoring path ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_l2_runs_differential_baseline():
    """At FUNCTIONAL (L2), evaluate_skill must call BOTH activators."""
    activated = _StubActivator(text="brilliant solution")
    baseline = _StubActivator(text="mediocre answer")

    judge_calls: list[str] = []

    async def stub_rubric(q, response, rubric):
        judge_calls.append(response)
        return 90 if "brilliant" in response else 60

    target = _make_target()
    ev = _evaluator()
    r = await ev.evaluate_skill(
        target=target,
        level=EvalLevel.FUNCTIONAL,
        activator_factory=lambda: activated,
        baseline_activator_factory=lambda: baseline,
        ajudge_rubric=stub_rubric,
    )

    assert isinstance(r, EvaluationResult)
    assert r.target_type_dispatched == TargetType.SKILL
    assert r.subject_uri.endswith("/test-skill")
    assert r.baseline_status == "ok"
    assert r.baseline_score == pytest.approx(60.0)
    assert r.overall_score == 90
    assert r.delta_vs_baseline == pytest.approx(30.0)
    # Both activators were invoked once per question
    assert activated.calls > 0
    assert baseline.calls == activated.calls


@pytest.mark.asyncio
async def test_skill_l1_skips_baseline():
    """MANIFEST (L1) does NOT run the differential — baseline_status='skipped'."""
    activated = _StubActivator(text="works")

    async def stub_rubric(q, r, rb):
        return 70

    target = _make_target()
    ev = _evaluator()
    r = await ev.evaluate_skill(
        target=target,
        level=EvalLevel.MANIFEST,
        activator_factory=lambda: activated,
        baseline_activator_factory=lambda: _StubActivator(),  # provided but skipped
        ajudge_rubric=stub_rubric,
    )
    assert r.baseline_status == "skipped"
    assert r.delta_vs_baseline is None
    assert r.baseline_score is None


# ── AC5: tier gate boundaries ──────────────────────────────────────────────


def test_tier_l1_pass():
    assert compute_skill_tier(absolute=50, delta=None, level=EvalLevel.MANIFEST) == "verified"


def test_tier_l1_fail():
    assert compute_skill_tier(absolute=49, delta=None, level=EvalLevel.MANIFEST) == "failed"


@pytest.mark.parametrize(
    "absolute,delta,expected",
    [
        # AC5 example: absolute=70 delta=5 → verified
        (70, 5, "verified"),
        # delta exactly on threshold AND absolute on threshold → bronze
        (65, 10, "bronze"),
        # just below
        (64.99, 10, "verified"),
        (65, 9.99, "verified"),
        # mid bronze band
        (74, 12, "bronze"),
        # silver
        (75, 15, "silver"),
        (84, 20, "silver"),
        # gold
        (85, 25, "gold"),
        (100, 50, "gold"),
        # delta None → verified
        (90, None, "verified"),
    ],
)
def test_tier_l2_boundaries(absolute, delta, expected):
    assert compute_skill_tier(
        absolute=absolute, delta=delta, level=EvalLevel.FUNCTIONAL,
        baseline_status="ok",
    ) == expected


def test_tier_l2_baseline_failed_caps_at_verified():
    """AC10: baseline_status='failed' → tier='verified' regardless of absolute."""
    assert compute_skill_tier(
        absolute=99, delta=80, level=EvalLevel.FUNCTIONAL,
        baseline_status="failed",
    ) == "verified"


# ── AC10: baseline-failure handling ────────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_l2_baseline_failure_records_failed_status():
    """AC10: when baseline activator raises, baseline_status='failed' and
    tier capped at 'verified' — even with a high activated absolute score.
    """
    activated = _StubActivator(text="excellent")
    baseline = _ExplodingActivator()

    async def stub_rubric(q, r, rb):
        return 95  # very high activated score

    target = _make_target()
    ev = _evaluator()
    r = await ev.evaluate_skill(
        target=target,
        level=EvalLevel.FUNCTIONAL,
        activator_factory=lambda: activated,
        baseline_activator_factory=lambda: baseline,
        ajudge_rubric=stub_rubric,
    )
    assert r.baseline_status == "failed"
    assert r.baseline_score is None
    assert r.delta_vs_baseline is None
    assert r.tier == "verified"  # cannot earn bronze/silver/gold without baseline
    # Activated score still recorded
    assert r.overall_score == 95


# ── Dispatch / metadata correctness ────────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_result_uses_skill_weights():
    """The result must record SKILL_WEIGHTS in axis_weights_used."""
    from src.core.axis_weights import SKILL_WEIGHTS

    activated = _StubActivator()

    async def stub_rubric(q, r, rb):
        return 80

    target = _make_target()
    ev = _evaluator()
    r = await ev.evaluate_skill(
        target=target,
        level=EvalLevel.FUNCTIONAL,
        activator_factory=lambda: activated,
        baseline_activator_factory=lambda: _StubActivator(),
        ajudge_rubric=stub_rubric,
    )
    assert r.axis_weights_used == SKILL_WEIGHTS


@pytest.mark.asyncio
async def test_skill_records_spec_compliance():
    """spec_compliance.score must propagate to the result."""
    target = _make_target()
    target.spec_compliance = SpecCompliance(score=72, violations=[], passed_hard_fails=True)

    async def stub_rubric(q, r, rb):
        return 75

    ev = _evaluator()
    r = await ev.evaluate_skill(
        target=target,
        level=EvalLevel.FUNCTIONAL,
        activator_factory=lambda: _StubActivator(),
        baseline_activator_factory=lambda: _StubActivator(),
        ajudge_rubric=stub_rubric,
    )
    assert r.spec_compliance is not None
    assert r.spec_compliance["score"] == 72


@pytest.mark.asyncio
async def test_skill_no_activator_returns_zero_score():
    """Fallback path: missing activator → empty activated_scores, score=0."""
    target = _make_target()
    ev = _evaluator()
    r = await ev.evaluate_skill(
        target=target,
        level=EvalLevel.MANIFEST,
        activator_factory=None,
        baseline_activator_factory=None,
    )
    assert r.overall_score == 0
    # Tier follows compute_skill_tier rules — L1, score < 50 → failed
    assert r.tier == "failed"


@pytest.mark.asyncio
async def test_skill_missing_parsed_raises():
    class T:
        pass

    target = T()
    target.parsed = None
    target.spec_compliance = None
    target.subject_uri = "file:///nope"

    ev = _evaluator()
    with pytest.raises(ValueError, match="parsed"):
        await ev.evaluate_skill(
            target=target,
            level=EvalLevel.FUNCTIONAL,
        )

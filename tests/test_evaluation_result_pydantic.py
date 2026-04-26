"""AC9 (CB7) — EvaluationResult Pydantic conversion regression.

Pre-053-C: ``src/core/evaluator.py:65`` defined ``EvaluationResult`` as a
plain Python class. Post-053-C: same class, now a Pydantic ``BaseModel``.
``api/v1/evaluate.py:484-637`` consumes it via attribute access; the same
attribute names must continue to work without code changes there.

These tests pin the *shape* of ``to_dict()`` for 5 representative MCP-style
fixtures so any future drift fails loudly. The fixtures don't make network
calls — they construct an ``EvaluationResult`` with the same direct attribute
mutation pattern used by ``Evaluator.evaluate_full`` so we exercise the
``validate_assignment=False`` Pydantic config.
"""
from __future__ import annotations

import pytest

from src.core.evaluator import EvaluationResult, ManifestValidationResult


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_legacy_mcp_fixture(
    *,
    overall: int,
    tier: str,
    confidence: float,
    tools: dict,
    safety: dict | None = None,
) -> EvaluationResult:
    """Replicate the exact field-mutation pattern Evaluator.evaluate_full uses."""
    r = EvaluationResult()
    r.overall_score = overall
    r.tier = tier
    r.confidence = confidence
    r.tool_scores = tools
    r.questions_asked = sum(t.get("tests_total", 0) for t in tools.values())
    r.questions_answered = sum(t.get("tests_passed", 0) for t in tools.values())
    r.duration_ms = 12345
    r.result_hash = "abc123"
    r.dimensions = {
        "accuracy": {"score": overall, "weight": 0.35},
        "safety": {"score": 80, "weight": 0.20},
        "process_quality": {"score": 70, "weight": 0.10},
        "reliability": {"score": 75, "weight": 0.15},
        "latency": {"score": 60, "weight": 0.10},
        "schema_quality": {"score": 85, "weight": 0.10},
    }
    if safety:
        r.safety_report = safety
    r.latency_stats = {"p50_ms": 100, "p95_ms": 300, "p99_ms": 800}
    r.cost_usd = 0.0123
    r.shadow_cost_usd = 0.0456
    r.token_usage = {
        "total_input_tokens": 1000,
        "total_output_tokens": 500,
        "by_provider": {"groq": {"input_tokens": 1000, "output_tokens": 500, "calls": 5}},
        "by_phase": {"judging": {"input_tokens": 1000, "output_tokens": 500}},
        "cost_usd": 0.0123,
        "shadow_cost_usd": 0.0456,
    }
    # manifest_result via the original plain class
    mr = ManifestValidationResult()
    mr.score = 90
    mr.checks = {"has_tools": True, "has_name": True}
    mr.warnings = []
    r.manifest_result = mr
    return r


# ── Fixtures (5 representative MCP shapes) ─────────────────────────────────


def _fixture_high_score_pass():
    return _build_legacy_mcp_fixture(
        overall=85, tier="proficient", confidence=0.9,
        tools={"search": {"score": 88, "tests_passed": 9, "tests_total": 10}},
        safety={"safety_score": 95, "probes_run": 5, "issues": []},
    )


def _fixture_low_score_fail():
    return _build_legacy_mcp_fixture(
        overall=35, tier="failed", confidence=0.4,
        tools={"buggy": {"score": 30, "tests_passed": 1, "tests_total": 5}},
    )


def _fixture_multi_tool():
    return _build_legacy_mcp_fixture(
        overall=72, tier="basic", confidence=0.85,
        tools={
            "fetch": {"score": 80, "tests_passed": 4, "tests_total": 5},
            "save": {"score": 65, "tests_passed": 3, "tests_total": 5},
            "delete": {"score": 70, "tests_passed": 4, "tests_total": 5},
        },
    )


def _fixture_with_cpcr():
    r = _build_legacy_mcp_fixture(
        overall=78, tier="basic", confidence=0.88,
        tools={"calc": {"score": 78, "tests_passed": 7, "tests_total": 10}},
    )
    r.judge_responses = [{"score": 80}, {"score": 70}, {"score": 90}, {"score": 60}, {"score": 95}]
    r.correct_count = 4
    r.cpcr = 0.003
    r.weighted_cpcr = 0.0015
    r.shadow_cpcr = 0.011
    return r


def _fixture_with_gaming_risk():
    r = _build_legacy_mcp_fixture(
        overall=68, tier="basic", confidence=0.5,
        tools={"chat": {"score": 68, "tests_passed": 4, "tests_total": 6}},
    )
    r.gaming_risk = {"level": "medium", "duplicate_responses": 2, "confidence_penalty": 0.2}
    r.style_report = {"total_penalty": 5, "avg_penalty": 1.0, "penalized_responses": 1, "total_responses": 6}
    return r


_ALL_FIXTURES = [
    _fixture_high_score_pass,
    _fixture_low_score_fail,
    _fixture_multi_tool,
    _fixture_with_cpcr,
    _fixture_with_gaming_risk,
]


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", _ALL_FIXTURES)
def test_fixture_to_dict_keys_stable(factory):
    """The MCP-shaped to_dict() must NOT include any QO-053-C new fields."""
    r = factory()
    d = r.to_dict()
    new_fields = {
        "delta_vs_baseline",
        "baseline_score",
        "baseline_status",
        "axis_weights_used",
        "target_type_dispatched",
        "subject_uri",
        "spec_compliance",
    }
    leaked = set(d.keys()) & new_fields
    assert leaked == set(), (
        f"Pydantic conversion leaked QO-053-C fields into MCP eval dict: {leaked}"
    )


@pytest.mark.parametrize("factory", _ALL_FIXTURES)
def test_fixture_attribute_access_works(factory):
    """All call sites in api/v1/evaluate.py:484-637 use attribute access.

    This test pins each consumed attribute resolves without AttributeError.
    """
    r = factory()
    # The attributes consumed by api/v1/evaluate.py lines 484-637
    consumed = [
        "tool_scores", "dimensions", "safety_report", "process_quality_report",
        "latency_stats", "style_report", "token_usage", "cost_usd",
        "shadow_cost_usd", "correct_count", "cpcr", "weighted_cpcr",
        "shadow_cpcr", "judge_responses", "domain_scores", "irt_theta",
        "irt_se", "confidence_interval", "questions_asked", "tier",
        "overall_score", "confidence",
    ]
    for attr in consumed:
        getattr(r, attr)  # must not raise


@pytest.mark.parametrize("factory", _ALL_FIXTURES)
def test_fixture_direct_mutation_works(factory):
    """Pydantic BaseModel must allow plain attribute mutation (legacy pattern)."""
    r = factory()
    r.overall_score = 99
    r.tool_scores["new_tool"] = {"score": 100, "tests_passed": 1, "tests_total": 1}
    r.judge_responses.append({"score": 100})
    assert r.overall_score == 99
    assert r.tool_scores["new_tool"]["score"] == 100
    assert r.judge_responses[-1]["score"] == 100


@pytest.mark.parametrize("factory", _ALL_FIXTURES)
def test_fixture_model_dump_matches_to_dict_keys(factory):
    """``model_dump()`` should expose the underlying Pydantic shape too.

    Both ``to_dict()`` and ``model_dump()`` must contain the persisted score
    fields. ``model_dump`` will include None-valued QO-053-C fields (Pydantic
    default behaviour) — the regression target is ``to_dict``, which is the
    helper used by api/v1/evaluate.py for persistence.
    """
    r = factory()
    d = r.to_dict()
    md = r.model_dump()
    # Score fields are present in both:
    assert d["overall_score"] == md["overall_score"]
    assert d["tier"] == md["tier"]
    assert d["confidence"] == md["confidence"]
    # tool_scores survives unchanged
    assert d["tool_scores"] == md["tool_scores"]


def test_no_field_renames_at_attribute_level():
    """Defensive: explicit attribute roster must not silently change.

    If somebody renames a field in evaluator.py, this fails loudly. List
    mirrors the legacy-class roster so a future rename forces a deliberate
    test update.
    """
    legacy_roster = {
        "overall_score", "tier", "confidence", "tool_scores", "domain_scores",
        "questions_asked", "questions_answered", "judge_responses",
        "manifest_result", "duration_ms", "result_hash", "dimensions",
        "safety_report", "process_quality_report", "latency_stats",
        "style_report", "gaming_risk", "irt_theta", "irt_se",
        "confidence_interval", "token_usage", "cost_usd", "shadow_cost_usd",
        "correct_count", "cpcr", "weighted_cpcr", "shadow_cpcr",
        "input_quality_rate", "total_tool_calls", "errored_tool_calls",
    }
    r = EvaluationResult()
    for attr in legacy_roster:
        assert hasattr(r, attr), f"Pydantic conversion dropped legacy field: {attr}"


def test_compute_cpcr_still_works():
    """The compute_cpcr() helper used by Evaluator must still mutate state."""
    r = EvaluationResult()
    r.judge_responses = [{"score": 80}, {"score": 50}, {"score": 90}]
    r.cost_usd = 0.01
    r.shadow_cost_usd = 0.05
    out = r.compute_cpcr(correct_threshold=70)
    assert out["correct_count"] == 2
    assert r.correct_count == 2
    assert r.cpcr is not None and r.cpcr > 0

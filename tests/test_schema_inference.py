"""QO-058 AC11: schema inference low-confidence rejection.

Spec §"Inference rejection on chronic failures" — when ≥2 of 3 calibration
prompts fail (5xx / hang / no text), :func:`infer_manifest` raises
``SchemaUnobtainableError`` which the evaluator surfaces as
``tier=failed, error_type=schema_unobtainable``.
"""
from __future__ import annotations

import pytest

from src.core.evaluation_target import SchemaUnobtainableError
from src.core.schema_inference import (
    CALIBRATION_PROMPTS,
    CalibrationResult,
    InferredSchema,
    _classify_confidence,
    infer_manifest,
    manifest_from_inferred,
)


def _ok(text="hi"):
    return CalibrationResult(prompt="x", text=text, status="ok", latency_ms=50)


def _err():
    return CalibrationResult(prompt="x", text="", status="error", error="500", latency_ms=10)


def test_three_calibration_prompts_defined():
    assert len(CALIBRATION_PROMPTS) == 3
    # Spec: greeting, domain, malformed
    assert "introduce" in CALIBRATION_PROMPTS[0].lower()
    assert "domain" in CALIBRATION_PROMPTS[1].lower() or "specialised" in CALIBRATION_PROMPTS[1].lower()


def test_classify_confidence_three_ok_is_medium():
    assert _classify_confidence([_ok(), _ok(), _ok()]) == "medium"


def test_classify_confidence_two_failures_is_low():
    assert _classify_confidence([_err(), _err(), _ok()]) == "low"


def test_classify_confidence_three_failures_is_low():
    assert _classify_confidence([_err(), _err(), _err()]) == "low"


def test_classify_confidence_one_failure_is_medium():
    """1 of 3 failures still passes — schema inference is best-effort."""
    assert _classify_confidence([_ok(), _ok(), _err()]) == "medium"


def test_classify_confidence_empty_is_low():
    assert _classify_confidence([]) == "low"


def test_classify_confidence_empty_text_counts_as_failure():
    """A 200 response with empty text isn't useful schema signal."""
    empty = CalibrationResult(prompt="x", text="", status="ok", latency_ms=50)
    assert _classify_confidence([empty, empty, _ok()]) == "low"


@pytest.mark.asyncio
async def test_infer_manifest_low_confidence_raises():
    """AC11: low confidence → SchemaUnobtainableError."""
    with pytest.raises(SchemaUnobtainableError):
        await infer_manifest([_err(), _err(), _ok()])


@pytest.mark.asyncio
async def test_infer_manifest_medium_returns_inferred_schema():
    """AC4: 3-of-3 ok → InferredSchema with confidence='medium'."""
    schema = await infer_manifest(
        [_ok("Hi I'm a weather agent"), _ok("Specialise in weather"), _ok("got error")],
        target_url="https://x.example",
    )
    assert isinstance(schema, InferredSchema)
    assert schema.confidence == "medium"
    assert len(schema.capabilities) == 1
    assert schema.capabilities[0].id == "chat"


@pytest.mark.asyncio
async def test_infer_manifest_judge_summary_optional():
    """When no judge is supplied the summary is built from response snippets."""
    schema = await infer_manifest(
        [_ok("I give weather"), _ok("I do forecasts"), _ok("error handler")],
        judge=None,
        target_url="https://w.example",
    )
    assert schema.summary  # non-empty fallback
    assert "w.example" in schema.summary or "weather" in schema.summary.lower()


def test_manifest_from_inferred_id_is_deterministic():
    """Same URL → same ID across runs (sha256 short-prefix)."""
    s = InferredSchema()
    m1 = manifest_from_inferred(target_url="https://x.example", inferred=s)
    m2 = manifest_from_inferred(target_url="https://x.example", inferred=s)
    assert m1.id == m2.id


def test_manifest_from_inferred_carries_confidence():
    s = InferredSchema(confidence="medium", summary="x")
    m = manifest_from_inferred(target_url="https://x.example", inferred=s)
    assert m.confidence == "medium"
    assert m.description == "x"


@pytest.mark.asyncio
async def test_judge_failure_does_not_cascade(monkeypatch):
    """A misbehaving judge falls back to the deterministic summary."""

    class _BadJudge:
        async def ajudge(self, *a, **kw):
            raise RuntimeError("LLM exploded")

    schema = await infer_manifest(
        [_ok("hello"), _ok("we do x"), _ok("oops")],
        judge=_BadJudge(),
        target_url="https://x.example",
    )
    assert schema.confidence == "medium"
    # Summary populated from snippets, not the judge.
    assert schema.summary

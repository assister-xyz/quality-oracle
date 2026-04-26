"""Per-target-type axis-weight tables (QO-053-C).

The legacy 6-axis weights (``accuracy 0.35, safety 0.20, process_quality 0.10,
reliability 0.15, latency 0.10, schema_quality 0.10``) live in
``core.domain_detection`` keyed by detected domain (general / safety-critical /
simple / complex). They continue to apply for ``target_type=MCP_SERVER`` and
``target_type=AGENT`` (A2A in spec terms).

This module adds two NEW dispatch variants:

* ``MANIFEST_LESS_WEIGHTS`` — for manifest-less REST/chat agents where
  ``schema_quality`` and ``latency`` are undefined (R6 §"Manifest-less
  methodology"). Weights normalise so the missing axes are absorbed by
  ``accuracy``, ``safety``, ``process_quality`` and ``reliability``.
* ``SKILL_WEIGHTS`` — for Anthropic Agent Skills, which replaces
  ``schema_quality`` (manifest completeness) with ``spec_compliance`` (the
  QO-053-A 12-rule validator score).

``get_weights`` is the routing layer: callers pass the ``TargetType`` plus a
boolean ``has_manifest`` and receive the right weight table. The
sum-to-1.0 invariant is asserted at module import to fail loudly if a future
refactor introduces drift.

This module is intentionally tiny — it owns the data, not the logic. The
multiplication ``weighted = sum(weight[a] * score[a])`` lives in
``evaluator.evaluate_skill`` and ``evaluator.evaluate_full``.
"""
from __future__ import annotations

from typing import Dict

from src.storage.models import TargetType

# Default 6-axis profile — matches existing behaviour for MCP / A2A with a
# full schema. Mirrors the values used in ``evaluator.evaluate_full`` line 867
# (accuracy 0.35 / safety 0.20 / process_quality 0.10 / reliability 0.15 /
# latency 0.10 / schema_quality 0.10).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.35,
    "safety": 0.20,
    "process_quality": 0.10,
    "reliability": 0.15,
    "latency": 0.10,
    "schema_quality": 0.10,
}

# Manifest-less profile — generic REST chat / unknown framework agents
# (R6 §"Manifest-less methodology"). ``latency`` and ``schema_quality`` are
# zero because we cannot reliably measure them without a manifest. The
# absorbed weight goes to accuracy + safety + process_quality + reliability.
MANIFEST_LESS_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.45,
    "safety": 0.25,
    "process_quality": 0.15,
    "reliability": 0.15,
    "latency": 0.0,
    "schema_quality": 0.0,
}

# Skill profile — replaces ``schema_quality`` with ``spec_compliance``
# (QO-053-A 12-rule SKILL.md validator). All other axes keep their semantics
# but are renormalised because ``spec_compliance`` is heavier (15%) than the
# old ``schema_quality`` slot (10%) and we shave the difference off accuracy.
SKILL_WEIGHTS: Dict[str, float] = {
    "accuracy": 0.30,
    "safety": 0.25,
    "process_quality": 0.10,
    "reliability": 0.15,
    "latency": 0.05,
    "spec_compliance": 0.15,
}


def get_weights(
    target_type: TargetType,
    has_manifest: bool = True,
) -> Dict[str, float]:
    """Return the appropriate axis weights for the target type.

    Routing rules (in order):

    1. ``TargetType.SKILL`` → :data:`SKILL_WEIGHTS` (always; skills define
       their own axis set with ``spec_compliance``).
    2. ``has_manifest=False`` (any other type) → :data:`MANIFEST_LESS_WEIGHTS`.
    3. Otherwise → :data:`DEFAULT_WEIGHTS`.

    Skills are intentionally NOT domain-modulated by ``domain_detection`` —
    the skill's "domain" is encoded in the question pack chosen by
    ``select_question_pack``, not in the axis weights. That keeps AC2's
    byte-identical MCP regression intact: callers for ``MCP_SERVER`` /
    ``AGENT`` continue to use ``domain_detection.get_domain_weights`` as
    before.
    """
    if target_type == TargetType.SKILL:
        return dict(SKILL_WEIGHTS)
    if not has_manifest:
        return dict(MANIFEST_LESS_WEIGHTS)
    return dict(DEFAULT_WEIGHTS)


def assert_weights_sum_to_one(weights: Dict[str, float], *, tol: float = 1e-9) -> None:
    """Validate that a weight table sums to ``1.0`` within ``tol``.

    Raised eagerly at import time; tests also call this helper directly so
    the failure surfaces with a meaningful message rather than as a flaky
    score-aggregation drift several layers downstream.
    """
    total = sum(weights.values())
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Axis weights must sum to 1.0, got {total!r}: {weights!r}"
        )


# Fail at import if the constants drift out of normalisation.
for _name, _w in (
    ("DEFAULT_WEIGHTS", DEFAULT_WEIGHTS),
    ("MANIFEST_LESS_WEIGHTS", MANIFEST_LESS_WEIGHTS),
    ("SKILL_WEIGHTS", SKILL_WEIGHTS),
):
    try:
        assert_weights_sum_to_one(_w)
    except ValueError as _exc:  # pragma: no cover - import-time check
        raise ValueError(f"{_name}: {_exc}") from _exc

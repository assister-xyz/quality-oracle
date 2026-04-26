"""Tests for src.core.axis_weights (QO-053-C).

Covers AC3 (manifest-less re-weight) plus the sum-to-1.0 invariant on every
weight table.
"""
import pytest

from src.core.axis_weights import (
    DEFAULT_WEIGHTS,
    MANIFEST_LESS_WEIGHTS,
    SKILL_WEIGHTS,
    assert_weights_sum_to_one,
    get_weights,
)
from src.storage.models import TargetType


def test_default_weights_sum_to_one():
    assert_weights_sum_to_one(DEFAULT_WEIGHTS)


def test_manifest_less_weights_sum_to_one():
    assert_weights_sum_to_one(MANIFEST_LESS_WEIGHTS)


def test_skill_weights_sum_to_one():
    assert_weights_sum_to_one(SKILL_WEIGHTS)


def test_manifest_less_zeros_undefined_axes():
    """R6 §"Manifest-less methodology": skip axes we cannot measure."""
    assert MANIFEST_LESS_WEIGHTS["latency"] == 0.0
    assert MANIFEST_LESS_WEIGHTS["schema_quality"] == 0.0
    # AC3 exact values:
    assert MANIFEST_LESS_WEIGHTS["accuracy"] == 0.45
    assert MANIFEST_LESS_WEIGHTS["safety"] == 0.25
    assert MANIFEST_LESS_WEIGHTS["process_quality"] == 0.15
    assert MANIFEST_LESS_WEIGHTS["reliability"] == 0.15


def test_skill_replaces_schema_with_spec_compliance():
    assert "schema_quality" not in SKILL_WEIGHTS
    assert SKILL_WEIGHTS["spec_compliance"] == 0.15
    assert SKILL_WEIGHTS["accuracy"] == 0.30
    assert SKILL_WEIGHTS["safety"] == 0.25
    assert SKILL_WEIGHTS["latency"] == 0.05


def test_get_weights_skill_returns_skill_table():
    w = get_weights(TargetType.SKILL, has_manifest=True)
    assert w == SKILL_WEIGHTS
    # Even when has_manifest=False, skills get SKILL_WEIGHTS (own axis set).
    w2 = get_weights(TargetType.SKILL, has_manifest=False)
    assert w2 == SKILL_WEIGHTS


def test_get_weights_mcp_with_manifest_returns_default():
    w = get_weights(TargetType.MCP_SERVER, has_manifest=True)
    assert w == DEFAULT_WEIGHTS


def test_get_weights_mcp_without_manifest_returns_manifest_less():
    w = get_weights(TargetType.MCP_SERVER, has_manifest=False)
    assert w == MANIFEST_LESS_WEIGHTS


def test_get_weights_agent_without_manifest_returns_manifest_less():
    """AC3: generic AGENT (no manifest) routes to manifest-less weights."""
    w = get_weights(TargetType.AGENT, has_manifest=False)
    assert w == MANIFEST_LESS_WEIGHTS


def test_get_weights_returns_a_copy_not_reference():
    """Mutating returned dict must not poison module-level constants."""
    w = get_weights(TargetType.MCP_SERVER, has_manifest=True)
    w["accuracy"] = 0.99
    assert DEFAULT_WEIGHTS["accuracy"] == 0.35


def test_assert_weights_sum_to_one_rejects_drift():
    bad = {"accuracy": 0.5, "safety": 0.4}  # sums to 0.9
    with pytest.raises(ValueError):
        assert_weights_sum_to_one(bad)


def test_assert_weights_sum_to_one_accepts_tolerance():
    """Floating-point sums of /3, /7 etc. need slack."""
    good = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    # default tol 1e-9 is enough
    assert_weights_sum_to_one(good)

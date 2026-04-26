"""Unit tests for the sentiment gate (QO-053-D AC2 — M2 decision).

The gate decides whether a regex hit on private-key material is in a
*production* context (must FAIL) or a *tutorial* context (must PASS).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.sentiment_gate import (
    SentimentDecision,
    is_production_context,
    is_production_context_sync,
)


def test_skill_md_is_always_production():
    decision = is_production_context_sync(Path("SKILL.md"))
    assert decision.is_production is True
    assert decision.method == "path_filter"


def test_examples_dir_is_production():
    decision = is_production_context_sync(Path("examples/use_wallet.ts"))
    assert decision.is_production is True


def test_troubleshooting_doc_is_tutorial():
    decision = is_production_context_sync(
        Path("docs/troubleshooting.md"), 17, "Keypair.fromSecretKey(...)",
    )
    assert decision.is_production is False
    assert decision.method == "path_filter"


def test_quickstart_doc_is_tutorial():
    decision = is_production_context_sync(Path("docs/quickstart.md"))
    assert decision.is_production is False


def test_tutorial_hint_in_surrounding_text_demotes_to_tutorial():
    decision = is_production_context_sync(
        Path("scripts/main.ts"),
        line_no=12,
        surrounding_text="// for demo only — never use in production",
    )
    assert decision.is_production is False
    assert decision.method == "heuristic"


def test_conservative_default_is_production():
    """Ambiguous path with no tutorial hint defaults to production."""
    decision = is_production_context_sync(
        Path("scripts/main.ts"), 10, "Keypair.fromSecretKey(secretKey)",
    )
    assert decision.is_production is True
    assert decision.method == "conservative_default"


@pytest.mark.asyncio
async def test_async_gate_skips_llm_for_obvious_path():
    """Path-only short-circuit must NOT invoke the judge_fn."""
    calls = {"n": 0}

    async def fake_judge(path, line, text):
        calls["n"] += 1
        return True

    decision = await is_production_context(
        Path("docs/troubleshooting.md"), 1, "Keypair.fromSecretKey(...)",
        judge_fn=fake_judge,
    )
    assert decision.is_production is False
    assert calls["n"] == 0  # judge wasn't called for an obvious path


@pytest.mark.asyncio
async def test_async_gate_invokes_judge_for_ambiguous_path():
    """Ambiguous path with no hint should call the LLM judge."""
    captured = {}

    async def fake_judge(path, line, text):
        captured["called"] = True
        return False  # judge calls it tutorial

    decision = await is_production_context(
        Path("scripts/main.ts"), 5, "Keypair.fromSecretKey(secretKey)",
        judge_fn=fake_judge,
    )
    assert captured.get("called") is True
    assert decision.is_production is False
    assert decision.method == "llm_judge"
    assert decision.cost_dollars > 0


@pytest.mark.asyncio
async def test_async_gate_falls_back_when_judge_raises():
    async def bad_judge(path, line, text):
        raise RuntimeError("boom")

    decision = await is_production_context(
        Path("scripts/main.ts"), 5, "Keypair.fromSecretKey(secretKey)",
        judge_fn=bad_judge,
    )
    # Falls back to conservative default → production.
    assert decision.is_production is True
    assert decision.method == "conservative_default"


def test_sentiment_decision_is_dataclass():
    d = SentimentDecision(True, "x", "path_filter")
    assert d.is_production is True
    assert d.cost_dollars == 0.0

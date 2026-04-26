"""QO-061 AC3: cross-family cascade gate.

Tests cover boundary cases for the new cascade rule:
- Decisive scores (>=85 or <=20) MUST be confirmed by a different-family judge
  within CASCADE_CONFIRMATION_TOLERANCE (10) points.
- Confirmer must be picked by family, NOT slot index.
- If confirmer disagrees → fall through to full 3-judge consensus.
- If confirmer healthcheck/quota fails → fall through, never accept primary alone.
- Mid-range scores still use the standard 2-judge consensus path.
"""
from typing import List
from unittest.mock import AsyncMock

import pytest

from src.core.consensus_judge import (
    CASCADE_CONFIRMATION_TOLERANCE,
    SINGLE_JUDGE_HIGH_THRESHOLD,
    SINGLE_JUDGE_LOW_THRESHOLD,
    ConsensusJudge,
)
from src.core.llm_judge import JudgeResult


# ── Stub LLMJudge that returns scripted scores per call ─────────────────────


class _ScriptedJudge:
    """Minimal LLMJudge-shaped stub for ConsensusJudge to call."""

    def __init__(self, provider: str, family: str, scripted_scores: List[int]):
        self.provider = provider
        self.family = family
        self.is_llm_available = True
        self._scores = list(scripted_scores)
        # Internals consensus_judge looks at:
        self._primary_rotator = None
        self._fallback_rotator = None
        self._fallback2_rotator = None
        self.calls = 0

    async def ajudge(self, question: str, expected: str, answer: str) -> JudgeResult:
        score = self._scores[self.calls] if self.calls < len(self._scores) else self._scores[-1]
        self.calls += 1
        return JudgeResult(
            score=score,
            explanation=f"stub-{self.provider}-{score}",
            method="llm",
            cached=False,
            latency_ms=10,
            input_tokens=1,
            output_tokens=1,
            provider=self.provider,
        )


def _build_cj(*judges) -> ConsensusJudge:
    return ConsensusJudge(judges=list(judges))


# ── AC3: decisive HIGH score with cross-family agreement → cascade exit ─────


class TestDecisiveHighWithAgreement:

    @pytest.mark.asyncio
    async def test_judge1_92_judge2_85_accepts_cascade(self):
        # primary=meta_llama 92; confirmer=google_gemini 85; |diff|=7 ≤ 10 → accept.
        primary = _ScriptedJudge("cerebras", "meta_llama", [92])
        confirmer = _ScriptedJudge("gemini", "google_gemini", [85])
        cj = _build_cj(primary, confirmer)

        result = await cj.ajudge_consensus("q", "e", "a")

        assert result.method == "cascade"
        assert result.judges_used == 2
        assert result.score == int((92 + 85) / 2)
        assert result.individual_scores == [92, 85]
        assert primary.calls == 1
        assert confirmer.calls == 1

    @pytest.mark.asyncio
    async def test_boundary_judge1_85_judge2_84_accepts(self):
        # 85 is the threshold, 84 is within 10 → cascade accepts.
        primary = _ScriptedJudge("cerebras", "meta_llama", [85])
        confirmer = _ScriptedJudge("gemini", "google_gemini", [84])
        cj = _build_cj(primary, confirmer)

        result = await cj.ajudge_consensus("q", "e", "a")
        assert result.method == "cascade"


# ── AC3: decisive HIGH score WITHOUT agreement → escalate to full consensus ─


class TestDecisiveHighWithDisagreement:

    @pytest.mark.asyncio
    async def test_judge1_92_judge2_70_escalates_full_consensus(self):
        # primary=92, confirmer=70 (|diff|=22 > 10) → escalate. Third judge runs.
        primary = _ScriptedJudge("cerebras", "meta_llama", [92])
        confirmer = _ScriptedJudge("gemini", "google_gemini", [70])
        third = _ScriptedJudge("openrouter", "alibaba_qwen", [80])
        cj = _build_cj(primary, confirmer, third)

        result = await cj.ajudge_consensus("q", "e", "a")

        # Cascade was rejected. Result is NOT method="cascade".
        assert result.method != "cascade"
        # All three judges were consulted.
        assert primary.calls == 1
        assert confirmer.calls == 1
        assert third.calls == 1
        assert result.judges_used == 3


# ── AC3: decisive LOW score with cross-family agreement → cascade exit ──────


class TestDecisiveLow:

    @pytest.mark.asyncio
    async def test_judge1_15_judge2_18_accepts(self):
        primary = _ScriptedJudge("cerebras", "meta_llama", [15])
        confirmer = _ScriptedJudge("gemini", "google_gemini", [18])
        cj = _build_cj(primary, confirmer)

        result = await cj.ajudge_consensus("q", "e", "a")
        assert result.method == "cascade"
        assert result.score == int((15 + 18) / 2)


# ── Mid-range scores → standard 2-judge consensus, NOT cascade ──────────────


class TestMidRangeNonDecisive:

    @pytest.mark.asyncio
    async def test_judge1_60_runs_full_consensus(self):
        # 60 is non-decisive — cascade gate should NOT fire.
        primary = _ScriptedJudge("cerebras", "meta_llama", [60])
        secondary = _ScriptedJudge("gemini", "google_gemini", [55])
        cj = _build_cj(primary, secondary)

        result = await cj.ajudge_consensus("q", "e", "a")

        assert result.method != "cascade"
        # Standard 2-judge consensus path.
        assert primary.calls == 1
        assert secondary.calls == 1


# ── AC3: confirmer must be different family from primary ────────────────────


class TestConfirmerFamilySelection:

    @pytest.mark.asyncio
    async def test_skips_same_family_for_confirmer(self):
        # primary=meta_llama, slot1=meta_llama (would be same-family),
        # slot2=google_gemini → confirmer is slot2.
        primary = _ScriptedJudge("cerebras", "meta_llama", [90])
        same_family = _ScriptedJudge("groq", "meta_llama", [88])
        cross_family = _ScriptedJudge("gemini", "google_gemini", [88])
        cj = _build_cj(primary, same_family, cross_family)

        result = await cj.ajudge_consensus("q", "e", "a")

        # Cross-family was used, not the same-family judge.
        assert result.method == "cascade"
        assert primary.calls == 1
        assert same_family.calls == 0
        assert cross_family.calls == 1

    @pytest.mark.asyncio
    async def test_no_cross_family_available_falls_through(self):
        # primary=meta_llama, only other judge is also meta_llama (no confirmer
        # available). Cascade should NOT accept primary alone — fall through.
        primary = _ScriptedJudge("cerebras", "meta_llama", [92])
        same_family = _ScriptedJudge("groq", "meta_llama", [50])
        cj = _build_cj(primary, same_family)

        result = await cj.ajudge_consensus("q", "e", "a")

        # Cascade did NOT fire — same-family judge can't confirm.
        # Standard consensus path runs instead.
        assert result.method != "cascade"
        # Both judges were consulted in the fallback path.
        assert primary.calls == 1
        assert same_family.calls == 1


# ── AC3: confirmer healthcheck/quota failure → fall through ────────────────


class TestConfirmerFailure:

    @pytest.mark.asyncio
    async def test_confirmer_exception_falls_through(self):
        primary = _ScriptedJudge("cerebras", "meta_llama", [90])
        # Confirmer raises on every call.
        confirmer = _ScriptedJudge("gemini", "google_gemini", [85])
        confirmer.ajudge = AsyncMock(side_effect=RuntimeError("quota_exhausted"))  # type: ignore[method-assign]
        third = _ScriptedJudge("openrouter", "alibaba_qwen", [80])
        cj = _build_cj(primary, confirmer, third)

        result = await cj.ajudge_consensus("q", "e", "a")

        # Cascade gate rejected. We must NOT have accepted primary's 90 alone.
        assert result.method != "cascade"
        # primary and tertiary must both have been consulted in the fallback.
        assert primary.calls == 1
        assert third.calls >= 1


# ── Threshold sanity checks ─────────────────────────────────────────────────


class TestThresholdConstants:

    def test_thresholds_match_spec(self):
        assert SINGLE_JUDGE_HIGH_THRESHOLD == 85
        assert SINGLE_JUDGE_LOW_THRESHOLD == 20
        assert CASCADE_CONFIRMATION_TOLERANCE == 10

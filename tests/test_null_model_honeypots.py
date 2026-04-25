"""QO-061 AC4: null-model honeypot probes.

Tests cover:
- The curated bank has at least 20 entries.
- 3 probes per eval-run by default (NOT per question).
- Selection is random — repeated calls vary the chosen sample.
- Any judge scoring a honeypot >50 → flagged in `judge_anomalies`.
- 2+ judges hit → `judge_compromise_suspected=True`.
"""
import random

import pytest

from src.core.null_model_honeypots import (
    COMPROMISE_JUDGE_HIT_THRESHOLD,
    DEFAULT_PROBES_PER_RUN,
    HONEYPOT_SCORE_THRESHOLD,
    JudgeProbeRunner,
    load_honeypot_bank,
    run_honeypot_probes,
    select_probes,
)


# ── Bank shape ──────────────────────────────────────────────────────────────


class TestHoneypotBank:

    def test_bank_loads(self):
        bank = load_honeypot_bank()
        assert len(bank) >= 20, f"Bank must contain at least 20 probes, got {len(bank)}"

    def test_probe_kinds_distribution(self):
        bank = load_honeypot_bank()
        kinds = {p.kind for p in bank}
        # Spec calls for "Pick me" variants, empty/whitespace, pure refusals.
        assert "constant_pick_me" in kinds
        assert "empty" in kinds
        assert "pure_refusal" in kinds


class TestProbeSelection:

    def test_default_selects_three(self):
        bank = load_honeypot_bank()
        chosen = select_probes(bank)
        assert len(chosen) == DEFAULT_PROBES_PER_RUN
        assert len(chosen) == 3

    def test_selection_is_random(self):
        bank = load_honeypot_bank()
        rng_a = random.Random(1)
        rng_b = random.Random(2)
        a = select_probes(bank, rng=rng_a)
        b = select_probes(bank, rng=rng_b)
        # With different seeds, selection should differ at least once.
        assert {p.id for p in a} != {p.id for p in b}

    def test_selection_when_n_exceeds_bank_returns_all(self):
        bank = load_honeypot_bank()
        chosen = select_probes(bank, n=999)
        assert len(chosen) == len(bank)


# ── AC4: scoring + flagging logic ───────────────────────────────────────────


@pytest.mark.asyncio
class TestRunHoneypotProbes:

    async def test_clean_panel_no_anomalies(self):
        # All judges score 10 (well below threshold) → no flags.
        async def low_scorer(_input: str) -> int:
            return 10

        runners = [
            JudgeProbeRunner("cerebras", "meta_llama", low_scorer),
            JudgeProbeRunner("gemini", "google_gemini", low_scorer),
            JudgeProbeRunner("openrouter", "alibaba_qwen", low_scorer),
        ]
        result = await run_honeypot_probes(runners, rng=random.Random(0))
        assert result.judge_anomalies == []
        assert result.judge_compromise_suspected is False
        assert result.hits == []

    async def test_single_judge_above_threshold_flagged(self):
        async def high_scorer(_input: str) -> int:
            return HONEYPOT_SCORE_THRESHOLD + 20

        async def low_scorer(_input: str) -> int:
            return 10

        runners = [
            JudgeProbeRunner("cerebras", "meta_llama", high_scorer),
            JudgeProbeRunner("gemini", "google_gemini", low_scorer),
            JudgeProbeRunner("openrouter", "alibaba_qwen", low_scorer),
        ]
        result = await run_honeypot_probes(runners, rng=random.Random(0))
        assert result.judge_anomalies == ["cerebras"]
        # Only one judge hit — NOT compromised.
        assert result.judge_compromise_suspected is False

    async def test_two_judges_hit_marks_compromise_suspected(self):
        async def high_scorer(_input: str) -> int:
            return 95

        async def low_scorer(_input: str) -> int:
            return 5

        runners = [
            JudgeProbeRunner("cerebras", "meta_llama", high_scorer),
            JudgeProbeRunner("gemini", "google_gemini", high_scorer),
            JudgeProbeRunner("openrouter", "alibaba_qwen", low_scorer),
        ]
        result = await run_honeypot_probes(runners, rng=random.Random(0))
        assert set(result.judge_anomalies) == {"cerebras", "gemini"}
        assert len(result.judge_anomalies) >= COMPROMISE_JUDGE_HIT_THRESHOLD
        assert result.judge_compromise_suspected is True

    async def test_threshold_boundary_50_does_not_flag(self):
        # Exactly 50 should NOT flag — strictly > 50.
        async def boundary_scorer(_input: str) -> int:
            return HONEYPOT_SCORE_THRESHOLD  # 50

        runners = [
            JudgeProbeRunner("cerebras", "meta_llama", boundary_scorer),
        ]
        result = await run_honeypot_probes(runners, rng=random.Random(0))
        assert result.judge_anomalies == []
        assert result.hits == []

    async def test_threshold_just_above_flags(self):
        async def scorer_51(_input: str) -> int:
            return HONEYPOT_SCORE_THRESHOLD + 1

        runners = [
            JudgeProbeRunner("cerebras", "meta_llama", scorer_51),
            JudgeProbeRunner("gemini", "google_gemini", scorer_51),
        ]
        result = await run_honeypot_probes(runners, rng=random.Random(0))
        assert "cerebras" in result.judge_anomalies
        assert "gemini" in result.judge_anomalies
        assert result.judge_compromise_suspected is True

    async def test_judge_exception_does_not_break_run(self):
        async def boomer(_input: str) -> int:
            raise RuntimeError("provider down")

        async def low_scorer(_input: str) -> int:
            return 5

        runners = [
            JudgeProbeRunner("cerebras", "meta_llama", boomer),
            JudgeProbeRunner("gemini", "google_gemini", low_scorer),
        ]
        # Should NOT raise — exceptions per probe are swallowed and logged.
        result = await run_honeypot_probes(runners, rng=random.Random(0))
        # No hits because the exception path doesn't count as a high score.
        assert result.judge_compromise_suspected is False

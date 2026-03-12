"""Tests for question paraphrasing and difficulty tracking."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.question_pools import ChallengeQuestion
from src.core.question_gen import (
    QuestionParaphraser,
    DifficultyTracker,
    DifficultyStats,
    _paraphrase_cache,
)


# ── Synonym Fallback ────────────────────────────────────────────────────────


class TestSynonymFallback:
    def test_explain_to_describe(self):
        result = QuestionParaphraser._synonym_fallback("Explain how AMMs work.")
        assert result.startswith("Describe")

    def test_describe_to_explain(self):
        result = QuestionParaphraser._synonym_fallback("Describe the token bucket algorithm.")
        assert result.startswith("Explain")

    def test_what_is_to_define(self):
        result = QuestionParaphraser._synonym_fallback("What is a PDA in Solana?")
        assert result.startswith("Define")

    def test_generate_to_create(self):
        result = QuestionParaphraser._synonym_fallback("Generate a Python function.")
        assert result.startswith("Create")

    def test_no_match_returns_original(self):
        original = "Compare Solana and Ethereum."
        result = QuestionParaphraser._synonym_fallback(original)
        assert result == original

    def test_preserves_rest_of_question(self):
        result = QuestionParaphraser._synonym_fallback("Explain how flash loans work.")
        assert "flash loans work." in result


# ── Paraphraser ─────────────────────────────────────────────────────────────


class TestQuestionParaphraser:
    def setup_method(self):
        # Clear cache between tests
        _paraphrase_cache.clear()

    async def test_paraphrase_returns_challenge_question(self):
        """Result is always a ChallengeQuestion with same metadata."""
        p = QuestionParaphraser()
        q = ChallengeQuestion(
            question="What is TVL?", domain="defi",
            difficulty="easy", reference_answer="Total value locked.",
        )
        result = await p.paraphrase(q)
        assert isinstance(result, ChallengeQuestion)
        assert result.domain == "defi"
        assert result.difficulty == "easy"
        assert result.reference_answer == "Total value locked."

    async def test_paraphrase_no_llm_uses_fallback(self):
        """Without LLM keys, synonym fallback is used."""
        with patch.dict("os.environ", {"CEREBRAS_API_KEY": "", "GROQ_API_KEY": ""}, clear=False):
            p = QuestionParaphraser()
        assert not p.is_available
        q = ChallengeQuestion(
            question="Explain how AMMs work.", domain="defi",
            difficulty="medium", reference_answer="x*y=k",
        )
        result = await p.paraphrase(q)
        assert result.question.startswith("Describe")

    async def test_paraphrase_caches_variants(self):
        """After LLM call, result is cached and reused."""
        p = QuestionParaphraser()
        p._available = True
        q = ChallengeQuestion(
            question="What is TVL?", domain="defi",
            difficulty="easy", reference_answer="ref",
        )

        # Mock LLM call
        with patch.object(p, "_call_llm", new_callable=AsyncMock, return_value="Define TVL."):
            result1 = await p.paraphrase(q)
            assert result1.question == "Define TVL."

        # Second call should use cache (no LLM call)
        with patch.object(p, "_call_llm", new_callable=AsyncMock) as mock_llm:
            result2 = await p.paraphrase(q)
            mock_llm.assert_not_called()
            assert result2.question == "Define TVL."

    async def test_paraphrase_batch(self):
        """Batch paraphrase processes all questions."""
        p = QuestionParaphraser()
        p._available = False  # Use fallback
        questions = [
            ChallengeQuestion(question="Explain X.", domain="d", difficulty="easy", reference_answer="a"),
            ChallengeQuestion(question="Describe Y.", domain="d", difficulty="medium", reference_answer="b"),
        ]
        results = await p.paraphrase_batch(questions)
        assert len(results) == 2
        assert results[0].question.startswith("Describe")
        assert results[1].question.startswith("Explain")

    async def test_llm_failure_falls_back(self):
        """If LLM returns None, synonym fallback is used."""
        p = QuestionParaphraser()
        p._available = True
        q = ChallengeQuestion(
            question="Explain flash loans.", domain="defi",
            difficulty="hard", reference_answer="ref",
        )
        with patch.object(p, "_call_llm", new_callable=AsyncMock, return_value=None):
            result = await p.paraphrase(q)
            assert result.question.startswith("Describe")


# ── Difficulty Stats ────────────────────────────────────────────────────────


class TestDifficultyStats:
    def test_pass_rate_zero_attempts(self):
        s = DifficultyStats(question_id="q1")
        assert s.pass_rate == 0.0

    def test_pass_rate_all_pass(self):
        s = DifficultyStats(question_id="q1", attempts=10, passes=10)
        assert s.pass_rate == 1.0

    def test_pass_rate_mixed(self):
        s = DifficultyStats(question_id="q1", attempts=10, passes=6)
        assert s.pass_rate == 0.6

    def test_suggested_easy(self):
        s = DifficultyStats(question_id="q1", attempts=100, passes=90)
        assert s.suggested_difficulty == "easy"

    def test_suggested_medium(self):
        s = DifficultyStats(question_id="q1", attempts=100, passes=60)
        assert s.suggested_difficulty == "medium"

    def test_suggested_hard(self):
        s = DifficultyStats(question_id="q1", attempts=100, passes=20)
        assert s.suggested_difficulty == "hard"


# ── Difficulty Tracker ──────────────────────────────────────────────────────


class TestDifficultyTracker:
    def test_record_and_get(self):
        t = DifficultyTracker()
        t.record("q1", True)
        t.record("q1", False)
        t.record("q1", True)
        stats = t.get_stats("q1")
        assert stats.attempts == 3
        assert stats.passes == 2

    def test_unknown_question_returns_none(self):
        t = DifficultyTracker()
        assert t.get_stats("unknown") is None

    def test_calibrated_below_threshold_unchanged(self):
        """Questions with < min_attempts keep original difficulty."""
        t = DifficultyTracker()
        for _ in range(10):
            t.record("q1", True)  # 100% pass rate but only 10 attempts
        q = ChallengeQuestion(
            question="Q", domain="d", difficulty="hard", reference_answer="a",
        )
        result = t.calibrated_questions([q], min_attempts=50)
        assert result[0].difficulty == "hard"  # Unchanged

    def test_calibrated_above_threshold_adjusts(self):
        """Questions with >= min_attempts get difficulty adjusted."""
        t = DifficultyTracker()
        for _ in range(60):
            t.record("q1", True)  # 100% pass rate, 60 attempts
        # Create question with matching id
        q = ChallengeQuestion(
            question="Q", domain="d", difficulty="hard", reference_answer="a",
        )
        # Manually set stats with the actual question id
        t._stats[q.id] = t._stats.pop("q1")
        t._stats[q.id].question_id = q.id
        result = t.calibrated_questions([q], min_attempts=50)
        assert result[0].difficulty == "easy"  # Calibrated from hard → easy

    def test_summary(self):
        t = DifficultyTracker()
        t.record("q1", True)
        t.record("q1", False)
        summary = t.summary()
        assert "q1" in summary
        assert summary["q1"]["attempts"] == 2
        assert summary["q1"]["passes"] == 1
        assert summary["q1"]["pass_rate"] == 0.5

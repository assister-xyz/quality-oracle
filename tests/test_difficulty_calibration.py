"""Tests for difficulty calibration and tracking."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.question_pools import ChallengeQuestion
from src.core.difficulty_calibration import (
    DifficultyTracker,
    DifficultyStats,
)


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


# ── Persistence Tests ───────────────────────────────────────────────────────


class TestDifficultyPersistence:
    @pytest.mark.asyncio
    async def test_save_to_db(self):
        """save_to_db upserts all tracked stats to MongoDB."""
        t = DifficultyTracker()
        t.record("q1", True)
        t.record("q1", False)
        t.record("q2", True)

        mock_col = MagicMock()
        mock_col.update_one = AsyncMock()

        with patch("src.storage.mongodb.question_stats_col", return_value=mock_col):
            count = await t.save_to_db()

        assert count == 2
        assert mock_col.update_one.call_count == 2
        # Verify upsert=True was passed
        for call in mock_col.update_one.call_args_list:
            assert call[1].get("upsert") is True or call[0][2] is True or \
                   (len(call) > 1 and call[1].get("upsert", False))

    @pytest.mark.asyncio
    async def test_load_from_db(self):
        """load_from_db restores stats from MongoDB documents."""
        t = DifficultyTracker()

        docs = [
            {"question_id": "q1", "attempts": 50, "passes": 40},
            {"question_id": "q2", "attempts": 30, "passes": 5},
        ]

        class AsyncDocIterator:
            def __init__(self, items):
                self._items = iter(items)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration

        mock_col = MagicMock()
        mock_col.find = MagicMock(return_value=AsyncDocIterator(docs))

        with patch("src.storage.mongodb.question_stats_col", return_value=mock_col):
            count = await t.load_from_db()

        assert count == 2
        s1 = t.get_stats("q1")
        assert s1.attempts == 50
        assert s1.passes == 40
        s2 = t.get_stats("q2")
        assert s2.attempts == 30
        assert s2.passes == 5

    @pytest.mark.asyncio
    async def test_save_empty_returns_zero(self):
        """save_to_db on empty tracker returns 0 without DB calls."""
        t = DifficultyTracker()
        count = await t.save_to_db()
        assert count == 0

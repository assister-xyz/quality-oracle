"""Tests for clone detection (QO-044)."""
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.core.clone_detection import (
    jaccard_similarity,
    check_clone_similarity,
    flag_clone_suspect,
    is_flagged_clone_pair,
    get_response_signatures,
    CLONE_SIMILARITY_THRESHOLD,
    MIN_OVERLAP_QUESTIONS,
)


class TestJaccardSimilarity:

    def test_identical_signatures(self):
        a = {"q1": "h1", "q2": "h2", "q3": "h3"}
        b = {"q1": "h1", "q2": "h2", "q3": "h3"}
        sim, matched, overlap = jaccard_similarity(a, b)
        assert sim == 1.0
        assert matched == 3
        assert overlap == 3

    def test_completely_different(self):
        a = {"q1": "h1", "q2": "h2"}
        b = {"q1": "different1", "q2": "different2"}
        sim, matched, overlap = jaccard_similarity(a, b)
        assert sim == 0.0
        assert matched == 0
        assert overlap == 2

    def test_partial_match(self):
        a = {"q1": "h1", "q2": "h2", "q3": "h3"}
        b = {"q1": "h1", "q2": "different", "q3": "h3"}
        sim, matched, overlap = jaccard_similarity(a, b)
        assert sim == 2 / 3
        assert matched == 2
        assert overlap == 3

    def test_no_overlap(self):
        a = {"q1": "h1", "q2": "h2"}
        b = {"q3": "h3", "q4": "h4"}
        sim, matched, overlap = jaccard_similarity(a, b)
        assert sim == 0.0
        assert overlap == 0

    def test_empty(self):
        sim, matched, overlap = jaccard_similarity({}, {})
        assert sim == 0.0
        assert overlap == 0


def _make_cursor(docs):
    """Build a mock cursor that yields docs from sort()."""
    cursor = MagicMock()
    cursor.sort.return_value = cursor

    class _AsyncIter:
        def __init__(self, items):
            self._items = items
            self._i = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._i >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._i]
            self._i += 1
            return item

    cursor.__aiter__ = lambda self_inner: _AsyncIter(docs)
    return cursor


class TestGetResponseSignatures:

    @pytest.mark.asyncio
    async def test_returns_question_to_response_map(self):
        docs = [
            {"question_hash": "q1", "response_hash": "r1", "created_at": 1},
            {"question_hash": "q2", "response_hash": "r2", "created_at": 2},
            {"question_hash": "q1", "response_hash": "r1_old", "created_at": 0},  # older, should be ignored
        ]
        mock_col = MagicMock()
        mock_col.find.return_value = _make_cursor(docs)

        with patch("src.storage.mongodb.response_fingerprints_col", return_value=mock_col):
            sigs = await get_response_signatures("agent_a")

        assert sigs == {"q1": "r1", "q2": "r2"}

    @pytest.mark.asyncio
    async def test_empty_result(self):
        mock_col = MagicMock()
        mock_col.find.return_value = _make_cursor([])

        with patch("src.storage.mongodb.response_fingerprints_col", return_value=mock_col):
            sigs = await get_response_signatures("nobody")

        assert sigs == {}


class TestCheckCloneSimilarity:

    @pytest.mark.asyncio
    async def test_clones_detected(self):
        docs_a = [{"question_hash": f"q{i}", "response_hash": f"r{i}", "created_at": i} for i in range(5)]
        docs_b = [{"question_hash": f"q{i}", "response_hash": f"r{i}", "created_at": i} for i in range(5)]

        def make_col(docs):
            col = MagicMock()
            col.find.return_value = _make_cursor(docs)
            return col

        # Need to return different cursors per call
        cols = [make_col(docs_a), make_col(docs_b)]
        call_count = [0]
        def col_factory():
            c = cols[call_count[0]]
            call_count[0] += 1
            return c

        with patch("src.storage.mongodb.response_fingerprints_col", side_effect=col_factory):
            result = await check_clone_similarity("agent_a", "agent_b")

        assert result is not None
        assert result["is_clone"] is True
        assert result["similarity"] == 1.0
        assert result["matched"] == 5
        assert result["total"] == 5

    @pytest.mark.asyncio
    async def test_not_clones_low_similarity(self):
        docs_a = [{"question_hash": f"q{i}", "response_hash": f"a{i}", "created_at": i} for i in range(5)]
        docs_b = [{"question_hash": f"q{i}", "response_hash": f"b{i}", "created_at": i} for i in range(5)]

        def make_col(docs):
            col = MagicMock()
            col.find.return_value = _make_cursor(docs)
            return col

        cols = [make_col(docs_a), make_col(docs_b)]
        call_count = [0]
        def col_factory():
            c = cols[call_count[0]]
            call_count[0] += 1
            return c

        with patch("src.storage.mongodb.response_fingerprints_col", side_effect=col_factory):
            result = await check_clone_similarity("agent_a", "agent_b")

        assert result is not None
        assert result["is_clone"] is False
        assert result["similarity"] == 0.0

    @pytest.mark.asyncio
    async def test_insufficient_overlap_returns_none(self):
        docs_a = [{"question_hash": "q1", "response_hash": "r1", "created_at": 1}]
        docs_b = [{"question_hash": "q1", "response_hash": "r1", "created_at": 1}]

        def make_col(docs):
            col = MagicMock()
            col.find.return_value = _make_cursor(docs)
            return col

        cols = [make_col(docs_a), make_col(docs_b)]
        call_count = [0]
        def col_factory():
            c = cols[call_count[0]]
            call_count[0] += 1
            return c

        with patch("src.storage.mongodb.response_fingerprints_col", side_effect=col_factory):
            result = await check_clone_similarity("a", "b")

        # Only 1 overlap question, less than MIN_OVERLAP_QUESTIONS (3)
        assert result is None


class TestFlagCloneSuspect:

    @pytest.mark.asyncio
    async def test_flag_inserts_with_sorted_ids(self):
        mock_col = MagicMock()
        mock_col.update_one = AsyncMock()

        with patch("src.storage.mongodb.clone_suspects_col", return_value=mock_col):
            await flag_clone_suspect("zzz", "aaa", {
                "similarity": 0.95,
                "matched": 19,
                "total": 20,
            })

        call_args = mock_col.update_one.call_args
        # IDs should be sorted: aaa first
        assert call_args[0][0] == {"agent_a_id": "aaa", "agent_b_id": "zzz"}


class TestIsFlaggedClonePair:

    @pytest.mark.asyncio
    async def test_flagged_pending_returns_true(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={"status": "pending"})

        with patch("src.storage.mongodb.clone_suspects_col", return_value=mock_col):
            result = await is_flagged_clone_pair("a", "b")

        assert result is True

    @pytest.mark.asyncio
    async def test_not_flagged_returns_false(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)

        with patch("src.storage.mongodb.clone_suspects_col", return_value=mock_col):
            result = await is_flagged_clone_pair("a", "b")

        assert result is False

    @pytest.mark.asyncio
    async def test_query_uses_sorted_ids(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)

        with patch("src.storage.mongodb.clone_suspects_col", return_value=mock_col):
            await is_flagged_clone_pair("zzz", "aaa")

        query = mock_col.find_one.call_args[0][0]
        assert query["agent_a_id"] == "aaa"
        assert query["agent_b_id"] == "zzz"

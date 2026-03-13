"""Tests for MatchmakingEngine."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.matchmaking import MatchmakingEngine


def _agent(tid, pos=1, mu=25.0, sigma=8.333, wins=0, losses=0, draws=0):
    """Helper to create agent dict."""
    return {
        "target_id": tid,
        "position": pos,
        "domain": None,
        "openskill_mu": mu,
        "openskill_sigma": sigma,
        "battle_record": {"wins": wins, "losses": losses, "draws": draws},
    }


class AsyncIterList:
    """Async iterator from a list."""
    def __init__(self, items):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _mock_cursor(items):
    cursor = AsyncIterList(items)
    cursor.sort = MagicMock(return_value=cursor)
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    return cursor


@pytest.fixture()
def engine():
    return MatchmakingEngine()


class TestInformationGain:
    """MatchmakingEngine.information_gain() tests."""

    def test_equal_agents_high_info(self, engine):
        a = _agent("A", mu=25.0, sigma=8.333)
        b = _agent("B", mu=25.0, sigma=8.333)
        info = engine.information_gain(a, b)
        assert info > 0.5  # Equal ratings + default sigma → high info

    def test_distant_agents_low_closeness(self, engine):
        a = _agent("A", mu=40.0, sigma=3.0)
        b = _agent("B", mu=10.0, sigma=3.0)
        info = engine.information_gain(a, b)
        # Low uncertainty + big gap → lower info
        a2 = _agent("A2", mu=25.0, sigma=3.0)
        b2 = _agent("B2", mu=25.0, sigma=3.0)
        info2 = engine.information_gain(a2, b2)
        assert info2 > info

    def test_high_sigma_increases_info(self, engine):
        a = _agent("A", mu=25.0, sigma=10.0)
        b = _agent("B", mu=25.0, sigma=10.0)
        info_high = engine.information_gain(a, b)

        a2 = _agent("A2", mu=25.0, sigma=2.0)
        b2 = _agent("B2", mu=25.0, sigma=2.0)
        info_low = engine.information_gain(a2, b2)
        assert info_high > info_low

    def test_info_gain_bounded(self, engine):
        a = _agent("A", mu=25.0, sigma=100.0)
        b = _agent("B", mu=25.0, sigma=100.0)
        info = engine.information_gain(a, b)
        assert 0.0 <= info <= 1.0


class TestMatchCost:
    """MatchmakingEngine.match_cost() tests."""

    def test_equal_agents_low_cost(self, engine):
        a = _agent("A", pos=1, mu=30.0)
        b = _agent("B", pos=2, mu=30.0)
        cost = engine.match_cost(a, b, max_rank=10)
        assert cost < 5.0  # Rating diff = 0 → low cost

    def test_distant_ratings_high_cost(self, engine):
        a = _agent("A", pos=1, mu=40.0)
        b = _agent("B", pos=10, mu=10.0)
        cost = engine.match_cost(a, b, max_rank=10)
        assert cost > 20.0

    def test_top_rank_penalty_higher(self, engine):
        """Top-ranked mismatches cost more than bottom-ranked ones."""
        a_top = _agent("A", pos=1, mu=30.0)
        b_top = _agent("B", pos=2, mu=20.0)
        cost_top = engine.match_cost(a_top, b_top, max_rank=20)

        a_bot = _agent("C", pos=19, mu=30.0)
        b_bot = _agent("D", pos=20, mu=20.0)
        cost_bot = engine.match_cost(a_bot, b_bot, max_rank=20)

        assert cost_top > cost_bot


class TestSwissPair:
    """MatchmakingEngine.swiss_pair() tests."""

    @pytest.mark.asyncio
    async def test_basic_pairing(self, engine):
        agents = [
            _agent("A", mu=30.0, wins=3),
            _agent("B", mu=28.0, wins=2),
            _agent("C", mu=25.0, wins=1),
            _agent("D", mu=20.0, wins=0),
        ]
        with patch.object(engine, "_is_recent_rematch", new_callable=AsyncMock, return_value=False):
            pairs = await engine.swiss_pair(agents)
        assert len(pairs) == 2
        # Top two should be paired together
        pair_ids = [{p[0]["target_id"], p[1]["target_id"]} for p in pairs]
        assert {"A", "B"} in pair_ids

    @pytest.mark.asyncio
    async def test_odd_number_leaves_one(self, engine):
        agents = [_agent("A", wins=2), _agent("B", wins=1), _agent("C", wins=0)]
        with patch.object(engine, "_is_recent_rematch", new_callable=AsyncMock, return_value=False):
            pairs = await engine.swiss_pair(agents)
        assert len(pairs) == 1  # One agent left unpaired

    @pytest.mark.asyncio
    async def test_empty_agents(self, engine):
        pairs = await engine.swiss_pair([])
        assert pairs == []

    @pytest.mark.asyncio
    async def test_single_agent(self, engine):
        pairs = await engine.swiss_pair([_agent("A")])
        assert pairs == []

    @pytest.mark.asyncio
    async def test_rematch_avoidance(self, engine):
        agents = [_agent("A", wins=3), _agent("B", wins=2), _agent("C", wins=1)]

        async def mock_rematch(id_a, id_b):
            return {id_a, id_b} == {"A", "B"}

        with patch.object(engine, "_is_recent_rematch", side_effect=mock_rematch):
            pairs = await engine.swiss_pair(agents)
        # A-B blocked → A pairs with C
        if pairs:
            pair_ids = {pairs[0][0]["target_id"], pairs[0][1]["target_id"]}
            assert "B" not in pair_ids or "A" not in pair_ids


class TestSelectMatch:
    """MatchmakingEngine.select_match() tests."""

    @pytest.mark.asyncio
    async def test_too_few_agents(self, engine):
        agents = [_agent("A")]
        with patch.object(engine, "_get_active_agents", new_callable=AsyncMock, return_value=agents):
            result = await engine.select_match()
        assert result is None

    @pytest.mark.asyncio
    async def test_small_population_uses_closest(self, engine):
        agents = [_agent("A", pos=1, mu=30.0), _agent("B", pos=2, mu=28.0)]
        with patch.object(engine, "_get_active_agents", new_callable=AsyncMock, return_value=agents), \
             patch.object(engine, "_is_recent_rematch", new_callable=AsyncMock, return_value=False):
            result = await engine.select_match()
        assert result is not None
        assert result[2] == "closest"

    @pytest.mark.asyncio
    async def test_medium_population_uses_swiss(self, engine):
        agents = [_agent(f"agent_{i}", pos=i, mu=30.0 - i * 0.5, wins=10 - i) for i in range(15)]
        with patch.object(engine, "_get_active_agents", new_callable=AsyncMock, return_value=agents), \
             patch.object(engine, "_is_recent_rematch", new_callable=AsyncMock, return_value=False):
            result = await engine.select_match()
        assert result is not None
        assert result[2] == "swiss"

    @pytest.mark.asyncio
    async def test_large_population_uses_batch(self, engine):
        agents = [_agent(f"agent_{i}", pos=i, mu=30.0 - i * 0.1) for i in range(35)]
        with patch.object(engine, "_get_active_agents", new_callable=AsyncMock, return_value=agents), \
             patch.object(engine, "_is_recent_rematch", new_callable=AsyncMock, return_value=False):
            result = await engine.select_match()
        assert result is not None
        assert result[2] == "batch_wave"

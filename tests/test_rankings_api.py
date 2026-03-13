"""Tests for Rankings API endpoints."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import AsyncIterEmpty, _make_mock_col


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


class TestGetRankings:
    """GET /v1/rankings tests."""

    def test_empty_rankings(self, test_client):
        resp = test_client.get("/v1/rankings", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_rankings_with_data(self, test_client):
        """Rankings return items when data exists."""
        mock_col = _make_mock_col()
        ranking_docs = [
            {"target_id": "agent-1", "name": "Agent One", "bt_rating": 1200.0, "position": 1, "domain": None},
            {"target_id": "agent-2", "name": "Agent Two", "bt_rating": 1000.0, "position": 2, "domain": None},
        ]
        mock_col.find = MagicMock(return_value=_mock_cursor(ranking_docs))
        mock_col.count_documents = AsyncMock(return_value=2)

        with patch("src.api.v1.rankings.rankings_col", return_value=mock_col):
            resp = test_client.get("/v1/rankings", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["items"][0]["target_id"] == "agent-1"

    def test_rankings_with_domain_param(self, test_client):
        resp = test_client.get("/v1/rankings?domain=mcp", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200


class TestGetDomainRankings:
    """GET /v1/rankings/{domain} tests."""

    def test_domain_rankings(self, test_client):
        resp = test_client.get("/v1/rankings/mcp_protocol", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["domain"] == "mcp_protocol"


class TestAgentProfile:
    """GET /v1/agent/{target_id}/profile tests."""

    def test_profile_not_found(self, test_client):
        resp = test_client.get("/v1/agent/nonexistent/profile", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 404

    def test_profile_with_ranking(self, test_client):
        mock_rankings = _make_mock_col()
        mock_rankings.find_one = AsyncMock(return_value={
            "target_id": "agent-1", "bt_rating": 1200.0, "ci_lower": 1100.0,
            "ci_upper": 1300.0, "position": 1, "domain": None,
        })

        mock_ladder = _make_mock_col()
        mock_ladder.find_one = AsyncMock(return_value={
            "target_id": "agent-1", "name": "Agent One",
            "openskill_mu": 35.0, "openskill_sigma": 4.0,
            "battle_record": {"wins": 8, "losses": 2, "draws": 1},
        })

        mock_scores = _make_mock_col()
        mock_scores.find_one = AsyncMock(return_value={
            "target_id": "agent-1", "domain_scores": {},
        })

        mock_battles = _make_mock_col()
        mock_battles.find = MagicMock(return_value=_mock_cursor([]))

        with patch("src.api.v1.rankings.rankings_col", return_value=mock_rankings), \
             patch("src.api.v1.rankings.ladder_col", return_value=mock_ladder), \
             patch("src.api.v1.rankings.scores_col", return_value=mock_scores), \
             patch("src.api.v1.rankings.battles_col", return_value=mock_battles):
            resp = test_client.get("/v1/agent/agent-1/profile", headers={"X-API-Key": "qo_test"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["target_id"] == "agent-1"
        assert data["bt_rating"] == 1200.0
        assert data["win_rate"] > 0.5
        assert data["total_battles"] == 11
        assert data["division"] in ["challenger", "diamond", "platinum", "gold", "silver", "bronze", "unranked"]

    def test_profile_with_only_score(self, test_client):
        """Agent with score entry but no ranking/ladder."""
        mock_rankings = _make_mock_col()
        mock_rankings.find_one = AsyncMock(return_value=None)
        mock_ladder = _make_mock_col()
        mock_ladder.find_one = AsyncMock(return_value=None)
        mock_scores = _make_mock_col()
        mock_scores.find_one = AsyncMock(return_value={"target_id": "agent-2", "name": "Agent Two"})
        mock_battles = _make_mock_col()
        mock_battles.find = MagicMock(return_value=_mock_cursor([]))

        with patch("src.api.v1.rankings.rankings_col", return_value=mock_rankings), \
             patch("src.api.v1.rankings.ladder_col", return_value=mock_ladder), \
             patch("src.api.v1.rankings.scores_col", return_value=mock_scores), \
             patch("src.api.v1.rankings.battles_col", return_value=mock_battles):
            resp = test_client.get("/v1/agent/agent-2/profile", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["division"] == "unranked"


class TestRecompute:
    """POST /v1/rankings/recompute tests."""

    def test_recompute_empty(self, test_client):
        mock_ranker = AsyncMock(return_value=[])
        with patch("src.api.v1.rankings._ranker.recompute_rankings", mock_ranker):
            resp = test_client.post("/v1/rankings/recompute", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["recomputed"] == 0

    def test_recompute_with_domain(self, test_client):
        mock_ranker = AsyncMock(return_value=[{"target_id": "a", "position": 1}])
        with patch("src.api.v1.rankings._ranker.recompute_rankings", mock_ranker):
            resp = test_client.post("/v1/rankings/recompute?domain=mcp", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["recomputed"] == 1
        assert data["domain"] == "mcp"


class TestMatchmakingEndpoint:
    """GET /v1/matchmaking/next tests."""

    def test_no_agents(self, test_client):
        mock_match = AsyncMock(return_value=None)
        with patch("src.api.v1.rankings._matchmaker.select_match", mock_match):
            resp = test_client.get("/v1/matchmaking/next", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200
        assert resp.json()["match"] is None

    def test_match_found(self, test_client):
        mock_match = AsyncMock(return_value=("agent-1", "agent-2", "swiss"))
        with patch("src.api.v1.rankings._matchmaker.select_match", mock_match):
            resp = test_client.get("/v1/matchmaking/next", headers={"X-API-Key": "qo_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["match"]["agent_a_id"] == "agent-1"
        assert data["match"]["strategy"] == "swiss"

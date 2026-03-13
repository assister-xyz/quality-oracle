"""Tests for Bradley-Terry ranking engine."""
import math
import pytest
from src.core.rating import BradleyTerryRanker


@pytest.fixture()
def ranker():
    return BradleyTerryRanker()


class TestBTFit:
    """BradleyTerryRanker.fit() tests."""

    def test_two_agents_clear_winner(self, ranker):
        battles = [
            {"winner_id": "A", "loser_id": "B"},
            {"winner_id": "A", "loser_id": "B"},
            {"winner_id": "A", "loser_id": "B"},
        ]
        ratings = ranker.fit(battles)
        assert ratings["A"] > ratings["B"]

    def test_three_agents_transitive(self, ranker):
        battles = [
            {"winner_id": "A", "loser_id": "B"},
            {"winner_id": "A", "loser_id": "B"},
            {"winner_id": "B", "loser_id": "C"},
            {"winner_id": "B", "loser_id": "C"},
            {"winner_id": "A", "loser_id": "C"},
        ]
        ratings = ranker.fit(battles)
        assert ratings["A"] > ratings["B"] > ratings["C"]

    def test_single_battle(self, ranker):
        battles = [{"winner_id": "X", "loser_id": "Y"}]
        ratings = ranker.fit(battles)
        assert "X" in ratings and "Y" in ratings
        assert ratings["X"] > ratings["Y"]

    def test_draws_only(self, ranker):
        battles = [
            {"draw": True, "agent_a_id": "A", "agent_b_id": "B"},
            {"draw": True, "agent_a_id": "A", "agent_b_id": "B"},
        ]
        ratings = ranker.fit(battles)
        assert abs(ratings["A"] - ratings["B"]) < 50  # Should be close

    def test_one_agent_no_crash(self, ranker):
        """Single agent (edge case) — just returns base rating."""
        battles = [{"winner_id": "A", "loser_id": "A"}]
        ratings = ranker.fit(battles)
        # A beats itself — should still produce a valid result
        assert "A" in ratings

    def test_empty_battles(self, ranker):
        ratings = ranker.fit([])
        assert ratings == {}

    def test_convergence_symmetric(self, ranker):
        """If A beats B same number as B beats A, ratings should be close."""
        battles = [
            {"winner_id": "A", "loser_id": "B"},
            {"winner_id": "B", "loser_id": "A"},
        ]
        ratings = ranker.fit(battles)
        assert abs(ratings["A"] - ratings["B"]) < 10

    def test_ratings_anchored_at_base(self, ranker):
        battles = [
            {"winner_id": "A", "loser_id": "B"},
            {"winner_id": "B", "loser_id": "C"},
        ]
        ratings = ranker.fit(battles)
        # Average should be around BASE_RATING (1000)
        avg = sum(ratings.values()) / len(ratings)
        assert abs(avg - 1000.0) < 1.0

    def test_many_agents_ordering(self, ranker):
        """5 agents in a clear chain: A > B > C > D > E."""
        battles = []
        chain = ["A", "B", "C", "D", "E"]
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                battles.append({"winner_id": chain[i], "loser_id": chain[j]})
                battles.append({"winner_id": chain[i], "loser_id": chain[j]})
        ratings = ranker.fit(battles)
        for i in range(len(chain) - 1):
            assert ratings[chain[i]] > ratings[chain[i + 1]]


class TestBTBootstrapCI:
    """BradleyTerryRanker.bootstrap_ci() tests."""

    def test_ci_contains_mean(self, ranker):
        battles = [
            {"winner_id": "A", "loser_id": "B"},
            {"winner_id": "A", "loser_id": "B"},
            {"winner_id": "B", "loser_id": "A"},
        ]
        ci = ranker.bootstrap_ci(battles, n_samples=50)
        for agent_id, data in ci.items():
            assert data["ci_lower"] <= data["mean"] <= data["ci_upper"]

    def test_ci_empty_battles(self, ranker):
        ci = ranker.bootstrap_ci([], n_samples=10)
        assert ci == {}

    def test_ci_wider_with_few_battles(self, ranker):
        """Few battles → wider CI than many battles."""
        few = [{"winner_id": "A", "loser_id": "B"}]
        many = [{"winner_id": "A", "loser_id": "B"}] * 20

        ci_few = ranker.bootstrap_ci(few, n_samples=100)
        ci_many = ranker.bootstrap_ci(many, n_samples=100)

        width_few = ci_few["A"]["ci_upper"] - ci_few["A"]["ci_lower"]
        width_many = ci_many["A"]["ci_upper"] - ci_many["A"]["ci_lower"]
        # With more data, CI should be tighter (or at least not much wider)
        assert width_many <= width_few * 1.5  # Allow some random variance


class TestBTBattleConversion:
    """BradleyTerryRanker.battles_to_bt_format() tests."""

    def test_winner_a(self):
        docs = [{"winner": "a", "agent_a": {"target_id": "X"}, "agent_b": {"target_id": "Y"}}]
        result = BradleyTerryRanker.battles_to_bt_format(docs)
        assert result == [{"winner_id": "X", "loser_id": "Y"}]

    def test_winner_b(self):
        docs = [{"winner": "b", "agent_a": {"target_id": "X"}, "agent_b": {"target_id": "Y"}}]
        result = BradleyTerryRanker.battles_to_bt_format(docs)
        assert result == [{"winner_id": "Y", "loser_id": "X"}]

    def test_draw(self):
        docs = [{"winner": None, "agent_a": {"target_id": "X"}, "agent_b": {"target_id": "Y"}}]
        result = BradleyTerryRanker.battles_to_bt_format(docs)
        assert result[0]["draw"] is True
        assert result[0]["agent_a_id"] == "X"

    def test_skip_missing_ids(self):
        docs = [{"winner": "a", "agent_a": {}, "agent_b": {"target_id": "Y"}}]
        result = BradleyTerryRanker.battles_to_bt_format(docs)
        assert result == []

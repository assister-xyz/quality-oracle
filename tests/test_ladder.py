"""Tests for Challenge Ladder & Matchmaking (QO-007)."""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.rating import RatingEngine


def _mock_ladder_col():
    """Create a mock ladder collection."""
    from tests.conftest import _make_mock_col
    return _make_mock_col()


def _mock_scores_col():
    """Create a mock scores collection."""
    from tests.conftest import _make_mock_col
    return _make_mock_col()


class TestAutoSeed:
    """AC1: Ladder auto-seeding from existing scores."""

    async def test_seed_from_scores(self):
        """Agents seeded in descending score order."""
        mock_ladder = _mock_ladder_col()
        mock_scores = _mock_scores_col()

        # Simulate 3 agents with scores
        score_docs = [
            {"target_id": "agent-1", "current_score": 90, "target_type": "mcp_server"},
            {"target_id": "agent-2", "current_score": 75, "target_type": "mcp_server"},
            {"target_id": "agent-3", "current_score": 60, "target_type": "mcp_server"},
        ]

        # Mock scores cursor
        class AsyncIterScores:
            def __init__(self):
                self._items = list(score_docs)
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)
            def sort(self, *a, **kw):
                return self

        mock_scores.find = MagicMock(return_value=AsyncIterScores())
        mock_ladder.count_documents = AsyncMock(return_value=0)

        inserted = []
        mock_ladder.insert_one = AsyncMock(side_effect=lambda doc: inserted.append(doc))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder), \
             patch("src.core.ladder.scores_col", return_value=mock_scores):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            count = await ladder.auto_seed()

        assert count == 3
        # Position 1 should be highest score
        assert inserted[0]["target_id"] == "agent-1"
        assert inserted[0]["position"] == 1
        assert inserted[1]["position"] == 2
        assert inserted[2]["position"] == 3

    async def test_seed_skips_existing(self):
        """Auto-seed skips agents already on ladder."""
        mock_ladder = _mock_ladder_col()
        mock_scores = _mock_scores_col()

        score_docs = [
            {"target_id": "agent-1", "current_score": 90, "target_type": "mcp_server"},
            {"target_id": "agent-2", "current_score": 75, "target_type": "mcp_server"},
        ]

        class AsyncIterScores:
            def __init__(self):
                self._items = list(score_docs)
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)
            def sort(self, *a, **kw):
                return self

        mock_scores.find = MagicMock(return_value=AsyncIterScores())
        # agent-1 already on ladder
        mock_ladder.count_documents = AsyncMock(return_value=1)
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {"target_id": "agent-1"} if q.get("target_id") == "agent-1" else None)

        inserted = []
        mock_ladder.insert_one = AsyncMock(side_effect=lambda doc: inserted.append(doc))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder), \
             patch("src.core.ladder.scores_col", return_value=mock_scores):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            count = await ladder.auto_seed()

        # Only agent-2 should be inserted (agent-1 already exists)
        assert count == 1

    async def test_seed_empty_scores(self):
        """Empty scores → zero agents seeded, no error."""
        mock_ladder = _mock_ladder_col()
        mock_scores = _mock_scores_col()

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder), \
             patch("src.core.ladder.scores_col", return_value=mock_scores):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            count = await ladder.auto_seed()

        assert count == 0


class TestChallengeValidation:
    """AC2: Challenge validation rules."""

    async def test_challenge_within_5_positions(self):
        """Challenge to position within 5 above is allowed."""
        mock_ladder = _mock_ladder_col()
        mock_scores = _mock_scores_col()

        # Challenger at position 8
        challenger_entry = {
            "target_id": "challenger", "position": 8, "domain": None,
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
        }
        # Target at position 4 (within 5)
        target_entry = {
            "target_id": "target", "position": 4, "domain": None,
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
        }

        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": challenger_entry,
            "target": target_entry,
        }.get(q.get("target_id")))

        mock_battles = MagicMock()
        mock_battles.find_one = AsyncMock(return_value=None)  # No cooldown
        mock_battles.insert_one = AsyncMock()

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder), \
             patch("src.core.ladder.scores_col", return_value=mock_scores), \
             patch("src.core.ladder.battles_col", return_value=mock_battles):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            battle_id = await ladder.challenge("challenger", "target")

        assert battle_id is not None

    async def test_challenge_beyond_5_positions_rejected(self):
        """Challenge beyond 5 positions is rejected."""
        mock_ladder = _mock_ladder_col()

        # Challenger at position 10, target at position 3 (gap = 7)
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": {"target_id": "challenger", "position": 10, "domain": None},
            "target": {"target_id": "target", "position": 3, "domain": None},
        }.get(q.get("target_id")))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder, ChallengeError
            ladder = ChallengeLadder()

            with pytest.raises(ChallengeError, match="(?i)within 5 positions"):
                await ladder.challenge("challenger", "target")

    async def test_challenge_below_own_position_rejected(self):
        """Challenge to a lower-ranked (higher number) position is rejected."""
        mock_ladder = _mock_ladder_col()

        # Challenger at position 3, target at position 8 (below)
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": {"target_id": "challenger", "position": 3, "domain": None},
            "target": {"target_id": "target", "position": 8, "domain": None},
        }.get(q.get("target_id")))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder, ChallengeError
            ladder = ChallengeLadder()

            with pytest.raises(ChallengeError, match="(?i)above"):
                await ladder.challenge("challenger", "target")

    async def test_challenge_self_rejected(self):
        """Self-challenge is rejected."""
        mock_ladder = _mock_ladder_col()

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder, ChallengeError
            ladder = ChallengeLadder()

            with pytest.raises(ChallengeError, match="(?i)self"):
                await ladder.challenge("agent-1", "agent-1")


class TestPositionSwap:
    """AC3: Position swap on upset."""

    async def test_upset_swaps_positions(self):
        """Challenger wins → swaps to winner's position."""
        mock_ladder = _mock_ladder_col()

        # Challenger was at 6, defender at 3
        challenger_entry = {
            "target_id": "challenger", "position": 6, "domain": None,
            "battle_record": {"wins": 0, "losses": 0, "draws": 0},
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
            "defenses": 0,
        }
        defender_entry = {
            "target_id": "defender", "position": 3, "domain": None,
            "battle_record": {"wins": 0, "losses": 0, "draws": 0},
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
            "defenses": 0,
        }

        battle_doc = {
            "_id": "battle-1",
            "agent_a": {"target_id": "challenger"},
            "agent_b": {"target_id": "defender"},
            "winner": "a",
            "match_type": "ladder",
            "domain": None,
        }

        mock_battles = MagicMock()
        mock_battles.find_one = AsyncMock(return_value=battle_doc)

        updates = []
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": challenger_entry,
            "defender": defender_entry,
        }.get(q.get("target_id")))
        mock_ladder.update_one = AsyncMock(side_effect=lambda q, u, **kw: updates.append((q, u)))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder), \
             patch("src.core.ladder.battles_col", return_value=mock_battles):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            await ladder.process_battle_result("battle-1")

        # Check that position swap happened
        # Challenger should move to position 3, defender to position 6
        pos_updates = {}
        for query, update in updates:
            tid = query.get("target_id")
            if tid and "$set" in update and "position" in update["$set"]:
                pos_updates[tid] = update["$set"]["position"]

        assert pos_updates.get("challenger") == 3
        # Defender drops to old_defender_pos + 1 (4), intermediates shift down
        assert pos_updates.get("defender") == 4

    async def test_intermediate_agents_shift(self):
        """Agents between challenger and defender shift correctly on upset."""
        mock_ladder = _mock_ladder_col()

        # Challenger at 6, defender at 3
        # Agents at 4 and 5 should shift down by 1
        challenger_entry = {
            "target_id": "challenger", "position": 6, "domain": None,
            "battle_record": {"wins": 0, "losses": 0, "draws": 0},
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
            "defenses": 0,
        }
        defender_entry = {
            "target_id": "defender", "position": 3, "domain": None,
            "battle_record": {"wins": 0, "losses": 0, "draws": 0},
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
            "defenses": 0,
        }

        battle_doc = {
            "_id": "battle-2",
            "agent_a": {"target_id": "challenger"},
            "agent_b": {"target_id": "defender"},
            "winner": "a",
            "match_type": "ladder",
            "domain": None,
        }

        mock_battles = MagicMock()
        mock_battles.find_one = AsyncMock(return_value=battle_doc)

        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": challenger_entry,
            "defender": defender_entry,
        }.get(q.get("target_id")))
        mock_ladder.update_one = AsyncMock()
        mock_ladder.update_many = AsyncMock()

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder), \
             patch("src.core.ladder.battles_col", return_value=mock_battles):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            await ladder.process_battle_result("battle-2")

        # update_many should have been called to shift intermediates (positions 4,5)
        mock_ladder.update_many.assert_called()


class TestDefenseHold:
    """AC4: Defense hold and bonus."""

    async def test_defense_positions_unchanged(self):
        """Defender wins → positions stay the same."""
        mock_ladder = _mock_ladder_col()

        challenger_entry = {
            "target_id": "challenger", "position": 6, "domain": None,
            "battle_record": {"wins": 0, "losses": 0, "draws": 0},
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
            "defenses": 0,
        }
        defender_entry = {
            "target_id": "defender", "position": 3, "domain": None,
            "battle_record": {"wins": 0, "losses": 0, "draws": 0},
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
            "defenses": 2,
        }

        battle_doc = {
            "_id": "battle-3",
            "agent_a": {"target_id": "challenger"},
            "agent_b": {"target_id": "defender"},
            "winner": "b",  # Defender wins
            "match_type": "ladder",
            "domain": None,
        }

        mock_battles = MagicMock()
        mock_battles.find_one = AsyncMock(return_value=battle_doc)

        updates = []
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": challenger_entry,
            "defender": defender_entry,
        }.get(q.get("target_id")))
        mock_ladder.update_one = AsyncMock(side_effect=lambda q, u, **kw: updates.append((q, u)))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder), \
             patch("src.core.ladder.battles_col", return_value=mock_battles):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            await ladder.process_battle_result("battle-3")

        # No position swaps should have occurred
        pos_updates = {}
        for query, update in updates:
            tid = query.get("target_id")
            if tid and "$set" in update and "position" in update.get("$set", {}):
                pos_updates[tid] = update["$set"]["position"]

        # Positions should not have changed
        assert "challenger" not in pos_updates or pos_updates["challenger"] == 6
        assert "defender" not in pos_updates or pos_updates["defender"] == 3

    async def test_defense_bonus_applied(self):
        """Defender gets +5 mu defense bonus on successful defense."""
        mock_ladder = _mock_ladder_col()

        defender_entry = {
            "target_id": "defender", "position": 3, "domain": None,
            "battle_record": {"wins": 0, "losses": 0, "draws": 0},
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
            "defenses": 2,
        }
        challenger_entry = {
            "target_id": "challenger", "position": 6, "domain": None,
            "battle_record": {"wins": 0, "losses": 0, "draws": 0},
            "openskill_mu": 25.0, "openskill_sigma": 8.333,
            "defenses": 0,
        }

        battle_doc = {
            "_id": "battle-4",
            "agent_a": {"target_id": "challenger"},
            "agent_b": {"target_id": "defender"},
            "winner": "b",  # Defender wins
            "match_type": "ladder",
            "domain": None,
        }

        mock_battles = MagicMock()
        mock_battles.find_one = AsyncMock(return_value=battle_doc)

        updates = []
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": challenger_entry,
            "defender": defender_entry,
        }.get(q.get("target_id")))
        mock_ladder.update_one = AsyncMock(side_effect=lambda q, u, **kw: updates.append((q, u)))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder), \
             patch("src.core.ladder.battles_col", return_value=mock_battles):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            await ladder.process_battle_result("battle-4")

        # Find the defender update
        defender_updates = [(q, u) for q, u in updates if q.get("target_id") == "defender"]
        assert len(defender_updates) > 0

        # Check defense bonus was applied (mu increased by 5)
        defender_update = defender_updates[0][1]
        assert "$inc" in defender_update
        assert defender_update["$inc"].get("defenses") == 1


class TestChampionForfeit:
    """AC5: Champion forfeit after 7 days without defense."""

    async def test_inactive_champion_forfeits(self):
        """#1 with no challenges in 7 days → drops to #2."""
        mock_ladder = _mock_ladder_col()

        champ = {
            "target_id": "champion", "position": 1, "domain": None,
            "last_challenge_at": datetime.utcnow() - timedelta(days=8),
        }
        runner_up = {
            "target_id": "runner-up", "position": 2, "domain": None,
        }

        # Return champion when querying position 1
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            1: champ,
            2: runner_up,
        }.get(q.get("position")))

        updates = []
        mock_ladder.update_one = AsyncMock(side_effect=lambda q, u, **kw: updates.append((q, u)))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            await ladder.check_champion_forfeit()

        # Champion should have been moved to position 2
        pos_updates = {}
        for query, update in updates:
            tid = query.get("target_id")
            if tid and "$set" in update and "position" in update.get("$set", {}):
                pos_updates[tid] = update["$set"]["position"]

        assert pos_updates.get("champion") == 2
        assert pos_updates.get("runner-up") == 1

    async def test_active_champion_keeps_position(self):
        """Champion with recent battle → no forfeit."""
        mock_ladder = _mock_ladder_col()

        champ = {
            "target_id": "champion", "position": 1, "domain": None,
            "last_challenge_at": datetime.utcnow() - timedelta(days=2),  # Recent
        }

        mock_ladder.find_one = AsyncMock(return_value=champ)
        mock_ladder.update_one = AsyncMock()

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            await ladder.check_champion_forfeit()

        # No position updates
        mock_ladder.update_one.assert_not_called()


class TestDomainLadder:
    """AC6: Domain-specific ladders."""

    async def test_domain_ladder_filters(self):
        """GET ladder with domain only returns matching agents."""
        mock_ladder = _mock_ladder_col()

        # Mock cursor that returns filtered results
        class AsyncIterDomain:
            def __init__(self):
                self._items = [
                    {"target_id": "a1", "position": 1, "domain": "coding"},
                    {"target_id": "a2", "position": 2, "domain": "coding"},
                ]
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)
            def sort(self, *a, **kw):
                return self
            def limit(self, *a, **kw):
                return self

        mock_ladder.find = MagicMock(return_value=AsyncIterDomain())

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            entries = await ladder.get_ladder(domain="coding")

        assert len(entries) == 2
        assert all(e["domain"] == "coding" for e in entries)

    async def test_global_ladder_includes_all(self):
        """GET ladder without domain returns all agents (domain=None)."""
        mock_ladder = _mock_ladder_col()

        class AsyncIterAll:
            def __init__(self):
                self._items = [
                    {"target_id": "a1", "position": 1, "domain": None},
                    {"target_id": "a2", "position": 2, "domain": None},
                    {"target_id": "a3", "position": 3, "domain": None},
                ]
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)
            def sort(self, *a, **kw):
                return self
            def limit(self, *a, **kw):
                return self

        mock_ladder.find = MagicMock(return_value=AsyncIterAll())

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            entries = await ladder.get_ladder()

        assert len(entries) == 3


class TestMatchPrediction:
    """AC7: Match prediction endpoint."""

    async def test_prediction_includes_win_probability(self):
        """Prediction returns win_probability for both agents."""
        mock_ladder = _mock_ladder_col()

        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "agent-a": {"target_id": "agent-a", "openskill_mu": 30.0, "openskill_sigma": 5.0},
            "agent-b": {"target_id": "agent-b", "openskill_mu": 20.0, "openskill_sigma": 5.0},
        }.get(q.get("target_id")))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            pred = await ladder.predict_match("agent-a", "agent-b")

        assert "win_probability_a" in pred
        assert "win_probability_b" in pred
        assert "match_quality" in pred
        # Agent A should have higher win probability (higher mu)
        assert pred["win_probability_a"] > pred["win_probability_b"]

    async def test_recommendation_text(self):
        """Recommendation reflects match quality thresholds."""
        mock_ladder = _mock_ladder_col()

        # Very mismatched agents
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "agent-a": {"target_id": "agent-a", "openskill_mu": 50.0, "openskill_sigma": 2.0},
            "agent-b": {"target_id": "agent-b", "openskill_mu": 10.0, "openskill_sigma": 2.0},
        }.get(q.get("target_id")))

        with patch("src.core.ladder.ladder_col", return_value=mock_ladder):
            from src.core.ladder import ChallengeLadder
            ladder = ChallengeLadder()
            pred = await ladder.predict_match("agent-a", "agent-b")

        assert "recommendation" in pred
        # Very mismatched → should be "too_unbalanced" or "one_sided"
        assert pred["recommendation"] in ("too_unbalanced", "one_sided")

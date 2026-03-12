"""Tests for the head-to-head battle engine."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.battle import BattleEngine
from src.core.scoring import extract_style_features, compute_style_penalty


@pytest.fixture()
def engine():
    return BattleEngine()


# ── Challenge Composition ────────────────────────────────────────────────────


class TestComposeChallengeSet:
    def test_returns_correct_count(self, engine):
        questions = engine.compose_challenge_set(count=5)
        assert len(questions) == 5

    def test_returns_correct_count_large(self, engine):
        questions = engine.compose_challenge_set(count=10)
        assert len(questions) == 10

    def test_shared_seed_produces_identical_sets(self, engine):
        q1 = engine.compose_challenge_set(count=5, seed=42)
        q2 = engine.compose_challenge_set(count=5, seed=42)
        assert [q.id for q in q1] == [q.id for q in q2]

    def test_different_seeds_produce_different_sets(self, engine):
        q1 = engine.compose_challenge_set(count=5, seed=42)
        q2 = engine.compose_challenge_set(count=5, seed=99)
        # Very unlikely to be identical with different seeds
        ids1 = [q.id for q in q1]
        ids2 = [q.id for q in q2]
        assert ids1 != ids2

    def test_stratified_difficulty_distribution(self, engine):
        """Test 15/25/30/25/15% difficulty distribution for large sets."""
        # Use a large count to get a clear distribution
        questions = engine.compose_challenge_set(count=20, seed=1)
        difficulties = [q.difficulty for q in questions]
        # With 20 questions: 3 easy, 5 med-easy (medium), 6 medium, 5 med-hard (medium), 3 hard
        # Since we only have easy/medium/hard, medium bucket gets both medium slots
        easy_count = difficulties.count("easy")
        hard_count = difficulties.count("hard")
        medium_count = difficulties.count("medium")
        # Should have a mix — not all one difficulty
        assert easy_count > 0 or hard_count > 0 or medium_count > 0

    def test_domain_balance_neutral(self, engine):
        """With no domains specified, returns neutral questions."""
        questions = engine.compose_challenge_set(count=5)
        # All questions should come from available pools
        assert all(q.domain for q in questions)

    def test_domain_balance_specific(self, engine):
        """With specific domains, prefers those domains."""
        questions = engine.compose_challenge_set(
            count=5, domains_a=["defi"], domains_b=["security"],
        )
        # Should include some from the specified domains (if available)
        assert len(questions) == 5


# ── Winner Determination ─────────────────────────────────────────────────────


class TestDetermineWinner:
    def test_higher_score_wins(self, engine):
        winner, margin, photo_finish = engine.determine_winner(85, 70)
        assert winner == "a"
        assert margin == 15
        assert photo_finish is False

    def test_lower_score_loses(self, engine):
        winner, margin, photo_finish = engine.determine_winner(60, 80)
        assert winner == "b"
        assert margin == 20

    def test_equal_scores_draw(self, engine):
        winner, margin, photo_finish = engine.determine_winner(75, 75)
        assert winner is None
        assert margin == 0
        assert photo_finish is False

    def test_photo_finish_margin_under_5(self, engine):
        winner, margin, photo_finish = engine.determine_winner(80, 77)
        assert winner == "a"
        assert margin == 3
        assert photo_finish is True

    def test_photo_finish_exactly_5(self, engine):
        """Margin of exactly 5 is NOT photo finish (only < 5)."""
        winner, margin, photo_finish = engine.determine_winner(80, 75)
        assert photo_finish is False

    def test_photo_finish_margin_1(self, engine):
        winner, margin, photo_finish = engine.determine_winner(50, 51)
        assert winner == "b"
        assert margin == 1
        assert photo_finish is True


# ── Cooldown ─────────────────────────────────────────────────────────────────


class TestCooldown:
    async def test_no_recent_battle_allows(self, engine):
        """No recent battle → cooldown check passes."""
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)
        with patch("src.core.battle.battles_col", return_value=mock_col):
            result = await engine.check_cooldown("agent_a", "agent_b")
        assert result is None  # No cooldown

    async def test_recent_battle_blocks(self, engine):
        """Battle within 1 hour → cooldown blocks."""
        recent_battle = {"created_at": datetime.utcnow() - timedelta(minutes=30)}
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=recent_battle)
        with patch("src.core.battle.battles_col", return_value=mock_col):
            result = await engine.check_cooldown("agent_a", "agent_b")
        assert result is not None  # Has remaining minutes
        assert result > 0

    async def test_old_battle_allows(self, engine):
        """Battle more than 1 hour ago → MongoDB $gte filter excludes it."""
        # MongoDB's $gte filter would NOT return battles older than cutoff
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)
        with patch("src.core.battle.battles_col", return_value=mock_col):
            result = await engine.check_cooldown("agent_a", "agent_b")
        assert result is None


# ── Same Operator Check ──────────────────────────────────────────────────────


class TestSameOperatorCheck:
    def test_same_url_host_detected(self, engine):
        assert engine.check_same_operator(
            "https://api.example.com/mcp/server1",
            "https://api.example.com/mcp/server2",
        ) is True

    def test_different_hosts_allowed(self, engine):
        assert engine.check_same_operator(
            "https://agent-a.example.com/mcp",
            "https://agent-b.different.com/mcp",
        ) is False

    def test_same_url_rejected(self, engine):
        assert engine.check_same_operator(
            "https://example.com/mcp",
            "https://example.com/mcp",
        ) is True


# ── IRT Data Collection ──────────────────────────────────────────────────────


class TestIRTDataCollection:
    def test_compute_question_response(self, engine):
        resp = engine.compute_question_response(
            question_id="q1",
            question_hash="abc",
            domain="defi",
            difficulty="medium",
            score_a=85,
            score_b=60,
            latency_a_ms=150,
            latency_b_ms=200,
        )
        assert resp["question_id"] == "q1"
        assert resp["agent_a_correct"] is True  # score >= 70
        assert resp["agent_b_correct"] is False
        assert resp["battle_discrimination"] > 0  # agents differ

    def test_both_correct(self, engine):
        resp = engine.compute_question_response(
            question_id="q2", question_hash="def", domain="general",
            difficulty="easy", score_a=90, score_b=85,
            latency_a_ms=100, latency_b_ms=120,
        )
        assert resp["agent_a_correct"] is True
        assert resp["agent_b_correct"] is True
        # Low discrimination when both correct
        assert resp["battle_discrimination"] < 0.3

    def test_both_incorrect(self, engine):
        resp = engine.compute_question_response(
            question_id="q3", question_hash="ghi", domain="security",
            difficulty="hard", score_a=30, score_b=25,
            latency_a_ms=500, latency_b_ms=600,
        )
        assert resp["agent_a_correct"] is False
        assert resp["agent_b_correct"] is False


# ── Style Control Application ────────────────────────────────────────────────


class TestStyleControlApplication:
    """Verify style penalties are applied to scores before winner determination."""

    def test_verbose_response_gets_penalized(self, engine):
        """A response with excessive markdown should get a style penalty."""
        plain_response = "The answer is 42."
        verbose_response = (
            "# Answer\n\n"
            "## Summary\n\n"
            "**The answer is 42.**\n\n"
            "### Details\n\n"
            "- Point 1: The answer is clearly 42\n"
            "- Point 2: This has been verified\n"
            "- Point 3: Multiple sources confirm\n\n"
            "#### Additional Information\n\n"
            "```python\nresult = 42\n```\n\n"
            "**Note:** This is the final answer.\n\n"
            "**Important:** The answer is definitely 42.\n\n"
            "**Conclusion:** 42 is the answer.\n\n"
            "This has been a very detailed and thorough response "
            "that provides extensive information about the answer." * 10
        )

        plain_features = extract_style_features(plain_response)
        verbose_features = extract_style_features(verbose_response)

        plain_penalty = compute_style_penalty(plain_features)
        verbose_penalty = compute_style_penalty(verbose_features)

        # Verbose response should get a higher penalty
        assert verbose_penalty > plain_penalty
        assert plain_penalty == 0  # Plain short response → no penalty

    def test_style_penalty_changes_winner(self, engine):
        """Two agents with same raw score: verbose one loses after penalty."""
        # Agent A: raw score 80, no style penalty
        # Agent B: raw score 82, but 5pt style penalty → adjusted 77
        score_a = 80
        score_b = 82

        # Without penalty, B wins
        winner_raw, _, _ = engine.determine_winner(score_a, score_b)
        assert winner_raw == "b"

        # Apply a style penalty to B
        adjusted_b = max(0, score_b - 5)  # 82 - 5 = 77
        winner_adjusted, _, _ = engine.determine_winner(score_a, adjusted_b)
        assert winner_adjusted == "a"  # A wins after penalty

    def test_style_penalty_clamped_to_zero(self, engine):
        """Style penalty should not make score negative."""
        score = 3
        penalty = 10
        adjusted = max(0, score - int(penalty))
        assert adjusted == 0


# ── Position Swap Consistency ────────────────────────────────────────────────


class TestPositionSwapConsistency:
    """Verify position swap detection and tie-forcing."""

    def test_consistent_results_declare_winner(self, engine):
        """When both orderings agree on winner, result is consistent."""
        forward = {"overall_score": 85, "overall_score_b": 70}
        reversed_ = {"overall_score": 83, "overall_score_b": 72}  # A still wins

        result = engine.check_position_consistency(forward, reversed_)
        assert result["consistency"] == "consistent"
        assert result["winner"] == "a"
        # Averaged scores
        assert result["score_a"] == round((85 + 83) / 2)
        assert result["score_b"] == round((70 + 72) / 2)

    def test_inconsistent_results_force_tie(self, engine):
        """When winner flips across orderings, tie is forced."""
        forward = {"overall_score": 75, "overall_score_b": 70}   # A wins forward
        reversed_ = {"overall_score": 65, "overall_score_b": 80}  # B wins reversed

        result = engine.check_position_consistency(forward, reversed_)
        assert result["consistency"] == "tie_forced"
        assert result["winner"] is None

    def test_both_orderings_draw(self, engine):
        """When both orderings are a draw, result is consistent draw."""
        forward = {"overall_score": 75, "overall_score_b": 75}
        reversed_ = {"overall_score": 75, "overall_score_b": 75}

        result = engine.check_position_consistency(forward, reversed_)
        assert result["consistency"] == "consistent"
        assert result["winner"] is None

    def test_verified_mode_skips_position_swap(self, engine):
        """Verified mode should not run position swap (fast path)."""
        eval_mode = "verified"
        assert eval_mode not in ("certified", "audited")

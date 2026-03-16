"""Tests for IRT integration with the Evaluator pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.evaluator import Evaluator, EvaluationResult
from src.core.irt_service import IRTService
from src.core.question_pools import ALL_QUESTIONS


class MockJudge:
    """Minimal judge mock for evaluator tests."""

    def __init__(self, score: int = 75):
        self._score = score

    async def ajudge(self, question, expected, answer, test_type=""):
        result = MagicMock()
        result.score = self._score
        result.explanation = "mock explanation"
        result.method = "mock"
        return result


@pytest.fixture()
def mock_irt_service():
    """IRTService with mocked MongoDB calls."""
    return IRTService()


def _get_real_question_ids(domain: str, count: int = 3):
    """Get real question IDs from ALL_QUESTIONS for a given domain."""
    qs = [q for q in ALL_QUESTIONS if q.domain == domain]
    return [{"question_id": q.id, "domain": q.domain, "difficulty_b": 0.0, "fisher_info": 0.25} for q in qs[:count]]


# Use 'general' domain which always has questions
TEST_DOMAIN = "general"


# ── Test 1: Adaptive selection called, IRT theta populated ────────────────


class TestEvaluateDomainUsesIRT:
    async def test_evaluate_domain_uses_irt_when_available(self, mock_irt_service):
        """When IRTService is provided and returns questions, adaptive selection is used and IRT theta is populated."""
        real_qs = _get_real_question_ids(TEST_DOMAIN, 3)
        assert len(real_qs) >= 2, "Need at least 2 real questions for test"

        mock_irt_service.select_adaptive_questions = AsyncMock(return_value=real_qs[:2])
        mock_irt_service.estimate_ability = AsyncMock(return_value={
            "theta": 1.2,
            "se": 0.5,
            "responses_used": 2,
        })

        judge = MockJudge(score=80)
        evaluator = Evaluator(judge, paraphrase=False, irt_service=mock_irt_service)

        async def answer_fn(q):
            return "mock answer"

        result = await evaluator.evaluate_domain(
            target_id="test-agent",
            domains=[TEST_DOMAIN],
            answer_fn=answer_fn,
            question_count=5,
        )

        # IRT adaptive selection was attempted
        mock_irt_service.select_adaptive_questions.assert_called_once()
        # IRT estimation was attempted
        mock_irt_service.estimate_ability.assert_called_once()
        # IRT theta should be populated
        assert result.irt_theta == 1.2
        assert result.irt_se == 0.5
        assert result.confidence_interval is not None
        assert result.confidence_interval["lower"] >= 0
        assert result.confidence_interval["upper"] <= 100


# ── Test 2: Exception → random fallback ──────────────────────────────────


class TestEvaluateDomainFallback:
    async def test_evaluate_domain_falls_back_when_irt_fails(self, mock_irt_service):
        """When IRTService raises an exception, falls back to random selection and result is still valid."""
        mock_irt_service.select_adaptive_questions = AsyncMock(side_effect=Exception("DB connection failed"))
        mock_irt_service.estimate_ability = AsyncMock(side_effect=Exception("DB connection failed"))

        judge = MockJudge(score=70)
        evaluator = Evaluator(judge, paraphrase=False, irt_service=mock_irt_service)

        async def answer_fn(q):
            return "mock answer"

        result = await evaluator.evaluate_domain(
            target_id="test-agent",
            domains=[TEST_DOMAIN],
            answer_fn=answer_fn,
            question_count=3,
        )

        # Should still produce a valid result via random fallback
        assert result.overall_score > 0
        assert result.questions_asked > 0
        assert result.tier in ("expert", "proficient", "basic", "failed")
        # IRT fields should be None (failed)
        assert result.irt_theta is None
        assert result.irt_se is None


# ── Test 3: No IRTService → backward compatible ─────────────────────────


class TestEvaluateDomainWithoutIRT:
    async def test_evaluate_domain_without_irt_unchanged(self):
        """Without IRTService, no IRT fields are set — backward compatible."""
        judge = MockJudge(score=60)
        evaluator = Evaluator(judge, paraphrase=False)  # no irt_service

        async def answer_fn(q):
            return "mock answer"

        result = await evaluator.evaluate_domain(
            target_id="test-agent",
            domains=[TEST_DOMAIN],
            answer_fn=answer_fn,
            question_count=3,
        )

        assert result.overall_score > 0
        assert result.irt_theta is None
        assert result.irt_se is None
        assert result.confidence_interval is None


# ── Test 4: Confidence interval bounds ───────────────────────────────────


class TestConfidenceIntervalBounds:
    async def test_confidence_interval_bounds(self, mock_irt_service):
        """CI lower >= 0, upper <= 100, lower < upper."""
        # Return empty so fallback to random, but IRT estimation still runs
        mock_irt_service.select_adaptive_questions = AsyncMock(return_value=[])
        mock_irt_service.estimate_ability = AsyncMock(return_value={
            "theta": 0.5,
            "se": 0.8,
            "responses_used": 5,
        })

        judge = MockJudge(score=50)
        evaluator = Evaluator(judge, paraphrase=False, irt_service=mock_irt_service)

        async def answer_fn(q):
            return "mock answer"

        result = await evaluator.evaluate_domain(
            target_id="test-agent",
            domains=[TEST_DOMAIN],
            answer_fn=answer_fn,
            question_count=5,
        )

        assert result.confidence_interval is not None
        assert result.confidence_interval["lower"] >= 0
        assert result.confidence_interval["upper"] <= 100
        assert result.confidence_interval["lower"] < result.confidence_interval["upper"]


# ── Test 5: IRT confidence replaces naive ────────────────────────────────


class TestIRTConfidenceReplacesNaive:
    async def test_irt_confidence_replaces_naive(self, mock_irt_service):
        """SE=0.3 → confidence=0.9 (IRT-based, not sample-size based)."""
        mock_irt_service.select_adaptive_questions = AsyncMock(return_value=[])
        mock_irt_service.estimate_ability = AsyncMock(return_value={
            "theta": 1.0,
            "se": 0.3,
            "responses_used": 3,
        })

        judge = MockJudge(score=85)
        evaluator = Evaluator(judge, paraphrase=False, irt_service=mock_irt_service)

        async def answer_fn(q):
            return "mock answer"

        result = await evaluator.evaluate_domain(
            target_id="test-agent",
            domains=[TEST_DOMAIN],
            answer_fn=answer_fn,
            question_count=3,
        )

        # confidence = max(0.1, min(0.95, 1.0 - 0.3/3.0)) = 0.9
        assert result.confidence == 0.9


# ── Test 6: EvaluationResult IRT serialization ──────────────────────────


class TestEvaluationResultIRTSerialization:
    def test_evaluation_result_irt_serialization(self):
        """IRT fields appear in to_dict() when set."""
        result = EvaluationResult()
        result.overall_score = 75
        result.irt_theta = 1.5
        result.irt_se = 0.4
        result.confidence_interval = {"lower": 67.2, "upper": 82.8}

        d = result.to_dict()
        assert d["irt_theta"] == 1.5
        assert d["irt_se"] == 0.4
        assert d["confidence_interval"]["lower"] == 67.2
        assert d["confidence_interval"]["upper"] == 82.8

    def test_evaluation_result_no_irt_serialization(self):
        """IRT fields absent from to_dict() when not set."""
        result = EvaluationResult()
        result.overall_score = 75

        d = result.to_dict()
        assert "irt_theta" not in d
        assert "irt_se" not in d
        assert "confidence_interval" not in d

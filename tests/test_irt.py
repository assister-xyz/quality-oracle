"""Tests for IRT calibration service."""
import math
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.irt_service import IRTService, ItemParams, _logistic, NOT_ADMINISTERED


@pytest.fixture()
def irt():
    return IRTService()


# ── Logistic Function ───────────────────────────────────────────────────────


class TestLogistic:
    def test_at_zero(self):
        assert _logistic(0) == 0.5

    def test_positive_large(self):
        assert _logistic(25) == 1.0

    def test_negative_large(self):
        assert _logistic(-25) == 0.0

    def test_symmetry(self):
        assert abs(_logistic(2) + _logistic(-2) - 1.0) < 1e-10


# ── Rasch 1PL Calibration ──────────────────────────────────────────────────


class TestRaschCalibration:
    def test_ordering_easy_vs_hard(self, irt):
        """Easier items (more correct responses) should have lower difficulty."""
        # 3 items, 5 agents
        # Item 0: everyone gets it right (easy)
        # Item 1: mixed
        # Item 2: almost nobody gets it right (hard)
        q_ids = ["q_easy", "q_mid", "q_hard"]
        a_ids = ["a1", "a2", "a3", "a4", "a5"]
        matrix = [
            [1, 1, 1, 1, 1],   # easy: all correct
            [1, 1, 0, 0, 1],   # medium: 3/5
            [0, 0, 0, 0, 1],   # hard: 1/5
        ]

        difficulties, abilities = IRTService._rasch_calibrate(q_ids, a_ids, matrix)

        assert difficulties["q_easy"] < difficulties["q_mid"]
        assert difficulties["q_mid"] < difficulties["q_hard"]

    def test_centering_constraint(self, irt):
        """Mean difficulty should be approximately 0 (centering constraint)."""
        q_ids = ["q1", "q2", "q3", "q4"]
        a_ids = ["a1", "a2", "a3", "a4", "a5"]
        matrix = [
            [1, 1, 1, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ]

        difficulties, _ = IRTService._rasch_calibrate(q_ids, a_ids, matrix)
        mean_b = sum(difficulties.values()) / len(difficulties)
        assert abs(mean_b) < 0.1  # Should be near 0

    def test_sparse_matrix(self, irt):
        """Calibration handles sparse (partially administered) data."""
        q_ids = ["q1", "q2"]
        a_ids = ["a1", "a2", "a3"]
        matrix = [
            [1, NOT_ADMINISTERED, 0],  # q1 only administered to a1 and a3
            [NOT_ADMINISTERED, 1, 1],  # q2 only administered to a2 and a3
        ]

        difficulties, abilities = IRTService._rasch_calibrate(q_ids, a_ids, matrix)
        # Should return values without error
        assert len(difficulties) == 2
        assert len(abilities) == 3

    def test_empty_input(self, irt):
        """Empty input returns empty dicts."""
        d, a = IRTService._rasch_calibrate([], [], [])
        assert d == {}
        assert a == {}

    def test_degenerate_all_correct(self, irt):
        """All-correct matrix: difficulties should be low/negative."""
        q_ids = ["q1", "q2"]
        a_ids = ["a1", "a2", "a3"]
        matrix = [
            [1, 1, 1],
            [1, 1, 1],
        ]
        difficulties, abilities = IRTService._rasch_calibrate(q_ids, a_ids, matrix)
        # Both items easy → negative difficulty
        for b in difficulties.values():
            assert b < 2.0  # Should be low

    def test_degenerate_all_wrong(self, irt):
        """All-wrong matrix: difficulties should be high/positive."""
        q_ids = ["q1", "q2"]
        a_ids = ["a1", "a2", "a3"]
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
        ]
        difficulties, abilities = IRTService._rasch_calibrate(q_ids, a_ids, matrix)
        for b in difficulties.values():
            assert b > -2.0  # Should be high

    def test_ability_ordering(self, irt):
        """Better-performing agents should have higher ability estimates."""
        q_ids = ["q1", "q2", "q3"]
        a_ids = ["weak", "mid", "strong"]
        matrix = [
            [0, 1, 1],   # q1
            [0, 0, 1],   # q2
            [0, 1, 1],   # q3
        ]
        _, abilities = IRTService._rasch_calibrate(q_ids, a_ids, matrix)
        assert abilities["weak"] < abilities["mid"]
        assert abilities["mid"] < abilities["strong"]


# ── Fisher Information ──────────────────────────────────────────────────────


class TestFisherInformation:
    def test_max_at_theta_equals_b(self, irt):
        """Fisher info is maximized when theta = b (P=0.5)."""
        b = 1.5
        info_at_b = irt.fisher_information(theta=b, b=b)
        info_away = irt.fisher_information(theta=b + 2, b=b)
        assert info_at_b > info_away

    def test_symmetry(self, irt):
        """Info should be symmetric around theta = b."""
        b = 0.0
        info_above = irt.fisher_information(theta=1.0, b=b)
        info_below = irt.fisher_information(theta=-1.0, b=b)
        assert abs(info_above - info_below) < 1e-10

    def test_discrimination_scaling(self, irt):
        """Higher discrimination → more information."""
        b = 0.0
        theta = 0.0
        info_a1 = irt.fisher_information(theta, b, a=1.0)
        info_a2 = irt.fisher_information(theta, b, a=2.0)
        assert info_a2 > info_a1
        # At theta=b, info = a^2 * 0.25
        assert abs(info_a1 - 0.25) < 1e-10
        assert abs(info_a2 - 1.0) < 1e-10

    def test_max_value(self, irt):
        """At theta=b with a=1, Fisher info = 0.25."""
        info = irt.fisher_information(0.0, 0.0, 1.0)
        assert abs(info - 0.25) < 1e-10


# ── Point-Biserial ──────────────────────────────────────────────────────────


class TestPointBiserial:
    def test_perfect_discrimination(self, irt):
        """When correct = high ability and incorrect = low ability, rpb is positive."""
        # Item 0: agents 0,1 wrong (low ability), agents 2,3,4 correct (high ability)
        matrix = [[0, 0, 1, 1, 1]]
        agent_ids = ["a1", "a2", "a3", "a4", "a5"]
        abilities = {"a1": -2.0, "a2": -1.0, "a3": 1.0, "a4": 1.5, "a5": 2.0}

        rpb = IRTService._point_biserial(0, matrix, agent_ids, abilities)
        assert rpb > 0.3  # Good discrimination

    def test_no_variance(self, irt):
        """All same response → low/zero point-biserial."""
        matrix = [[1, 1, 1, 1]]
        agent_ids = ["a1", "a2", "a3", "a4"]
        abilities = {"a1": 0.0, "a2": 1.0, "a3": 2.0, "a4": 3.0}

        rpb = IRTService._point_biserial(0, matrix, agent_ids, abilities)
        assert rpb == 0.0  # n0 = 0, returns 0

    def test_few_responses(self, irt):
        """Too few responses returns 0."""
        matrix = [[1, NOT_ADMINISTERED, NOT_ADMINISTERED, 0]]
        agent_ids = ["a1", "a2", "a3", "a4"]
        abilities = {"a1": 1.0, "a2": 0.0, "a3": 0.0, "a4": -1.0}

        # Only 2 administered → < 3, returns 0
        rpb = IRTService._point_biserial(0, matrix, agent_ids, abilities)
        assert rpb == 0.0


# ── Ability Estimation (EAP) ───────────────────────────────────────────────


class TestAbilityEstimation:
    async def test_all_correct_high_theta(self, irt):
        """All correct answers → positive theta."""
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={
            "question_id": "q1", "difficulty_b": 0.0, "discrimination_a": 1.0,
            "calibration_model": "rasch_1pl",
        })
        with patch("src.core.irt_service.item_params_col", return_value=mock_col):
            result = await irt.estimate_ability([
                {"question_id": "q1", "correct": True},
                {"question_id": "q2", "correct": True},
                {"question_id": "q3", "correct": True},
            ])
        assert result["theta"] > 0

    async def test_all_wrong_low_theta(self, irt):
        """All wrong answers → negative theta."""
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={
            "question_id": "q1", "difficulty_b": 0.0, "discrimination_a": 1.0,
            "calibration_model": "rasch_1pl",
        })
        with patch("src.core.irt_service.item_params_col", return_value=mock_col):
            result = await irt.estimate_ability([
                {"question_id": "q1", "correct": False},
                {"question_id": "q2", "correct": False},
                {"question_id": "q3", "correct": False},
            ])
        assert result["theta"] < 0

    async def test_mixed_responses(self, irt):
        """Mixed responses → theta near 0."""
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={
            "question_id": "q1", "difficulty_b": 0.0, "discrimination_a": 1.0,
            "calibration_model": "rasch_1pl",
        })
        with patch("src.core.irt_service.item_params_col", return_value=mock_col):
            result = await irt.estimate_ability([
                {"question_id": "q1", "correct": True},
                {"question_id": "q2", "correct": False},
            ])
        assert -2.0 < result["theta"] < 2.0
        assert result["responses_used"] == 2

    async def test_uncalibrated_skipped(self, irt):
        """Items without calibration are skipped."""
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={
            "question_id": "q1", "difficulty_b": 0.0, "discrimination_a": 1.0,
            "calibration_model": "none",
        })
        with patch("src.core.irt_service.item_params_col", return_value=mock_col):
            result = await irt.estimate_ability([
                {"question_id": "q1", "correct": True},
            ])
        assert result["responses_used"] == 0
        assert result["theta"] == 0.0

    async def test_no_responses(self, irt):
        """Empty responses list returns default."""
        result = await irt.estimate_ability([])
        assert result["theta"] == 0.0
        assert result["responses_used"] == 0


# ── Adaptive Question Selection ─────────────────────────────────────────────


class TestAdaptiveSelection:
    async def test_selects_near_ability(self, irt):
        """Selected questions should have difficulty near the given theta."""
        # Create mock items at various difficulties
        items = [
            {"question_id": f"q{i}", "domain": "general", "difficulty_b": b,
             "discrimination_a": 1.0, "status": "active"}
            for i, b in enumerate([-3.0, -1.0, 0.0, 1.0, 3.0])
        ]

        class MockCursor:
            def __init__(self, data):
                self._data = iter(data)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._data)
                except StopIteration:
                    raise StopAsyncIteration

        mock_col = MagicMock()
        mock_col.find = MagicMock(return_value=MockCursor(items))

        with patch("src.core.irt_service.item_params_col", return_value=mock_col):
            selected = await irt.select_adaptive_questions(theta=0.0, count=2)

        assert len(selected) == 2
        # Should prefer items with difficulty near 0 (max Fisher info)
        selected_bs = [q["difficulty_b"] for q in selected]
        assert any(abs(b) <= 1.5 for b in selected_bs)

    async def test_excludes_administered(self, irt):
        """Already-administered items should not be selected."""
        items = [
            {"question_id": "q0", "domain": "general", "difficulty_b": 0.0,
             "discrimination_a": 1.0, "status": "active"},
            {"question_id": "q1", "domain": "general", "difficulty_b": 0.5,
             "discrimination_a": 1.0, "status": "active"},
        ]

        class MockCursor:
            def __init__(self, data):
                self._data = iter(data)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._data)
                except StopIteration:
                    raise StopAsyncIteration

        mock_col = MagicMock()
        mock_col.find = MagicMock(return_value=MockCursor(items))

        with patch("src.core.irt_service.item_params_col", return_value=mock_col):
            selected = await irt.select_adaptive_questions(
                theta=0.0, administered=["q0"], count=5,
            )

        selected_ids = [q["question_id"] for q in selected]
        assert "q0" not in selected_ids
        assert "q1" in selected_ids

    async def test_empty_pool(self, irt):
        """Empty item pool returns empty list."""
        class MockCursor:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration

        mock_col = MagicMock()
        mock_col.find = MagicMock(return_value=MockCursor())

        with patch("src.core.irt_service.item_params_col", return_value=mock_col):
            selected = await irt.select_adaptive_questions(theta=0.0, count=5)
        assert selected == []


# ── ItemParams Persistence ──────────────────────────────────────────────────


class TestItemParamsPersistence:
    def test_to_dict(self):
        """ItemParams.to_dict() should produce a serializable dict."""
        params = ItemParams(
            question_id="q1",
            domain="defi",
            difficulty_b=1.5,
            discrimination_a=1.0,
            p_value=0.4,
            point_biserial=0.35,
            exposure_count=50,
            total_responses=50,
            status="active",
            calibration_model="rasch_1pl",
            last_calibrated=datetime(2026, 3, 1, 12, 0),
        )
        d = params.to_dict()
        assert d["question_id"] == "q1"
        assert d["difficulty_b"] == 1.5
        assert "2026-03-01" in d["last_calibrated"]

    async def test_save_and_load(self, irt):
        """Save then load should return matching params."""
        mock_col = MagicMock()
        mock_col.update_one = AsyncMock()
        mock_col.find_one = AsyncMock(return_value={
            "question_id": "q1",
            "domain": "defi",
            "difficulty_b": 1.5,
            "discrimination_a": 1.0,
            "calibration_model": "rasch_1pl",
        })

        with patch("src.core.irt_service.item_params_col", return_value=mock_col):
            params = ItemParams(
                question_id="q1", domain="defi", difficulty_b=1.5,
            )
            await irt._save_item_param(params)
            mock_col.update_one.assert_called_once()

            loaded = await irt._load_item_param("q1")
            assert loaded["question_id"] == "q1"
            assert loaded["difficulty_b"] == 1.5

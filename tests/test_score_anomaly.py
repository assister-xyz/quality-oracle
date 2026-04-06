"""Tests for score anomaly detection (QO-043)."""
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.core.score_anomaly import (
    check_score_anomaly,
    record_anomaly,
    AnomalyAlert,
    _compute_mean_stdev,
    FIRST_EVAL_EXTREME_THRESHOLD,
    Z_SCORE_THRESHOLD,
    MIN_HISTORY_FOR_ZSCORE,
)


class TestComputeMeanStdev:
    """Test the helper statistics function."""

    def test_empty_list(self):
        mean, stdev = _compute_mean_stdev([])
        assert mean == 0.0
        assert stdev == 0.0

    def test_single_value(self):
        mean, stdev = _compute_mean_stdev([70.0])
        assert mean == 70.0
        assert stdev == 0.0

    def test_two_values(self):
        mean, stdev = _compute_mean_stdev([60.0, 80.0])
        assert mean == 70.0
        assert abs(stdev - 14.14) < 0.1

    def test_uniform_values(self):
        mean, stdev = _compute_mean_stdev([50.0, 50.0, 50.0])
        assert mean == 50.0
        assert stdev == 0.0

    def test_known_distribution(self):
        values = [65.0, 70.0, 68.0, 72.0, 66.0]
        mean, stdev = _compute_mean_stdev(values)
        assert abs(mean - 68.2) < 0.1
        assert stdev > 0


class TestAnomalyAlert:
    """Test AnomalyAlert data class."""

    def test_to_dict(self):
        alert = AnomalyAlert(
            anomaly_type="first_eval_extreme",
            severity="medium",
            target_id="http://example.com",
            current_score=98.0,
            details={"threshold": 95},
        )
        d = alert.to_dict()
        assert d["anomaly_type"] == "first_eval_extreme"
        assert d["severity"] == "medium"
        assert d["target_id"] == "http://example.com"
        assert d["current_score"] == 98.0
        assert "detected_at" in d


def _make_mock_col(history_docs):
    """Create a mock collection that returns history_docs from find().sort().limit().to_list()."""
    mock_col = MagicMock()
    mock_find = MagicMock()
    mock_sort = MagicMock()
    mock_limit = AsyncMock()
    mock_col.find.return_value = mock_find
    mock_find.sort.return_value = mock_sort
    mock_sort.limit.return_value = mock_limit
    mock_limit.to_list = AsyncMock(return_value=history_docs)
    return mock_col


class TestCheckScoreAnomaly:
    """Test anomaly detection logic with mocked MongoDB."""

    @pytest.mark.asyncio
    async def test_first_eval_normal_score(self):
        mock_col = _make_mock_col([])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://new-server.com", 75.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_first_eval_extreme_score(self):
        mock_col = _make_mock_col([])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://suspicious.com", 98.0)
        assert result is not None
        assert result.anomaly_type == "first_eval_extreme"
        assert result.severity == "medium"

    @pytest.mark.asyncio
    async def test_first_eval_at_threshold(self):
        mock_col = _make_mock_col([])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://edge.com", float(FIRST_EVAL_EXTREME_THRESHOLD))
        assert result is not None
        assert result.anomaly_type == "first_eval_extreme"

    @pytest.mark.asyncio
    async def test_normal_score_with_history(self):
        mock_col = _make_mock_col([
            {"score": 70}, {"score": 72}, {"score": 68}, {"score": 71},
        ])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://stable.com", 73.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_zscore_deviation_high(self):
        mock_col = _make_mock_col([
            {"score": 70}, {"score": 72}, {"score": 68}, {"score": 71}, {"score": 69},
        ])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://jumped.com", 95.0)
        assert result is not None
        assert result.anomaly_type == "z_score_deviation"
        assert result.details["direction"] == "increase"

    @pytest.mark.asyncio
    async def test_zscore_deviation_drop(self):
        mock_col = _make_mock_col([
            {"score": 85}, {"score": 87}, {"score": 83}, {"score": 86},
        ])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://dropped.com", 30.0)
        assert result is not None
        assert result.anomaly_type == "z_score_deviation"
        assert result.details["direction"] == "decrease"

    @pytest.mark.asyncio
    async def test_zscore_high_severity(self):
        mock_col = _make_mock_col([
            {"score": 50}, {"score": 52}, {"score": 48}, {"score": 51},
        ])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://extreme.com", 100.0)
        assert result is not None
        assert result.severity == "high"

    @pytest.mark.asyncio
    async def test_insufficient_history_for_zscore(self):
        mock_col = _make_mock_col([{"score": 70}, {"score": 72}])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://new-ish.com", 95.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_unchanged_manifest_large_jump(self):
        mock_col = _make_mock_col([
            {"score": 65, "recorded_at": "2026-01-01", "manifest_hash": "abc123"},
            {"score": 63, "recorded_at": "2026-01-02", "manifest_hash": "abc123"},
        ])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://suspicious.com", 92.0, manifest_hash="abc123")
        assert result is not None
        assert result.anomaly_type == "unchanged_manifest_jump"
        assert result.details["score_delta"] == 27

    @pytest.mark.asyncio
    async def test_changed_manifest_large_jump_ok(self):
        mock_col = _make_mock_col([
            {"score": 65, "recorded_at": "2026-01-01", "manifest_hash": "old_hash"},
        ])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://improved.com", 90.0, manifest_hash="new_hash")
        assert result is None

    @pytest.mark.asyncio
    async def test_unchanged_manifest_small_change_ok(self):
        mock_col = _make_mock_col([
            {"score": 70, "recorded_at": "2026-01-01", "manifest_hash": "abc123"},
        ])
        with patch("src.storage.mongodb.score_history_col", return_value=mock_col):
            result = await check_score_anomaly("http://stable.com", 78.0, manifest_hash="abc123")
        assert result is None


class TestRecordAnomaly:

    @pytest.mark.asyncio
    async def test_record_anomaly_inserts(self):
        alert = AnomalyAlert(
            anomaly_type="first_eval_extreme",
            severity="medium",
            target_id="http://test.com",
            current_score=98.0,
            details={"threshold": 95},
        )
        mock_col = AsyncMock()
        with patch("src.storage.mongodb.score_anomalies_col", return_value=mock_col):
            await record_anomaly(alert)
        mock_col.insert_one.assert_called_once()
        inserted = mock_col.insert_one.call_args[0][0]
        assert inserted["anomaly_type"] == "first_eval_extreme"

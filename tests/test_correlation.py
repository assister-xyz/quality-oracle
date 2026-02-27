"""Tests for production correlation engine and feedback endpoints."""
import pytest
from src.core.correlation import (
    pearson_correlation,
    classify_alignment,
    detect_sandbagging,
    compute_confidence_adjustment,
    compute_correlation_report,
    CorrelationReport,
)


# ── Pearson Correlation ──────────────────────────────────────────────────────

class TestPearsonCorrelation:

    def test_perfect_positive(self):
        r = pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r is not None
        assert abs(r - 1.0) < 0.001

    def test_perfect_negative(self):
        r = pearson_correlation([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r is not None
        assert abs(r - (-1.0)) < 0.001

    def test_no_correlation(self):
        # Zigzag pattern — near-zero correlation
        r = pearson_correlation([1, 2, 3, 4, 5, 6], [10, 1, 10, 1, 10, 1])
        assert r is not None
        assert abs(r) < 0.3

    def test_insufficient_data(self):
        assert pearson_correlation([1], [2]) is None
        assert pearson_correlation([], []) is None

    def test_zero_variance(self):
        assert pearson_correlation([5, 5, 5], [1, 2, 3]) is None

    def test_mismatched_lengths(self):
        assert pearson_correlation([1, 2, 3], [1, 2]) is None


# ── Alignment Classification ─────────────────────────────────────────────────

class TestClassifyAlignment:

    def test_strong(self):
        assert classify_alignment(0.8) == "strong"
        assert classify_alignment(0.7) == "strong"

    def test_moderate(self):
        assert classify_alignment(0.5) == "moderate"
        assert classify_alignment(0.4) == "moderate"

    def test_weak(self):
        assert classify_alignment(0.2) == "weak"
        assert classify_alignment(0.1) == "weak"

    def test_none(self):
        assert classify_alignment(0.05) == "none"
        assert classify_alignment(0.0) == "none"
        assert classify_alignment(-0.05) == "none"

    def test_negative(self):
        assert classify_alignment(-0.3) == "negative"
        assert classify_alignment(-0.8) == "negative"

    def test_insufficient_data(self):
        assert classify_alignment(None) == "insufficient_data"


# ── Sandbagging Detection ────────────────────────────────────────────────────

class TestDetectSandbagging:

    def test_high_risk(self):
        """High eval + very low production = sandbagging."""
        assert detect_sandbagging(85, 30, 10) == "high"

    def test_medium_risk(self):
        """Large gap but production not critically low."""
        assert detect_sandbagging(80, 45, 10) == "medium"

    def test_low_risk_aligned(self):
        """Scores are close — no sandbagging."""
        assert detect_sandbagging(75, 70, 10) == "low"

    def test_low_risk_insufficient_data(self):
        """Not enough feedback to flag."""
        assert detect_sandbagging(90, 20, 3) == "low"

    def test_production_higher_than_eval(self):
        """Production outperforming eval — no sandbagging."""
        assert detect_sandbagging(60, 80, 10) == "low"


# ── Confidence Adjustment ────────────────────────────────────────────────────

class TestConfidenceAdjustment:

    def test_positive_correlation_boosts(self):
        adj = compute_confidence_adjustment(0.8, 20)
        assert adj > 0
        assert adj <= 0.10

    def test_negative_correlation_penalizes(self):
        adj = compute_confidence_adjustment(-0.7, 20)
        assert adj < 0
        assert adj >= -0.15

    def test_zero_correlation_no_change(self):
        adj = compute_confidence_adjustment(0.0, 20)
        assert adj == 0.0

    def test_insufficient_feedback(self):
        adj = compute_confidence_adjustment(0.9, 2)
        assert adj == 0.0

    def test_none_correlation(self):
        adj = compute_confidence_adjustment(None, 20)
        assert adj == 0.0

    def test_scales_with_sample_size(self):
        adj_small = compute_confidence_adjustment(0.8, 5)
        adj_large = compute_confidence_adjustment(0.8, 20)
        assert adj_large >= adj_small


# ── Full Correlation Report ──────────────────────────────────────────────────

class TestCorrelationReport:

    def test_empty_feedback(self):
        report = compute_correlation_report("test-server", 80, [])
        assert report.eval_score == 80
        assert report.production_score == 0
        assert report.feedback_count == 0
        assert report.alignment == "insufficient_data"
        assert report.sandbagging_risk == "low"
        assert report.confidence_adjustment == 0.0

    def test_aligned_scores(self):
        """Production matches eval — strong alignment."""
        feedback = [
            {"outcome": "success", "outcome_score": 78},
            {"outcome": "success", "outcome_score": 82},
            {"outcome": "success", "outcome_score": 80},
            {"outcome": "partial", "outcome_score": 75},
            {"outcome": "success", "outcome_score": 85},
        ]
        report = compute_correlation_report("server-a", 80, feedback)
        assert report.alignment == "strong"
        assert report.sandbagging_risk == "low"
        assert report.feedback_count == 5
        assert report.production_score == 80

    def test_sandbagging_detected(self):
        """High eval but low production = sandbagging."""
        feedback = [
            {"outcome": "failure", "outcome_score": 20},
            {"outcome": "failure", "outcome_score": 25},
            {"outcome": "failure", "outcome_score": 15},
            {"outcome": "partial", "outcome_score": 30},
            {"outcome": "failure", "outcome_score": 10},
        ]
        report = compute_correlation_report("sketchy-server", 85, feedback)
        assert report.sandbagging_risk == "high"
        assert report.alignment == "negative"

    def test_moderate_alignment(self):
        feedback = [
            {"outcome": "success", "outcome_score": 55},
            {"outcome": "partial", "outcome_score": 60},
            {"outcome": "success", "outcome_score": 65},
        ]
        report = compute_correlation_report("server-b", 75, feedback)
        assert report.alignment in ("moderate", "weak")

    def test_outcome_breakdown(self):
        feedback = [
            {"outcome": "success", "outcome_score": 90},
            {"outcome": "success", "outcome_score": 85},
            {"outcome": "failure", "outcome_score": 20},
            {"outcome": "partial", "outcome_score": 60},
        ]
        report = compute_correlation_report("server-c", 70, feedback)
        assert report.outcome_breakdown["success"] == 2
        assert report.outcome_breakdown["failure"] == 1
        assert report.outcome_breakdown["partial"] == 1

    def test_to_dict_format(self):
        report = compute_correlation_report("test", 75, [
            {"outcome": "success", "outcome_score": 70},
        ])
        d = report.to_dict()
        assert "target_id" in d
        assert "eval_score" in d
        assert "production_score" in d
        assert "correlation" in d
        assert "alignment" in d
        assert "sandbagging_risk" in d
        assert "confidence_adjustment" in d
        assert "outcome_breakdown" in d

    def test_single_feedback_item(self):
        report = compute_correlation_report("single", 80, [
            {"outcome": "success", "outcome_score": 90},
        ])
        assert report.feedback_count == 1
        assert report.production_score == 90
        # Single item → no Pearson correlation
        assert report.correlation is None


# ── Feedback API integration ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_endpoint_registered():
    """Feedback router should be registered in the app."""
    from src.main import app
    paths = [route.path for route in app.routes]
    assert "/v1/feedback" in paths
    assert any("/v1/correlation/" in p for p in paths)


@pytest.mark.asyncio
async def test_feedback_models():
    """Feedback models should validate correctly."""
    from src.storage.models import FeedbackRequest, FeedbackOutcome

    req = FeedbackRequest(
        target_id="https://mcp.example.com",
        outcome=FeedbackOutcome.SUCCESS,
        outcome_score=85,
        context="payment_agent",
        details="Completed in 3 retries",
    )
    assert req.outcome_score == 85
    assert req.outcome == FeedbackOutcome.SUCCESS

    # Score bounds
    with pytest.raises(Exception):
        FeedbackRequest(
            target_id="test",
            outcome=FeedbackOutcome.SUCCESS,
            outcome_score=101,
        )

    with pytest.raises(Exception):
        FeedbackRequest(
            target_id="test",
            outcome=FeedbackOutcome.FAILURE,
            outcome_score=-1,
        )


@pytest.mark.asyncio
async def test_correlation_response_model():
    """CorrelationResponse should serialize properly."""
    from src.storage.models import CorrelationResponse

    resp = CorrelationResponse(
        target_id="test",
        eval_score=80,
        production_score=75,
        correlation=0.85,
        feedback_count=10,
        alignment="strong",
        confidence_adjustment=0.05,
        sandbagging_risk="low",
        outcome_breakdown={"success": 8, "partial": 2},
    )
    data = resp.model_dump()
    assert data["alignment"] == "strong"
    assert data["sandbagging_risk"] == "low"
    assert data["outcome_breakdown"]["success"] == 8

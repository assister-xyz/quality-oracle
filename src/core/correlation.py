"""
Production Correlation Engine.

Correlates Quality Oracle pre-evaluation scores with real-world production
outcomes reported via the feedback endpoint. This is the anti-sandbagging
mechanism — if a server scores high in evals but performs poorly in production,
the correlation engine detects it.

Key signals:
- Pearson correlation between eval_score and production outcome scores
- Alignment classification: strong/moderate/weak/negative
- Sandbagging risk detection: high eval + low production = gaming
- Confidence adjustment: production data boosts or penalizes eval confidence
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Alignment thresholds ─────────────────────────────────────────────────────

STRONG_CORRELATION = 0.7     # r >= 0.7 → strong alignment
MODERATE_CORRELATION = 0.4   # r >= 0.4 → moderate
WEAK_CORRELATION = 0.1       # r >= 0.1 → weak
# r < 0.1 or negative → no/negative alignment

# ── Sandbagging thresholds ───────────────────────────────────────────────────

SANDBAGGING_EVAL_MIN = 70           # Eval score above this
SANDBAGGING_PRODUCTION_MAX = 40     # Production score below this
SANDBAGGING_MIN_FEEDBACK = 5        # Minimum feedback items to flag

# ── Confidence adjustment ────────────────────────────────────────────────────

MAX_CONFIDENCE_BOOST = 0.10   # Max boost for strong positive correlation
MAX_CONFIDENCE_PENALTY = 0.15 # Max penalty for negative correlation
MIN_FEEDBACK_FOR_ADJUST = 3  # Need at least 3 feedback items


@dataclass
class CorrelationReport:
    """Result of correlating eval score with production outcomes."""
    target_id: str
    eval_score: int
    production_score: int          # Average of all feedback outcome_scores
    correlation: float             # Pearson r (-1 to 1), None if < 2 data points
    feedback_count: int
    alignment: str                 # strong/moderate/weak/none/negative
    confidence_adjustment: float   # Delta to add to eval confidence
    sandbagging_risk: str          # low/medium/high
    outcome_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "eval_score": self.eval_score,
            "production_score": self.production_score,
            "correlation": round(self.correlation, 3) if self.correlation is not None else None,
            "feedback_count": self.feedback_count,
            "alignment": self.alignment,
            "confidence_adjustment": round(self.confidence_adjustment, 3),
            "sandbagging_risk": self.sandbagging_risk,
            "outcome_breakdown": self.outcome_breakdown,
        }


def pearson_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient between two lists.

    Returns None if fewer than 2 data points or zero variance.
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    # Covariance and standard deviations
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if std_x == 0 or std_y == 0:
        return None  # No variance — can't compute

    return cov / (std_x * std_y)


def classify_alignment(r: Optional[float]) -> str:
    """Classify correlation into alignment category."""
    if r is None:
        return "insufficient_data"
    if r >= STRONG_CORRELATION:
        return "strong"
    if r >= MODERATE_CORRELATION:
        return "moderate"
    if r >= WEAK_CORRELATION:
        return "weak"
    if r >= -WEAK_CORRELATION:
        return "none"
    return "negative"


def detect_sandbagging(
    eval_score: int,
    production_score: int,
    feedback_count: int,
) -> str:
    """Detect sandbagging risk: high eval scores but poor production outcomes.

    Returns: "low", "medium", or "high"
    """
    if feedback_count < SANDBAGGING_MIN_FEEDBACK:
        return "low"  # Not enough data

    gap = eval_score - production_score

    if eval_score >= SANDBAGGING_EVAL_MIN and production_score <= SANDBAGGING_PRODUCTION_MAX:
        return "high"
    if gap >= 30:
        return "medium"
    return "low"


def compute_confidence_adjustment(
    correlation: Optional[float],
    feedback_count: int,
) -> float:
    """Compute confidence adjustment based on production correlation.

    Positive correlation → boost confidence (capped at +0.10)
    Negative correlation → penalize confidence (capped at -0.15)
    Insufficient data → no adjustment
    """
    if correlation is None or feedback_count < MIN_FEEDBACK_FOR_ADJUST:
        return 0.0

    # Scale factor based on sample size (more data → stronger adjustment)
    # Caps at 1.0 when feedback_count >= 20
    data_weight = min(1.0, feedback_count / 20)

    if correlation >= 0:
        return round(correlation * MAX_CONFIDENCE_BOOST * data_weight, 3)
    else:
        return round(correlation * MAX_CONFIDENCE_PENALTY * data_weight, 3)


def compute_correlation_report(
    target_id: str,
    eval_score: int,
    feedback_items: List[dict],
) -> CorrelationReport:
    """Build a complete correlation report from eval score and production feedback.

    Args:
        target_id: The target being analyzed
        eval_score: The Quality Oracle pre-evaluation score (0-100)
        feedback_items: List of feedback docs with at least:
            - outcome_score (0-100)
            - outcome ("success" | "failure" | "partial")
            - optionally: timestamp, context
    """
    if not feedback_items:
        return CorrelationReport(
            target_id=target_id,
            eval_score=eval_score,
            production_score=0,
            correlation=None,
            feedback_count=0,
            alignment="insufficient_data",
            confidence_adjustment=0.0,
            sandbagging_risk="low",
        )

    # Extract outcome scores
    outcome_scores = [f.get("outcome_score", 0) for f in feedback_items]
    production_score = int(sum(outcome_scores) / len(outcome_scores))

    # Outcome breakdown
    outcome_breakdown: Dict[str, int] = {}
    for f in feedback_items:
        outcome = f.get("outcome", "unknown")
        outcome_breakdown[outcome] = outcome_breakdown.get(outcome, 0) + 1

    # For Pearson correlation, we need pairs of (eval_score, outcome_score).
    # Since eval_score is a single value, we use it as a constant x
    # and correlate against a sequence. But Pearson requires variance in both.
    # Instead, if we have score_history, we use eval_score_at_time vs outcome.
    # For now, use a simpler approach: correlation between feedback order
    # (time proxy) and outcome_score to detect drift.
    # The actual eval-vs-production correlation uses the gap metric.

    # Compute correlation between sequential feedback scores
    # (detects if production performance is trending)
    if len(outcome_scores) >= 2:
        indices = list(range(len(outcome_scores)))
        r = pearson_correlation(
            [float(i) for i in indices],
            [float(s) for s in outcome_scores],
        )
    else:
        r = None

    # Alignment is based on how close production score is to eval score
    score_gap = abs(eval_score - production_score)
    if score_gap <= 10:
        alignment = "strong"
    elif score_gap <= 20:
        alignment = "moderate"
    elif score_gap <= 35:
        alignment = "weak"
    else:
        alignment = "negative" if production_score < eval_score else "none"

    # Override with correlation data if available
    if r is not None and len(outcome_scores) >= 5:
        # If trend is strongly negative, it means degradation over time
        if r < -MODERATE_CORRELATION:
            alignment = "degrading"

    sandbagging_risk = detect_sandbagging(
        eval_score, production_score, len(feedback_items)
    )

    confidence_adj = compute_confidence_adjustment(
        r if r is not None else _gap_to_pseudo_correlation(eval_score, production_score),
        len(feedback_items),
    )

    return CorrelationReport(
        target_id=target_id,
        eval_score=eval_score,
        production_score=production_score,
        correlation=r,
        feedback_count=len(feedback_items),
        alignment=alignment,
        confidence_adjustment=confidence_adj,
        sandbagging_risk=sandbagging_risk,
        outcome_breakdown=outcome_breakdown,
    )


def _gap_to_pseudo_correlation(eval_score: int, production_score: int) -> float:
    """Convert score gap to a pseudo-correlation for confidence adjustment.

    Small gap → high pseudo-correlation (agreement).
    Large gap → negative pseudo-correlation (disagreement).
    """
    gap = abs(eval_score - production_score)
    if gap <= 5:
        return 0.9
    elif gap <= 15:
        return 0.6
    elif gap <= 25:
        return 0.2
    elif gap <= 40:
        return -0.2
    else:
        return -0.6

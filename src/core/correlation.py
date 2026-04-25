"""
Production Correlation Engine — REBUILT for QO-061.

Correlates AgentTrust pre-evaluation scores with real-world production
outcomes reported via the feedback endpoint. This is the anti-sandbagging
mechanism — if a server scores high in evals but performs poorly in
production, the correlation engine detects it.

QO-061 fixes the previous misnamed implementation:
- The OLD code computed Pearson on `(feedback_index, outcome_score)` —
  that's a drift detector, NOT anti-sandbagging. The eval score never
  even entered the formula.
- The NEW code requires each feedback row to carry an `eval_score_at_time`
  snapshot taken at submission time, then computes WEIGHTED Pearson on
  `(snapshot.eval_score_at_time, feedback.outcome_score)` pairs — the
  honest signal.
- KYA-weighted: reports from verified team members weigh more than from
  free-tier anonymous reporters (reduces brigading).
- Inverse-gap flag: eval≤40 AND prod≥80 → "reverse_sandbagging" anomaly
  (soft signal, manual review only — does NOT change displayed tier).

Public entry point: `compute_correlation(snapshots: list[FeedbackSnapshot])`.
The legacy `compute_correlation_report(target_id, eval_score, feedback_items)`
is kept as a thin adapter that builds snapshots from raw feedback dicts —
this is the migration path for callers that don't have access to a snapshot
join yet.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.storage.models import FeedbackSnapshot

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

# ── Reverse sandbagging (QO-061) ─────────────────────────────────────────────
REVERSE_SANDBAGGING_EVAL_MAX = 40
REVERSE_SANDBAGGING_PRODUCTION_MIN = 80

# ── Confidence adjustment ────────────────────────────────────────────────────

MAX_CONFIDENCE_BOOST = 0.10   # Max boost for strong positive correlation
MAX_CONFIDENCE_PENALTY = 0.15 # Max penalty for negative correlation
MIN_FEEDBACK_FOR_ADJUST = 3  # Need at least 3 feedback items

# ── KYA tier weights (QO-061) ────────────────────────────────────────────────
KYA_TIER_WEIGHTS: Dict[int, float] = {
    1: 1.0,  # free
    2: 2.0,  # builder
    3: 3.0,  # team
}

# ── Pearson sample-size minimum (QO-061) ─────────────────────────────────────
# Below this we report status="insufficient_data" rather than a meaningless r.
MIN_SNAPSHOTS_FOR_PEARSON = 10


# ── Result dataclasses ──────────────────────────────────────────────────────


@dataclass
class CorrelationResult:
    """QO-061 result of correlating a list of FeedbackSnapshot rows."""
    target_id: str
    status: str  # "ok" | "insufficient_data"
    n: int
    r: Optional[float] = None
    avg_eval: Optional[float] = None
    avg_outcome: Optional[float] = None
    alignment: str = "insufficient_data"
    confidence_adjustment: float = 0.0
    sandbagging_risk: str = "low"
    anomaly_type: Optional[str] = None  # "sandbagging" | "reverse_sandbagging" | None
    data_quality_warning: Optional[str] = None
    outcome_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "status": self.status,
            "n": self.n,
            "r": round(self.r, 3) if self.r is not None else None,
            "avg_eval": round(self.avg_eval, 2) if self.avg_eval is not None else None,
            "avg_outcome": round(self.avg_outcome, 2) if self.avg_outcome is not None else None,
            "alignment": self.alignment,
            "confidence_adjustment": round(self.confidence_adjustment, 3),
            "sandbagging_risk": self.sandbagging_risk,
            "anomaly_type": self.anomaly_type,
            "data_quality_warning": self.data_quality_warning,
            "outcome_breakdown": self.outcome_breakdown,
        }


@dataclass
class CorrelationReport:
    """Legacy shape consumed by existing API/tests; populated from CorrelationResult."""
    target_id: str
    eval_score: int
    production_score: int          # Average of all feedback outcome_scores
    correlation: Optional[float]   # Weighted Pearson r (QO-061)
    feedback_count: int
    alignment: str                 # strong/moderate/weak/none/negative
    confidence_adjustment: float
    sandbagging_risk: str
    outcome_breakdown: Dict[str, int] = field(default_factory=dict)
    anomaly_type: Optional[str] = None
    data_quality_warning: Optional[str] = None

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
            "anomaly_type": self.anomaly_type,
            "data_quality_warning": self.data_quality_warning,
        }


# ── Pearson primitives ──────────────────────────────────────────────────────


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Compute (unweighted) Pearson correlation coefficient between two lists.

    Returns None if fewer than 2 data points or zero variance.
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if std_x == 0 or std_y == 0:
        return None  # No variance — can't compute

    return cov / (std_x * std_y)


def weighted_pearson(
    xs: Sequence[float], ys: Sequence[float], weights: Sequence[float]
) -> Optional[float]:
    """Weighted Pearson correlation. Returns None on insufficient/degenerate data."""
    n = len(xs)
    if n < 2 or len(ys) != n or len(weights) != n:
        return None
    total_w = sum(weights)
    if total_w == 0:
        return None

    mean_x = sum(w * x for w, x in zip(weights, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total_w

    cov = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys))
    var_x = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    var_y = sum(w * (y - mean_y) ** 2 for w, y in zip(weights, ys))
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


# ── Classification helpers ──────────────────────────────────────────────────


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

    data_weight = min(1.0, feedback_count / 20)

    if correlation >= 0:
        return round(correlation * MAX_CONFIDENCE_BOOST * data_weight, 3)
    else:
        return round(correlation * MAX_CONFIDENCE_PENALTY * data_weight, 3)


# ── QO-061 public entry point ───────────────────────────────────────────────


def compute_correlation(snapshots: List[FeedbackSnapshot]) -> CorrelationResult:
    """Honest production-vs-eval correlation (QO-061).

    Args:
        snapshots: list of FeedbackSnapshot rows. Each carries the eval_score
            that was current when the user submitted the feedback (the
            ANTI-sandbagging signal).

    Returns:
        CorrelationResult with weighted Pearson r over
        (eval_score_at_time, feedback_outcome) pairs, KYA-weighted, plus
        anomaly type if detected.

    AC5: Pearson r is on `(snapshot.eval_score_at_time, feedback.outcome_score)`,
         NOT on `(feedback_index, outcome)`.
    AC6: inverse-gap flag (eval≤40 AND prod≥80) → `anomaly_type='reverse_sandbagging'`.
    AC7: weights = KYA_TIER_WEIGHTS[reporter_kya_tier].
    """
    target_id = snapshots[0].target_id if snapshots else ""
    n = len(snapshots)

    # Aggregate any data-quality warnings on input rows. A single legacy row in
    # the window taints the whole correlation result (per AC7 migration note).
    dq_warnings = {s.data_quality_warning for s in snapshots if s.data_quality_warning}
    dq_warning = sorted(dq_warnings)[0] if dq_warnings else None

    if n < MIN_SNAPSHOTS_FOR_PEARSON:
        return CorrelationResult(
            target_id=target_id,
            status="insufficient_data",
            n=n,
            data_quality_warning=dq_warning,
        )

    eval_scores = [float(s.eval_score_at_time) for s in snapshots]
    outcomes = [float(s.feedback_outcome) for s in snapshots]
    weights = [
        # Prefer the cached `weight` field if present, else look up KYA tier.
        s.weight if s.weight else KYA_TIER_WEIGHTS.get(s.reporter_kya_tier, 1.0)
        for s in snapshots
    ]

    r = weighted_pearson(eval_scores, outcomes, weights)
    total_w = sum(weights)
    avg_eval = sum(w * x for w, x in zip(weights, eval_scores)) / total_w if total_w else 0.0
    avg_outcome = sum(w * y for w, y in zip(weights, outcomes)) / total_w if total_w else 0.0

    # Anomaly detection — soft signal only per AC6, no auto tier change.
    anomaly: Optional[str] = None
    if avg_eval >= SANDBAGGING_EVAL_MIN and avg_outcome <= SANDBAGGING_PRODUCTION_MAX:
        anomaly = "sandbagging"
    elif (
        avg_eval <= REVERSE_SANDBAGGING_EVAL_MAX
        and avg_outcome >= REVERSE_SANDBAGGING_PRODUCTION_MIN
    ):
        anomaly = "reverse_sandbagging"

    sandbagging_risk = detect_sandbagging(int(avg_eval), int(avg_outcome), n)
    confidence_adj = compute_confidence_adjustment(r, n)
    alignment = classify_alignment(r)

    # Outcome breakdown — bucket on outcome_score quartiles (no `outcome` enum
    # in FeedbackSnapshot since it's a raw correlation primitive).
    breakdown: Dict[str, int] = {"success": 0, "partial": 0, "failure": 0}
    for s in snapshots:
        if s.feedback_outcome >= 70:
            breakdown["success"] += 1
        elif s.feedback_outcome >= 40:
            breakdown["partial"] += 1
        else:
            breakdown["failure"] += 1

    return CorrelationResult(
        target_id=target_id,
        status="ok",
        n=n,
        r=r,
        avg_eval=avg_eval,
        avg_outcome=avg_outcome,
        alignment=alignment,
        confidence_adjustment=confidence_adj,
        sandbagging_risk=sandbagging_risk,
        anomaly_type=anomaly,
        data_quality_warning=dq_warning,
        outcome_breakdown=breakdown,
    )


# ── Compatibility adapter for existing API/tests ────────────────────────────


def _snapshot_from_feedback_dict(
    target_id: str, current_eval_score: int, item: dict
) -> FeedbackSnapshot:
    """Adapter: build a FeedbackSnapshot from a raw feedback dict.

    If the row carries `eval_score_at_time` (post-migration), use it. Otherwise
    fall back to `current_eval_score` and tag the row `legacy_eval_unknown`.
    """
    snap_eval = item.get("eval_score_at_time")
    dq: Optional[str] = item.get("data_quality_warning")
    if snap_eval is None:
        snap_eval = current_eval_score
        dq = dq or "legacy_eval_unknown"

    tier = int(item.get("reporter_kya_tier", 1))
    return FeedbackSnapshot(
        target_id=target_id,
        eval_score_at_time=float(snap_eval),
        feedback_outcome=float(item.get("outcome_score", 0)),
        reporter_kya_tier=tier,
        weight=KYA_TIER_WEIGHTS.get(tier, 1.0),
        timestamp=item.get("created_at"),
        data_quality_warning=dq,
    )


def compute_correlation_report(
    target_id: str,
    eval_score: int,
    feedback_items: List[dict],
) -> CorrelationReport:
    """Build a complete correlation report from eval score and production feedback.

    QO-061: this is now an ADAPTER over `compute_correlation(snapshots)`. It
    accepts the legacy raw-feedback-dict shape (used by `feedback.py` API +
    existing tests) and constructs `FeedbackSnapshot` rows on the fly. Each
    raw row that lacks `eval_score_at_time` is back-filled with the current
    eval score AND tagged `legacy_eval_unknown` in the result.

    This adapter lets the API endpoint keep working while we migrate writers
    to the snapshot schema. New callers MUST use `compute_correlation()` with
    pre-built FeedbackSnapshot lists.
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

    # Outcome breakdown over the legacy `outcome` enum (success/partial/failure)
    outcome_breakdown: Dict[str, int] = {}
    for f in feedback_items:
        outcome = f.get("outcome", "unknown")
        outcome_breakdown[outcome] = outcome_breakdown.get(outcome, 0) + 1

    outcome_scores = [f.get("outcome_score", 0) for f in feedback_items]
    production_score = int(sum(outcome_scores) / len(outcome_scores)) if outcome_scores else 0

    snapshots = [
        _snapshot_from_feedback_dict(target_id, eval_score, item)
        for item in feedback_items
    ]
    result = compute_correlation(snapshots)

    # Below the Pearson threshold, fall back to gap-based alignment so the
    # legacy CorrelationReport stays informative for small-sample callers.
    if result.status == "insufficient_data":
        score_gap = abs(eval_score - production_score)
        if score_gap <= 10:
            alignment = "strong"
        elif score_gap <= 20:
            alignment = "moderate"
        elif score_gap <= 35:
            alignment = "weak"
        else:
            alignment = "negative" if production_score < eval_score else "none"
        sandbagging_risk = detect_sandbagging(
            eval_score, production_score, len(feedback_items)
        )
        # Soft anomaly detection still applies on small samples (AC6 guidance)
        anomaly = None
        if eval_score >= SANDBAGGING_EVAL_MIN and production_score <= SANDBAGGING_PRODUCTION_MAX:
            anomaly = "sandbagging"
        elif (
            eval_score <= REVERSE_SANDBAGGING_EVAL_MAX
            and production_score >= REVERSE_SANDBAGGING_PRODUCTION_MIN
        ):
            anomaly = "reverse_sandbagging"
        confidence_adj = compute_confidence_adjustment(
            _gap_to_pseudo_correlation(eval_score, production_score),
            len(feedback_items),
        )
        return CorrelationReport(
            target_id=target_id,
            eval_score=eval_score,
            production_score=production_score,
            correlation=None,
            feedback_count=len(feedback_items),
            alignment=alignment,
            confidence_adjustment=confidence_adj,
            sandbagging_risk=sandbagging_risk,
            outcome_breakdown=outcome_breakdown,
            anomaly_type=anomaly,
            data_quality_warning=result.data_quality_warning,
        )

    return CorrelationReport(
        target_id=target_id,
        eval_score=eval_score,
        production_score=production_score,
        correlation=result.r,
        feedback_count=result.n,
        alignment=result.alignment,
        confidence_adjustment=result.confidence_adjustment,
        sandbagging_risk=result.sandbagging_risk,
        outcome_breakdown=outcome_breakdown,
        anomaly_type=result.anomaly_type,
        data_quality_warning=result.data_quality_warning,
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

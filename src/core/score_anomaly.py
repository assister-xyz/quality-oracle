"""Score distribution anomaly detection for QO-043 (Judge Hardening).

Detects suspicious score patterns that may indicate adversarial manipulation:
- Sudden score jumps (z-score > 2 sigma from historical mean)
- First-evaluation extreme scores (>95 on first eval is statistically unlikely)
- Rapid score improvement without server changes (manifest hash unchanged)

References:
- Chatbot Arena vote rigging (ICML 2025): statistical anomaly detection
- "Cheating Automatic LLM Benchmarks" (ICLR 2025): null-model style attacks
"""
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# A new server scoring above this on first evaluation is flagged
FIRST_EVAL_EXTREME_THRESHOLD = 95

# Z-score threshold for flagging score deviation
Z_SCORE_THRESHOLD = 2.0

# Minimum history entries before z-score detection activates
MIN_HISTORY_FOR_ZSCORE = 3


@dataclass
class AnomalyAlert:
    """Describes a detected score anomaly."""
    anomaly_type: str    # "first_eval_extreme" | "z_score_deviation" | "unchanged_manifest_jump"
    severity: str        # "low" | "medium" | "high"
    target_id: str
    current_score: float
    details: dict

    def to_dict(self) -> dict:
        return {
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "target_id": self.target_id,
            "current_score": self.current_score,
            "details": self.details,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }


def _compute_mean_stdev(values: List[float]) -> tuple:
    """Compute mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance)


async def check_score_anomaly(
    target_id: str,
    new_score: float,
    manifest_hash: Optional[str] = None,
) -> Optional[AnomalyAlert]:
    """Check if a new score is anomalous compared to historical scores.

    Args:
        target_id: The MCP server/agent identifier
        new_score: The newly computed overall score
        manifest_hash: Current manifest hash (to detect unchanged-server jumps)

    Returns:
        AnomalyAlert if anomaly detected, None otherwise.
    """
    from src.storage.mongodb import score_history_col

    # Fetch score history (newest first)
    cursor = score_history_col().find(
        {"target_id": target_id},
        {"score": 1, "recorded_at": 1, "manifest_hash": 1, "_id": 0},
    ).sort("recorded_at", -1).limit(20)

    history_docs = await cursor.to_list(length=20)
    history_scores = [doc["score"] for doc in history_docs if "score" in doc]

    # ── Check 1: First evaluation extreme score ──────────────────────────
    if len(history_scores) == 0:
        if new_score >= FIRST_EVAL_EXTREME_THRESHOLD:
            return AnomalyAlert(
                anomaly_type="first_eval_extreme",
                severity="medium",
                target_id=target_id,
                current_score=new_score,
                details={
                    "threshold": FIRST_EVAL_EXTREME_THRESHOLD,
                    "message": f"First evaluation scored {new_score}, above {FIRST_EVAL_EXTREME_THRESHOLD} threshold",
                },
            )
        return None

    # ── Check 2: Z-score deviation ───────────────────────────────────────
    if len(history_scores) >= MIN_HISTORY_FOR_ZSCORE:
        mean, stdev = _compute_mean_stdev(history_scores)

        if stdev > 0:
            z_score = (new_score - mean) / stdev
        else:
            # All previous scores identical — any change is noteworthy
            z_score = abs(new_score - mean) * 10 if new_score != mean else 0.0

        if abs(z_score) > Z_SCORE_THRESHOLD:
            severity = "high" if abs(z_score) > 3.0 else "medium"
            return AnomalyAlert(
                anomaly_type="z_score_deviation",
                severity=severity,
                target_id=target_id,
                current_score=new_score,
                details={
                    "z_score": round(z_score, 2),
                    "historical_mean": round(mean, 1),
                    "historical_stdev": round(stdev, 1),
                    "history_count": len(history_scores),
                    "direction": "increase" if z_score > 0 else "decrease",
                    "message": f"Score {new_score} deviates {abs(z_score):.1f} sigma from mean {mean:.1f}",
                },
            )

    # ── Check 3: Unchanged manifest but large score jump ─────────────────
    if manifest_hash and len(history_docs) > 0:
        last_doc = history_docs[0]
        last_manifest = last_doc.get("manifest_hash")
        last_score = last_doc.get("score", 0)

        if last_manifest and last_manifest == manifest_hash:
            score_delta = abs(new_score - last_score)
            if score_delta > 20:
                return AnomalyAlert(
                    anomaly_type="unchanged_manifest_jump",
                    severity="medium",
                    target_id=target_id,
                    current_score=new_score,
                    details={
                        "previous_score": last_score,
                        "score_delta": score_delta,
                        "manifest_hash": manifest_hash,
                        "message": f"Score changed by {score_delta} points but manifest unchanged",
                    },
                )

    return None


async def record_anomaly(alert: AnomalyAlert) -> None:
    """Persist anomaly alert to MongoDB for admin review."""
    from src.storage.mongodb import score_anomalies_col

    doc = alert.to_dict()
    await score_anomalies_col().insert_one(doc)
    logger.warning(
        f"Score anomaly recorded: {alert.anomaly_type} for {alert.target_id} "
        f"(score={alert.current_score}, severity={alert.severity})"
    )

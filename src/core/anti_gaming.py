"""
Anti-gaming detection for AgentTrust evaluations.

Detects evaluation gaming through:
1. Response fingerprinting — flag identical/near-identical responses across evaluations
2. Timing analysis — flag suspiciously fast or uniform response times
3. Gaming risk scoring — aggregate signals into overall risk assessment

MongoDB collections:
- quality__response_fingerprints: stores hashed responses per target+question
- quality__paraphrase_log: tracks paraphrase variants used per evaluation
"""
import hashlib
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Timing Thresholds ────────────────────────────────────────────────────────

# Responses faster than this (ms) for non-trivial questions are suspicious
FAST_RESPONSE_THRESHOLD_MS = 100
# Response time standard deviation below this (ms) is suspicious (too uniform)
UNIFORM_TIMING_THRESHOLD_MS = 50
# Minimum responses needed for meaningful timing analysis
MIN_RESPONSES_FOR_TIMING = 4


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class TimingAnalysis:
    """Result of response timing analysis."""
    is_suspicious: bool = False
    fast_responses: int = 0
    total_responses: int = 0
    mean_ms: float = 0.0
    std_dev_ms: float = 0.0
    is_uniform: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "is_suspicious": self.is_suspicious,
            "fast_responses": self.fast_responses,
            "total_responses": self.total_responses,
            "mean_ms": round(self.mean_ms, 1),
            "std_dev_ms": round(self.std_dev_ms, 1),
            "is_uniform": self.is_uniform,
            "reason": self.reason,
        }


@dataclass
class FingerprintResult:
    """Result of response fingerprinting for a single response."""
    question_hash: str
    response_hash: str
    is_duplicate: bool = False
    prior_eval_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question_hash": self.question_hash,
            "response_hash": self.response_hash,
            "is_duplicate": self.is_duplicate,
            "prior_eval_id": self.prior_eval_id,
        }


@dataclass
class GamingRisk:
    """Aggregate gaming risk assessment."""
    level: str = "none"  # none, low, medium, high
    confidence_penalty: float = 0.0
    timing_anomaly: bool = False
    duplicate_responses: int = 0
    total_responses: int = 0
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "confidence_penalty": self.confidence_penalty,
            "timing_anomaly": self.timing_anomaly,
            "duplicate_responses": self.duplicate_responses,
            "total_responses": self.total_responses,
            "details": self.details,
        }


# ── Core Functions ───────────────────────────────────────────────────────────

def _hash_text(text: str) -> str:
    """Deterministic hash of normalized text."""
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:24]


def analyze_response_timing(
    response_times_ms: List[float],
) -> TimingAnalysis:
    """Analyze response times for gaming signals.

    Flags:
    - Suspiciously fast responses (< 100ms for complex questions)
    - Suspiciously uniform timing (std_dev < 50ms across all responses)
    """
    result = TimingAnalysis(total_responses=len(response_times_ms))

    if len(response_times_ms) < MIN_RESPONSES_FOR_TIMING:
        return result

    result.mean_ms = statistics.mean(response_times_ms)
    result.std_dev_ms = statistics.stdev(response_times_ms) if len(response_times_ms) >= 2 else 0.0

    # Check for fast responses
    fast_count = sum(1 for t in response_times_ms if t < FAST_RESPONSE_THRESHOLD_MS)
    result.fast_responses = fast_count

    # Check for uniform timing
    result.is_uniform = result.std_dev_ms < UNIFORM_TIMING_THRESHOLD_MS

    # Determine if suspicious
    fast_ratio = fast_count / len(response_times_ms)
    reasons = []

    if fast_ratio > 0.5:
        reasons.append(f"{fast_count}/{len(response_times_ms)} responses under {FAST_RESPONSE_THRESHOLD_MS}ms")
        result.is_suspicious = True

    if result.is_uniform and result.mean_ms < 500:
        reasons.append(f"uniform timing (std_dev={result.std_dev_ms:.0f}ms, mean={result.mean_ms:.0f}ms)")
        result.is_suspicious = True

    result.reason = "; ".join(reasons)
    return result


def fingerprint_response(
    question_text: str,
    response_text: str,
) -> FingerprintResult:
    """Create a fingerprint for a question-response pair.

    The actual duplicate detection against MongoDB is done in
    check_fingerprints_batch() to enable batched DB queries.
    """
    return FingerprintResult(
        question_hash=_hash_text(question_text),
        response_hash=_hash_text(response_text),
    )


async def check_fingerprints_batch(
    target_id: str,
    evaluation_id: str,
    fingerprints: List[FingerprintResult],
) -> List[FingerprintResult]:
    """Check a batch of fingerprints against stored history in MongoDB.

    For each fingerprint, check if a prior evaluation of the same target
    produced an identical response hash for the same question hash.

    Also stores the new fingerprints for future lookups.
    """
    try:
        from src.storage.mongodb import response_fingerprints_col

        col = response_fingerprints_col()

        # Batch query: find prior fingerprints for this target
        question_hashes = [fp.question_hash for fp in fingerprints]
        cursor = col.find({
            "target_id": target_id,
            "question_hash": {"$in": question_hashes},
            "evaluation_id": {"$ne": evaluation_id},  # Exclude current eval
        })
        prior_map: Dict[str, dict] = {}
        async for doc in cursor:
            key = f"{doc['question_hash']}:{doc['response_hash']}"
            prior_map[key] = doc

        # Check each fingerprint
        for fp in fingerprints:
            key = f"{fp.question_hash}:{fp.response_hash}"
            if key in prior_map:
                fp.is_duplicate = True
                fp.prior_eval_id = prior_map[key].get("evaluation_id")

        # Store new fingerprints (batch insert)
        docs = [
            {
                "target_id": target_id,
                "evaluation_id": evaluation_id,
                "question_hash": fp.question_hash,
                "response_hash": fp.response_hash,
                "created_at": datetime.utcnow(),
            }
            for fp in fingerprints
        ]
        if docs:
            await col.insert_many(docs)

    except Exception as e:
        logger.warning(f"Fingerprint check failed (non-fatal): {e}")

    return fingerprints


def compute_gaming_risk(
    timing: TimingAnalysis,
    fingerprints: List[FingerprintResult],
) -> GamingRisk:
    """Aggregate timing and fingerprint signals into gaming risk.

    Risk levels:
    - none: No signals detected
    - low: Minor timing anomalies OR 1 duplicate response
    - medium: Multiple signals OR 2+ duplicate responses
    - high: Strong timing + fingerprint signals, high confidence of gaming
    """
    risk = GamingRisk(total_responses=timing.total_responses)

    duplicates = [fp for fp in fingerprints if fp.is_duplicate]
    risk.duplicate_responses = len(duplicates)
    risk.timing_anomaly = timing.is_suspicious

    signals = 0

    # Timing signals
    if timing.is_suspicious:
        signals += 1
        risk.details["timing"] = timing.to_dict()

    # Fingerprint signals
    dup_ratio = len(duplicates) / len(fingerprints) if fingerprints else 0
    if len(duplicates) >= 3 or dup_ratio > 0.3:
        signals += 2  # Strong signal
        risk.details["duplicates"] = {
            "count": len(duplicates),
            "ratio": round(dup_ratio, 2),
            "prior_evals": list({fp.prior_eval_id for fp in duplicates if fp.prior_eval_id}),
        }
    elif len(duplicates) >= 1:
        signals += 1  # Weak signal
        risk.details["duplicates"] = {
            "count": len(duplicates),
            "ratio": round(dup_ratio, 2),
        }

    # Classify risk level
    if signals >= 3:
        risk.level = "high"
        risk.confidence_penalty = 0.15
    elif signals == 2:
        risk.level = "medium"
        risk.confidence_penalty = 0.10
    elif signals == 1:
        risk.level = "low"
        risk.confidence_penalty = 0.05
    else:
        risk.level = "none"
        risk.confidence_penalty = 0.0

    return risk


async def log_paraphrase(
    evaluation_id: str,
    target_id: str,
    entries: List[dict],
):
    """Store paraphrase audit trail in MongoDB.

    Each entry: {original, paraphrased, method, question_hash, response_hash}
    """
    if not entries:
        return

    try:
        from src.storage.mongodb import paraphrase_log_col

        doc = {
            "evaluation_id": evaluation_id,
            "target_id": target_id,
            "entries": entries,
            "created_at": datetime.utcnow(),
        }
        await paraphrase_log_col().insert_one(doc)
    except Exception as e:
        logger.warning(f"Paraphrase log failed (non-fatal): {e}")

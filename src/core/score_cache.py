"""
Score caching with TTL and exponential decay for fast A2A lookups.

Implements three-tier evaluation architecture:
- Tier 0: Cache lookup (<10ms) — serve pre-computed score if fresh
- Tier 1: Schema check (<100ms) — manifest validation only
- Tier 2: Full functional eval (<3s target) — tool probing + LLM judging

Scores decay exponentially over time:
  effective_score = cached_score * e^(-lambda * time_since_eval)

When effective score drops below confidence threshold, triggers re-evaluation.
"""
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default TTLs (seconds)
DEFAULT_SCORE_TTL = 3600  # 1 hour
HIGH_CONFIDENCE_TTL = 7200  # 2 hours (confidence > 0.8)
LOW_CONFIDENCE_TTL = 900  # 15 minutes (confidence < 0.4)

# Decay rate (higher = faster decay)
DEFAULT_DECAY_LAMBDA = 0.0003  # ~50% confidence after 38 minutes


@dataclass
class CachedScore:
    """A cached evaluation score with metadata."""
    target_url: str
    score: int  # 0-100
    tier: str
    confidence: float
    dimensions: Optional[Dict[str, int]] = None
    tools_count: int = 0
    transport: str = "unknown"
    evaluated_at: float = 0.0  # Unix timestamp
    ttl: float = DEFAULT_SCORE_TTL
    eval_duration_ms: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.evaluated_at

    @property
    def is_expired(self) -> bool:
        return self.age_seconds > self.ttl

    @property
    def effective_score(self) -> float:
        """Score with exponential decay applied."""
        decay = math.exp(-DEFAULT_DECAY_LAMBDA * self.age_seconds)
        return self.score * decay

    @property
    def effective_confidence(self) -> float:
        """Confidence with time decay."""
        decay = math.exp(-DEFAULT_DECAY_LAMBDA * self.age_seconds)
        return self.confidence * decay

    @property
    def freshness(self) -> str:
        """Human-readable freshness indicator."""
        age = self.age_seconds
        if age < 300:
            return "fresh"  # < 5 min
        elif age < 3600:
            return "recent"  # < 1 hour
        elif age < 86400:
            return "stale"  # < 1 day
        return "expired"

    def to_dict(self) -> dict:
        return {
            "target_url": self.target_url,
            "score": self.score,
            "effective_score": round(self.effective_score),
            "tier": self.tier,
            "confidence": round(self.confidence, 2),
            "effective_confidence": round(self.effective_confidence, 2),
            "freshness": self.freshness,
            "age_seconds": round(self.age_seconds),
            "dimensions": self.dimensions,
            "tools_count": self.tools_count,
            "transport": self.transport,
        }


class ScoreCache:
    """
    In-memory score cache for fast A2A quality lookups.

    Supports:
    - TTL-based expiration with confidence-aware TTL
    - Exponential decay for progressive staleness
    - Background re-evaluation hints when score is decaying
    - Max entries with LRU eviction
    """

    def __init__(self, max_entries: int = 1000):
        self._cache: Dict[str, CachedScore] = {}
        self._max_entries = max_entries

    def get(self, target_url: str) -> Optional[CachedScore]:
        """
        Get cached score for a target.

        Returns None if no cache entry or expired.
        Returns CachedScore with decayed values if within TTL.
        """
        entry = self._cache.get(target_url)
        if entry is None:
            return None

        if entry.is_expired:
            del self._cache[target_url]
            return None

        return entry

    def get_effective(self, target_url: str) -> Optional[dict]:
        """
        Get effective score with decay applied.

        Returns dict with effective_score, freshness, needs_refresh flag.
        """
        entry = self.get(target_url)
        if entry is None:
            return None

        result = entry.to_dict()
        result["needs_refresh"] = entry.effective_confidence < 0.5
        return result

    def put(
        self,
        target_url: str,
        score: int,
        tier: str,
        confidence: float,
        dimensions: Optional[Dict[str, int]] = None,
        tools_count: int = 0,
        transport: str = "unknown",
        eval_duration_ms: int = 0,
    ) -> CachedScore:
        """Store a score with confidence-aware TTL."""
        # Confidence-aware TTL
        if confidence > 0.8:
            ttl = HIGH_CONFIDENCE_TTL
        elif confidence < 0.4:
            ttl = LOW_CONFIDENCE_TTL
        else:
            ttl = DEFAULT_SCORE_TTL

        entry = CachedScore(
            target_url=target_url,
            score=score,
            tier=tier,
            confidence=confidence,
            dimensions=dimensions,
            tools_count=tools_count,
            transport=transport,
            evaluated_at=time.time(),
            ttl=ttl,
            eval_duration_ms=eval_duration_ms,
        )

        # LRU eviction if at capacity
        if len(self._cache) >= self._max_entries and target_url not in self._cache:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].evaluated_at)
            del self._cache[oldest_key]

        self._cache[target_url] = entry
        logger.debug(f"Cached score for {target_url}: {score}/100 (TTL={ttl}s)")
        return entry

    def invalidate(self, target_url: str) -> bool:
        """Remove a cached score."""
        if target_url in self._cache:
            del self._cache[target_url]
            return True
        return False

    def stats(self) -> dict:
        """Cache statistics."""
        time.time()
        entries = list(self._cache.values())
        fresh = sum(1 for e in entries if e.freshness == "fresh")
        recent = sum(1 for e in entries if e.freshness == "recent")
        stale = sum(1 for e in entries if e.freshness == "stale")

        return {
            "total_entries": len(entries),
            "fresh": fresh,
            "recent": recent,
            "stale": stale,
            "avg_score": round(sum(e.score for e in entries) / len(entries)) if entries else 0,
            "avg_effective_score": round(sum(e.effective_score for e in entries) / len(entries)) if entries else 0,
        }


# Global singleton for use across the application
_global_cache: Optional[ScoreCache] = None


def get_score_cache() -> ScoreCache:
    """Get or create the global score cache singleton."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ScoreCache()
    return _global_cache

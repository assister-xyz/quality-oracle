"""Redis cache for scores, rate limiting, and API key management."""
import logging
import redis.asyncio as redis

from src.config import settings

logger = logging.getLogger(__name__)

_redis: redis.Redis | None = None

# Cache TTLs (per architecture caching policy)
SCORE_TTL = 3600       # 1 hour for score lookups
BADGE_TTL = 21600      # 6 hours for badge SVGs
ATTESTATION_TTL = 86400  # 24 hours for attestation verification


async def connect_redis():
    global _redis
    _redis = redis.from_url(settings.redis_url, decode_responses=True)
    await _redis.ping()
    logger.info("Connected to Redis")


async def close_redis():
    global _redis
    if _redis:
        await _redis.close()
        logger.info("Redis connection closed")


def get_redis() -> redis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized. Call connect_redis() first.")
    return _redis


async def cache_score(target_id: str, score_data: dict, ttl: int = SCORE_TTL):
    """Cache a quality score (1 hour TTL)."""
    import json
    r = get_redis()
    await r.set(f"qo:score:{target_id}", json.dumps(score_data), ex=ttl)


async def get_cached_score(target_id: str) -> dict | None:
    """Get cached quality score."""
    import json
    r = get_redis()
    data = await r.get(f"qo:score:{target_id}")
    return json.loads(data) if data else None


async def cache_badge(target_id: str, svg: str, ttl: int = BADGE_TTL):
    """Cache a badge SVG (6 hour TTL)."""
    r = get_redis()
    await r.set(f"qo:badge:{target_id}", svg, ex=ttl)


async def get_cached_badge(target_id: str) -> str | None:
    """Get cached badge SVG."""
    r = get_redis()
    return await r.get(f"qo:badge:{target_id}")


async def cache_attestation_verify(attestation_id: str, result: dict, ttl: int = ATTESTATION_TTL):
    """Cache attestation verification result (24 hour TTL)."""
    import json
    r = get_redis()
    await r.set(f"qo:attest:{attestation_id}", json.dumps(result), ex=ttl)


async def get_cached_attestation_verify(attestation_id: str) -> dict | None:
    """Get cached attestation verification result."""
    import json
    r = get_redis()
    data = await r.get(f"qo:attest:{attestation_id}")
    return json.loads(data) if data else None


async def check_rate_limit(api_key: str, limit: int, window: str = "month") -> tuple[bool, int, int]:
    """
    Check if API key is within rate limit.

    Returns: (allowed, remaining, limit)
    """
    r = get_redis()
    key = f"qo:rate:{window}:{api_key}"
    current = await r.get(key)
    current_count = int(current) if current else 0

    if current_count >= limit:
        return False, 0, limit

    pipe = r.pipeline()
    pipe.incr(key)
    if window == "month":
        pipe.expire(key, 2592000)  # 30 days
    elif window == "day":
        pipe.expire(key, 86400)    # 24 hours
    elif window == "minute":
        pipe.expire(key, 60)
    await pipe.execute()

    return True, limit - current_count - 1, limit

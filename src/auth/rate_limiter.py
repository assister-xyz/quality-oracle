"""Rate limiting middleware for API keys — per architecture Section 2.5.1."""
from fastapi import Response

from src.storage.cache import check_rate_limit

# Evaluation limits (per month)
EVAL_LIMITS = {
    "free": 10,
    "developer": 100,
    "team": 500,
    "marketplace": 999999,
}

# Score lookup limits (per minute)
SCORE_LOOKUP_LIMITS = {
    "free": 5,
    "developer": 30,
    "team": 100,
    "marketplace": 500,
}

# Evaluation level restrictions per tier
TIER_EVAL_LEVELS = {
    "free": [1],
    "developer": [1, 2],
    "team": [1, 2, 3],
    "marketplace": [1, 2, 3],
}


async def check_eval_rate_limit(api_key: str, tier: str) -> tuple[bool, int, int]:
    """
    Check if an API key has exceeded its evaluation rate limit.

    Returns: (allowed, remaining, limit)
    """
    limit = EVAL_LIMITS.get(tier, 10)
    return await check_rate_limit(api_key, limit, window="month")


async def check_score_lookup_limit(api_key: str, tier: str) -> tuple[bool, int, int]:
    """
    Check if an API key has exceeded its score lookup rate limit.

    Returns: (allowed, remaining, limit)
    """
    limit = SCORE_LOOKUP_LIMITS.get(tier, 5)
    return await check_rate_limit(f"score:{api_key}", limit, window="minute")


def is_eval_level_allowed(tier: str, level: int) -> bool:
    """Check if the evaluation level is allowed for this tier."""
    allowed_levels = TIER_EVAL_LEVELS.get(tier, [1])
    return level in allowed_levels


def add_rate_limit_headers(
    response: Response,
    tier: str,
    limit: int,
    remaining: int,
    reset_timestamp: int | None = None,
):
    """Add rate limit headers to response per architecture spec."""
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    if reset_timestamp:
        response.headers["X-RateLimit-Reset"] = str(reset_timestamp)
    response.headers["X-Quality-Oracle-Tier"] = tier

"""Tests for API key management and rate limiting."""
import pytest
from unittest.mock import AsyncMock, patch

from src.auth.api_keys import generate_api_key, hash_api_key
from src.auth.rate_limiter import (
    is_eval_level_allowed,
    EVAL_LIMITS,
    SCORE_LOOKUP_LIMITS,
    TIER_EVAL_LEVELS,
)


def test_generate_api_key_prefix():
    """Generated keys should start with 'qo_'."""
    key = generate_api_key()
    assert key.startswith("qo_")
    assert len(key) > 10  # Should be substantial


def test_generate_api_key_unique():
    """Each generated key should be unique."""
    keys = {generate_api_key() for _ in range(100)}
    assert len(keys) == 100


def test_hash_api_key_deterministic():
    """Same key should always produce the same hash."""
    key = "qo_test-key-12345"
    h1 = hash_api_key(key)
    h2 = hash_api_key(key)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex digest


def test_hash_api_key_different_keys():
    """Different keys should produce different hashes."""
    h1 = hash_api_key("qo_key-one")
    h2 = hash_api_key("qo_key-two")
    assert h1 != h2


def test_eval_level_allowed_free():
    """Free tier should only allow Level 1."""
    assert is_eval_level_allowed("free", 1) is True
    assert is_eval_level_allowed("free", 2) is False
    assert is_eval_level_allowed("free", 3) is False


def test_eval_level_allowed_developer():
    """Developer tier should allow Level 1 and 2."""
    assert is_eval_level_allowed("developer", 1) is True
    assert is_eval_level_allowed("developer", 2) is True
    assert is_eval_level_allowed("developer", 3) is False


def test_eval_level_allowed_team():
    """Team tier should allow all levels."""
    assert is_eval_level_allowed("team", 1) is True
    assert is_eval_level_allowed("team", 2) is True
    assert is_eval_level_allowed("team", 3) is True


def test_eval_level_allowed_marketplace():
    """Marketplace tier should allow all levels."""
    assert is_eval_level_allowed("marketplace", 1) is True
    assert is_eval_level_allowed("marketplace", 2) is True
    assert is_eval_level_allowed("marketplace", 3) is True


def test_eval_level_unknown_tier():
    """Unknown tier should default to Level 1 only."""
    assert is_eval_level_allowed("unknown", 1) is True
    assert is_eval_level_allowed("unknown", 2) is False


def test_eval_limits_structure():
    """Rate limits should be defined for all tiers."""
    for tier in ["free", "developer", "team", "marketplace"]:
        assert tier in EVAL_LIMITS
        assert tier in SCORE_LOOKUP_LIMITS
        assert tier in TIER_EVAL_LEVELS


def test_rate_limits_ascending():
    """Higher tiers should have higher limits."""
    assert EVAL_LIMITS["free"] < EVAL_LIMITS["developer"]
    assert EVAL_LIMITS["developer"] < EVAL_LIMITS["team"]
    assert SCORE_LOOKUP_LIMITS["free"] < SCORE_LOOKUP_LIMITS["developer"]


@pytest.mark.asyncio
async def test_validate_api_key_valid():
    """Valid key should return the key document."""
    mock_doc = {"_id": "hash", "tier": "developer", "active": True}
    with patch("src.auth.api_keys.api_keys_col") as mock_col:
        mock_col.return_value.find_one = AsyncMock(return_value=mock_doc)
        mock_col.return_value.update_one = AsyncMock()

        from src.auth.api_keys import validate_api_key
        result = await validate_api_key("qo_test-key")
    assert result is not None
    assert result["tier"] == "developer"


@pytest.mark.asyncio
async def test_validate_api_key_invalid():
    """Invalid key should return None."""
    with patch("src.auth.api_keys.api_keys_col") as mock_col:
        mock_col.return_value.find_one = AsyncMock(return_value=None)

        from src.auth.api_keys import validate_api_key
        result = await validate_api_key("qo_bad-key")
    assert result is None

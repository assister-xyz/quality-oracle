"""Shared test fixtures for Quality Oracle."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_api_key_doc():
    """A valid API key document as returned by validate_api_key."""
    return {
        "_id": "hashed_key_abc123",
        "key_prefix": "qo_test12",
        "owner_email": "test@example.com",
        "tier": "developer",
        "created_at": datetime(2026, 1, 1),
        "last_used_at": None,
        "active": True,
        "used_this_month": 0,
    }


@pytest.fixture()
def auth_headers():
    """Headers with a valid API key for authenticated requests."""
    return {"X-API-Key": "qo_test-valid-key-for-testing"}


class AsyncIterEmpty:
    """Async iterator that yields nothing — simulates empty cursor."""
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _make_mock_col():
    """Create a mock MongoDB collection with async methods."""
    mock_col = MagicMock()
    mock_col.find_one = AsyncMock(return_value=None)
    mock_col.insert_one = AsyncMock()
    mock_col.update_one = AsyncMock()
    mock_col.count_documents = AsyncMock(return_value=0)
    mock_col.create_index = AsyncMock()

    mock_cursor = AsyncIterEmpty()
    mock_cursor.sort = MagicMock(return_value=mock_cursor)
    mock_cursor.skip = MagicMock(return_value=mock_cursor)
    mock_cursor.limit = MagicMock(return_value=mock_cursor)
    mock_col.find = MagicMock(return_value=mock_cursor)

    return mock_col


@pytest.fixture()
def test_client(mock_api_key_doc):
    """FastAPI TestClient with mocked DB, Redis, and auth."""
    mock_col = _make_mock_col()

    patches = [
        # MongoDB collection accessors
        patch("src.storage.mongodb.evaluations_col", return_value=mock_col),
        patch("src.storage.mongodb.scores_col", return_value=mock_col),
        patch("src.storage.mongodb.score_history_col", return_value=mock_col),
        patch("src.storage.mongodb.attestations_col", return_value=mock_col),
        patch("src.storage.mongodb.question_banks_col", return_value=mock_col),
        patch("src.storage.mongodb.api_keys_col", return_value=mock_col),
        # Also patch where imported (evaluate.py, scores.py, etc.)
        patch("src.api.v1.evaluate.evaluations_col", return_value=mock_col),
        patch("src.api.v1.evaluate.scores_col", return_value=mock_col),
        patch("src.api.v1.evaluate.score_history_col", return_value=mock_col),
        patch("src.api.v1.scores.scores_col", return_value=mock_col),
        patch("src.api.v1.attestations.attestations_col", return_value=mock_col),
        patch("src.api.v1.enrichment.scores_col", return_value=mock_col),
        # Auth — validate_api_key where the dependency imports it
        patch("src.auth.dependencies.validate_api_key", new_callable=AsyncMock, return_value=mock_api_key_doc),
        # Cache functions — patch where imported
        patch("src.api.v1.scores.get_cached_score", new_callable=AsyncMock, return_value=None),
        patch("src.api.v1.scores.cache_score", new_callable=AsyncMock),
        patch("src.storage.cache.get_cached_badge", new_callable=AsyncMock, return_value=None),
        patch("src.storage.cache.cache_badge", new_callable=AsyncMock),
        patch("src.api.v1.attestations.get_cached_attestation_verify", new_callable=AsyncMock, return_value=None),
        patch("src.api.v1.attestations.cache_attestation_verify", new_callable=AsyncMock),
        # Rate limiting — patch where imported in rate_limiter module
        patch("src.auth.rate_limiter.check_rate_limit", new_callable=AsyncMock, return_value=(True, 99, 100)),
        # Lifecycle
        patch("src.storage.mongodb.connect_db", new_callable=AsyncMock),
        patch("src.storage.mongodb.close_db", new_callable=AsyncMock),
        patch("src.storage.cache.connect_redis", new_callable=AsyncMock),
        patch("src.storage.cache.close_redis", new_callable=AsyncMock),
    ]

    for p in patches:
        p.start()

    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    for p in reversed(patches):
        p.stop()

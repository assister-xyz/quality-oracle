"""Tests for Arena API endpoints (Challenge Ladder)."""
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import _make_mock_col


def _make_ladder_col_with_entries(entries):
    """Create a mock ladder col that returns entries from find()."""
    col = _make_mock_col()

    class AsyncIterEntries:
        def __init__(self):
            self._items = list(entries)
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)
        def sort(self, *a, **kw):
            return self
        def limit(self, *a, **kw):
            return self

    col.find = MagicMock(return_value=AsyncIterEntries())
    return col


def _arena_client(mock_api_key_doc, mock_ladder=None, mock_scores=None):
    """Create test client with arena-specific mocks."""
    if mock_ladder is None:
        mock_ladder = _make_mock_col()
    if mock_scores is None:
        mock_scores = _make_mock_col()

    patches = [
        # MongoDB
        patch("src.storage.mongodb.evaluations_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.scores_col", return_value=mock_scores),
        patch("src.storage.mongodb.score_history_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.attestations_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.question_banks_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.api_keys_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.battles_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.ladder_col", return_value=mock_ladder),
        # Patch where imported
        patch("src.api.v1.evaluate.evaluations_col", return_value=_make_mock_col()),
        patch("src.api.v1.evaluate.scores_col", return_value=mock_scores),
        patch("src.api.v1.evaluate.score_history_col", return_value=_make_mock_col()),
        patch("src.api.v1.scores.scores_col", return_value=mock_scores),
        patch("src.api.v1.attestations.attestations_col", return_value=_make_mock_col()),
        patch("src.api.v1.enrichment.scores_col", return_value=mock_scores),
        patch("src.api.v1.battles.battles_col", return_value=_make_mock_col()),
        # Arena / ladder patches
        patch("src.core.ladder.ladder_col", return_value=mock_ladder),
        patch("src.core.ladder.scores_col", return_value=mock_scores),
        patch("src.core.ladder.battles_col", return_value=_make_mock_col()),
        # Battle engine patches
        patch("src.core.battle.battles_col", return_value=_make_mock_col()),
        patch("src.core.battle.scores_col", return_value=mock_scores),
        # Auth
        patch("src.auth.dependencies.validate_api_key", new_callable=AsyncMock, return_value=mock_api_key_doc),
        # Cache
        patch("src.api.v1.scores.get_cached_score", new_callable=AsyncMock, return_value=None),
        patch("src.api.v1.scores.cache_score", new_callable=AsyncMock),
        patch("src.storage.cache.get_cached_badge", new_callable=AsyncMock, return_value=None),
        patch("src.storage.cache.cache_badge", new_callable=AsyncMock),
        patch("src.api.v1.attestations.get_cached_attestation_verify", new_callable=AsyncMock, return_value=None),
        patch("src.api.v1.attestations.cache_attestation_verify", new_callable=AsyncMock),
        # Rate limiting
        patch("src.auth.rate_limiter.check_rate_limit", new_callable=AsyncMock, return_value=(True, 99, 100)),
        # Lifecycle
        patch("src.storage.mongodb.connect_db", new_callable=AsyncMock),
        patch("src.storage.mongodb.close_db", new_callable=AsyncMock),
        patch("src.storage.cache.connect_redis", new_callable=AsyncMock),
        patch("src.storage.cache.close_redis", new_callable=AsyncMock),
        # Prevent lifespan from hitting real DB for API key seeding
        patch("src.main._ensure_default_api_key", new_callable=AsyncMock),
    ]

    for p in patches:
        p.start()

    from fastapi.testclient import TestClient
    from src.main import app
    client = TestClient(app, raise_server_exceptions=False)
    return client, patches


class TestGetLadder:
    def test_global_ladder_empty(self, mock_api_key_doc, auth_headers):
        """GET /v1/arena/ladder returns empty list."""
        client, patches = _arena_client(mock_api_key_doc)
        try:
            resp = client.get("/v1/arena/ladder", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["items"] == []
            assert data["domain"] is None
            assert data["count"] == 0
        finally:
            for p in reversed(patches):
                p.stop()

    def test_global_ladder_with_entries(self, mock_api_key_doc, auth_headers):
        """GET /v1/arena/ladder returns ranked entries."""
        entries = [
            {"target_id": "a1", "position": 1, "name": "Champion", "domain": None},
            {"target_id": "a2", "position": 2, "name": "Runner-up", "domain": None},
        ]
        mock_ladder = _make_ladder_col_with_entries(entries)

        client, patches = _arena_client(mock_api_key_doc, mock_ladder=mock_ladder)
        try:
            resp = client.get("/v1/arena/ladder", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["items"]) == 2
            assert data["items"][0]["position"] == 1
        finally:
            for p in reversed(patches):
                p.stop()

    def test_domain_ladder(self, mock_api_key_doc, auth_headers):
        """GET /v1/arena/ladder/{domain} returns domain-filtered entries."""
        entries = [
            {"target_id": "a1", "position": 1, "domain": "coding"},
        ]
        mock_ladder = _make_ladder_col_with_entries(entries)

        client, patches = _arena_client(mock_api_key_doc, mock_ladder=mock_ladder)
        try:
            resp = client.get("/v1/arena/ladder/coding", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["domain"] == "coding"
            assert len(data["items"]) == 1
        finally:
            for p in reversed(patches):
                p.stop()


class TestChallenge:
    def test_challenge_success(self, mock_api_key_doc, auth_headers):
        """POST /v1/arena/challenge creates a ladder battle."""
        mock_ladder = _make_mock_col()
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": {"target_id": "challenger", "position": 6, "domain": None,
                           "openskill_mu": 25.0, "openskill_sigma": 8.333},
            "target": {"target_id": "target", "position": 3, "domain": None,
                       "openskill_mu": 25.0, "openskill_sigma": 8.333},
        }.get(q.get("target_id")))

        mock_battles = _make_mock_col()
        mock_battles.find_one = AsyncMock(return_value=None)  # No cooldown

        client, patches = _arena_client(mock_api_key_doc, mock_ladder=mock_ladder)
        # Override battles_col for ladder
        p_battles = patch("src.core.ladder.battles_col", return_value=mock_battles)
        p_battles.start()
        patches.append(p_battles)

        try:
            resp = client.post(
                "/v1/arena/challenge",
                json={"challenger_id": "challenger", "target_id": "target"},
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "battle_id" in data
            assert data["status"] == "pending"
        finally:
            for p in reversed(patches):
                p.stop()

    def test_challenge_self_rejected(self, mock_api_key_doc, auth_headers):
        """POST /v1/arena/challenge self → 422."""
        client, patches = _arena_client(mock_api_key_doc)
        try:
            resp = client.post(
                "/v1/arena/challenge",
                json={"challenger_id": "agent-1", "target_id": "agent-1"},
                headers=auth_headers,
            )
            assert resp.status_code == 422
            assert "self" in resp.json()["detail"].lower()
        finally:
            for p in reversed(patches):
                p.stop()

    def test_challenge_beyond_distance_rejected(self, mock_api_key_doc, auth_headers):
        """POST /v1/arena/challenge beyond 5 positions → 422."""
        mock_ladder = _make_mock_col()
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "challenger": {"target_id": "challenger", "position": 10, "domain": None},
            "target": {"target_id": "target", "position": 2, "domain": None},
        }.get(q.get("target_id")))

        client, patches = _arena_client(mock_api_key_doc, mock_ladder=mock_ladder)
        try:
            resp = client.post(
                "/v1/arena/challenge",
                json={"challenger_id": "challenger", "target_id": "target"},
                headers=auth_headers,
            )
            assert resp.status_code == 422
            assert "5 positions" in resp.json()["detail"].lower()
        finally:
            for p in reversed(patches):
                p.stop()


class TestPredict:
    def test_predict_match(self, mock_api_key_doc, auth_headers):
        """GET /v1/arena/predict/{a}/{b} returns prediction."""
        mock_ladder = _make_mock_col()
        mock_ladder.find_one = AsyncMock(side_effect=lambda q: {
            "agent-a": {"target_id": "agent-a", "openskill_mu": 30.0, "openskill_sigma": 5.0},
            "agent-b": {"target_id": "agent-b", "openskill_mu": 20.0, "openskill_sigma": 5.0},
        }.get(q.get("target_id")))

        client, patches = _arena_client(mock_api_key_doc, mock_ladder=mock_ladder)
        try:
            resp = client.get("/v1/arena/predict/agent-a/agent-b", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "win_probability_a" in data
            assert "win_probability_b" in data
            assert "match_quality" in data
            assert "recommendation" in data
            # Agent A has higher mu → higher win probability
            assert data["win_probability_a"] > data["win_probability_b"]
        finally:
            for p in reversed(patches):
                p.stop()


class TestSeed:
    def test_seed_ladder(self, mock_api_key_doc, auth_headers):
        """POST /v1/arena/seed returns seeded count."""
        client, patches = _arena_client(mock_api_key_doc)
        try:
            resp = client.post("/v1/arena/seed", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "seeded" in data
            assert data["seeded"] == 0  # Empty scores
        finally:
            for p in reversed(patches):
                p.stop()

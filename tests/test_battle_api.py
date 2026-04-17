"""Tests for battle API endpoints."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from tests.conftest import _make_mock_col


def _mock_verified_operator():
    """Build a mock verified operator for require_verified_operator override."""
    from src.storage.models import Operator, OperatorStatus
    return Operator(
        operator_id="op_test_verified",
        display_name="Test Operator",
        email="test@example.com",
        auth_provider="github",
        github_user_id=99999,
        github_username="testbot",
        github_avatar_url="https://example.com/avatar.png",
        github_account_age_days=365,
        github_public_repos=10,
        github_followers=20,
        verified=True,
        agent_target_ids=[],
        max_agents=5,
        max_battles_per_day=15,
        status=OperatorStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture()
def mock_battles_col():
    return _make_mock_col()


@pytest.fixture()
def battle_api_client(mock_api_key_doc, mock_battles_col):
    """Test client with battle-specific mocks."""
    mock_scores_col = _make_mock_col()

    patches = [
        # MongoDB
        patch("src.storage.mongodb.evaluations_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.scores_col", return_value=mock_scores_col),
        patch("src.storage.mongodb.score_history_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.attestations_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.question_banks_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.api_keys_col", return_value=_make_mock_col()),
        patch("src.storage.mongodb.battles_col", return_value=mock_battles_col),
        patch("src.storage.mongodb.ladder_col", return_value=_make_mock_col()),
        # Patch where imported in API modules
        patch("src.api.v1.evaluate.evaluations_col", return_value=_make_mock_col()),
        patch("src.api.v1.evaluate.scores_col", return_value=mock_scores_col),
        patch("src.api.v1.evaluate.score_history_col", return_value=_make_mock_col()),
        patch("src.api.v1.scores.scores_col", return_value=mock_scores_col),
        patch("src.api.v1.attestations.attestations_col", return_value=_make_mock_col()),
        patch("src.api.v1.enrichment.scores_col", return_value=mock_scores_col),
        patch("src.api.v1.battles.battles_col", return_value=mock_battles_col),
        # Battle engine internal patches
        patch("src.core.battle.battles_col", return_value=mock_battles_col),
        patch("src.core.battle.scores_col", return_value=mock_scores_col),
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
        # QO-046: patch add_agent_to_operator (battle.py calls it to auto-claim agents)
        patch("src.api.v1.battles.add_agent_to_operator", new_callable=AsyncMock),
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
    from src.auth.dependencies import require_verified_operator

    # QO-046: Override require_verified_operator to return a mock verified operator
    # so existing battle tests don't need to set up full OAuth session
    app.dependency_overrides[require_verified_operator] = lambda: _mock_verified_operator()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()

    for p in reversed(patches):
        p.stop()


class TestCreateBattle:
    def test_create_battle_success(self, battle_api_client, auth_headers, mock_battles_col):
        """POST /v1/battle creates a battle and returns battle_id."""
        mock_battles_col.find_one = AsyncMock(return_value=None)  # No cooldown
        mock_battles_col.insert_one = AsyncMock()

        resp = battle_api_client.post(
            "/v1/battle",
            json={
                "agent_a_url": "https://agent-a.example.com/mcp",
                "agent_b_url": "https://agent-b.different.com/mcp",
            },
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "battle_id" in data
        assert data["status"] == "pending"
        assert "poll_url" in data

    def test_create_battle_same_operator_rejected(self, battle_api_client, auth_headers):
        """Same host → 422."""
        resp = battle_api_client.post(
            "/v1/battle",
            json={
                "agent_a_url": "https://same-host.com/mcp/a",
                "agent_b_url": "https://same-host.com/mcp/b",
            },
            headers=auth_headers,
        )

        assert resp.status_code == 422
        assert "same operator" in resp.json()["detail"].lower()

    def test_create_battle_cooldown_rejected(self, battle_api_client, auth_headers, mock_battles_col):
        """Recent battle between same pair → 429."""
        mock_battles_col.find_one = AsyncMock(return_value={
            "created_at": datetime.utcnow(),
        })

        resp = battle_api_client.post(
            "/v1/battle",
            json={
                "agent_a_url": "https://agent-a.example.com/mcp",
                "agent_b_url": "https://agent-b.different.com/mcp",
            },
            headers=auth_headers,
        )

        assert resp.status_code == 429
        assert "cooldown" in resp.json()["detail"].lower()


class TestGetBattle:
    def test_get_battle_found(self, battle_api_client, auth_headers, mock_battles_col):
        """GET /v1/battle/{id} returns battle doc."""
        mock_battles_col.find_one = AsyncMock(return_value={
            "_id": "test-battle-id",
            "battle_id": "test-battle-id",
            "status": "completed",
            "winner": "a",
            "margin": 15,
        })

        resp = battle_api_client.get(
            "/v1/battle/test-battle-id",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["battle_id"] == "test-battle-id"
        assert data["status"] == "completed"
        assert "_id" not in data  # _id stripped

    def test_get_battle_not_found(self, battle_api_client, auth_headers, mock_battles_col):
        """GET /v1/battle/{id} with invalid ID → 404."""
        mock_battles_col.find_one = AsyncMock(return_value=None)

        resp = battle_api_client.get(
            "/v1/battle/nonexistent",
            headers=auth_headers,
        )

        assert resp.status_code == 404


class TestListBattles:
    def test_list_battles_empty(self, battle_api_client, auth_headers, mock_battles_col):
        """GET /v1/battles returns empty list."""
        resp = battle_api_client.get(
            "/v1/battles",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_battles_with_pagination(self, battle_api_client, auth_headers, mock_battles_col):
        """GET /v1/battles?page=1&limit=10 respects pagination."""
        resp = battle_api_client.get(
            "/v1/battles?page=1&limit=10",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 1
        assert data["limit"] == 10


class TestGetAgentBattles:
    def test_agent_battles_empty(self, battle_api_client, auth_headers, mock_battles_col):
        """GET /v1/battles/agent/{id} returns empty for unknown agent."""
        resp = battle_api_client.get(
            "/v1/battles/agent/unknown-agent",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["target_id"] == "unknown-agent"


class TestBattleCard:
    def test_battle_card_completed(self, battle_api_client, mock_battles_col):
        """GET /v1/battle/{id}/card.svg returns SVG for completed battle."""
        mock_battles_col.find_one = AsyncMock(return_value={
            "_id": "card-test",
            "status": "completed",
            "agent_a": {"name": "Agent Alpha", "overall_score": 85, "scores": {}},
            "agent_b": {"name": "Agent Beta", "overall_score": 72, "scores": {}},
            "winner": "a",
            "margin": 13,
            "photo_finish": False,
            "match_quality": 0.85,
        })

        resp = battle_api_client.get("/v1/battle/card-test/card.svg")

        assert resp.status_code == 200
        assert "image/svg+xml" in resp.headers["content-type"]
        assert "<svg" in resp.text
        assert "Agent Alpha" in resp.text

    def test_battle_card_not_completed(self, battle_api_client, mock_battles_col):
        """GET /v1/battle/{id}/card.svg → 400 if not completed."""
        mock_battles_col.find_one = AsyncMock(return_value={
            "_id": "running-battle",
            "status": "running",
        })

        resp = battle_api_client.get("/v1/battle/running-battle/card.svg")
        assert resp.status_code == 400

    def test_battle_card_not_found(self, battle_api_client, mock_battles_col):
        """GET /v1/battle/{id}/card.svg → 404 if missing."""
        mock_battles_col.find_one = AsyncMock(return_value=None)

        resp = battle_api_client.get("/v1/battle/missing/card.svg")
        assert resp.status_code == 404

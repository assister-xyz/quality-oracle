"""Tests for require_verified_operator middleware (QO-046)."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.auth.dependencies import require_verified_operator
from src.auth.session import create_session_token
from src.storage.models import Operator, OperatorStatus


def _mock_request(cookies: dict, path: str = "/v1/battle"):
    """Build a minimal Request mock with cookies."""
    mock = MagicMock()
    mock.cookies = cookies
    mock_url = MagicMock()
    mock_url.path = path
    mock.url = mock_url
    return mock


def _sample_verified_operator(**overrides) -> Operator:
    base = {
        "operator_id": "op_verified_123",
        "display_name": "Test User",
        "email": "test@example.com",
        "auth_provider": "github",
        "github_user_id": 12345,
        "github_username": "testuser",
        "github_avatar_url": "https://example.com/avatar.png",
        "github_account_age_days": 365,
        "github_public_repos": 10,
        "github_followers": 20,
        "verified": True,
        "agent_target_ids": [],
        "max_agents": 5,
        "max_battles_per_day": 15,
        "status": OperatorStatus.ACTIVE,
        "created_at": datetime.now(timezone.utc),
    }
    base.update(overrides)
    return Operator(**base)


class TestRequireVerifiedOperator:

    @pytest.mark.asyncio
    async def test_no_cookie_returns_403(self):
        req = _mock_request({})
        with pytest.raises(HTTPException) as exc_info:
            await require_verified_operator(req)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "verified_operator_required"
        assert "login_url" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_cookie_returns_403(self):
        req = _mock_request({"laureum_session": "garbage.token.here"})
        with pytest.raises(HTTPException) as exc_info:
            await require_verified_operator(req)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "invalid_session"

    @pytest.mark.asyncio
    async def test_valid_session_returns_operator(self):
        operator = _sample_verified_operator()
        token = create_session_token(
            operator.operator_id,
            operator.github_username,
        )
        req = _mock_request({"laureum_session": token})

        with patch(
            "src.core.operator_identity.get_operator_by_id",
            new_callable=AsyncMock,
            return_value=operator,
        ):
            result = await require_verified_operator(req)

        assert result.operator_id == operator.operator_id
        assert result.github_username == operator.github_username

    @pytest.mark.asyncio
    async def test_operator_not_found_returns_403(self):
        token = create_session_token("op_missing", "ghostuser")
        req = _mock_request({"laureum_session": token})

        with patch(
            "src.core.operator_identity.get_operator_by_id",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await require_verified_operator(req)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "operator_not_found"

    @pytest.mark.asyncio
    async def test_unverified_operator_returns_403(self):
        operator = _sample_verified_operator(verified=False, auth_provider="email")
        token = create_session_token(operator.operator_id, "someuser")
        req = _mock_request({"laureum_session": token})

        with patch(
            "src.core.operator_identity.get_operator_by_id",
            new_callable=AsyncMock,
            return_value=operator,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await require_verified_operator(req)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "operator_not_verified"

    @pytest.mark.asyncio
    async def test_suspended_operator_returns_403(self):
        operator = _sample_verified_operator(status=OperatorStatus.SUSPENDED)
        token = create_session_token(operator.operator_id, operator.github_username)
        req = _mock_request({"laureum_session": token})

        with patch(
            "src.core.operator_identity.get_operator_by_id",
            new_callable=AsyncMock,
            return_value=operator,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await require_verified_operator(req)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["error"] == "operator_suspended"

    @pytest.mark.asyncio
    async def test_login_url_includes_return_to(self):
        req = _mock_request({}, path="/v1/battle")
        with pytest.raises(HTTPException) as exc_info:
            await require_verified_operator(req)
        assert "return_to=/v1/battle" in exc_info.value.detail["login_url"]

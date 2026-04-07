"""Tests for operator identity (QO-044)."""
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.core.operator_identity import (
    register_operator,
    get_operator_by_id,
    get_operator_by_email,
    get_operator_for_agent,
    add_agent_to_operator,
    are_same_operator,
    check_operator_battle_limit,
    url_to_target_id,
    _normalize_email,
    _validate_email,
    _generate_operator_id,
    DuplicateOperatorError,
    OperatorNotFoundError,
    MaxAgentsExceededError,
    OperatorError,
)


class TestHelpers:

    def test_normalize_email(self):
        assert _normalize_email("  Test@Example.COM  ") == "test@example.com"
        assert _normalize_email("user@domain.com") == "user@domain.com"

    def test_validate_email_valid(self):
        _validate_email("user@example.com")
        _validate_email("test.user+tag@sub.domain.org")

    def test_validate_email_invalid(self):
        with pytest.raises(OperatorError):
            _validate_email("not-an-email")
        with pytest.raises(OperatorError):
            _validate_email("missing@tld")
        with pytest.raises(OperatorError):
            _validate_email("@nodomain.com")

    def test_generate_operator_id_format(self):
        op_id = _generate_operator_id()
        assert op_id.startswith("op_")
        assert len(op_id) == 19  # "op_" + 16 hex chars

    def test_generate_operator_id_unique(self):
        ids = {_generate_operator_id() for _ in range(100)}
        assert len(ids) == 100

    def test_url_to_target_id(self):
        tid = url_to_target_id("https://example.com/mcp")
        assert len(tid) == 16
        assert tid == url_to_target_id("https://example.com/mcp")

    def test_url_to_target_id_different(self):
        a = url_to_target_id("https://example.com/mcp")
        b = url_to_target_id("https://other.com/mcp")
        assert a != b


class TestRegisterOperator:

    @pytest.mark.asyncio
    async def test_register_success(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)
        mock_col.insert_one = AsyncMock()

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            operator = await register_operator(
                display_name="Test User",
                email="test@example.com",
            )

        assert operator.display_name == "Test User"
        assert operator.email == "test@example.com"
        assert operator.operator_id.startswith("op_")
        assert operator.max_agents == 5
        assert operator.max_battles_per_day == 15

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={"email": "test@example.com"})

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            with pytest.raises(DuplicateOperatorError):
                await register_operator("Test", "test@example.com")

    @pytest.mark.asyncio
    async def test_register_short_name(self):
        with pytest.raises(OperatorError, match="display_name"):
            await register_operator("X", "test@example.com")

    @pytest.mark.asyncio
    async def test_register_invalid_email(self):
        with pytest.raises(OperatorError, match="email"):
            await register_operator("Test User", "not-an-email")

    @pytest.mark.asyncio
    async def test_register_normalizes_email(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)
        mock_col.insert_one = AsyncMock()

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            operator = await register_operator("Test", "  USER@EXAMPLE.COM  ")

        assert operator.email == "user@example.com"


class TestGetOperator:

    @pytest.mark.asyncio
    async def test_get_by_id_found(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={
            "operator_id": "op_test123",
            "display_name": "Test",
            "email": "test@example.com",
            "auth_provider": "email",
            "agent_target_ids": [],
            "max_agents": 5,
            "max_battles_per_day": 15,
            "status": "active",
            "created_at": __import__("datetime").datetime.now(),
        })

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            op = await get_operator_by_id("op_test123")

        assert op is not None
        assert op.operator_id == "op_test123"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            op = await get_operator_by_id("op_missing")

        assert op is None

    @pytest.mark.asyncio
    async def test_get_for_agent_returns_owner(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={
            "operator_id": "op_owner",
            "display_name": "Owner",
            "email": "owner@example.com",
            "auth_provider": "email",
            "agent_target_ids": ["abc123"],
            "max_agents": 5,
            "max_battles_per_day": 15,
            "status": "active",
            "created_at": __import__("datetime").datetime.now(),
        })

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            op = await get_operator_for_agent("abc123")

        assert op is not None
        assert op.operator_id == "op_owner"


class TestAddAgentToOperator:

    @pytest.mark.asyncio
    async def test_add_agent_success(self):
        from datetime import datetime as dt

        mock_col = MagicMock()
        # First find_one (in get_operator_for_agent) → not owned
        # Second find_one (in get_operator_by_id) → operator found
        operator_doc = {
            "operator_id": "op_test",
            "display_name": "Test",
            "email": "test@example.com",
            "auth_provider": "email",
            "agent_target_ids": [],
            "max_agents": 5,
            "max_battles_per_day": 15,
            "status": "active",
            "created_at": dt.now(),
        }
        mock_col.find_one = AsyncMock(side_effect=[None, operator_doc])
        mock_col.update_one = AsyncMock()

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            op = await add_agent_to_operator("op_test", "https://example.com/mcp")

        assert len(op.agent_target_ids) == 1

    @pytest.mark.asyncio
    async def test_add_agent_max_exceeded(self):
        from datetime import datetime as dt

        mock_col = MagicMock()
        operator_doc = {
            "operator_id": "op_full",
            "display_name": "Full",
            "email": "full@example.com",
            "auth_provider": "email",
            "agent_target_ids": ["a", "b", "c", "d", "e"],
            "max_agents": 5,
            "max_battles_per_day": 15,
            "status": "active",
            "created_at": dt.now(),
        }
        mock_col.find_one = AsyncMock(side_effect=[None, operator_doc])

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            with pytest.raises(MaxAgentsExceededError):
                await add_agent_to_operator("op_full", "https://example.com/mcp")

    @pytest.mark.asyncio
    async def test_add_agent_owned_by_other(self):
        from datetime import datetime as dt

        mock_col = MagicMock()
        # First find_one returns a different operator
        other_op = {
            "operator_id": "op_other",
            "display_name": "Other",
            "email": "other@example.com",
            "auth_provider": "email",
            "agent_target_ids": ["target"],
            "max_agents": 5,
            "max_battles_per_day": 15,
            "status": "active",
            "created_at": dt.now(),
        }
        mock_col.find_one = AsyncMock(return_value=other_op)

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            with pytest.raises(OperatorError, match="already owned"):
                await add_agent_to_operator("op_test", "https://example.com/mcp")


class TestSameOperator:

    @pytest.mark.asyncio
    async def test_same_operator_true(self):
        from datetime import datetime as dt
        op = {
            "operator_id": "op_same",
            "display_name": "Same",
            "email": "same@example.com",
            "auth_provider": "email",
            "agent_target_ids": ["a", "b"],
            "max_agents": 5,
            "max_battles_per_day": 15,
            "status": "active",
            "created_at": dt.now(),
        }

        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=op)

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            result = await are_same_operator("a", "b")

        assert result is True

    @pytest.mark.asyncio
    async def test_same_operator_false(self):
        from datetime import datetime as dt

        def make_op(op_id, agents):
            return {
                "operator_id": op_id,
                "display_name": "Op",
                "email": f"{op_id}@example.com",
                "auth_provider": "email",
                "agent_target_ids": agents,
                "max_agents": 5,
                "max_battles_per_day": 15,
                "status": "active",
                "created_at": dt.now(),
            }

        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(side_effect=[
            make_op("op_a", ["a"]),
            make_op("op_b", ["b"]),
        ])

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            result = await are_same_operator("a", "b")

        assert result is False

    @pytest.mark.asyncio
    async def test_same_operator_one_unregistered(self):
        from datetime import datetime as dt

        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(side_effect=[None, None])

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            result = await are_same_operator("a", "b")

        assert result is False  # Unknown operators don't count as same


class TestBattleLimit:

    @pytest.mark.asyncio
    async def test_check_battle_limit_allowed(self):
        with patch(
            "src.core.operator_identity.check_rate_limit",
            new_callable=AsyncMock,
            return_value=(True, 14, 15),
        ):
            allowed, remaining, limit = await check_operator_battle_limit("op_test")

        assert allowed is True
        assert remaining == 14
        assert limit == 15

    @pytest.mark.asyncio
    async def test_check_battle_limit_exceeded(self):
        with patch(
            "src.core.operator_identity.check_rate_limit",
            new_callable=AsyncMock,
            return_value=(False, 0, 15),
        ):
            allowed, _, _ = await check_operator_battle_limit("op_test")

        assert allowed is False

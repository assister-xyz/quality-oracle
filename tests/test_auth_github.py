"""Tests for GitHub operator upsert & anti-abuse (QO-046)."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.operator_identity import (
    check_github_anti_abuse,
    upsert_github_operator,
    log_rejected_registration,
    get_operator_by_github_id,
    AntiAbuseError,
)


class TestCheckGitHubAntiAbuse:

    def test_healthy_account_passes(self):
        check_github_anti_abuse(
            account_age_days=365,
            public_repos=10,
            followers=20,
        )  # no raise

    def test_minimum_age_passes(self):
        check_github_anti_abuse(
            account_age_days=30,
            public_repos=1,
            followers=0,
        )  # no raise

    def test_account_too_young_rejected(self):
        with pytest.raises(AntiAbuseError, match="too young"):
            check_github_anti_abuse(
                account_age_days=15,
                public_repos=5,
                followers=10,
            )

    def test_empty_profile_rejected(self):
        with pytest.raises(AntiAbuseError, match="no public repositories"):
            check_github_anti_abuse(
                account_age_days=365,
                public_repos=0,
                followers=0,
            )

    def test_only_repos_passes(self):
        check_github_anti_abuse(
            account_age_days=365,
            public_repos=1,
            followers=0,
        )  # no raise

    def test_only_followers_passes(self):
        check_github_anti_abuse(
            account_age_days=365,
            public_repos=0,
            followers=1,
        )  # no raise

    def test_custom_min_age(self):
        with pytest.raises(AntiAbuseError):
            check_github_anti_abuse(
                account_age_days=45,
                public_repos=5,
                followers=5,
                min_age_days=60,
            )

    def test_disable_repos_follower_check(self):
        # When require_repos_or_followers=False, empty profile passes
        check_github_anti_abuse(
            account_age_days=365,
            public_repos=0,
            followers=0,
            require_repos_or_followers=False,
        )  # no raise


class TestUpsertGitHubOperator:

    @pytest.mark.asyncio
    async def test_create_new_operator(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)  # Not exists
        mock_col.insert_one = AsyncMock()

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            op = await upsert_github_operator(
                github_user_id=12345,
                github_username="vitamin33",
                github_avatar_url="https://avatars.githubusercontent.com/u/12345",
                display_name="Vitalii",
                account_age_days=1000,
                public_repos=50,
                followers=100,
                email="vitalii@example.com",
            )

        assert op.github_user_id == 12345
        assert op.github_username == "vitamin33"
        assert op.auth_provider == "github"
        assert op.verified is True
        assert op.operator_id.startswith("op_")
        assert op.email == "vitalii@example.com"
        mock_col.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_existing_operator(self):
        existing_doc = {
            "_id": "mongo_id_abc",
            "operator_id": "op_existing",
            "display_name": "Old Name",
            "email": "old@example.com",
            "auth_provider": "github",
            "github_user_id": 12345,
            "github_username": "old_username",
            "github_avatar_url": "old_url",
            "github_account_age_days": 500,
            "github_public_repos": 10,
            "github_followers": 20,
            "verified": True,
            "agent_target_ids": ["agent_1"],
            "max_agents": 5,
            "max_battles_per_day": 15,
            "status": "active",
            "created_at": datetime.now(timezone.utc),
        }

        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=existing_doc)
        mock_col.update_one = AsyncMock()

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            op = await upsert_github_operator(
                github_user_id=12345,
                github_username="new_username",
                github_avatar_url="new_url",
                display_name="New Name",
                account_age_days=1000,
                public_repos=50,
                followers=100,
                email="new@example.com",
            )

        assert op.operator_id == "op_existing"
        assert op.github_username == "new_username"  # Updated
        assert op.github_avatar_url == "new_url"
        assert op.email == "new@example.com"
        mock_col.update_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_anti_abuse_young_account(self):
        mock_col = MagicMock()
        mock_attempts = MagicMock()
        mock_attempts.insert_one = AsyncMock()

        with patch("src.storage.mongodb.operators_col", return_value=mock_col), \
             patch("src.storage.mongodb.operator_registration_attempts_col", return_value=mock_attempts):
            with pytest.raises(AntiAbuseError, match="too young"):
                await upsert_github_operator(
                    github_user_id=999,
                    github_username="throwaway",
                    github_avatar_url="",
                    display_name="Throwaway",
                    account_age_days=5,
                    public_repos=1,
                    followers=0,
                )

        # Rejection should be logged
        mock_attempts.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_anti_abuse_empty_profile(self):
        mock_col = MagicMock()
        mock_attempts = MagicMock()
        mock_attempts.insert_one = AsyncMock()

        with patch("src.storage.mongodb.operators_col", return_value=mock_col), \
             patch("src.storage.mongodb.operator_registration_attempts_col", return_value=mock_attempts):
            with pytest.raises(AntiAbuseError, match="no public repositories"):
                await upsert_github_operator(
                    github_user_id=888,
                    github_username="empty",
                    github_avatar_url="",
                    display_name="Empty",
                    account_age_days=365,
                    public_repos=0,
                    followers=0,
                )

        mock_attempts.insert_one.assert_called_once()


class TestGetOperatorByGitHubId:

    @pytest.mark.asyncio
    async def test_found(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value={
            "operator_id": "op_found",
            "display_name": "Found",
            "email": "found@example.com",
            "auth_provider": "github",
            "github_user_id": 777,
            "github_username": "founduser",
            "verified": True,
            "agent_target_ids": [],
            "max_agents": 5,
            "max_battles_per_day": 15,
            "status": "active",
            "created_at": datetime.now(timezone.utc),
        })

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            op = await get_operator_by_github_id(777)

        assert op is not None
        assert op.github_user_id == 777

    @pytest.mark.asyncio
    async def test_not_found(self):
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)

        with patch("src.storage.mongodb.operators_col", return_value=mock_col):
            op = await get_operator_by_github_id(999)

        assert op is None


class TestLogRejectedRegistration:

    @pytest.mark.asyncio
    async def test_inserts_document(self):
        mock_col = MagicMock()
        mock_col.insert_one = AsyncMock()

        with patch("src.storage.mongodb.operator_registration_attempts_col", return_value=mock_col):
            await log_rejected_registration(
                github_user_id=123,
                github_username="rejected",
                reason="account_too_young",
                details={"age_days": 5},
            )

        mock_col.insert_one.assert_called_once()
        doc = mock_col.insert_one.call_args[0][0]
        assert doc["github_user_id"] == 123
        assert doc["github_username"] == "rejected"
        assert doc["reason"] == "account_too_young"
        assert doc["details"] == {"age_days": 5}
        assert "rejected_at" in doc

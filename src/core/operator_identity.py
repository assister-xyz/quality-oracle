"""Operator identity for QO-044 Sybil Defense.

Provides email-based operator registration and agent ownership tracking
to prevent single actors from creating multiple fake agent identities
that manipulate the leaderboard.

References:
- Google DeepMind "AI Agent Traps" (2026): Sybil Attacks
- Chatbot Arena vote rigging (ICML 2025)
- ERC-8004 minimum bond proposals
"""
import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from src.storage.cache import check_rate_limit
from src.storage.models import (
    Operator,
    OperatorStatus,
)

logger = logging.getLogger(__name__)


# Defaults
DEFAULT_MAX_AGENTS = 5
DEFAULT_MAX_BATTLES_PER_DAY = 15

# Email validation (simple, RFC-lite)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


class OperatorError(Exception):
    """Base error for operator operations."""


class DuplicateOperatorError(OperatorError):
    """Operator with this email already exists."""


class OperatorNotFoundError(OperatorError):
    """Operator not found."""


class MaxAgentsExceededError(OperatorError):
    """Operator already owns max_agents agents."""


class OperatorRateLimitError(OperatorError):
    """Operator exceeded daily battle limit."""


class AntiAbuseError(OperatorError):
    """GitHub account failed anti-abuse heuristics."""


def _normalize_email(email: str) -> str:
    """Lowercase and strip whitespace."""
    return email.strip().lower()


def _validate_email(email: str) -> None:
    if not _EMAIL_RE.match(email):
        raise OperatorError(f"Invalid email format: {email}")


def _generate_operator_id() -> str:
    """Generate a unique operator ID."""
    return "op_" + uuid.uuid4().hex[:16]


def url_to_target_id(url: str) -> str:
    """Convert an MCP server URL to a stable target_id (matches battle.py)."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


# ── Registration ─────────────────────────────────────────────────────────────


async def register_operator(
    display_name: str,
    email: str,
    max_agents: int = DEFAULT_MAX_AGENTS,
    max_battles_per_day: int = DEFAULT_MAX_BATTLES_PER_DAY,
) -> Operator:
    """Register a new operator with email-based identity.

    Raises:
        OperatorError: invalid input
        DuplicateOperatorError: email already registered
    """
    from src.storage.mongodb import operators_col

    if not display_name or len(display_name.strip()) < 2:
        raise OperatorError("display_name must be at least 2 characters")

    email = _normalize_email(email)
    _validate_email(email)

    col = operators_col()

    # Check for existing email
    existing = await col.find_one({"email": email})
    if existing:
        raise DuplicateOperatorError(f"Operator with email {email} already exists")

    operator_id = _generate_operator_id()
    now = datetime.now(timezone.utc)

    doc = {
        "operator_id": operator_id,
        "display_name": display_name.strip(),
        "email": email,
        "auth_provider": "email",
        "agent_target_ids": [],
        "max_agents": max_agents,
        "max_battles_per_day": max_battles_per_day,
        "status": OperatorStatus.ACTIVE.value,
        "created_at": now,
    }

    await col.insert_one(doc)
    logger.info(f"Registered operator {operator_id} ({display_name}, {email})")

    return Operator(**doc)


async def get_operator_by_id(operator_id: str) -> Optional[Operator]:
    """Lookup operator by ID."""
    from src.storage.mongodb import operators_col

    doc = await operators_col().find_one({"operator_id": operator_id})
    if doc:
        doc.pop("_id", None)
        return Operator(**doc)
    return None


async def get_operator_by_email(email: str) -> Optional[Operator]:
    """Lookup operator by email."""
    from src.storage.mongodb import operators_col

    email = _normalize_email(email)
    doc = await operators_col().find_one({"email": email})
    if doc:
        doc.pop("_id", None)
        return Operator(**doc)
    return None


async def get_operator_for_agent(target_id: str) -> Optional[Operator]:
    """Find which operator owns a given agent target_id."""
    from src.storage.mongodb import operators_col

    doc = await operators_col().find_one({"agent_target_ids": target_id})
    if doc:
        doc.pop("_id", None)
        return Operator(**doc)
    return None


# ── Agent Assignment ─────────────────────────────────────────────────────────


async def add_agent_to_operator(operator_id: str, agent_url: str) -> Operator:
    """Assign an MCP server (by URL) to an operator.

    Raises:
        OperatorNotFoundError
        MaxAgentsExceededError
        OperatorError if agent already owned by another operator
    """
    from src.storage.mongodb import operators_col

    target_id = url_to_target_id(agent_url)

    # Check if already owned
    existing_owner = await get_operator_for_agent(target_id)
    if existing_owner:
        if existing_owner.operator_id == operator_id:
            return existing_owner  # Already owned by this operator
        raise OperatorError(
            f"Agent {target_id} already owned by operator {existing_owner.operator_id}"
        )

    operator = await get_operator_by_id(operator_id)
    if not operator:
        raise OperatorNotFoundError(f"Operator {operator_id} not found")

    if len(operator.agent_target_ids) >= operator.max_agents:
        raise MaxAgentsExceededError(
            f"Operator {operator_id} already has {operator.max_agents} agents (max)"
        )

    now = datetime.now(timezone.utc)
    await operators_col().update_one(
        {"operator_id": operator_id},
        {
            "$addToSet": {"agent_target_ids": target_id},
            "$set": {"updated_at": now},
        },
    )

    operator.agent_target_ids.append(target_id)
    operator.updated_at = now
    logger.info(f"Added agent {target_id} to operator {operator_id}")
    return operator


# ── Same-Operator Detection ──────────────────────────────────────────────────


async def are_same_operator(target_id_a: str, target_id_b: str) -> bool:
    """Return True if both agents are owned by the same operator.

    Returns False if either agent has no operator (legacy/unregistered).
    """
    op_a = await get_operator_for_agent(target_id_a)
    op_b = await get_operator_for_agent(target_id_b)

    if not op_a or not op_b:
        return False
    return op_a.operator_id == op_b.operator_id


# ── Per-Operator Battle Rate Limit ───────────────────────────────────────────


async def check_operator_battle_limit(operator_id: str, limit: int = DEFAULT_MAX_BATTLES_PER_DAY) -> tuple:
    """Check if operator has remaining daily battle quota.

    Returns: (allowed, remaining, limit)
    """
    key = f"operator_battles:{operator_id}"
    return await check_rate_limit(key, limit, window="day")


# ── GitHub OAuth (QO-046) ────────────────────────────────────────────────────


def check_github_anti_abuse(
    account_age_days: int,
    public_repos: int,
    followers: int,
    min_age_days: int = 30,
    require_repos_or_followers: bool = True,
) -> None:
    """Apply anti-abuse heuristics to a GitHub profile.

    Raises AntiAbuseError with a reason code if the profile fails checks.
    """
    if account_age_days < min_age_days:
        raise AntiAbuseError(
            f"GitHub account too young: {account_age_days} days < {min_age_days} required"
        )

    if require_repos_or_followers and public_repos == 0 and followers == 0:
        raise AntiAbuseError(
            "GitHub account has no public repositories or followers (likely throwaway)"
        )


async def log_rejected_registration(
    github_user_id: int,
    github_username: str,
    reason: str,
    details: Optional[dict] = None,
) -> None:
    """Log a rejected GitHub OAuth registration attempt for audit."""
    from src.storage.mongodb import operator_registration_attempts_col

    doc = {
        "github_user_id": github_user_id,
        "github_username": github_username,
        "reason": reason,
        "details": details or {},
        "rejected_at": datetime.now(timezone.utc),
    }
    try:
        await operator_registration_attempts_col().insert_one(doc)
    except Exception as e:
        logger.error(f"Failed to log rejected registration: {e}")


async def upsert_github_operator(
    github_user_id: int,
    github_username: str,
    github_avatar_url: str,
    display_name: str,
    account_age_days: int,
    public_repos: int,
    followers: int,
    email: Optional[str] = None,
) -> Operator:
    """Create or update an operator from a GitHub profile.

    Applies anti-abuse checks. Raises AntiAbuseError if rejected.
    On existing github_user_id, updates profile fields and last_login_at.
    """
    from src.config import settings
    from src.storage.mongodb import operators_col

    # 1. Apply anti-abuse heuristics
    try:
        check_github_anti_abuse(
            account_age_days=account_age_days,
            public_repos=public_repos,
            followers=followers,
            min_age_days=settings.github_min_account_age_days,
            require_repos_or_followers=settings.github_require_repos_or_followers,
        )
    except AntiAbuseError as e:
        await log_rejected_registration(
            github_user_id=github_user_id,
            github_username=github_username,
            reason=str(e),
            details={
                "account_age_days": account_age_days,
                "public_repos": public_repos,
                "followers": followers,
            },
        )
        raise

    col = operators_col()
    now = datetime.now(timezone.utc)

    # 2. Check if operator exists by github_user_id
    existing = await col.find_one({"github_user_id": github_user_id})

    if existing:
        # Update profile data + last_login_at
        update_doc = {
            "$set": {
                "github_username": github_username,
                "github_avatar_url": github_avatar_url,
                "github_account_age_days": account_age_days,
                "github_public_repos": public_repos,
                "github_followers": followers,
                "email": email or existing.get("email"),
                "verified": True,
                "last_login_at": now,
                "updated_at": now,
            }
        }
        await col.update_one({"github_user_id": github_user_id}, update_doc)
        existing.update(update_doc["$set"])
        existing.pop("_id", None)
        logger.info(f"Updated GitHub operator {existing['operator_id']} ({github_username})")
        return Operator(**existing)

    # 3. Create new operator
    operator_id = _generate_operator_id()
    doc = {
        "operator_id": operator_id,
        "display_name": display_name or github_username,
        "email": email,
        "auth_provider": "github",
        "github_user_id": github_user_id,
        "github_username": github_username,
        "github_avatar_url": github_avatar_url,
        "github_account_age_days": account_age_days,
        "github_public_repos": public_repos,
        "github_followers": followers,
        "verified": True,
        "agent_target_ids": [],
        "max_agents": DEFAULT_MAX_AGENTS,
        "max_battles_per_day": DEFAULT_MAX_BATTLES_PER_DAY,
        "status": OperatorStatus.ACTIVE.value,
        "created_at": now,
        "last_login_at": now,
    }

    await col.insert_one(doc)
    logger.info(f"Created GitHub operator {operator_id} ({github_username}) age={account_age_days}d repos={public_repos} followers={followers}")
    return Operator(**doc)


async def get_operator_by_github_id(github_user_id: int) -> Optional[Operator]:
    """Lookup operator by GitHub user ID."""
    from src.storage.mongodb import operators_col

    doc = await operators_col().find_one({"github_user_id": github_user_id})
    if doc:
        doc.pop("_id", None)
        return Operator(**doc)
    return None

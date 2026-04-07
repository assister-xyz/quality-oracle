"""JWT session management for QO-046 GitHub OAuth.

Signs session cookies with HS256 using settings.session_secret. Tokens
include operator_id, github_username, and verified flag.
"""
import logging
import time
from typing import Optional

import jwt

from src.config import settings

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"


def create_session_token(
    operator_id: str,
    github_username: str,
    verified: bool = True,
    ttl_seconds: Optional[int] = None,
) -> str:
    """Create a signed JWT session token."""
    now = int(time.time())
    ttl = ttl_seconds if ttl_seconds is not None else settings.session_cookie_max_age
    payload = {
        "sub": operator_id,
        "github_username": github_username,
        "verified": verified,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, settings.session_secret, algorithm=ALGORITHM)


def decode_session_token(token: str) -> Optional[dict]:
    """Decode and validate a session JWT. Returns payload or None if invalid/expired."""
    if not token:
        return None
    try:
        payload = jwt.decode(
            token,
            settings.session_secret,
            algorithms=[ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.info("Session token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid session token: {e}")
        return None

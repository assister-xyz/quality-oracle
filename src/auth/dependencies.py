"""FastAPI auth dependencies for AgentTrust endpoints."""
import logging

from fastapi import Header, HTTPException, Request

from src.auth.api_keys import validate_api_key
from src.auth.session import decode_session_token
from src.config import settings

logger = logging.getLogger(__name__)


async def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> dict:
    """
    Validate API key from X-API-Key header.

    Returns the key document if valid.
    Raises 401 if missing or invalid.
    """
    doc = await validate_api_key(x_api_key)
    if not doc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    return doc


async def require_verified_operator(request: Request):
    """
    QO-046: Require a GitHub-verified operator session.

    Reads the laureum_session HttpOnly cookie, validates the JWT, and loads
    the operator from the database. Returns 403 with a structured error
    including a login_url for the frontend to redirect to.
    """
    # Late import to avoid circular dependency with operator_identity
    from src.core.operator_identity import get_operator_by_id

    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "verified_operator_required",
                "message": "Sign in with GitHub to access this feature",
                "login_url": f"/v1/auth/github?return_to={request.url.path}",
            },
        )

    payload = decode_session_token(token)
    if not payload:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "invalid_session",
                "message": "Session expired or invalid. Please sign in again.",
                "login_url": f"/v1/auth/github?return_to={request.url.path}",
            },
        )

    operator_id = payload.get("sub")
    if not operator_id:
        raise HTTPException(status_code=403, detail={"error": "invalid_session"})

    operator = await get_operator_by_id(operator_id)
    if not operator:
        raise HTTPException(
            status_code=403,
            detail={"error": "operator_not_found"},
        )

    if not operator.verified:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "operator_not_verified",
                "message": "Your operator account is not verified. Re-authenticate with GitHub.",
                "login_url": f"/v1/auth/github?return_to={request.url.path}",
            },
        )

    if operator.status != "active":
        raise HTTPException(
            status_code=403,
            detail={
                "error": "operator_suspended",
                "message": f"Your operator account is {operator.status}.",
            },
        )

    return operator

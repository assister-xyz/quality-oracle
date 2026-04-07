"""GitHub OAuth authentication endpoints (QO-046).

Flow:
    GET /v1/auth/github?return_to=/battle
        → generates state, stores in Redis
        → redirects to github.com/login/oauth/authorize

    GET /v1/auth/github/callback?code=X&state=Y
        → validates state
        → exchanges code for access_token
        → fetches GitHub profile + email
        → runs anti-abuse checks
        → upserts operator
        → sets session cookie
        → redirects to return_to

    GET /v1/auth/me
        → returns current verified operator info (or 403)

    POST /v1/auth/logout
        → clears session cookie
"""
import logging
import secrets
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from src.auth.dependencies import require_verified_operator
from src.auth.session import create_session_token
from src.config import settings
from src.core.operator_identity import (
    AntiAbuseError,
    upsert_github_operator,
)
from src.storage.cache import get_redis
from src.storage.models import AuthMeResponse, Operator

logger = logging.getLogger(__name__)
router = APIRouter()

GITHUB_OAUTH_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_API = "https://api.github.com/user"
GITHUB_EMAILS_API = "https://api.github.com/user/emails"
OAUTH_STATE_TTL_SECONDS = 300  # 5 minutes


def _is_safe_return_to(return_to: str) -> bool:
    """Only allow relative paths starting with / and not containing //."""
    if not return_to or not return_to.startswith("/"):
        return False
    if return_to.startswith("//"):
        return False
    # Reject javascript: and data: schemes
    if ":" in return_to.split("/", 2)[1] if len(return_to.split("/", 2)) > 1 else False:
        return False
    return True


@router.get("/auth/github")
async def github_login(return_to: str = "/battle"):
    """Initiate GitHub OAuth flow."""
    if not settings.github_client_id:
        raise HTTPException(
            status_code=503,
            detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID.",
        )

    if not _is_safe_return_to(return_to):
        return_to = "/battle"

    # Generate CSRF state token, store in Redis with return_to
    state = secrets.token_urlsafe(32)
    try:
        redis = get_redis()
        await redis.set(
            f"qo:oauth_state:{state}",
            return_to,
            ex=OAUTH_STATE_TTL_SECONDS,
        )
    except Exception as e:
        logger.error(f"Failed to store OAuth state in Redis: {e}")
        raise HTTPException(status_code=503, detail="OAuth state storage unavailable")

    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": settings.github_oauth_redirect_url,
        "state": state,
        "scope": "read:user user:email",
        "allow_signup": "true",
    }
    url = f"{GITHUB_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"
    logger.info(f"[oauth] Initiating GitHub OAuth for return_to={return_to}")
    return RedirectResponse(url, status_code=302)


@router.get("/auth/github/callback")
async def github_callback(
    code: str,
    state: str,
    error: str = None,
    error_description: str = None,
):
    """Handle GitHub OAuth callback — exchange code, fetch profile, create session."""
    # User denied or GitHub error
    if error:
        logger.info(f"[oauth] GitHub OAuth error: {error} — {error_description}")
        return _redirect_to_frontend("/battle?auth_error=denied")

    # 1. Validate state token
    try:
        redis = get_redis()
        return_to = await redis.get(f"qo:oauth_state:{state}")
        if return_to:
            await redis.delete(f"qo:oauth_state:{state}")
    except Exception as e:
        logger.error(f"Redis error validating OAuth state: {e}")
        return _redirect_to_frontend("/battle?auth_error=state_storage")

    if not return_to:
        logger.warning(f"[oauth] Invalid or expired state token: {state[:8]}...")
        return _redirect_to_frontend("/battle?auth_error=invalid_state")

    # 2. Exchange code for access token
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            token_resp = await client.post(
                GITHUB_TOKEN_URL,
                data={
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret,
                    "code": code,
                    "redirect_uri": settings.github_oauth_redirect_url,
                },
                headers={"Accept": "application/json"},
            )
            token_data = token_resp.json()
        except Exception as e:
            logger.error(f"[oauth] Token exchange failed: {e}")
            return _redirect_to_frontend("/battle?auth_error=token_exchange")

        access_token = token_data.get("access_token")
        if not access_token:
            logger.warning(f"[oauth] No access_token in response: {token_data}")
            return _redirect_to_frontend("/battle?auth_error=no_token")

        # 3. Fetch user profile
        try:
            user_resp = await client.get(
                GITHUB_USER_API,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "Laureum-Evaluator/1.0",
                },
            )
            if user_resp.status_code != 200:
                logger.warning(f"[oauth] GitHub user API {user_resp.status_code}")
                return _redirect_to_frontend("/battle?auth_error=user_fetch")
            gh_user = user_resp.json()
        except Exception as e:
            logger.error(f"[oauth] User fetch failed: {e}")
            return _redirect_to_frontend("/battle?auth_error=user_fetch")

        # 4. Fetch primary email (may not be public on profile)
        primary_email = gh_user.get("email")
        if not primary_email:
            try:
                emails_resp = await client.get(
                    GITHUB_EMAILS_API,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github+json",
                        "User-Agent": "Laureum-Evaluator/1.0",
                    },
                )
                if emails_resp.status_code == 200:
                    emails = emails_resp.json()
                    primary_email = next(
                        (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                        None,
                    )
            except Exception as e:
                logger.warning(f"[oauth] Email fetch failed: {e}")

    # 5. Compute account age
    try:
        created_at_str = gh_user.get("created_at", "")
        if created_at_str.endswith("Z"):
            created_at_str = created_at_str[:-1] + "+00:00"
        gh_created_at = datetime.fromisoformat(created_at_str)
        account_age_days = (datetime.now(timezone.utc) - gh_created_at).days
    except Exception as e:
        logger.error(f"[oauth] Failed to parse GitHub created_at: {e}")
        account_age_days = 0

    # 6. Upsert operator (applies anti-abuse checks internally)
    try:
        operator = await upsert_github_operator(
            github_user_id=gh_user["id"],
            github_username=gh_user["login"],
            github_avatar_url=gh_user.get("avatar_url", ""),
            display_name=gh_user.get("name") or gh_user["login"],
            account_age_days=account_age_days,
            public_repos=gh_user.get("public_repos", 0),
            followers=gh_user.get("followers", 0),
            email=primary_email,
        )
    except AntiAbuseError as e:
        logger.info(f"[oauth] Anti-abuse rejection for {gh_user.get('login')}: {e}")
        reason = "account_too_young" if "too young" in str(e) else "empty_profile"
        return _redirect_to_frontend(f"/battle?auth_error={reason}")
    except Exception as e:
        logger.error(f"[oauth] Operator upsert failed: {e}")
        return _redirect_to_frontend("/battle?auth_error=upsert_failed")

    # 7. Create session JWT
    session_token = create_session_token(
        operator_id=operator.operator_id,
        github_username=operator.github_username,
        verified=True,
    )

    # 8. Redirect to return_to with session cookie
    redirect_url = _build_frontend_url(return_to)
    response = RedirectResponse(redirect_url, status_code=302)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_token,
        max_age=settings.session_cookie_max_age,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        path="/",
    )
    logger.info(f"[oauth] Session issued for {operator.github_username} (op_id={operator.operator_id})")
    return response


@router.get("/auth/me", response_model=AuthMeResponse)
async def auth_me(operator: Operator = Depends(require_verified_operator)):
    """Return current verified operator info. 403 if not signed in."""
    return AuthMeResponse(
        operator_id=operator.operator_id,
        display_name=operator.display_name,
        github_username=operator.github_username or "",
        github_avatar_url=operator.github_avatar_url,
        email=operator.email,
        verified=operator.verified,
        agent_count=len(operator.agent_target_ids),
        max_agents=operator.max_agents,
        max_battles_per_day=operator.max_battles_per_day,
        created_at=operator.created_at,
        last_login_at=operator.last_login_at,
    )


@router.post("/auth/logout")
async def auth_logout():
    """Clear session cookie."""
    response = JSONResponse({"ok": True, "message": "Logged out"})
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
    )
    return response


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_frontend_url(path: str) -> str:
    """Build an absolute URL on the frontend base for OAuth redirects."""
    base = settings.frontend_base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def _redirect_to_frontend(path: str) -> RedirectResponse:
    """Redirect to a frontend path with error query params."""
    return RedirectResponse(_build_frontend_url(path), status_code=302)

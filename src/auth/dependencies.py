"""FastAPI auth dependencies for Quality Oracle endpoints."""
from fastapi import Header, HTTPException

from src.auth.api_keys import validate_api_key


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

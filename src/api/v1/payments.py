"""Payment and pricing endpoints.

GET  /v1/pricing — Get evaluation pricing table
GET  /v1/pricing/{level} — Get price quote for specific level
"""
import logging

from fastapi import APIRouter, Depends

from src.auth.dependencies import get_api_key
from src.payments.pricing import get_price_quote, get_pricing_table

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/pricing")
async def pricing_table(
    api_key_doc: dict = Depends(get_api_key),
):
    """Get the full pricing table for all evaluation levels.

    Prices are adjusted based on the caller's API key tier.
    """
    tier = api_key_doc.get("tier", "free")
    return {
        "tier": tier,
        "pricing": get_pricing_table(tier),
        "note": "Level 1 is always free. Prices in USD, payable via x402 protocol.",
    }


@router.get("/pricing/{level}")
async def price_quote(
    level: int,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get a price quote for a specific evaluation level.

    Returns x402-compatible payment details if level requires payment.
    """
    if level not in (1, 2, 3):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Level must be 1, 2, or 3")

    tier = api_key_doc.get("tier", "free")
    quote = get_price_quote(level, tier)

    if quote.is_free:
        return {
            "level": level,
            "price_usd": 0,
            "is_free": True,
            "message": f"Level {level} evaluation is free.",
        }

    from src.payments.x402 import build_402_response
    return {
        "level": level,
        "price_usd": quote.final_price_usd,
        "is_free": False,
        "x402": build_402_response(quote),
    }

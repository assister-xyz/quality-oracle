"""
x402 Protocol implementation for Quality Oracle.

The x402 protocol uses HTTP 402 Payment Required to gate API access.
Flow:
1. Client sends request without payment
2. Server returns 402 with payment details (price, receiver, tokens)
3. Client makes payment (e.g., Solana SPL transfer)
4. Client retries request with X-Payment header containing tx signature
5. Server verifies payment and processes request

This module provides:
- x402 response builder (402 Payment Required with payment instructions)
- Payment header parser (X-Payment)
- Payment verification dependency for FastAPI
- Payment receipt storage
- Pricing endpoint

x402 spec reference: https://x402.org
Solana has 77% of x402 payment volume.
"""
import logging
import time as _time
from typing import Optional

from fastapi import HTTPException

from src.payments.pricing import (
    PriceQuote,
    PaymentReceipt,
    get_price_quote,
    ACCEPTED_TOKENS,
)

logger = logging.getLogger(__name__)


# ── x402 Response Builder ────────────────────────────────────────────────────

def build_402_response(
    quote: PriceQuote,
    description: str = "Payment required for evaluation",
) -> dict:
    """Build x402-compliant 402 Payment Required response body.

    Per x402 spec, the response includes:
    - paymentRequirements: array of accepted payment methods
    - description: human-readable explanation
    - resource: what the payment grants access to
    """
    payment_requirements = []
    for token_name in quote.accepted_tokens:
        token_info = ACCEPTED_TOKENS.get(token_name, {})
        payment_requirements.append({
            "type": "exact",
            "network": token_info.get("network", "solana"),
            "token": token_name,
            "mint": token_info.get("mint", ""),
            "decimals": token_info.get("decimals", 6),
            "amount": _usd_to_token_amount(quote.final_price_usd, token_name),
            "amount_usd": quote.final_price_usd,
            "receiver": quote.receiver_address,
        })

    return {
        "error": "payment_required",
        "status": 402,
        "description": description,
        "x402_version": "1",
        "payment_requirements": payment_requirements,
        "resource": f"evaluation/level-{quote.level}",
        "pricing": quote.to_dict(),
    }


# SOL price cache (5 min TTL)
_sol_price_cache: dict = {"price": None, "fetched_at": 0.0}
_SOL_PRICE_CACHE_TTL = 300  # 5 minutes


async def _fetch_sol_price() -> float:
    """Fetch SOL/USD price from Jupiter Price API v2 with 5-min cache.

    Falls back to CoinGecko simple price API, then to a conservative default.
    """
    now = _time.time()
    if _sol_price_cache["price"] and (now - _sol_price_cache["fetched_at"]) < _SOL_PRICE_CACHE_TTL:
        return _sol_price_cache["price"]

    import httpx

    # Try Jupiter Price API v2 (free, no auth)
    try:
        sol_mint = "So11111111111111111111111111111111111111112"
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"https://api.jup.ag/price/v2?ids={sol_mint}"
            )
            if resp.status_code == 200:
                data = resp.json()
                price = float(data["data"][sol_mint]["price"])
                _sol_price_cache["price"] = price
                _sol_price_cache["fetched_at"] = now
                logger.info(f"SOL price from Jupiter: ${price:.2f}")
                return price
    except Exception as e:
        logger.warning(f"Jupiter price API failed: {e}")

    # Fallback: CoinGecko simple price
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
            )
            if resp.status_code == 200:
                price = float(resp.json()["solana"]["usd"])
                _sol_price_cache["price"] = price
                _sol_price_cache["fetched_at"] = now
                logger.info(f"SOL price from CoinGecko: ${price:.2f}")
                return price
    except Exception as e:
        logger.warning(f"CoinGecko price API failed: {e}")

    # Return cached price if available, otherwise conservative default
    if _sol_price_cache["price"]:
        return _sol_price_cache["price"]
    return 150.0  # Conservative fallback


def _usd_to_token_amount(usd: float, token: str) -> str:
    """Convert USD amount to token base units (string for precision).

    For USDC: 1 USD = 1_000_000 base units (6 decimals)
    For SOL: uses cached Jupiter/CoinGecko price (sync version with cached value)
    """
    token_info = ACCEPTED_TOKENS.get(token, {})
    decimals = token_info.get("decimals", 6)

    if token == "USDC":
        return str(int(usd * (10 ** decimals)))
    elif token == "SOL":
        sol_price = _sol_price_cache.get("price") or 150.0
        sol_amount = usd / sol_price
        return str(int(sol_amount * (10 ** decimals)))
    else:
        return str(int(usd * (10 ** decimals)))


# ── Payment Header Parser ────────────────────────────────────────────────────

def parse_payment_header(header_value: str) -> dict:
    """Parse X-Payment header from client.

    Expected format (x402 spec):
        X-Payment: <tx_signature>:<token>:<network>

    Or simplified:
        X-Payment: <tx_signature>

    Returns dict with tx_signature, token, network.
    """
    parts = header_value.strip().split(":")
    result = {
        "tx_signature": parts[0],
        "token": parts[1] if len(parts) > 1 else "USDC",
        "network": parts[2] if len(parts) > 2 else "solana",
    }
    return result


# ── Payment Verification ─────────────────────────────────────────────────────

_BASE58_CHARS = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def _is_valid_base58(s: str) -> bool:
    """Check if a string contains only valid base58 characters."""
    return all(c in _BASE58_CHARS for c in s)


def _is_valid_solana_signature(tx_signature: str) -> bool:
    """Validate a Solana transaction signature format.

    A real Solana tx signature is 86-88 base58 characters.
    """
    if not tx_signature:
        return False
    # Solana signatures are typically 87-88 base58 chars
    if not (64 <= len(tx_signature) <= 90):
        return False
    return _is_valid_base58(tx_signature)


async def verify_payment(
    tx_signature: str,
    expected_amount_usd: float,
    token: str = "USDC",
    network: str = "solana",
) -> PaymentReceipt:
    """Verify a payment transaction.

    Phase C: Format validation (base58, correct length).
    Phase D (future): Full Solana RPC verification via solders.

    Validates:
    1. Non-empty signature
    2. Base58 character set
    3. Correct length for Solana tx signatures (86-88 chars)
    """
    # Basic validation
    if not tx_signature or len(tx_signature) < 10:
        return PaymentReceipt(
            evaluation_id="",
            payer="unknown",
            amount_usd=expected_amount_usd,
            token=token,
            tx_signature=tx_signature,
            network=network,
            verified=False,
        )

    # Validate base58 format and length
    is_valid_format = _is_valid_solana_signature(tx_signature)

    logger.info(
        f"Payment verification: tx={tx_signature[:16]}... "
        f"len={len(tx_signature)} base58={is_valid_format} "
        f"amount=${expected_amount_usd} token={token}"
    )

    return PaymentReceipt(
        evaluation_id="",
        payer="unknown",  # Phase D: extract from tx via RPC
        amount_usd=expected_amount_usd,
        token=token,
        tx_signature=tx_signature,
        network=network,
        verified=is_valid_format,
    )


# ── FastAPI Dependency ────────────────────────────────────────────────────────

async def require_payment(
    level: int,
    tier: str,
    x_payment: Optional[str] = None,
) -> Optional[PaymentReceipt]:
    """FastAPI dependency that enforces x402 payment for paid evaluations.

    Returns:
        None if evaluation is free
        PaymentReceipt if payment was provided and verified

    Raises:
        HTTPException(402) if payment is required but not provided
        HTTPException(402) if payment verification fails
    """
    quote = get_price_quote(level, tier)

    if quote.is_free:
        return None

    # Pre-fetch SOL price for accurate 402 response
    try:
        await _fetch_sol_price()
    except Exception:
        pass

    if not x_payment:
        raise HTTPException(
            status_code=402,
            detail=build_402_response(
                quote,
                description=f"Payment required for Level {level} evaluation. "
                           f"Price: ${quote.final_price_usd} USD",
            ),
        )

    # Parse and verify payment
    payment_info = parse_payment_header(x_payment)
    receipt = await verify_payment(
        tx_signature=payment_info["tx_signature"],
        expected_amount_usd=quote.final_price_usd,
        token=payment_info["token"],
        network=payment_info["network"],
    )

    if not receipt.verified:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "payment_verification_failed",
                "status": 402,
                "description": "Payment could not be verified. "
                              "Ensure the transaction is finalized.",
                "tx_signature": payment_info["tx_signature"],
                "x402_version": "1",
                "payment_requirements": build_402_response(quote)["payment_requirements"],
            },
        )

    return receipt

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
- Payment verification (format + optional Solana RPC)
- Replay prevention via MongoDB receipt storage
- Pricing endpoint

x402 spec reference: https://x402.org
Solana has 77% of x402 payment volume.
"""
import logging
import time as _time
from datetime import datetime, timezone
from typing import Optional, Tuple

import httpx
from fastapi import HTTPException

from src.config import settings
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


# ── USDC Mint Addresses ─────────────────────────────────────────────────────

_USDC_MINTS = {
    "mainnet-beta": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "devnet": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU",
}


# ── Solana RPC Transaction Verification ──────────────────────────────────────

def _check_sol_transfer(
    instructions: list,
    receiver: str,
    expected_usd: float,
) -> Tuple[bool, str, str]:
    """Parse system program SOL transfer instructions.

    Returns (verified, error, payer_address).
    """
    sol_price = _sol_price_cache.get("price") or 150.0
    expected_lamports = int((expected_usd / sol_price) * 1_000_000_000)
    # Allow 5% slippage for SOL price fluctuation
    min_lamports = int(expected_lamports * 0.95)

    for ix in instructions:
        program = ix.get("program", "")
        if program != "system":
            continue
        parsed = ix.get("parsed", {})
        if parsed.get("type") != "transfer":
            continue
        info = parsed.get("info", {})
        dest = info.get("destination", "")
        lamports = info.get("lamports", 0)
        source = info.get("source", "")

        if dest == receiver and lamports >= min_lamports:
            return True, "", source

    return False, "No matching SOL transfer found", ""


def _check_spl_transfer(
    instructions: list,
    receiver: str,
    expected_usd: float,
    mint: str,
    token: str,
) -> Tuple[bool, str, str]:
    """Parse SPL token transferChecked instructions.

    Returns (verified, error, payer_address).
    """
    token_info = ACCEPTED_TOKENS.get(token, {})
    decimals = token_info.get("decimals", 6)
    expected_amount = int(expected_usd * (10 ** decimals))
    # Allow 1% slippage for USDC (price stable)
    min_amount = int(expected_amount * 0.99)

    for ix in instructions:
        program = ix.get("program", "")
        if program != "spl-token":
            continue
        parsed = ix.get("parsed", {})
        ix_type = parsed.get("type", "")
        if ix_type not in ("transferChecked", "transfer"):
            continue
        info = parsed.get("info", {})

        # Check mint matches
        ix_mint = info.get("mint", "")
        if ix_type == "transferChecked" and ix_mint != mint:
            continue

        dest = info.get("destination", "")
        authority = info.get("authority", "")
        amount_str = info.get("tokenAmount", {}).get("amount", "0") if ix_type == "transferChecked" else info.get("amount", "0")
        amount = int(amount_str)

        if dest == receiver and amount >= min_amount:
            return True, "", authority

    return False, "No matching SPL token transfer found", ""


async def _verify_solana_transaction(
    tx_signature: str,
    expected_amount_usd: float,
    token: str = "USDC",
) -> Tuple[bool, str, str]:
    """Verify a Solana transaction via JSON-RPC getTransaction.

    Uses httpx POST to Solana RPC — zero new deps.

    Returns (verified, error_message, payer_address).
    """
    receiver = settings.receiver_wallet_address
    cluster = settings.solana_cluster

    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            tx_signature,
            {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(settings.solana_rpc_url, json=body)

        if resp.status_code != 200:
            return False, f"RPC returned HTTP {resp.status_code}", ""

        data = resp.json()
        error = data.get("error")
        if error:
            return False, f"RPC error: {error.get('message', str(error))}", ""

        result = data.get("result")
        if not result:
            return False, "Transaction not found (not finalized yet?)", ""

        # Check transaction succeeded (no error)
        meta = result.get("meta", {})
        if meta.get("err") is not None:
            return False, f"Transaction failed: {meta['err']}", ""

        # Extract all instructions (including inner instructions)
        tx = result.get("transaction", {})
        message = tx.get("message", {})
        instructions = message.get("instructions", [])

        # Also check inner instructions
        inner_instructions = meta.get("innerInstructions", [])
        for inner in inner_instructions:
            instructions.extend(inner.get("instructions", []))

        # Determine verification based on token type
        if token == "SOL":
            return _check_sol_transfer(instructions, receiver, expected_amount_usd)
        else:
            mint = _USDC_MINTS.get(cluster, _USDC_MINTS["devnet"])
            return _check_spl_transfer(instructions, receiver, expected_amount_usd, mint, token)

    except httpx.TimeoutException:
        return False, "Solana RPC request timed out", ""
    except Exception as e:
        return False, f"RPC verification error: {e}", ""


# ── Main Verification Flow ───────────────────────────────────────────────────

async def verify_payment(
    tx_signature: str,
    expected_amount_usd: float,
    token: str = "USDC",
    network: str = "solana",
) -> PaymentReceipt:
    """Verify a payment transaction.

    Flow:
    1. Format validation (base58, correct length)
    2. Replay prevention (check payment_receipts collection)
    3. RPC verification (only if receiver_wallet_address is configured)
    4. Store receipt for replay prevention
    5. Return PaymentReceipt

    When receiver_wallet_address is empty (default), falls back to
    format-only validation — same as Phase C behavior.
    """
    # Step 1: Basic format validation
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

    is_valid_format = _is_valid_solana_signature(tx_signature)
    if not is_valid_format:
        logger.info(f"Payment format invalid: tx={tx_signature[:16]}... len={len(tx_signature)}")
        return PaymentReceipt(
            evaluation_id="",
            payer="unknown",
            amount_usd=expected_amount_usd,
            token=token,
            tx_signature=tx_signature,
            network=network,
            verified=False,
        )

    # Step 2: Replay prevention
    payer = "unknown"
    try:
        from src.storage.mongodb import payment_receipts_col
        existing = await payment_receipts_col().find_one({"tx_signature": tx_signature})
        if existing:
            logger.warning(f"Replay attempt: tx={tx_signature[:16]}... already used")
            return PaymentReceipt(
                evaluation_id="",
                payer=existing.get("payer", "unknown"),
                amount_usd=expected_amount_usd,
                token=token,
                tx_signature=tx_signature,
                network=network,
                verified=False,
            )
    except Exception as e:
        logger.warning(f"Replay check failed (proceeding): {e}")

    # Step 3: RPC verification (opt-in via receiver_wallet_address)
    rpc_verified = True  # Default: pass if no RPC configured
    if settings.receiver_wallet_address:
        rpc_verified, error, rpc_payer = await _verify_solana_transaction(
            tx_signature, expected_amount_usd, token,
        )
        if rpc_payer:
            payer = rpc_payer
        if not rpc_verified:
            logger.warning(f"RPC verification failed: {error}")
            return PaymentReceipt(
                evaluation_id="",
                payer=payer,
                amount_usd=expected_amount_usd,
                token=token,
                tx_signature=tx_signature,
                network=network,
                verified=False,
            )

    logger.info(
        f"Payment verified: tx={tx_signature[:16]}... "
        f"payer={payer} amount=${expected_amount_usd} token={token} "
        f"rpc={'on' if settings.receiver_wallet_address else 'off'}"
    )

    # Step 4: Store receipt for replay prevention
    try:
        from src.storage.mongodb import payment_receipts_col
        await payment_receipts_col().insert_one({
            "tx_signature": tx_signature,
            "payer": payer,
            "amount_usd": expected_amount_usd,
            "token": token,
            "network": network,
            "verified": True,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        # Don't fail payment verification if receipt storage fails
        logger.warning(f"Failed to store payment receipt: {e}")

    return PaymentReceipt(
        evaluation_id="",
        payer=payer,
        amount_usd=expected_amount_usd,
        token=token,
        tx_signature=tx_signature,
        network=network,
        verified=True,
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

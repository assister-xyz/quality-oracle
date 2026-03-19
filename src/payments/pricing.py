"""
Evaluation pricing model.

Defines per-level evaluation costs, tier-based discounts, and
payment receipt structures. Designed to integrate with x402 protocol
for agent-to-agent payments on Solana (77% of x402 volume).

Pricing tiers:
- Level 1 (Manifest): Free — schema validation only, no LLM calls
- Level 2 (Functional): $0.01 — MCP tool calls + LLM judging
- Level 3 (Domain Expert): $0.05 — calibrated question bank + deep eval

Tier discounts:
- free: no discount (pay-per-eval)
- developer: 20% off
- team: 40% off
- marketplace: 60% off (volume pricing)
"""
from dataclasses import dataclass
from typing import Optional


# ── Pricing constants ────────────────────────────────────────────────────────

# Base prices in USD (micro-payments via x402)
LEVEL_PRICES_USD = {
    1: 0.00,    # Level 1: Free
    2: 0.01,    # Level 2: Functional
    3: 0.05,    # Level 3: Domain Expert
}

# Tier discount rates
TIER_DISCOUNTS = {
    "free": 0.00,
    "developer": 1.00,  # 100% off during development (was 0.20)
    "team": 0.40,
    "marketplace": 0.60,
}

# Supported payment tokens (Solana SPL tokens)
ACCEPTED_TOKENS = {
    "USDC": {
        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "decimals": 6,
        "network": "solana",
    },
    "SOL": {
        "mint": "So11111111111111111111111111111111111111112",
        "decimals": 9,
        "network": "solana",
    },
}

def _get_default_receiver() -> str:
    """Get receiver wallet from config, with placeholder fallback for dev."""
    try:
        from src.config import settings
        if settings.receiver_wallet_address:
            return settings.receiver_wallet_address
    except Exception:
        pass
    return "NOT_CONFIGURED"


DEFAULT_RECEIVER = _get_default_receiver()


@dataclass
class PriceQuote:
    """Price quote for an evaluation."""
    level: int
    base_price_usd: float
    discount_rate: float
    final_price_usd: float
    is_free: bool
    accepted_tokens: list
    receiver_address: str

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "base_price_usd": self.base_price_usd,
            "discount_rate": self.discount_rate,
            "final_price_usd": self.final_price_usd,
            "is_free": self.is_free,
            "accepted_tokens": self.accepted_tokens,
            "receiver_address": self.receiver_address,
        }


@dataclass
class PaymentReceipt:
    """Record of a payment for an evaluation."""
    evaluation_id: str
    payer: str              # Payer wallet address or API key
    amount_usd: float
    token: str              # USDC, SOL, etc.
    tx_signature: Optional[str] = None  # Solana transaction signature
    network: str = "solana"
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "evaluation_id": self.evaluation_id,
            "payer": self.payer,
            "amount_usd": self.amount_usd,
            "token": self.token,
            "tx_signature": self.tx_signature,
            "network": self.network,
            "verified": self.verified,
        }


def get_price_quote(
    level: int,
    tier: str = "free",
    receiver: Optional[str] = None,
) -> PriceQuote:
    """Get a price quote for an evaluation at the given level and tier.

    Args:
        level: Evaluation level (1, 2, or 3)
        tier: API key tier for discount
        receiver: Override receiver address (default from config)
    """
    base_price = LEVEL_PRICES_USD.get(level, 0.01)
    discount = TIER_DISCOUNTS.get(tier, 0.0)
    final_price = round(base_price * (1 - discount), 4)

    # If no receiver wallet is configured, all evaluations are free
    # (development/demo mode — no payment infrastructure yet)
    receiver_addr = receiver or DEFAULT_RECEIVER
    if receiver_addr == "NOT_CONFIGURED" or not receiver_addr:
        final_price = 0

    return PriceQuote(
        level=level,
        base_price_usd=base_price,
        discount_rate=discount,
        final_price_usd=final_price,
        is_free=final_price == 0,
        accepted_tokens=list(ACCEPTED_TOKENS.keys()),
        receiver_address=receiver_addr,
    )


def get_pricing_table(tier: str = "free") -> list:
    """Get full pricing table for all levels at the given tier."""
    return [get_price_quote(level, tier).to_dict() for level in [1, 2, 3]]

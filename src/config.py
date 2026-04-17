"""AgentTrust configuration via environment variables."""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8002
    debug: bool = False
    log_level: str = "info"

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "assisterr"

    # Redis
    redis_url: str = "redis://localhost:6379/1"

    # LLM Judge - Primary (DeepSeek V3.2)
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # LLM Judge - Fallback (Groq)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # LLM Judge - Primary (OpenAI)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # LLM Judge - Cerebras (free: 1M TPD, Qwen3 235B)
    cerebras_api_key: str = ""
    cerebras_model: str = "gpt-oss-120b"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"

    # LLM Judge - Gemini (free: 250 RPD via OpenAI-compat)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # LLM Judge - OpenRouter (free: DeepSeek R1, 200 RPD)
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3-next-80b-a3b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # LLM Judge - Mistral (free: 2 RPM, all models)
    mistral_api_key: str = ""
    mistral_model: str = "mistral-large-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"

    # LLM Judge - Question Generation
    anthropic_api_key: str = ""

    # Consensus Judge
    consensus_enabled: bool = True
    consensus_min_judges: int = 2
    consensus_agreement_threshold: int = 15  # points within which judges "agree"

    # API Keys
    api_key_salt: str = "change-this-to-random-string"

    # JWT Attestation Signing (Ed25519)
    jwt_private_key_path: str = ""
    jwt_private_key: str = ""  # Base64-encoded key content (alternative to file path)
    jwt_issuer: str = "did:web:agenttrust.assisterr.ai"
    attestation_validity_days: int = 30

    # Payment — Receiver Wallet
    receiver_wallet_address: str = ""

    # Solana RPC (for payment verification)
    solana_rpc_url: str = "https://api.devnet.solana.com"
    solana_cluster: str = "devnet"

    # ── On-Chain: ERC-8004 (Base L2) ──────────────────────────────────────────
    erc8004_enabled: bool = False
    erc8004_evaluator_private_key: str = ""  # Hex-encoded private key for posting feedback
    base_rpc_url: str = "https://mainnet.base.org"
    base_chain_id: int = 8453
    # Deterministic CREATE2 addresses (same on all chains)
    erc8004_identity_registry: str = "0x1f55Fb5898c42dA6327172F197B03048ef5f0057"
    erc8004_reputation_registry: str = "0x82e28A65CE6079e3aDc3D4722Af1F917b3A26E34"

    # ── On-Chain: EAS (Ethereum Attestation Service) ────────────────────────
    eas_enabled: bool = False
    eas_private_key: str = ""  # Hex-encoded; shares wallet with ERC-8004 if empty
    eas_contract_address: str = "0x4200000000000000000000000000000000000021"  # Base
    eas_schema_registry: str = "0x4200000000000000000000000000000000000020"  # Base
    eas_schema_uid: str = ""  # Set after registering AQVC schema
    eas_onchain_min_score: int = 90  # Only on-chain for Audited tier (score >= 90)

    # ── On-Chain: IPFS ──────────────────────────────────────────────────────
    ipfs_gateway: str = "https://w3s.link/ipfs/"
    ipfs_upload_url: str = ""  # web3.storage or Pinata upload endpoint
    ipfs_api_token: str = ""

    # ── On-Chain: Gas Tracking ──────────────────────────────────────────────
    gas_tracking_enabled: bool = True
    onchain_async: bool = True  # Non-blocking on-chain posting

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001", "http://localhost:3003", "http://localhost:3004"]

    # Evaluation defaults
    default_eval_level: int = 2
    max_concurrent_evals: int = 5
    eval_timeout_seconds: int = 120
    evaluation_version: str = "v1.0"

    # QO-050: per-server cap for batch evaluation — prevents one slow MCP
    # server from hanging the whole batch job.
    batch_per_server_timeout_seconds: int = 180

    # Rate limiting (evaluations per month)
    rate_limit_free: int = 10
    rate_limit_developer: int = 100
    rate_limit_team: int = 500

    # Webhook
    webhook_timeout_seconds: int = 10
    webhook_max_retries: int = 3

    # Base URL for constructing links in responses (fallback only)
    base_url: str = "http://localhost:8002"

    # ── Audit logging (QO-047) ───────────────────────────────────────────────
    audit_logging_enabled: bool = True
    audit_max_response_kb: int = 5
    audit_max_prompt_kb: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()


def get_base_url(request=None) -> str:
    """Derive base URL from the incoming request, or fall back to settings.

    Handles AWS ALB x-forwarded-proto header for correct HTTPS detection.
    """
    if request is not None:
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("host", request.url.netloc)
        return f"{scheme}://{host}"
    return settings.base_url.rstrip("/")


# ── Provider Pricing (USD per 1M tokens) ────────────────────────────────────
# Updated 2026-03-19. Free tiers tracked as $0 for quota monitoring.

PROVIDER_PRICING = {
    # actual = what we pay (free tier); market = paid API rate for same model
    "cerebras": {"input_per_m": 0.0, "output_per_m": 0.0, "tier": "free",
                 "market_input_per_m": 0.10, "market_output_per_m": 0.10},   # llama3.1-8b
    "groq": {"input_per_m": 0.0, "output_per_m": 0.0, "tier": "free",
             "market_input_per_m": 0.05, "market_output_per_m": 0.08},       # llama-3.1-8b-instant
    "openrouter": {"input_per_m": 0.0, "output_per_m": 0.0, "tier": "free",
                   "market_input_per_m": 0.09, "market_output_per_m": 1.10}, # qwen3-80b-a3b-instruct
    "gemini": {"input_per_m": 0.0, "output_per_m": 0.0, "tier": "free",
               "market_input_per_m": 0.075, "market_output_per_m": 0.30},    # gemini-2.0-flash
    "mistral": {"input_per_m": 0.1, "output_per_m": 0.3, "tier": "free",
                "market_input_per_m": 0.1, "market_output_per_m": 0.3},
    "deepseek": {"input_per_m": 0.14, "output_per_m": 0.28, "tier": "paid",
                 "market_input_per_m": 0.14, "market_output_per_m": 0.28},
    "openai": {"input_per_m": 0.15, "output_per_m": 0.60, "tier": "paid",
               "market_input_per_m": 0.15, "market_output_per_m": 0.60},
}


def calculate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a given provider and token counts."""
    pricing = PROVIDER_PRICING.get(provider, {})
    input_cost = input_tokens * pricing.get("input_per_m", 0) / 1_000_000
    output_cost = output_tokens * pricing.get("output_per_m", 0) / 1_000_000
    return round(input_cost + output_cost, 8)


def calculate_market_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate what this would cost at paid API market rates."""
    pricing = PROVIDER_PRICING.get(provider, {})
    input_cost = input_tokens * pricing.get("market_input_per_m", pricing.get("input_per_m", 0)) / 1_000_000
    output_cost = output_tokens * pricing.get("market_output_per_m", pricing.get("output_per_m", 0)) / 1_000_000
    return round(input_cost + output_cost, 8)


def calculate_total_cost(by_provider: dict) -> dict:
    """Calculate cost breakdown from per-provider token usage.

    Args:
        by_provider: Dict of provider -> {input_tokens, output_tokens, calls}

    Returns:
        {total_cost_usd, shadow_cost_usd, by_provider: {provider: {actual, market}}}
    """
    cost_by_provider = {}
    total = 0.0
    shadow_total = 0.0
    for provider, usage in by_provider.items():
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        actual = calculate_cost(provider, in_tok, out_tok)
        market = calculate_market_cost(provider, in_tok, out_tok)
        cost_by_provider[provider] = {"actual": actual, "market": market}
        total += actual
        shadow_total += market
    return {
        "total_cost_usd": round(total, 6),
        "shadow_cost_usd": round(shadow_total, 6),
        "by_provider": cost_by_provider,
    }

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

    # CORS
    cors_origins: List[str] = ["http://localhost:3000"]

    # Evaluation defaults
    default_eval_level: int = 2
    max_concurrent_evals: int = 5
    eval_timeout_seconds: int = 120
    evaluation_version: str = "v1.0"

    # Rate limiting (evaluations per month)
    rate_limit_free: int = 10
    rate_limit_developer: int = 100
    rate_limit_team: int = 500

    # Webhook
    webhook_timeout_seconds: int = 10
    webhook_max_retries: int = 3

    # Base URL for constructing links in responses
    base_url: str = "http://localhost:8002"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()


# ── Provider Pricing (USD per 1M tokens) ────────────────────────────────────
# Updated 2026-03-19. Free tiers tracked as $0 for quota monitoring.

PROVIDER_PRICING = {
    "cerebras": {"input_per_m": 0.0, "output_per_m": 0.0, "tier": "free"},
    "groq": {"input_per_m": 0.0, "output_per_m": 0.0, "tier": "free"},
    "openrouter": {"input_per_m": 0.0, "output_per_m": 0.0, "tier": "free"},
    "gemini": {"input_per_m": 0.0, "output_per_m": 0.0, "tier": "free"},
    "mistral": {"input_per_m": 0.1, "output_per_m": 0.3, "tier": "free"},
    "deepseek": {"input_per_m": 0.14, "output_per_m": 0.28, "tier": "paid"},
    "openai": {"input_per_m": 0.15, "output_per_m": 0.60, "tier": "paid"},
}


def calculate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a given provider and token counts."""
    pricing = PROVIDER_PRICING.get(provider, {})
    input_cost = input_tokens * pricing.get("input_per_m", 0) / 1_000_000
    output_cost = output_tokens * pricing.get("output_per_m", 0) / 1_000_000
    return round(input_cost + output_cost, 8)


def calculate_total_cost(by_provider: dict) -> dict:
    """Calculate cost breakdown from per-provider token usage.

    Args:
        by_provider: Dict of provider -> {input_tokens, output_tokens, calls}

    Returns:
        {total_cost_usd, by_provider: {provider: cost_usd}}
    """
    cost_by_provider = {}
    total = 0.0
    for provider, usage in by_provider.items():
        cost = calculate_cost(
            provider,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        cost_by_provider[provider] = cost
        total += cost
    return {
        "total_cost_usd": round(total, 6),
        "by_provider": cost_by_provider,
    }

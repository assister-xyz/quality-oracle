# Laureum.ai (Quality Oracle)

> Part of **assisterr-workflow**. See `../assisterr-workflow/CLAUDE.md` for full workflow, sizing, spec lifecycle, memory entities, and agent routing.

## Project Context

- **Port:** 8002
- **Stack:** FastAPI + Motor (MongoDB) + Redis
- **Brand:** Laureum.ai (standard: AQVC — Agent Quality Verifiable Credential)
- **MongoDB prefix:** `quality__` (collections: evaluations, scores, score_history, attestations, question_banks, api_keys)
- **Redis prefix:** `qo:` (score cache, badge cache, attestation verify cache, rate limits)
- **LLM Judge:** Multi-provider (Cerebras, Groq, Gemini, DeepSeek, OpenAI, OpenRouter, Mistral) with ConsensusJudge (2-judge parallel + tiebreaker)
- **Domain agent:** `32-implement-py` handles this repo

## Architecture

3-level evaluation pipeline:
- **Level 1 (Manifest):** Schema completeness, descriptions, input schemas
- **Level 2 (Functional):** MCP SSE connection → list tools → generate test cases → call tools → LLM judge responses
- **Level 3 (Domain Expert):** Calibrated question bank with weighted scoring

6-axis scoring: accuracy(35%), safety(20%), process_quality(10%), reliability(15%), latency(10%), schema_quality(10%).
Production correlation: POST /v1/feedback → GET /v1/correlation/{target_id} (anti-sandbagging, confidence adjustment).

JWT attestation via Ed25519 (AQVC format). MCP SDK SSE + Streamable HTTP dual transport. A2A v0.3 compliant Agent Card. Webhook-first async delivery for Level 2+.

x402 payment layer: Level 1 free, Level 2 $0.01, Level 3 $0.05 (base). Tier discounts: developer 20%, team 40%, marketplace 60%. Tokens: USDC + SOL on Solana. `X-Payment` header with `tx_sig:token:network` format. `GET /v1/pricing` for pricing table.

## Running Locally

```bash
source .venv/bin/activate && unset GROQ_API_KEY && python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload
```

**Important:** Always `unset GROQ_API_KEY` before starting — a shell env variable overrides the `.env` key rotation pool.

LLM API keys in `.env` are comma-separated for rotation (e.g. `GROQ_API_KEY=key1,key2,key3`). The judge auto-rotates on 429/quota errors.

## Quality Gates

```bash
python3 -m pytest tests/ -v          # All tests pass
ruff check src/                       # No lint errors
python3 -c "from src.main import app" # App imports clean
```

## Auth

- API keys with `qo_` prefix, SHA256 hashed with salt
- `X-API-Key` header for protected endpoints
- Rate limiting per tier: free/developer/team/marketplace
- Public endpoints: health, badge SVG, agent card (/.well-known/agent.json)

## Key Paths

```
src/api/v1/          # FastAPI endpoint routers
src/core/            # MCP client, evaluator, scoring, attestation
src/auth/            # API key management, rate limiting
src/storage/         # MongoDB, Redis, Pydantic models
src/payments/        # x402 protocol, pricing model
src/standards/       # W3C VC, AQVC format, A2A extension
dev/                 # Mock MCP server, seed questions
tests/               # Pytest tests
```

## Forbidden Zones (Quick Reference)

| Path | Risk |
|------|------|
| `src/core/attestation.py` | JWT signing, key management |
| `src/core/scoring.py` | Score aggregation weights |
| `src/auth/api_keys.py` | API key generation, hashing |
| `src/storage/mongodb.py` | Collection accessors, indexes |
| `src/config.py` | All settings |

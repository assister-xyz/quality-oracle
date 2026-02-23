# Quality Oracle

> Part of **assisterr-workflow**. See `../assisterr-workflow/CLAUDE.md` for full workflow, sizing, spec lifecycle, memory entities, and agent routing.

## Project Context

- **Port:** 8002
- **Stack:** FastAPI + Motor (MongoDB) + Redis
- **MongoDB prefix:** `quality__` (collections: evaluations, scores, score_history, attestations, question_banks, api_keys)
- **Redis prefix:** `qo:` (score cache, badge cache, attestation verify cache, rate limits)
- **LLM Judge:** DeepSeek V3.2 primary, Groq Llama 3.3-70B fallback, fuzzy string matching fallback
- **Domain agent:** `32-implement-py` handles this repo

## Architecture

3-level evaluation pipeline:
- **Level 1 (Manifest):** Schema completeness, descriptions, input schemas
- **Level 2 (Functional):** MCP SSE connection → list tools → generate test cases → call tools → LLM judge responses
- **Level 3 (Domain Expert):** Calibrated question bank with weighted scoring

JWT attestation via Ed25519 (UAQA format). MCP SDK SSE client for server connections. Webhook-first async delivery for Level 2+.

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
src/standards/       # W3C VC, UAQA format, A2A extension
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

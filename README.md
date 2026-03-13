# AgentTrust

**Challenge-response quality verification for AI agents and MCP servers.**

AgentTrust evaluates AI agent competency *before* you trust them with real tasks or payments. It connects to any MCP server, runs challenge-response tests across 6 quality dimensions, and issues W3C Verifiable Credentials as proof.

## Why

The AI agent ecosystem has identity (ERC-8004, SATI), post-hoc reputation (TARS, Amiko), and payments (x402) — but no **pre-payment quality gate**. AgentTrust fills this gap: verify competency first, then trust.

## Features

**Evaluation Engine**
- 3-level pipeline: Manifest (schema) → Functional (tool calls) → Domain Expert (calibrated questions)
- 6-axis scoring: accuracy (35%), safety (20%), reliability (15%), process quality (10%), latency (10%), schema quality (10%)
- Consensus judging: 2-3 LLM judges in parallel with agreement threshold (saves 50-66% LLM calls)
- 7 LLM provider fallback chain: Cerebras → Groq → OpenRouter → Gemini → Mistral → DeepSeek → OpenAI
- 5 adversarial probe types: prompt injection, PII leakage, hallucination, overflow, system prompt extraction

**Battle Arena**
- Head-to-head blind evaluation with position-swap consistency
- OpenSkill (Bayesian ELO) rating system with divisions (Bronze → Grandmaster)
- Fair matchmaking: rating proximity + uncertainty bonus + cross-division challenges
- Style control penalties to prevent gaming via verbose/formatted responses

**IRT Adaptive Testing**
- Rasch 1PL calibration from battle data (pure Python, no numpy)
- Fisher information maximization for adaptive question selection
- EAP ability estimation with standard normal prior
- Reduces evaluation cost by 50-90% while maintaining accuracy

**Standards**
- W3C Verifiable Credentials (AQVC format) with Ed25519 DataIntegrityProof
- Google A2A v0.3 native support (AgentTrust IS an A2A agent)
- x402 Solana payment verification (USDC + SOL)
- AIUC-1 protocol mapping

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
# Add at least one LLM key (GROQ_API_KEY, CEREBRAS_API_KEY, etc.)
docker compose up -d
```

Services:
- API: http://localhost:8002
- MCP Server: http://localhost:8003
- Health: http://localhost:8002/health

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add LLM keys to .env

unset GROQ_API_KEY  # Shell env overrides .env rotation pool
python -m uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload
```

### MCP Server (for Claude, Cursor, Windsurf)

Add to your MCP client config:

```json
{
  "mcpServers": {
    "agenttrust": {
      "command": "python",
      "args": ["-m", "src.standards.mcp_server"],
      "env": {
        "GROQ_API_KEY": "your-key"
      }
    }
  }
}
```

Or connect to a running instance via SSE:
```
http://localhost:8003/sse
```

**Available MCP tools:**
| Tool | Description |
|------|-------------|
| `check_quality(server_url)` | Full evaluation: manifest + functional + judge scoring |
| `check_quality_fast(server_url)` | Cached score (<10ms) or manifest-only (<100ms) |
| `get_score(server_url)` | Lookup cached score with freshness decay |
| `verify_attestation(attestation_jwt)` | Verify AQVC JWT and decode payload |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/evaluate` | Submit target for evaluation |
| GET | `/v1/evaluate/{id}` | Poll evaluation status |
| GET | `/v1/score/{target_id}` | Get quality score |
| GET | `/v1/scores` | Search/list scores |
| GET | `/v1/badge/{target_id}.svg` | SVG quality badge |
| GET | `/v1/attestation/{id}` | Get signed attestation (JWT or W3C VC) |
| POST | `/v1/attestation/{id}/verify` | Verify attestation |
| POST | `/v1/feedback` | Submit production feedback (anti-sandbagging) |
| POST | `/v1/battles` | Create evaluation battle |
| GET | `/v1/arena/leaderboard` | Battle arena leaderboard |
| GET | `/v1/rankings` | Global rankings by domain/tier |
| POST | `/v1/irt/calibrate` | Trigger IRT batch calibration |
| GET | `/v1/irt/recommend` | Adaptive question selection |
| GET | `/v1/pricing` | x402 pricing table |
| GET | `/.well-known/agent.json` | A2A Agent Card |

## Architecture

```
src/
  api/v1/          # 14 FastAPI routers
  core/            # Evaluator, MCP client, scoring, IRT, battle arena
  auth/            # API keys (SHA256 + salt), rate limiting by tier
  storage/         # MongoDB (Motor) + Redis
  payments/        # x402 protocol, Solana verification
  standards/       # W3C VC issuer, A2A extension, MCP server, AIUC-1
```

**Stack:** FastAPI + MongoDB + Redis | 533 tests | 60 source files | 15 lean dependencies

## Tests

```bash
python -m pytest tests/ -q
# 533 passed in ~2s
```

## Configuration

See `.env.example` for all 60+ configuration options including:
- LLM API keys (7 providers, comma-separated for rotation)
- MongoDB/Redis connection
- JWT attestation (Ed25519 key, issuer DID, validity)
- Solana wallet for x402 payments
- Rate limit tiers and consensus judge settings

## License

MIT

## Links

- [Architecture](docs/ARCHITECTURE.md) — Full system design (845 lines)
- [Distribution Roadmap](docs/DISTRIBUTION_ROADMAP.md) — Partner and integration plan
- [A2A Agent Card](http://localhost:8002/.well-known/agent.json) — Machine-readable capabilities

Built by [Assisterr](https://assisterr.ai)

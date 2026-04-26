# AgentTrust — Final Architecture & Implementation Plan

## 1. IDEA (Short)

**AgentTrust = active competency verification for AI agents, skills, and MCP servers.**

Every agent economy needs three layers: Identity (who is this agent?), Quality (is this agent competent?), Payment (how to pay?). Identity exists (SATI, ERC-8004). Payments exist (x402, AP2). Quality is missing. We fill that gap.

How: challenge-response benchmarking where AgentTrust calls the agent/skill/server with calibrated test inputs, LLM-as-Judge evaluates responses, and results are published as signed AQVC attestations (JWT → W3C VC) portable across A2A, MCP, on-chain, and framework ecosystems.

First domain: MCP Servers (biggest pain point post-ClawHavoc, clearest test surface via JSON schemas, natural distribution as MCP Server itself).

---

## 2. ARCHITECTURE

### 2.1 System Overview

```
                    ┌──────────────────────────────────┐
                    │         CONSUMERS                │
                    │                                  │
                    │  IDE Users ──── MCP Server        │
                    │  Developers ─── REST API          │
                    │  Agents ─────── A2A Protocol      │
                    │  CI/CD ──────── GitHub Action      │
                    │  Marketplaces ─ Webhook/Badge API  │
                    └──────────┬───────────────────────┘
                               │
                    ┌──────────▼───────────────────────┐
                    │      AGENTTRUST SERVICE            │
                    │         (FastAPI, port 8002)      │
                    │                                  │
                    │  ┌─────────┐  ┌──────────────┐   │
                    │  │ REST API│  │ A2A Handler   │   │
                    │  │ /v1/*   │  │ JSON-RPC      │   │
                    │  └────┬────┘  └──────┬───────┘   │
                    │       │              │           │
                    │  ┌────▼──────────────▼───────┐   │
                    │  │    EVALUATION ENGINE       │   │
                    │  │                           │   │
                    │  │  ┌─────────────────────┐  │   │
                    │  │  │ Test Generator      │  │   │
                    │  │  │ (manifest → cases)  │  │   │
                    │  │  └─────────┬───────────┘  │   │
                    │  │            │              │   │
                    │  │  ┌─────────▼───────────┐  │   │
                    │  │  │ Challenge Runner    │  │   │
                    │  │  │ (MCP Client calls)  │  │   │
                    │  │  └─────────┬───────────┘  │   │
                    │  │            │              │   │
                    │  │  ┌─────────▼───────────┐  │   │
                    │  │  │ LLM Judge           │  │   │
                    │  │  │ (DeepSeek/Groq)     │  │   │
                    │  │  └─────────┬───────────┘  │   │
                    │  │            │              │   │
                    │  │  ┌─────────▼───────────┐  │   │
                    │  │  │ Scorer + Attestor   │  │   │
                    │  │  │ (AQVC / JWT → VC)   │  │   │
                    │  │  └─────────────────────┘  │   │
                    │  └───────────────────────────┘   │
                    │                                  │
                    │  ┌───────────┐  ┌─────────────┐  │
                    │  │ MongoDB   │  │ Redis       │  │
                    │  │ quality__ │  │ (cache,     │  │
                    │  │ prefix    │  │  rate limit) │  │
                    │  └───────────┘  └─────────────┘  │
                    └──────────────────────────────────┘
                               │
                    ┌──────────▼───────────────────────┐
                    │    TARGETS BEING EVALUATED        │
                    │                                  │
                    │  MCP Servers (via MCP Client)     │
                    │  ClawHub Skills (65% are MCP)     │
                    │  REST API Agents (via HTTP)       │
                    │  A2A Agents (via A2A protocol)    │
                    └──────────────────────────────────┘
```

### 2.2 Directory Structure

```
quality-oracle/
├── src/
│   ├── main.py                     # FastAPI app entry, port 8002
│   ├── config.py                   # Env config (pydantic-settings)
│   │
│   ├── api/
│   │   ├── v1/
│   │   │   ├── evaluate.py         # POST /v1/evaluate
│   │   │   ├── scores.py           # GET /v1/score/{id}
│   │   │   ├── attestations.py     # GET /v1/attestation/{id}
│   │   │   ├── badges.py           # GET /v1/badge/{id}.svg
│   │   │   ├── enrichment.py       # POST /v1/enrich-agent-card
│   │   │   └── health.py          # GET /health
│   │   ├── a2a.py                  # POST /v1/a2a (A2A JSON-RPC)
│   │   ├── agent_card.py           # GET /.well-known/agent.json
│   │   └── webhooks.py             # Webhook delivery for async results
│   │
│   ├── core/
│   │   ├── evaluator.py            # Orchestrates full evaluation flow
│   │   ├── llm_judge.py            # Multi-provider LLM judging (from agent-poi)
│   │   ├── challenge_handler.py    # Challenge-response logic (from agent-poi)
│   │   ├── question_pools.py       # Domain question bank (from agent-poi)
│   │   ├── test_generator.py       # NEW: manifest → test cases auto-generation
│   │   ├── mcp_client.py           # NEW: MCP client to call target servers (Strategy A: SSE/HTTP)
│   │   ├── scoring.py              # Score aggregation, tiers, confidence
│   │   └── attestation.py          # AQVC format, JWT signing (Phase 1), VC signing (Phase 2)
│   │
│   ├── standards/
│   │   ├── a2a_extension.py        # A2A Agent Card quality extension
│   │   ├── mcp_server.py           # AgentTrust as MCP Server (FastMCP)
│   │   ├── vc_issuer.py            # W3C Verifiable Credential issuance (Phase 2, Week 5+)
│   │   └── badge_renderer.py       # SVG badge generation
│   │
│   ├── storage/
│   │   ├── mongodb.py              # quality__ collections
│   │   ├── cache.py                # Redis score cache + rate limiting
│   │   └── models.py               # Pydantic models for DB documents
│   │
│   └── auth/
│       ├── api_keys.py             # API key management (Redis-backed)
│       └── rate_limiter.py         # Per-key rate limiting
│
├── mcp-server/
│   ├── server.py                   # Standalone MCP Server (for PyPI)
│   └── pyproject.toml              # mcp-quality-oracle package
│
├── dev/
│   ├── mock-mcp-server/
│   │   ├── server.py               # FastMCP server with predictable tool responses
│   │   ├── Dockerfile
│   │   └── README.md
│   └── seed-questions/             # Initial question bank seeds per domain
│       ├── code.json
│       ├── defi.json
│       ├── data.json
│       ├── search.json
│       └── general.json
│
├── tests/
│   ├── test_evaluator.py
│   ├── test_llm_judge.py
│   ├── test_api.py
│   └── fixtures/                   # Mock MCP server responses
│
├── Dockerfile
├── docker-compose.yml              # App + MongoDB + Redis + mock-mcp-server (dev)
├── requirements.txt                # ~15 lean dependencies
├── openapi.yaml                    # OpenAPI 3.1 specification
├── .env.example
└── README.md
```

### 2.3 Key Dependencies (~15)

```
# Core
fastapi==0.115.*
uvicorn[standard]
pydantic==2.*
pydantic-settings

# LLM
openai                  # DeepSeek V3.2 via OpenAI-compatible API
groq                    # Fallback judge

# Storage
motor                   # Async MongoDB driver
redis[hiredis]          # Caching + rate limiting

# MCP Client
mcp                     # Official MCP SDK (to call target servers)

# Attestation Signing
PyJWT                   # JWT signing (Ed25519) — Phase 1 attestations
cryptography            # Ed25519 key generation for JWT signing

# Utils
httpx                   # Async HTTP client
arq                     # Lightweight async Redis job queue (eval jobs)
                        # Start with FastAPI BackgroundTasks, migrate to arq at ~100 evals/day
                        # Celery is overkill until 1000+ evals/day
```

### 2.4 MongoDB Collections

```
quality__evaluations
  - _id, target_id, target_type (mcp_server|agent|skill)
  - target_url, target_manifest
  - status (pending|running|completed|failed)
  - level (1|2|3), questions_asked, questions_answered
  - scores: { overall, per_tool: { tool_name: score } }
  - report: { level1: {...}, level2: {...}, level3: null }   # Full evaluation report
  - llm_judge_model, llm_judge_responses
  - connection_strategy (sse|docker|self_report|a2a)
  - evaluation_version (string, e.g., "v1.0")
  - created_at, completed_at, duration_ms
  - attestation_id (ref to quality__attestations)

quality__scores
  - _id, target_id, target_type
  - current_score (0-100), tier (expert|proficient|basic|failed)
  - confidence, evaluation_count
  - trend — COMPUTED from quality__score_history (last 3+ entries), not stored as static value
  - domain_scores: { domain: { score, questions, se } }
  - tool_scores: { tool_name: { score, tests_passed, tests_total } }
  - evaluation_version (string, e.g., "v1.0")
  - first_evaluated_at, last_evaluated_at, next_evaluation_at
  - badge_url

quality__score_history
  - _id
  - target_id, target_type
  - evaluation_id (ref to quality__evaluations)
  - score, tier, confidence
  - evaluation_version (e.g., "v1.0")
  - domain_scores: { domain: score }
  - recorded_at
  - delta_from_previous (computed: current - previous score)

quality__attestations
  - _id, evaluation_id, target_id
  - attestation_jwt (signed JWT string — Phase 1)
  - vc_document (full W3C VC JSON — Phase 2, Week 5+)
  - aqvc_payload (raw AQVC JSON, used in both phases)
  - evaluation_version (string)
  - issued_at, expires_at
  - revoked (bool), revoked_reason

quality__question_banks
  - _id, domain, difficulty, question_text
  - expected_behavior, scoring_rubric
  - source (manual|auto_generated), generator_model
  - irt_params: { discrimination, difficulty, se }
  - usage_count, exposure_count, last_used_at
  - variant_group_id
  - variants: [string]                # 3-5 paraphrased variants (anti-gaming)
  - is_canary (bool)                  # Canary question flag

quality__api_keys
  - _id, key_hash, owner_email
  - tier (free|developer|team|marketplace)
  - rate_limit, monthly_quota, used_this_month
  - created_at, last_used_at, active (bool)
```

**Evaluation Versioning Policy:**
- `evaluation_version` tracks the methodology version (question bank, judge prompt, scoring formula)
- When methodology changes significantly → bump version (e.g., "v1.0" → "v1.1")
- Scores from different versions are flagged as non-comparable in the API response
- Major version change (v1 → v2): all targets re-evaluated within 14 days
- Minor version change (v1.0 → v1.1): scores remain valid, new evals use new version

### 2.5 API Endpoints

```
# Core Evaluation
POST   /v1/evaluate              # Submit target for evaluation
  Body: {
    target_url: string (required),
    target_type: "mcp_server" | "agent" | "skill" (required),
    level: 1 | 2 | 3 (default: 2),
    domains: string[] (optional, auto-detected from manifest if omitted),
    webhook_url: string (recommended — receives POST with full result on completion),
    callback_secret: string (optional — HMAC signature for webhook verification)
  }
  Returns: {
    evaluation_id: string,
    status: "pending",
    estimated_time_seconds: 45,
    poll_url: "/v1/evaluate/{evaluation_id}",
    message: "Webhook recommended over polling for Level 2+ evaluations"
  }

  Webhook payload (POST to webhook_url on completion):
  {
    "event": "evaluation.completed",
    "evaluation_id": "eval_abc",
    "target_id": "srv_123",
    "score": 82,
    "tier": "proficient",
    "report_url": "/v1/evaluate/eval_abc",
    "badge_url": "/v1/badge/srv_123.svg",
    "attestation_url": "/v1/attestation/att_xyz",
    "signature": "HMAC-SHA256 of payload using callback_secret"
  }

GET    /v1/evaluate/{eval_id}    # Poll evaluation status + get report
  Returns when completed:
  {
    "evaluation_id": "eval_abc",
    "status": "completed",
    "score": 47,
    "tier": "basic",
    "evaluation_version": "v1.0",
    "report": {
      "level1": {
        "manifest_score": 62,
        "tools_declared": 7,
        "tools_with_descriptions": 4,
        "tools_with_input_schemas": 5,
        "issues": [
          "3/7 tools missing input descriptions",
          "No error handling schema declared",
          "2 tools have empty description strings"
        ]
      },
      "level2": {
        "tools_tested": 7,
        "tools_passed": 4,
        "tools_failed": 3,
        "avg_latency_ms": 420,
        "tool_details": [
          {
            "tool": "search_docs",
            "score": 81,
            "latency_ms": 340,
            "tests": [
              {
                "test_type": "happy_path",
                "input_summary": "Search for 'authentication best practices'",
                "expected_behavior": "Return relevant documentation results",
                "actual_summary": "Returned 5 relevant docs with snippets",
                "judge_score": 8,
                "judge_reasoning": "Results relevant and well-formatted"
              }
            ]
          }
        ]
      },
      "level3": null
    },
    "attestation_jwt": "eyJ...",
    "badge_url": "https://laureum.ai/v1/badge/srv_123.svg"
  }

# Scores
GET    /v1/score/{target_id}     # Get quality score
  Returns: { score, tier, confidence, domains, last_evaluated, attestation_url, evaluation_version }

GET    /v1/scores                # List/search scores
  Query: ?domain=&min_score=&tier=&sort=&limit=
  Returns: { items[], total, page }

# Attestations (JWT Phase 1, W3C VC Phase 2)
GET    /v1/attestation/{id}      # Get signed attestation JWT
  Returns: JWT string (Phase 1) or W3C VC JSON (Phase 2 via /v1/attestation/{id}/vc)

GET    /v1/attestation/{id}/verify  # Verify attestation signature
  Returns: { valid, issuer, issued_at, expires_at }

# Badges
GET    /v1/badge/{target_id}.svg # SVG badge for README embedding
  Query: ?style=flat|flat-square|plastic
  Returns: SVG image

# Agent Card Enrichment
POST   /v1/enrich-agent-card     # Add quality data to an A2A Agent Card
  Body: { agent_card: <A2A Agent Card JSON> }
  Returns: {
    enriched_card: <original card + extensions.quality_oracle>,
    quality_data: { score, tier, confidence, last_evaluated }
  }
  Notes:
    - Looks up agent by matching card.url or card.name against evaluated targets
    - If not yet evaluated, returns original card + { quality_data: null, evaluate_url: "/v1/evaluate" }

# A2A Protocol
POST   /v1/a2a                   # A2A JSON-RPC handler
  Methods: tasks/send, tasks/get, tasks/cancel
  AgentTrust acts as A2A agent accepting evaluation tasks

# Agent Card (A2A Discovery)
GET    /.well-known/agent.json   # A2A Agent Card for AgentTrust itself
  Returns: A2A-compliant agent card with capabilities

# MCP Server (separate process)
# Tools: check_quality, find_best, verify_attestation
# Resources: quality://scores/{id}, quality://attestations/{id}

# Admin
GET    /health                   # Health check
GET    /metrics                  # Prometheus metrics
```

#### 2.5.1 Rate Limits by Tier

| Tier | Evaluations/month | Score Lookups/min | Evaluation Level | Badge Access |
|------|-------------------|-------------------|------------------|-------------|
| **Free** | 10 | 5 | Level 1 only | Yes (with "free tier" note) |
| **Developer ($29/mo)** | 100 | 30 | Level 1 + 2 | Yes |
| **Team ($99/mo)** | 500 | 100 | Level 1 + 2 + 3 | Yes |
| **Marketplace ($2-10K/mo)** | Unlimited | 500 | All levels | White-label |

**Caching policy:**
- Score lookups (`GET /v1/score/*`) cached in Redis with TTL of 1 hour
- Badge SVGs cached with TTL of 6 hours
- Attestation verification cached with TTL of 24 hours
- Evaluation results are immutable once completed (no cache expiry)

**Rate limit headers** (returned on every response):
```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 27
X-RateLimit-Reset: 1708700000
X-Quality-Oracle-Tier: developer
```

### 2.6 Evaluation Flow (3 Levels)

```
Level 1: Manifest Validation (instant, free)
  ├── Fetch target manifest (MCP: server capabilities, Skill: SKILL.md)
  ├── Validate JSON schema completeness
  ├── Check: tool descriptions present? input/output schemas defined?
  ├── Check: error handling declared? security patterns?
  ├── Score: 0-100 on manifest quality
  └── Result: pass/fail + warnings + detailed report

Level 2: Functional Testing (30-60s, paid)
  ├── Read manifest → extract tool definitions
  ├── Auto-generate test cases per tool:
  │   ├── Happy path (valid input → expected output type)
  │   ├── Edge case (boundary values, empty inputs)
  │   └── Error case (invalid input → graceful error)
  ├── Execute via MCP Client (connect → call tool → collect response)
  ├── For each response:
  │   ├── LLM Judge scores: relevance, correctness, completeness (0-10 each)
  │   ├── Measure latency (p50, p99)
  │   └── Check error handling
  ├── Aggregate: per-tool scores → overall score
  └── Result: 0-100 score + per-tool breakdown + full report with judge reasoning

Level 3: Domain Expert Testing (2-5min, premium)
  ├── Identify domain from manifest (defi, code, data, etc.)
  ├── Pull calibrated questions from question bank (with variant selection)
  ├── Include 10-15% canary questions for gaming detection
  ├── Challenge-response with domain-specific questions
  ├── LLM Judge with domain-specific rubric
  ├── IRT scoring (when enough data: adaptive termination)
  └── Result: Certification level (Expert/Proficient/Basic)
```

### 2.7 AQVC (Agent Quality Verifiable Credential) Format

AQVC is the canonical attestation format — the AgentTrust standard. It's designed to be forward-compatible with W3C Verifiable Credentials but starts as a simpler signed JWT for MVP.

**Phase 1 (Weeks 1-4): Signed JWT**
- Attestation payload structured as AQVC JSON (see below)
- Signed as JWT (Ed25519 via PyJWT) — simple, well-understood, easy to verify
- `GET /v1/attestation/{id}` returns JWT string
- `GET /v1/attestation/{id}/verify` decodes and validates signature
- No JSON-LD contexts, no VC data model overhead

**Phase 2 (Weeks 5+): W3C Verifiable Credential**
- When integrating with SATI/ERC-8004 which consume VCs natively
- Wrap existing AQVC payload in W3C VC envelope
- Add JSON-LD `@context`, `proof` block with Ed25519Signature2020
- Backward-compatible: JWT endpoint still works, VC endpoint added as `/v1/attestation/{id}/vc`

**AQVC Payload (used in both phases):**

```json
{
  "aqvc_version": "1.0",
  "issuer": "did:web:laureum.ai",
  "issued_at": "2026-02-23T12:00:00Z",
  "expires_at": "2026-03-25T12:00:00Z",
  "evaluation_version": "v1.0",
  "subject": {
    "id": "mcp://smithery.ai/servers/@example/my-server",
    "type": "mcp_server",
    "name": "My MCP Server"
  },
  "quality": {
    "score": 82,
    "tier": "proficient",
    "confidence": 0.91,
    "evaluation_level": 2,
    "domains": ["code-generation"],
    "tool_scores": {
      "generate_code": { "score": 87, "tests_passed": 8, "tests_total": 10 },
      "explain_code": { "score": 76, "tests_passed": 6, "tests_total": 8 }
    },
    "questions_asked": 18,
    "latency_p50_ms": 340,
    "latency_p99_ms": 1200
  },
  "evaluation": {
    "id": "eval_abc123",
    "method": "challenge-response-v1",
    "evaluated_at": "2026-02-23T12:00:00Z",
    "verification_mode": "oracle_verified",
    "connection_strategy": "sse"
  }
}
```

**Maps into (projection layer, same data, different formats):**
- **A2A Agent Card:** `extensions.quality_oracle` field
- **MCP Manifest:** `quality` section in server info
- **ERC-8004:** `validationResponse()` payload (Week 5+)
- **SATI:** Attestation schema data as W3C VC (Week 5+)
- **OpenAPI:** `x-quality-attestation` extension header
- **ClawHub:** Quality badge metadata
- **x402 Bazaar:** Quality score in discovery listing

### 2.8 Target Connection Strategies

MCP servers use different transport mechanisms. AgentTrust must handle all of them.

**Strategy A: SSE / Streamable HTTP (preferred, MVP default)**
- Target provides a publicly accessible URL (SSE or Streamable HTTP endpoint)
- AgentTrust connects via MCP SSE client from the `mcp` SDK
- Works for: hosted MCP servers, Smithery-proxied servers, any server with HTTP transport
- This is the only strategy needed for Week 1-4 MVP
- Most Smithery-listed servers expose HTTP/SSE endpoints

**Strategy B: Docker Sandbox (for stdio-only servers, post-MVP)**
- Pull server package (npm/pip) → run inside isolated Docker container
- Connect via stdio MCP client within the container
- Security constraints: no outbound network (except allowlisted APIs), no filesystem outside /tmp, 60s timeout per evaluation
- Required for: ClawHub skills, locally-distributed MCP servers
- Implementation: Week 5+ (requires sandboxed runner infrastructure)

**Strategy C: Self-Report (lowest trust, community-driven)**
- Server owner runs `quality-oracle evaluate --self` locally against their own server
- CLI connects to server locally via stdio, runs evaluation, uploads signed results
- Results marked as `verification_mode: "self_reported"` — lower trust badge
- Useful for: servers that can't expose HTTP endpoint, community quality mapping
- Implementation: Week 7+

**Strategy D: Agent-Initiated (for A2A agents)**
- AgentTrust sends A2A task to target agent via A2A protocol
- Agent processes challenge questions and returns responses
- No MCP needed — pure A2A JSON-RPC flow
- Implementation: Week 4 (alongside A2A integration)

**Connection priority for evaluation:**
```
1. Check if target has SSE/HTTP URL → Strategy A
2. Check if target is A2A agent → Strategy D
3. Check if Docker image/package available → Strategy B
4. Offer self-report CLI → Strategy C
```

**Week 1 scope impact:** Only Strategy A targets can be evaluated. When selecting the initial 20 MCP servers from Smithery, filter for servers with HTTP/SSE transport endpoints.

### 2.9 Anti-Gaming & Adversarial Resistance

An MCP server or agent could detect it's being evaluated and return artificially better responses. AgentTrust uses 5 layers of defense:

**Layer 1: Timing Randomization**
- Evaluations are not scheduled at predictable intervals
- Re-evaluation window: 7-30 days ± random jitter (up to 48h)
- No advance notification to target

**Layer 2: Input Variation**
- Each question in the bank has 3-5 paraphrased variants (same concept, different wording)
- Test inputs are drawn randomly from variant groups
- Same server never sees identical input twice within 90 days

**Layer 3: Canary Questions**
- 10-15% of test questions are "canaries" with deterministic expected outputs
- Example: "What is 2+2?" or domain-specific facts with known answers
- If server scores perfectly on canaries but poorly on novel questions → flag for review
- Canary performance tracked separately, not counted in final score

**Layer 4: Consistency Scoring**
- Servers with < 3 evaluations: confidence capped at 0.70
- High variance between evaluations (score delta > 20 across evals): confidence penalty of 0.15
- "Consistency bonus": servers scoring within ±5 across 3+ evals get confidence boost of 0.05
- Confidence formula: `base_confidence * consistency_factor * sample_size_factor`

**Layer 5: Cross-Validation (Phase 2+)**
- Compare evaluation results vs real-user feedback (when usage data available)
- Servers with high eval score but low user satisfaction → flagged
- Requires integration with marketplace usage data (Smithery, ClawHub)

**Gaming Detection Flags:**
- `gaming_suspected`: True if canary score > 95% but novel score < 60%
- `evaluation_aware`: True if response latency drops significantly during known evaluation windows
- `score_manipulation`: True if score variance > 3 standard deviations from domain mean
- Flagged servers get `verification_mode: "under_review"` badge instead of quality score

---

## 3. STANDARDS COMPATIBILITY

### 3.1 Integration Matrix

| Standard | How AgentTrust Integrates | MVP? |
|----------|------------------------------|------|
| **Google A2A** | AgentTrust IS an A2A agent; publishes Agent Card with quality extension | Yes |
| **Anthropic MCP** | Published as MCP Server on PyPI; evaluates MCP servers as primary target | Yes |
| **W3C VCs** | Phase 1: JWT attestations. Phase 2 (Week 5+): full W3C VC format for SATI/ERC-8004 | Phase 2 |
| **OpenAPI 3.1** | Full API spec with x-quality extensions | Yes |
| **ERC-8004** | Implements ValidationRegistry interface (validationRequest/Response) | Week 5 |
| **SATI/Cascade** | Quality attestations as Token-2022 NFTs on Solana | Week 5 |
| **x402** | HTTP 402 payment gate for premium evaluations | Week 4 |
| **LangChain** | QualityOracleMiddleware package | Week 7 |
| **OpenClaw/ClawHub** | Evaluate skills (65% are MCP wrappers), quality badges | Week 3 |

### 3.2 Design Principles

1. **Extension-only** — never fork standards, use official extension mechanisms
2. **Protocol-agnostic evaluation** — same engine evaluates MCP servers, ClawHub skills, REST agents, A2A agents
3. **AQVC as canonical** — one internal format, multiple external projections
4. **AgentTrust IS an A2A agent** — speaks A2A natively, not just extends it
5. **MCP-first distribution** — any IDE with MCP support gets instant access
6. **Progressive trust** — JWT → W3C VC → on-chain, complexity only when needed

---

## 4. IMPLEMENTATION PLAN (8 Weeks)

### Week 1: Core Service + Level 1 (Days 1-5)

**Day 1-2: Scaffold + Port**
- [ ] Create `quality-oracle/` directory in monorepo
- [ ] Setup FastAPI app with config, health endpoint
- [ ] Port from agent-poi: `llm_judge.py`, `challenge_handler.py`, `question_pools.py`
- [ ] Setup MongoDB connection (motor), create quality__ collections
- [ ] Setup Redis connection (cache + rate limiting)

**Day 3-4: Level 1 + MCP Client**
- [ ] Implement MCP Client wrapper with Strategy A (SSE/HTTP transport)
- [ ] Build target URL validator: check if server exposes SSE/Streamable HTTP endpoint
- [ ] Filter Smithery servers for HTTP/SSE transport when selecting initial 20 targets
- [ ] Graceful handling when server is unreachable: timeout 30s, retry 1x, then mark as "connection_failed"
- [ ] Level 1: manifest validation (schema check, description completeness)
- [ ] `POST /v1/evaluate` endpoint (async background task, webhook support)
- [ ] `GET /v1/evaluate/{id}` endpoint (poll status + full report)
- [ ] `GET /v1/score/{id}` endpoint
- [ ] API key management (Redis-backed, simple hash)

**Day 5: Test + Evaluate**
- [ ] Setup mock MCP server in dev/ for testing
- [ ] Evaluate 20 popular MCP servers from Smithery (HTTP/SSE only)
- [ ] Collect Level 1 results
- [ ] Basic rate limiting with tier headers
- [ ] Docker setup (Dockerfile + docker-compose.yml with mock server)

**Deliverable:** Working API that accepts MCP server URLs and returns manifest quality scores with full reports.

---

### Week 2: Level 2 + Badges (Days 6-10)

**Day 6-7: Test Auto-Generation**
- [ ] Test generator: read tool manifest → generate test cases (happy path, edge, error)
- [ ] Claude Sonnet for initial test case generation per domain ($0.05/domain)
- [ ] Challenge runner: connect to MCP server → call tools → collect responses

**Day 8-9: LLM Judging + Scoring**
- [ ] DeepSeek V3.2 as primary judge (relevance, correctness, completeness: 0-10 each)
- [ ] Groq Llama fallback
- [ ] Score aggregation: per-tool → per-domain → overall (0-100)
- [ ] Tier assignment: Expert (85+), Proficient (70-84), Basic (50-69), Failed (<50)
- [ ] SVG badge generation (`GET /v1/badge/{id}.svg`)
- [ ] JWT attestation signing (Ed25519)

**Day 10: First X Post**
- [ ] Evaluate top 20 MCP servers with Level 2
- [ ] Deploy to Fargate
- [ ] Prepare results visualization
- [ ] **X thread: "We evaluated the top 20 MCP servers. Results inside."**

**Deliverable:** Full Level 1+2 evaluation with quality scores, badges, and JWT attestations. First public announcement.

---

### Week 3: MCP Server + ClawHub Scan (Days 11-15)

**Day 11-12: AgentTrust as MCP Server**
- [ ] Implement FastMCP server with tools:
  - `check_quality(server_url)` → score + report
  - `find_best(domain, min_score)` → ranked list
  - `verify_attestation(attestation_id)` → validity
- [ ] `mcp-quality-oracle` PyPI package
- [ ] Publish on Smithery, Glama.ai, PulseMCP

**Day 13-14: Mass Scan**
- [ ] Scan all MCP servers on Smithery (1000+)
- [ ] Generate quality distribution report
- [ ] Identify worst offenders and best picks
- [ ] OpenClaw skill wrapper (evaluate ClawHub skills that are MCP wrappers)

**Day 15: Landing Page + Report**
- [ ] Landing page (Vercel): problem statement, live demo, "Get API Key"
- [ ] Full scan report with data visualizations
- [ ] **X thread: "We scanned 1,247 MCP servers. 23% fail basic tests."**

**Deliverable:** AgentTrust available as MCP Server in IDE. Mass scan report published.

---

### Week 4: x402 + Attestations + A2A (Days 16-20)

**Day 16-17: Agent Card Enrichment + A2A**
- [ ] `POST /v1/enrich-agent-card` endpoint
- [ ] `GET /.well-known/agent.json` — AgentTrust Agent Card
- [ ] `POST /v1/a2a` — JSON-RPC handler (tasks/send, tasks/get)
- [ ] Quality extension schema for other agents' Agent Cards
- [ ] A2A task flow: receive evaluation request → process → return result (Strategy D)

**Day 18-19: Webhooks + Polish**
- [ ] Webhook delivery on evaluation completion (HMAC-signed)
- [ ] Rate limit headers on all responses
- [ ] Score history tracking (quality__score_history collection)
- [ ] Trend computation from score history

**Day 20: x402 Integration**
- [ ] HTTP 402 payment gate for Level 2+ evaluations
- [ ] x402 facilitator integration for accepting payments
- [ ] Free tier: Level 1 only, 10 evals/month
- [ ] **X thread: "AgentTrust + x402 = trust before you pay."**

**Deliverable:** Standards-compliant service with JWT attestations, A2A, webhooks, and x402.

---

### Week 5-6: On-Chain + Assisterr Integration (Days 21-30)

**Day 21-23: Solana On-Chain + W3C VCs**
- [ ] SATI SDK integration for quality attestations
- [ ] Upgrade attestation format: wrap JWT payload in W3C VC envelope
- [ ] `GET /v1/attestation/{id}/vc` endpoint for W3C VC format
- [ ] Merkle audit trail (from hackathon code, 99.97% cost reduction)
- [ ] On-chain score lookup
- [ ] ERC-8004 compatibility layer

**Day 24-26: Assisterr Platform**
- [ ] Quality badges on SLM Store agents
- [ ] Add quality_score, quality_badges to incentive__slm_agents schema
- [ ] Dashboard endpoint for quality trends
- [ ] Strategy B implementation: Docker sandbox for stdio MCP servers
- [ ] **X thread: "AgentTrust goes on-chain via SATI."**

**Day 27-30: Level 3 + Question Bank**
- [ ] Domain-specific question banks (5 domains: code, defi, data, search, general)
- [ ] Auto-generation pipeline: domain description → Claude Sonnet → questions
- [ ] Question variants (3-5 per question) for anti-gaming
- [ ] Canary question system
- [ ] Level 3 certification flow
- [ ] 7-day per-agent re-evaluation cooldown with jitter
- [ ] **X thread: "Assisterr agents now verified by AgentTrust."**

**Deliverable:** On-chain attestations, Assisterr integration, Level 3 certification, anti-gaming.

---

### Week 7-8: Distribution + Frameworks (Days 31-40)

**Day 31-33: Framework SDKs**
- [ ] LangChain middleware: `pip install langchain-quality-oracle`
- [ ] OpenAI function calling tool definitions
- [ ] CrewAI guardrail function
- [ ] Code examples + tutorials

**Day 34-36: CI/CD + GitHub Action**
- [ ] GitHub Action: `quality-oracle/evaluate@v1`
- [ ] Quality gate: fail CI if score < threshold
- [ ] Publish on GitHub Marketplace
- [ ] Strategy C: self-report CLI (`quality-oracle evaluate --self`)

**Day 37-38: API Self-Service**
- [ ] API key signup on landing page (email + Stripe)
- [ ] Usage dashboard
- [ ] Documentation (API reference, tutorials, examples)

**Day 39-40: Partnerships**
- [ ] Smithery integration proposal (quality badges in registry)
- [ ] SendAI Agent Kit plugin
- [ ] Content: Show HN, r/LangChain, Dev.to tutorial
- [ ] **X thread: "AgentTrust is now a GitHub Action."**

**Deliverable:** Framework SDKs, CI/CD integration, self-service API, partnership outreach.

---

## 5. COST MODEL

### 5.1 Infrastructure

| Component | Monthly Cost |
|-----------|-------------|
| Fargate (0.25 vCPU, 0.5GB) | $15-30 |
| MongoDB Atlas (shared with Assisterr) | $0 |
| Redis (shared with Assisterr) | $0 |
| Domain + SSL | ~$1 |
| **Infra Total** | **~$20-30/mo** |

### 5.2 LLM Costs (per evaluation)

| Level | Model | Cost/eval |
|-------|-------|-----------|
| Level 1 (manifest) | No LLM needed | $0.00 |
| Level 2 (functional, 10 tools avg) | DeepSeek V3.2 | $0.006 |
| Level 3 (domain, 30 questions) | DeepSeek V3.2 | $0.018 |
| Question generation (one-time/domain) | Claude Sonnet | $0.05 |

### 5.3 Projected Monthly Costs

| Phase | Evals/day | LLM/mo | Infra/mo | Total/mo |
|-------|-----------|--------|----------|----------|
| Week 1-4 (launch) | 20-50 | $4-9 | $25 | ~$30-35 |
| Month 2-3 (growth) | 100-300 | $18-54 | $30 | ~$50-85 |
| Month 6 (scale) | 1,000 | $180 | $50 | ~$230 |

---

## 6. SUCCESS METRICS

| Metric | Week 2 | Week 4 | Week 8 |
|--------|--------|--------|--------|
| Servers/skills evaluated | 50 | 500 | 2,000 |
| API calls/day | 100 | 500 | 2,000 |
| API keys issued | 20 | 100 | 500 |
| X followers | 200 | 1,000 | 3,000 |
| MCP Server installs | 50 | 200 | 1,000 |
| Quality badges displayed | 0 | 20 | 100 |
| Revenue | $0 | $0 | $290+ |

---

## 7. TECH DECISIONS LOG

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Separate FastAPI in monorepo | Independent deploy, lean deps, spin-off ready |
| Primary judge | DeepSeek V3.2 | Best cost/quality ratio ($0.006/eval) |
| Fallback judge | Groq Llama 3.3 70B | Fast, cheap ($0.002/eval), good enough |
| Question gen | Claude Sonnet (one-time) | Best quality for test case generation |
| DB | MongoDB (quality__ prefix) | Shared with Assisterr, consistent patterns |
| Cache | Redis | Shared with Assisterr, fast score lookup |
| Attestation signing | Ed25519 JWT (Phase 1) → W3C VC (Phase 2) | Start simple, add VC complexity only when SATI/ERC-8004 need it |
| Async jobs | BackgroundTasks → arq → Celery | Start simplest, arq at 100/day, Celery at 1000+/day. arq has 1 dep vs Celery's 8+ |
| MCP Client | Official mcp SDK (Strategy A: SSE/HTTP) | Standard, maintained by Anthropic. Stdio via Docker sandbox post-MVP |
| First domain | MCP Servers | Biggest pain, clearest test surface, natural distribution |
| On-chain | SATI/Cascade on Solana | Cheapest ($0.002/attestation), Solana ecosystem fit |
| Anti-gaming | 5 layers: timing jitter, input variants, canaries, consistency, cross-validation | Progressive defense matching attacker sophistication |

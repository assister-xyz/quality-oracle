# T54 Labs (@t54ai) — Deep Competitive Analysis

> Last updated: 2026-03-23
> Context: Research for Laureum.ai (Quality Oracle) strategic positioning

## Company Overview

- **Name**: t54 Labs, Inc.
- **Website**: https://t54.ai
- **Twitter/X**: https://x.com/t54ai
- **Founded**: January 2025
- **HQ**: San Francisco + Panama + Hong Kong
- **Team**: 12 people (hiring 2 engineers + 1 BD/DevRel)
- **Funding**: $5M Seed (Feb 2026) — Anagram, PL Capital, Franklin Templeton, Ripple, Virtuals Ventures, Blockchain Coinvestors, ABCDE
- **Token**: Explicitly NO token planned
- **Tagline**: "Empowering Trusted Agentic Economy"

### Founders
- **Chandler Fang** — ex-JPMorgan Onyx (quant trading 2017-2021), ex-Ripple (cross-border payments). @chandler_agi
- **Sergio Chan** — @Sergio2Chan

### Advisors
- **Roberta Vassallo** — Legal & Compliance (ex-FDIC, SEC, FINRA)
- **Andi White, PhD** — AI Partnerships (Microsoft)
- **James Chie** — Financial Institutions (Ripple)

---

## What T54 Actually Is

T54 is the **financial infrastructure layer for autonomous AI agents**. NOT an agent quality/benchmarking platform. Their focus: enabling agents to safely **transact financially** — payments, credit, risk management, settlement.

**Core thesis**: "Financial systems were designed around human identity. As agents become autonomous participants, we need agent-native financial primitives."

**L4 Finance analogy** (like L4 self-driving):
- L1: Manual Finance (banks)
- L2: Algorithm-Recommended (Stripe, Visa) — human decides
- L3: AI-Assisted (Visa "Intelligent Commerce") — shared decision
- L4: **Agentic Autonomous (T54)** — agent decides

---

## Product Stack (4 Layers)

```
┌─────────────────────────────────────────────────────────┐
│  1. KYA (Know Your Agent) — Identity & Verification     │
│     Developer KYB, model provenance, intent attestation │
├─────────────────────────────────────────────────────────┤
│  2. Trustline — Risk Engine                             │
│     Real-time per-transaction risk assessment           │
│     tRadar (runtime) + tAudit (code integrity)          │
├─────────────────────────────────────────────────────────┤
│  3. tLedger — Settlement Platform                       │
│     Virtual accounts on Solana, XRPL, Base              │
│     x402 payment protocol (HTTP 402)                    │
├─────────────────────────────────────────────────────────┤
│  4. ClawCredit — Agent Credit Lines                     │
│     Instant $5 credit → grows with usage                │
│     30,209 agents signed up, $11,112 issued             │
└─────────────────────────────────────────────────────────┘
```

---

## Trustline — Risk Engine (Detail)

### What it scores
NOT agent quality. It's **per-transaction risk assessment**:
- LOW → auto-approve
- MEDIUM → additional validation needed
- HIGH → likely unsafe, stringent validation
- CRITICAL → automatic rejection

### Documented Architecture (from docs)

**Sequencer** → classifies risk requests by complexity (Easy/Medium/Hard), routes to tiers, escalates on low confidence.

**Validator Agent Network (VAN)** — 5 specialized committees (as documented):
1. **Injection Detection** — malicious prompt injections
2. **Identity Verification** — KYA, device posture, attestation
3. **Consent & Mandate** — AP2/x402 evidence validation
4. **Code & Step Validation** — tool calls, function signature integrity
5. **Anomaly Detection** — behavioral/contextual deviations

**Consensus Coordinator** → aggregates scores → Final Decision Artifact with liability mapping.

### IMPORTANT: VAN is Deprecated

From their own glossary:
> **Validator (Deprecated)**: Previously committee-based risk assessment; now replaced by Trustline's unified engine

The "5 AI validators reaching consensus" may no longer be the actual implementation. Marketing may lag behind reality.

### Decision Object Schema
```json
{
  "decision": "Approve | Decline",
  "risk_level": "Low | Medium | High | Critical",
  "confidence": 0.85,
  "ttl_seconds": 300,
  "rationales": ["validator reasoning summaries"],
  "challenge_eligibility": true,
  "liability_map": {"agent": 0.3, "operator": 0.5, "merchant": 0.2}
}
```

### Challenge-Response
Flagged transactions → dynamic gatekeeper pauses execution → requests clarification (proof-of-purpose, logs, metadata) → agent responds programmatically → re-evaluation.

---

## tAudit — Code Integrity (How It Really Works)

From actual code in `tpay-sdk-python`:

```python
@taudit_verifier
def my_payment_function():
    ...
```

1. Decorator captures function source code
2. Normalizes it (strips comments, whitespace, formatting)
3. SHA-256 hash of normalized source
4. Stored in global `AUDIT_SNAPSHOT` dict
5. On payment: `func_stack_hashes` sent with transaction
6. Backend compares against whitelist of approved hashes

**Reality**: Simple but effective tamper detection. If anyone modifies the logic of a payment function, the hash changes and the transaction is blocked. But it's just SHA-256 comparison, not deep code analysis.

---

## Trace Collection (Reasoning Capture)

Wraps OpenAI client via monkey-patch:

```python
from tpay import wrap_openai_client
wrap_openai_client(openai_client)
```

Captures:
- `user_input` — user prompts (+ SHA-256 hash)
- `system_prompt` — system instructions (+ hash + version)
- `agent_output` — agent responses (+ hash)
- `reasoning_summary` — o1/o3 extended thinking
- `function_call` — tool invocation (name + arguments)
- `tool_result` — tool return values

All sent via `POST /risk/trace` → gets `trace_id` → linked to transaction via `X-PAYMENT-SECURE` header (W3C Trace Context).

---

## x402 Payment Flow (Complete)

```
1. Agent → GET /api (no payment headers)
   ← 402 Payment Required + PaymentRequirements

2. Agent → POST /risk/session
   ← { sid, expires_at }

3. Agent → POST /risk/trace (reasoning events, model config)
   ← { tid }

4. Agent → GET /api + headers:
   X-PAYMENT: base64(EIP-3009 signed payment)
   X-PAYMENT-SECURE: W3C trace context + tid
   X-RISK-SESSION: sid
   X-AP2-EVIDENCE: mandate reference (optional)

5. Seller → POST /x402/verify
   ← { isValid, payer, invalidReason }

6. Seller → POST /x402/settle
   ← { success, payer, transaction, network }
```

### Custom Headers
- **X-PAYMENT**: Base64(EIP-3009 signed payment). Contains `{ authorization: { from, nonce, deadline, value }, signature }`
- **X-PAYMENT-SECURE**: `w3c.v1;tp=<traceparent>[;ts=<tracestate>]`. Max 4096 bytes.
- **X-RISK-SESSION**: UUID of risk session
- **X-AP2-EVIDENCE**: `evd.v1;mr=<ref>;ms=<sha256>;mt=application/json;sz=<bytes>`. Max 2048 bytes.

---

## ClawCredit — Growth Engine

### The Growth Hack
```
Agent has lobster.cash wallet
         ↓
Send $0.01 USDC to verification address (Solana)
  → AsEDd1vqP4AT4jnycoum22Z1cdKGKuwz3aWevUDbkmxE
         ↓
ClawCredit verifies on-chain transfer
         ↓
Instant $5 credit line (no pre-qualification wait)
         ↓
Agent can immediately pay for x402 services
```

**Result**: 30,209 agents signed up, $11,112 total credit issued (~$0.37/agent average).

### Repayment Urgency Tiers
| Urgency | Check Frequency |
|---------|----------------|
| Not urgent (>7 days) | Every 24h |
| Due this week | Every 12h |
| Due soon (2-3 days) | Every 6h |
| Due tomorrow | Every 4h |
| Due today | Every 2h |
| Overdue | Every 1h |

### Credit Scoring
- Scale: 200-850
- Signals: usage history, repayment behavior, integration depth, wallet proof
- Chains: XRPL (RLUSD), Base (USDC), Solana (USDC)

---

## ERC-ACP — Agent Commerce Protocol (Smart Contracts)

### Core Escrow Lifecycle

```
Open → Funded → Submitted → Completed ✓ (payment to provider)
                     ↓
                  Rejected (refund to client)
                     ↓
                  Expired (auto-refund, permissionless)
```

### Job Struct (Solidity)
```solidity
struct Job {
    address client;      // who pays
    address provider;    // who works
    address evaluator;   // who judges
    string description;
    uint256 budget;
    uint256 expiredAt;
    JobStatus status;
}
```

### Key Functions
- `createJob(provider, evaluator, expiredAt, description)` — client creates
- `setBudget(jobId, amount)` — price negotiation
- `fund(jobId, expectedBudget)` — client escrows (front-running protection)
- `submit(jobId, deliverable)` — provider submits work (deliverable = bytes32 hash)
- `complete(jobId, reason)` — **evaluator** releases payment (minus platform fee)
- `reject(jobId, reason)` — evaluator rejects, refunds client
- `claimRefund(jobId)` — permissionless safety net after expiry

### Hooks System (like Uniswap v4)
- `beforeAction` / `afterAction` callbacks on all functions
- Gas limit: 500K per hook call
- `claimRefund()` deliberately NOT hookable (safety: refunds can't be blocked)

### Hook Examples
1. **FundTransferHook** — Two-phase escrow (agent bridges/swaps tokens)
2. **BiddingHook** — Sealed-bid auction with EIP-712 signature verification

### MCU (Merchant Custody Underwriting) — Experimental
- Bond-backed merchant execution with underwriter oversight
- EIP-712 signatures for completion/rejection/dispute
- Delivery confirmation timeout → dispute → slash/release bond

### LAUREUM AS EVALUATOR
Laureum can be the `evaluator` in ERC-ACP:
- Client hires agent → creates job with escrow
- Laureum evaluates agent quality
- Approve → payment released to provider
- Reject → refund to client
- Hooks can require "Laureum Gold badge" as pre-condition

---

## Sandbox API — Tested End-to-End

### URLs
| Environment | API | Portal |
|---|---|---|
| Sandbox | `https://api-sandbox.t54.ai/api/v1` | `https://portal-sandbox.t54.ai` |
| Production | `https://api.t54.ai/api/v1` | `https://portal.t54.ai` |

### API Stats
- **82 endpoints** in sandbox (81 in production)
- **OpenAPI docs**: `https://api-sandbox.t54.ai/docs` (Swagger UI)
- **Health**: `{"status":"healthy","version":"portal-v3-20260213-02"}`

### Authentication (2 methods)
1. **Bearer Token (JWT)** — RS256, ~8 day expiry, for portal operations
   - Get via `POST /api/v1/login/access-token` (email + password)
2. **API Key/Secret** — for agent/payment operations
   - Headers: `X-API-Key` + `X-API-Secret`

### 22 API Scopes
`asset_account:read`, `deposit:read`, `withdraw:create`, `trade:submit`, `agent:profile:create`, `asset_account:swap:read`, `agent:limits:update`, `agent:profile:delete`, `trade:create`, `agent:profile:update`, `agent:account:read`, `withdraw:read`, `asset_account:swap:write`, `payments:read`, `payments:write`, `agent:limits:read`, `agent:profile:read`, `agent:limits:create`, `analytics:read`, `trade:read`, `agent:account:delete`, `balance:read`

### Endpoint Categories

**Payments (4)**:
- `POST /api/v1/payment` — create payment
- `GET /api/v1/payment/{payment_id}` — get status
- `GET /api/v1/payments` — list for project
- `GET /api/v1/agent/{agent_id}/payments` — list for agent

**Agents (8)**:
- CRUD on agent profiles (SDK + portal variants)
- Agent limits management

**Projects (8)**:
- CRUD on projects
- List agents in project

**Balances (3)**:
- Agent total balance (USD)
- Agent asset balance (per network/asset)
- Project total balance

**Virtual Accounts (10)**:
- Asset accounts listing
- Transaction history
- Deposit/withdraw operations
- Wallet verification

**API Keys (4)**:
- Generate, list, delete, get scopes

**Auth/Users (10)**:
- Signup (open, no auth), login, email verify, password reset
- Google OAuth
- KYC update

**Risk/Audit (1)**:
- `POST /api/v1/radar/audit` — submit code audit

**Payment Rules (5)**:
- CRUD on automated payment rules

**WebSocket (3)**:
- Connection stats, broadcast, force disconnect

**Token Operations (3)**:
- Token info, swap, swap status

**Analytics (1)**:
- Transaction volume

**OAuth2/OIDC (8)**:
- Full OAuth2 with dynamic client registration (RFC 7591)
- Supports Claude Desktop native integration

### Agent Creation — What Happens
- 9 asset accounts auto-provisioned:
  - Solana: SOL, USDT, USDC, ABC, AMN
  - XRPL: XRP, RLUSD
  - Base: ETH, USDC
- Auto-airdrop: 200 XRP (~$280 on testnet)
- Payment lifecycle: `initiated` → `reviewing` → `completed`/`failed`/`cancelled`

### Endpoints Requiring NO Auth
- `GET /api/v1/api_key/scopes/all`
- `GET /api/v1/websocket/ws/stats`
- `POST /api/v1/users/signup`
- `GET /.well-known/openid-configuration`
- `GET /.well-known/oauth-authorization-server`
- `GET /oauth2/jwks.json`
- `GET /health`

---

## Open-Source Repos (github.com/t54-labs)

| Repo | Language | Stars | What It Is |
|---|---|---|---|
| `x402-secure` | Python | 21 | Payment proxy + SDK (Apache 2.0). **Core is just a proxy to closed risk engine** |
| `clawcredit-blockrun-gateway` | TypeScript | 3 | OpenAI-compatible gateway paying via ClawCredit |
| `ERC-ACP` | Solidity | 0 | Agent Commerce Protocol smart contracts |
| `lobstercash-clawcredit` | TypeScript | 1 | ClawCredit + lobster.cash wallet integration |
| `tpay-sdk-python` | Python | 1 | tLedger Python SDK (NOT on public PyPI) |
| `payments-mcp` | TypeScript | Fork | Forked from coinbase/payments-mcp |
| `x402-xrpl` | TypeScript | 0 | x402 explorer for XRP Ledger |

### x402-secure Architecture (from code)
```
packages/x402-secure/    — SDK for buyers (trace collection) and sellers (verify/settle)
proxy/                   — FastAPI facilitator proxy
  routes.py              — 1070 lines, verify/settle endpoints
  risk_routes.py         — session/trace/evaluate (forwards to RISK_ENGINE_URL)
  headers.py             — X-PAYMENT-SECURE, X-AP2-EVIDENCE parsing
```

**Key finding**: Proxy has two modes:
- `PROXY_LOCAL_RISK=1` → always approve (testing)
- Default → forwards to external `RISK_ENGINE_URL` (Trustline, closed)

**Dependencies**: FastAPI, httpx, web3, eth-account, cryptography, opentelemetry, cachetools, x402>=0.2.1

### ClawCredit Gateway
- OpenAI-compatible endpoint at port 3402
- Routes to BlockRun API for inference
- Cost estimation: `defaultAmountUsd * 1M + maxTokens * 8`
- Models: nvidia/gpt-oss-120b (free), nvidia/kimi-k2.5 ($0.6/$3), anthropic/claude-sonnet-4.6 ($3/$15)

---

## Published Packages

| Package | Registry | Version | Downloads |
|---|---|---|---|
| `x402-secure` | PyPI | 0.1.4 | — |
| `@t54-labs/clawcredit-sdk` | npm | 0.2.55 | 1,392/month |
| `@t54-labs/clawcredit-blockrun-sdk` | npm | 0.1.8 | 985/month |
| `tpay` | Private | 0.1.1 | Not public |

---

## Supported Chains & Assets

| Chain | Assets | Sandbox Network |
|---|---|---|
| Solana | SOL, USDT, USDC, ABC, AMN | Devnet |
| XRPL | XRP, RLUSD | Testnet |
| Base (EVM) | ETH, USDC | Sepolia |

---

## Marketing vs Reality

| Claim | What Code Shows |
|---|---|
| "5 AI validators reaching consensus" (VAN) | **Deprecated** per their own glossary. Open-source = just proxy to closed API |
| "tAudit code verification" | SHA-256 hash of normalized function source. Simple but effective |
| "30K agents" | May include pre-qualifying (not yet active) agents |
| "$11K credit issued" | ~$0.37 per agent average |
| "Real-time risk assessment" | Actual Trustline engine is proprietary, can't verify sophistication |
| "Multi-chain settlement" | Real — Solana, XRPL, Base all work in sandbox |
| "Agent Commerce Protocol" | Real — solid Solidity contracts with hooks system |

---

## T54 vs Laureum — Comparison

| Dimension | T54 | Laureum |
|---|---|---|
| **Core question** | "Is this transaction safe?" | "Is this agent competent?" |
| **When** | Per-transaction, real-time | Pre-payment, certification |
| **What's measured** | Transaction risk, reasoning traces, code integrity | Accuracy, safety, reliability, process quality, latency, schema |
| **Methodology** | Unified risk engine (formerly VAN) | Multi-judge consensus (CollabEval) + IRT adaptive + adversarial probes |
| **Output** | Approve/Decline + risk level | Trust score 0-100 + AQVC credential + badge |
| **Scope** | Financial transactions only | General agent quality across any domain |
| **On-chain** | Hybrid (settlement on-chain, risk off-chain) | Off-chain eval, on-chain credential (planned) |
| **Chains** | Solana, XRPL, Base | Solana (planned) |
| **Standards** | x402, HTTP 402, ERC-ACP, AP2 | A2A, MCP, W3C VCs, ERC-8004 |
| **Business model** | Infrastructure/fintech (credit, settlement, risk-as-service) | Certification/SaaS (eval credits, badges, CI/CD gates) |
| **Growth mechanic** | "Want credit? Get verified" (pull) | "Want enterprise clients? Get badge" (push→pull) |
| **Funding** | $5M seed | Self-funded |
| **Token** | No | No |
| **Team** | 12 people | — |

### Overlapping Mechanisms
- Multi-agent consensus (VAN ≈ our multi-judge)
- Agent identity verification (KYA ≈ our evaluation)
- x402 payment protocol (both have it)
- Solana blockchain (both target it)
- Challenge-response (both use it)

### Where Laureum Wins
1. **Quality benchmarking** — T54 has zero agent capability testing
2. **Challenge-response evaluation** — we test knowledge, they test transaction legitimacy
3. **Battle arena / leaderboard** — competitive evaluation, engagement
4. **IRT/adaptive testing** — statistically rigorous, reduces cost 50-90%
5. **Adversarial probes** — prompt injection, PII leakage, hallucination detection
6. **Domain-specific evaluation** — quality across MCP, OpenAI, LangChain, CrewAI

### Where T54 Wins
1. **Financial infrastructure** — payments, settlement, virtual accounts
2. **Multi-chain support** — Solana + XRPL + Base, all working
3. **Credit system** — instant credit as growth engine (30K signups)
4. **Smart contracts** — ERC-ACP with hooks, production-grade
5. **Funding & team** — $5M, 12 people, Ripple/Franklin Templeton backing
6. **OAuth2/OIDC** — native Claude Desktop integration

---

## Integration Opportunities

### Option A: Laureum as Evaluator in ERC-ACP
```
Client creates job → escrow funded →
Provider submits → LAUREUM evaluates quality →
  ✓ Approve → payment released
  ✗ Reject → client refunded
```

### Option B: Laureum Quality Score as KYA Signal
```
Agent → Laureum certification (quality score + AQVC) →
  → T54 KYA (quality signal included in identity) →
  → ClawCredit (higher limit for certified agents) →
  → Trustline (lower friction for trusted agents)
```

### Option C: Co-marketing Partnership
- T54 handles financial trust, Laureum handles quality trust
- "Verified & Certified" dual badge (T54 KYA + Laureum AQVC)
- Joint pitch to agent marketplaces and enterprises

---

## What to Learn from T54

1. **Pull-based growth** — their "$0.01 → instant credit" is brilliant. We need equivalent pull mechanic for certification
2. **Trace collection** — capturing reasoning BEFORE evaluation. We evaluate outputs, they evaluate process. Could combine
3. **Code hashing** — simple SHA-256 tamper detection. Easy to add to our eval
4. **OAuth2/OIDC** — for native MCP client integration (Claude Desktop etc.)
5. **Sandbox with auto-airdrop** — lowers barrier to first experience
6. **ERC-ACP hooks** — extensible smart contract pattern worth adopting

---

## Sources

### Official
- [t54.ai](https://t54.ai)
- [t54 Documentation](https://t54.ai/docs/first-steps/overview)
- [Trustline Architecture](https://t54.ai/docs/risk-fraud/technical-architecture)
- [x402-secure Docs](https://t54.ai/docs/risk-fraud/x402-secure)
- [tLedger Docs](https://t54.ai/docs/platform/tledger)
- [Sandbox API Swagger](https://api-sandbox.t54.ai/docs)
- [Production API Swagger](https://api.t54.ai/docs)

### GitHub
- [x402-secure](https://github.com/t54-labs/x402-secure) — Python, 21 stars
- [ERC-ACP](https://github.com/t54-labs/ERC-ACP) — Solidity
- [tpay-sdk-python](https://github.com/t54-labs/tpay-sdk-python) — Python
- [clawcredit-blockrun-gateway](https://github.com/t54-labs/clawcredit-blockrun-gateway) — TypeScript
- [lobstercash-clawcredit](https://github.com/t54-labs/lobstercash-clawcredit) — TypeScript

### Press
- [The Block: $5M seed round](https://www.theblock.co/post/391273/ripple-franklin-templeton-ai-agent-trust-startup-t54-labs)
- [t54 Blog: Seed announcement](https://www.t54.ai/blog/t54-labs-raises-5m-seed-round)
- [Crowdfund Insider](https://www.crowdfundinsider.com/2026/03/264798-t54s-building-the-trust-layer-for-agentic-commerce/)
- [Bitcoinist: XRPL + Virtuals](https://bitcoinist.com/xrp-ledger-ai-agent-payments-virtuals-t54/)

### Packages
- [x402-secure on PyPI](https://pypi.org/project/x402-secure/) — v0.1.4
- [@t54-labs/clawcredit-sdk on npm](https://www.npmjs.com/package/@t54-labs/clawcredit-sdk) — v0.2.55
- [@t54-labs/clawcredit-blockrun-sdk on npm](https://www.npmjs.com/package/@t54-labs/clawcredit-blockrun-sdk) — v0.1.8
- [CryptoRank: T54 Labs](https://cryptorank.io/price/t-54-labs)

### Social
- [Twitter/X: @t54ai](https://x.com/t54ai)
- [Discord](https://discord.com/invite/t54ai)
- [Telegram](https://t.me/t54ai)
- [LinkedIn](https://www.linkedin.com/company/t54-labs/)
- [x402 Monopoly](https://x.com/x402monopoly) — live demo with 4 AI agents

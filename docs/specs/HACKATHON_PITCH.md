# AgentTrust — Hackathon Pitch Template

## One-Liner

AgentTrust: verify AI agent quality BEFORE you trust them.

## Elevator Pitch (30 seconds)

The AI agent ecosystem has identity, reputation, and payments — but no pre-payment quality gate. AgentTrust fills this gap: it connects to any MCP server, runs challenge-response tests with LLM judge consensus, and issues W3C Verifiable Credentials as proof of competency. Think "credit score for AI agents."

## Problem

- 17K+ MCP servers, 143K+ AI agents — how do you know which ones actually work?
- Current solutions are post-hoc (rate after use) or self-reported (trust badges from the agent itself)
- No standardized way to verify competency BEFORE delegation or payment
- ClawHavoc research found 1,184 malicious MCP skills — security is real

## Solution

AgentTrust provides **pre-trust verification** via:

1. **Challenge-Response Testing** — Connects to target server, runs automated tests
2. **6-Axis Scoring** — accuracy, safety, reliability, process quality, latency, schema quality
3. **Consensus Judging** — 2-3 LLM judges in parallel, majority vote
4. **Verifiable Credentials** — W3C VC format, Ed25519 signed, machine-readable
5. **Battle Arena** — Head-to-head blind evaluation with ELO ratings
6. **Adaptive Testing (IRT)** — Reduces eval cost by 50-90% using psychometric models

## Technical Stack

- **Backend:** FastAPI + MongoDB + Redis (Python)
- **MCP Server:** FastMCP (SSE transport, port 8003)
- **LLM Judges:** 7 provider fallback chain (all free tier: Cerebras → Groq → OpenRouter → Gemini → Mistral → DeepSeek → OpenAI)
- **Standards:** W3C VC v2.0, Google A2A v0.3, x402, AIUC-1
- **Tests:** 533 passing, 60 source files, 15 dependencies

## Key Metrics

- **Evaluation cost:** $0.006-0.013 per eval (30 questions)
- **Speed:** Manifest check <100ms, full eval ~30s
- **Accuracy:** Consensus judging reduces false positives by 50-66%
- **Coverage:** SSE + Streamable HTTP dual transport

## Demo Flow

1. Point AgentTrust at any MCP server URL
2. Watch it discover tools, generate test cases, run evaluations
3. See 6-axis scores with confidence levels
4. Get a signed W3C Verifiable Credential
5. Compare servers in Battle Arena with ELO rankings

## Market Context

- EU AI Act deadline: August 2026 (requires AI system quality documentation)
- Braintrust: $800M valuation (post-deploy observability — we're pre-deploy)
- LMArena: $1.7B valuation (LLM benchmarks — we're agent benchmarks)
- NIST AI Agent Standards Initiative launched Feb 2026

## What Makes This Different

| Existing | What They Do | Our Gap |
|----------|-------------|---------|
| TARS/Amiko | Post-hoc reputation | Pre-payment verification |
| SATI | Identity attestation | Quality verification |
| Braintrust | Production monitoring | Pre-deploy evaluation |
| LMArena | LLM benchmarks | Agent/MCP benchmarks |
| mcp-scan (Snyk) | Security scanning | Functional quality |

AgentTrust is the only tool that verifies **functional competency** before trust.

## Applicable Hackathon Categories

- **MCP Servers** — AgentTrust IS an MCP server
- **AI Agent Tools** — Quality verification tool
- **Security** — Adversarial probes, trust verification
- **Standards** — W3C VC, A2A, x402 integration
- **Solana/Web3** — On-chain payment verification, AQVC credentials

## Links

- GitHub: https://github.com/vitaliiserbynassisterr/quality-oracle
- MCP Server: `pip install mcp-agenttrust`
- Architecture: [docs/ARCHITECTURE.md](../ARCHITECTURE.md)

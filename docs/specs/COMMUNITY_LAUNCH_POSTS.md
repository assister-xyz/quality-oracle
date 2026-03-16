# Community Launch Posts

## 1. MCP Discord #showcase

```
🔍 AgentTrust — Quality verification for MCP servers

We built a tool that evaluates MCP server competency before you trust them with real tasks.

What it does:
• Connects to any MCP server via SSE
• Runs 3-level challenge-response tests (manifest → functional → domain expert)
• Scores across 6 dimensions: accuracy, safety, reliability, process quality, latency, schema quality
• Adversarial probes: prompt injection, PII leakage, hallucination detection
• Issues W3C Verifiable Credentials as proof

Install as MCP server:
pip install mcp-agenttrust

Or run full stack:
docker compose up -d  (API + MCP Server + MongoDB + Redis)

552 tests | MIT license | Python/FastAPI

GitHub: https://github.com/assister-xyz/quality-oracle
PyPI: https://pypi.org/project/mcp-agenttrust/
```

---

## 2. DEV Community Article

**Title:** How We Score MCP Server Quality Before Payment

**Tags:** ai, mcp, testing, python

```markdown
The AI agent ecosystem is exploding. MCP alone has 16,000+ registered servers. But how do you know if an agent is competent *before* trusting it with your data or money?

Identity exists (ERC-8004, SATI). Post-hoc reputation exists (TARS, Amiko). Payments are mature (x402). But there's a gap: **no pre-payment quality gate**.

We built AgentTrust to fill it.

## The Problem

When you connect to an MCP server, you're trusting it blindly. There's no standardized way to verify:
- Does it actually do what it claims?
- Is it safe (won't leak PII, resists prompt injection)?
- Is it reliable under load?
- Does its schema match reality?

## Our Approach: Challenge-Response Testing

AgentTrust uses a 3-level evaluation pipeline:

**Level 1 — Manifest Analysis** (~50ms)
Read the server's tool descriptions. Score schema completeness, description quality, input validation.

**Level 2 — Functional Testing** (~5-30s)
Actually connect via MCP SSE, list tools, generate test cases, call tools, and judge responses with LLM consensus.

**Level 3 — Domain Expert** (~30-60s)
96 calibrated questions across 5 domains, scored by 2-3 LLM judges in parallel with agreement threshold.

## 6-Axis Scoring

Every evaluation produces scores across:
- **Accuracy** (35%) — Are responses correct?
- **Safety** (20%) — Resists adversarial probes?
- **Reliability** (15%) — Consistent under repeated calls?
- **Process Quality** (10%) — Error handling, input validation?
- **Latency** (10%) — Response time within bounds?
- **Schema Quality** (10%) — Tool descriptions match behavior?

## Anti-Gaming

Agents that game evaluations are a real threat. We address this with:
- **Question paraphrasing** — Same question, different phrasing each time
- **Adversarial probes** — 5 types injected randomly
- **Production correlation** — Compare eval scores to real-world feedback
- **Style penalties** — Prevent gaming via verbose/formatted responses

## Battle Arena

Head-to-head blind evaluation with position-swap consistency. OpenSkill (Bayesian ELO) rating system with divisions from Bronze to Grandmaster.

## W3C Verifiable Credentials

Every evaluation issues an AQVC (Agent Quality Verifiable Credential) — a W3C VC with Ed25519 signature that any system can verify. Compatible with Google A2A, MCP, and ERC-8004.

## Try It

```bash
pip install mcp-agenttrust
```

Or the full stack:
```bash
git clone https://github.com/assister-xyz/quality-oracle
cd quality-oracle && cp .env.example .env
docker compose up -d
```

552 tests, MIT license, free LLM providers (Cerebras, Groq, OpenRouter).

[GitHub](https://github.com/assister-xyz/quality-oracle) | [PyPI](https://pypi.org/project/mcp-agenttrust/)
```

---

## 3. Twitter/X Thread

```
🧵 Thread: Why we built AgentTrust — quality verification for AI agents

1/ The AI agent ecosystem has 16,000+ MCP servers. But how do you know if one is competent BEFORE trusting it with your data?

Identity ✅ (ERC-8004)
Reputation ✅ (post-hoc)
Payments ✅ (x402)
Pre-payment quality gate ❌ ← we built this

2/ AgentTrust connects to any MCP server and runs challenge-response tests across 6 dimensions:

• Accuracy (35%)
• Safety (20%)
• Reliability (15%)
• Process Quality (10%)
• Latency (10%)
• Schema Quality (10%)

3/ Three evaluation levels:

L1: Schema analysis (~50ms) — Does the manifest match reality?
L2: Functional tests (~5-30s) — Call tools, judge responses
L3: Domain expert (~30-60s) — 96 calibrated questions, consensus judging

4/ Anti-gaming built in:

• Question paraphrasing (different phrasing each eval)
• 5 adversarial probe types
• Production correlation (catches sandbagging)
• Style penalties (no gaming via verbose responses)

5/ Every eval issues a W3C Verifiable Credential (AQVC) with Ed25519 signature.

Compatible with: Google A2A, MCP protocol, ERC-8004, SATI, x402.

One credential format, all standards.

6/ Battle Arena: head-to-head blind evaluation with OpenSkill ratings.

IRT Adaptive Testing: reduces eval cost by 50-90% while maintaining accuracy.

552 tests. MIT license. Free LLM providers.

7/ Try it now:

pip install mcp-agenttrust

Full stack: docker compose up -d

GitHub: https://github.com/assister-xyz/quality-oracle
PyPI: https://pypi.org/project/mcp-agenttrust/

Built by @assaborovna
```

---

## 4. Show HN

**Title:** AgentTrust – Quality verification for AI agents before you trust them

```
AgentTrust evaluates AI agent and MCP server competency BEFORE you trust them with real tasks or payments.

The AI agent ecosystem has identity (ERC-8004), post-hoc reputation (TARS), and payments (x402) — but no pre-payment quality gate. We built one.

How it works:
- Connects to any MCP server via SSE
- Runs 3-level challenge-response tests
- Scores across 6 quality dimensions
- Issues W3C Verifiable Credentials as proof

Key features:
- 7-provider LLM fallback chain (all free tiers)
- Consensus judging (2-3 judges, early-stop agreement)
- Battle arena with OpenSkill ratings
- IRT adaptive testing (50-90% cost reduction)
- 5 adversarial probe types
- Anti-gaming: paraphrasing, style penalties, production correlation

Stack: Python/FastAPI + MongoDB + Redis
Tests: 552, all mocked (no external deps needed)
License: MIT
Cost: $0.006-0.013 per evaluation

pip install mcp-agenttrust

GitHub: https://github.com/assister-xyz/quality-oracle
```

---

## 5. Reddit Posts

### r/MachineLearning

**Title:** [P] AgentTrust: Challenge-response quality verification for AI agents with IRT adaptive testing

```
We built a system for evaluating AI agent competency before deployment/payment.

Key technical contributions:
- Multi-judge consensus (CollabEval pattern): 2 judges parallel, 3rd tiebreaker if disagreement > 15pt threshold
- IRT (Item Response Theory) adaptive question selection: Rasch 1PL calibration from battle data, Fisher information maximization
- 6-axis evaluation: accuracy, safety, reliability, process quality, latency, schema quality
- Anti-sandbagging: production correlation feedback loop with Pearson correlation and alignment classification

The system connects to MCP servers, runs functional tests, and issues W3C Verifiable Credentials.

552 tests, MIT: https://github.com/assister-xyz/quality-oracle
```

### r/LocalLLaMA

**Title:** AgentTrust — Free tool to evaluate MCP server quality using Cerebras/Groq/OpenRouter (all free tiers)

```
Built a quality evaluation tool that works entirely on free LLM tiers:

- Cerebras (1M tokens/day, llama3.1-8b)
- Groq (500K tokens/day, llama-3.1-8b-instant)
- OpenRouter (free Qwen3 80B)

It connects to MCP servers, runs tests, and scores them across 6 dimensions. Cost per eval: $0.006-0.013.

pip install mcp-agenttrust

The LLM judge auto-rotates keys on rate limits and falls back through 7 providers.

GitHub: https://github.com/assister-xyz/quality-oracle
```

### r/LangChain

**Title:** AgentTrust — Quality gate for AI agents before delegation (MCP + LangChain + CrewAI support planned)

```
We're building a pre-payment quality verification system for AI agents. Currently supports MCP servers natively, with LangChain and CrewAI adapters in the roadmap.

Use case: Before your orchestrator delegates a task to an agent, verify its competency first.

Features: 3-level evaluation, 6-axis scoring, W3C Verifiable Credentials, battle arena.

pip install mcp-agenttrust

GitHub: https://github.com/assister-xyz/quality-oracle

Would love feedback from the LangChain community on what evaluation dimensions matter most for your agents.
```

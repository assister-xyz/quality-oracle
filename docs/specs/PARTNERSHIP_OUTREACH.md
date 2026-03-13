# Partnership Outreach Templates

## 1. MCP Evals (Mat Lenhard) — #1 Target

**Channel:** GitHub issue or DM

**Subject:** Complementary tools — quality verification + eval scoring

**Message:**
```
Hi Mat,

I've been following mcp-evals and love what you're building for tool-level evaluation.

I built AgentTrust (https://github.com/vitaliiserbynassisterr/quality-oracle) — it does
agent-level competency verification via challenge-response testing. 6-axis scoring,
consensus judging, battle arena, IRT adaptive testing.

Our tools are complementary, not competitive:
- mcp-evals: scores individual tool correctness (accuracy, completeness, relevance)
- AgentTrust: certifies overall agent competency (safety, reliability, adversarial robustness)

Would love to explore integration — e.g., mcp-evals scores could feed into AgentTrust's
scoring pipeline, or we could cross-reference results for higher confidence.

533 tests passing, MIT license, built on FastMCP.
```

---

## 2. AgentAudit (ecap0/starbuck100)

**Channel:** DEV Community comment or GitHub issue

**Subject:** Security + Quality = complete trust profile

**Message:**
```
Hi! Great work on the AgentAudit registry and multi-agent consensus auditing.

I built AgentTrust — it verifies functional competency (not just security) of MCP
servers and AI agents via challenge-response testing. We use a similar multi-judge
consensus approach (2-3 LLM judges).

Combined, our tools create a complete trust profile:
- AgentAudit → Is the code safe? (security trust score)
- AgentTrust → Does it actually work well? (competency trust score)

We could cross-reference scores, or build a joint "full trust profile" badge.
Would love to discuss.

Project: https://github.com/vitaliiserbynassisterr/quality-oracle
```

---

## 3. PulseMCP (orliesaurus)

**Channel:** GitHub issue or Twitter/X DM

**Subject:** Quality badges for PulseMCP listings

**Message:**
```
Hey! PulseMCP is a great aggregator for MCP servers.

I built AgentTrust — it evaluates MCP server quality via automated testing.
Would be cool to display quality scores/badges alongside PulseMCP listings,
so users can see at a glance which servers are actually reliable.

We already generate SVG badges (Apple-inspired design) and issue W3C
Verifiable Credentials. Could integrate as a simple API call from your indexer.

Open to any form of collaboration. MIT license.

https://github.com/vitaliiserbynassisterr/quality-oracle
```

---

## 4. MCPReady.ai

**Channel:** Email (info@mcpready.ai)

**Subject:** Joint certification — compliance + quality

**Message:**
```
Hi MCPReady team,

You're building MCP compliance certification. I built AgentTrust, which does
functional quality verification — challenge-response testing, adversarial probes,
6-axis scoring.

Our approaches are complementary:
- MCPReady: "Does this server follow MCP specs correctly?" (compliance)
- AgentTrust: "Does this server produce high-quality results?" (quality)

Could explore a joint certification badge — "MCP Ready + Quality Verified."

https://github.com/vitaliiserbynassisterr/quality-oracle
```

---

## 5. ZARQ.ai / Nerq

**Channel:** DEV Community or Twitter/X

**Subject:** Competency scores for your trust index

**Message:**
```
Impressive work on the State of AI Assets census — 143K agents scored!

I built AgentTrust, which does challenge-response competency verification
for MCP servers and AI agents. We could contribute functional quality scores
to enrich your trust index beyond the current signals (security, maintenance,
popularity, docs).

Our scores come from actual testing — LLM judge consensus on tool responses,
adversarial probes, and IRT-calibrated question banks.

https://github.com/vitaliiserbynassisterr/quality-oracle
```

---

## 6. DEV Community Article

**Title:** "How We Score MCP Server Quality Before Payment"

**Outline:**
1. The problem: 17K+ MCP servers, no quality gate
2. Our approach: challenge-response testing
3. 6-axis scoring explained (with examples)
4. Consensus judging: why 2-3 judges > 1
5. IRT adaptive testing: psychometrics for AI
6. Battle arena: competitive benchmarking
7. W3C VCs: machine-readable proof
8. Try it: `pip install mcp-agenttrust`

**Tags:** mcp, ai, evaluation, quality, agents

---

## Outreach Checklist

- [ ] Repo made public
- [ ] MCP Evals → GitHub issue/DM
- [ ] AgentAudit → DEV Community comment
- [ ] PulseMCP → GitHub issue
- [ ] MCPReady → Email
- [ ] ZARQ → DEV Community/Twitter
- [ ] DEV Community article published
- [ ] MCP Discord #showcase post

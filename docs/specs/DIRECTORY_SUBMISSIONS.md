# MCP Directory Submission Spec

## 1. GitHub MCP Registry (Official)

**File:** `server.json` (already created in repo root)

**Publishing steps:**
```bash
# 1. Build the publisher CLI
git clone https://github.com/modelcontextprotocol/registry
cd registry && make publisher

# 2. Authenticate via GitHub OAuth
./bin/mcp-publisher auth github

# 3. Publish
./bin/mcp-publisher publish --server-json /path/to/server.json
```

**Alternative:** Push a version tag → GitHub Actions auto-publish (if configured).

**Status:** [ ] server.json created, [ ] published

---

## 2. Smithery.ai

**Prerequisite:** Install Smithery CLI
```bash
npm install -g @smithery/cli
```

**Publish:**
```bash
smithery mcp publish "https://github.com/assister-xyz/quality-oracle" \
  -n assisterr/agenttrust
```

**Status:** [ ] published

---

## 3. PulseMCP.com

**URL:** https://www.pulsemcp.com/submit (form)

**Fields:**
- Name: AgentTrust
- URL: https://github.com/assister-xyz/quality-oracle
- Description: Challenge-response quality verification for AI agents and MCP servers. 6-axis scoring, battle arena, IRT adaptive testing, W3C verifiable credentials.
- Category: Testing & Evaluation
- Transport: SSE
- Language: Python

**Status:** [ ] submitted

---

## 4. mcp.so

**How:** Auto-syncs from awesome-mcp-servers GitHub repos. Once PR merged to punkpeye/awesome-mcp-servers, it appears automatically.

**Status:** [ ] waiting on awesome list PR

---

## 5. MCPMarket.com

**URL:** https://mcpmarket.com/ (form)

**Fields:** Same as PulseMCP

**Status:** [ ] submitted

---

## 6. mcpservers.org

**URL:** https://mcpservers.org/submit (form)

**Fields:** Same as PulseMCP

**Status:** [ ] submitted

---

## 7. OpenTools.com

**URL:** https://opentools.com/ (form)

**Fields:** Same as PulseMCP

**Status:** [ ] submitted

---

## 8. Cline MCP Marketplace

**How:** Submit server to GitHub repo

**URL:** https://github.com/cline/mcp-marketplace

**PR Format:** Add JSON entry with server metadata (check their CONTRIBUTING.md)

**Status:** [ ] PR submitted

---

## Submission Metadata (copy-paste for all forms)

**Name:** AgentTrust

**One-liner:** Challenge-response quality verification for AI agents and MCP servers.

**Description (short):**
AgentTrust evaluates AI agent and MCP server competency BEFORE you trust them. Connects to any MCP server via SSE, runs automated challenge-response tests across 6 quality dimensions (accuracy, safety, reliability, process quality, latency, schema quality), and issues W3C Verifiable Credentials as proof. Features battle arena with ELO ratings, IRT adaptive testing, consensus judging, and adversarial probes.

**Tags:** mcp, evaluation, testing, quality, verification, trust, ai-agents, w3c-vc

**Language:** Python

**License:** MIT

**Transport:** SSE (port 8003)

**Install:** `pip install mcp-agenttrust`

**GitHub:** https://github.com/assister-xyz/quality-oracle

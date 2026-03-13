# AgentTrust Distribution Roadmap

> Realistic partners and distribution channels for AgentTrust, prioritized by response probability for a small team.

## Key Insight

No MCP directory (82K+ stars combined) currently has a **quality verification / competency testing** MCP server. AgentTrust creates a new category rather than competing in an existing one.

---

## TIER 1: Highest Probability (This Week)

### 1.1 GitHub Awesome Lists (PR submissions, free, ~100% acceptance)

| Repo | Stars | Category | Status |
|------|-------|----------|--------|
| [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) | 82.9K | Testing & Evaluation | [ ] PR submitted |
| [punkpeye/awesome-mcp-devtools](https://github.com/punkpeye/awesome-mcp-devtools) | 430 | Evaluation | [ ] PR submitted |
| [e2b-dev/awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents) | 26.4K | Evaluation Tools | [ ] PR submitted |
| [appcypher/awesome-mcp-servers](https://github.com/appcypher/awesome-mcp-servers) | 5.2K | Testing | [ ] PR submitted |
| [kyrolabs/awesome-agents](https://github.com/kyrolabs/awesome-agents) | ~1.5K | Testing | [ ] PR submitted |
| [ottosulin/awesome-ai-security](https://github.com/ottosulin/awesome-ai-security) | - | Agent Testing | [ ] PR submitted |

**Prerequisites:** Public repo with README, clear description, working install instructions.

### 1.2 MCP Directory Submissions (form/CLI, free, ~95% acceptance)

| Platform | Servers Listed | How | Status |
|----------|---------------|-----|--------|
| [GitHub MCP Registry](https://github.com/modelcontextprotocol/registry) | Official | `server.json` + CLI publish | [ ] Published |
| [Glama.ai](https://glama.ai/mcp/servers) | 19K+ | Auto-syncs from awesome-mcp-servers | [ ] Listed |
| [mcp.so](https://mcp.so) | 18.4K+ | Auto-syncs from GitHub repos | [ ] Listed |
| [Smithery.ai](https://smithery.ai/) | 3.3K+ | `smithery mcp publish` CLI | [ ] Published |
| [PulseMCP.com](https://www.pulsemcp.com/servers) | 9K+ | Form at /submit | [ ] Submitted |
| [MCPMarket.com](https://mcpmarket.com/) | - | Form submission | [ ] Submitted |
| [mcpservers.org](https://mcpservers.org/) | - | Form at /submit | [ ] Submitted |
| [OpenTools.com](https://opentools.com/) | - | Form submission | [ ] Submitted |

**Prerequisites:** `server.json` in repo root, public GitHub repo.

### 1.3 Web2 Agent Directories (form, free)

| Platform | Agents Listed | Status |
|----------|--------------|--------|
| [aiagentsdirectory.com](https://aiagentsdirectory.com/submit-agent) | 2,218+ | [ ] Submitted |
| [aiagentstore.ai](https://aiagentstore.ai/) | - | [ ] Submitted |
| [agent.ai](https://agent.ai/) | "#1 AI Agent Network" | [ ] Submitted |

### 1.4 Hackathons

| Event | Prizes | Deadline | Status |
|-------|--------|----------|--------|
| [World's Biggest MCP Hackathon](https://biggest-mcp-hackathon.devpost.com/) | At YC, Anthropic-sponsored | May 17, 2026 | [ ] Registered |
| [AWS MCP & A2A Hackathon](https://mcp-and-a2a-hackathon.devpost.com/) | AWS prizes | Check Devpost | [ ] Registered |
| [GitLab AI Hackathon](https://gitlab.devpost.com/) | $27,000 | March 25, 2026 | [ ] Submitted |
| [AWS AI Agent Global Hackathon](https://aws-agent-hackathon.devpost.com/) | $45,000 | Check Devpost | [ ] Registered |
| [HuggingFace Agents-MCP](https://huggingface.co/Agents-MCP-Hackathon) | Community | Ongoing | [ ] Published Space |

---

## TIER 2: High Probability (Weeks 2-3)

### 2.1 Partnership Targets (small teams, will respond)

#### MCP Evals (mcpevals.io) — #1 Partnership Target
- **Maintainer:** Mat Lenhard (solo dev)
- **GitHub:** [mclenhard/mcp-evals](https://github.com/mclenhard/mcp-evals) — 125 stars
- **What they do:** Tool-level eval (accuracy, completeness, relevance scoring)
- **Complementary:** They score tools, we certify agents. Not competitive.
- **Pitch:** "mcp-evals scores your tools, AgentTrust certifies your agent"
- **Contact:** GitHub issues/DM
- **Status:** [ ] Outreach sent

#### AgentAudit (agentaudit.dev)
- **Maintainer:** ecap0/starbuck100 (small team)
- **GitHub:** [agentaudit-dev/agentaudit-skill](https://github.com/agentaudit-dev/agentaudit-skill) — 4 stars
- **What they do:** Security trust scores for MCP packages (CVE-like registry, multi-agent consensus audit)
- **Complementary:** They verify security, we verify competency. Combined = full trust profile.
- **Pitch:** "AgentAudit = security + AgentTrust = quality = complete trust profile"
- **Contact:** DEV Community, GitHub
- **Status:** [ ] Outreach sent

#### PulseMCP / orliesaurus
- **Maintainer:** orliesaurus (solo dev)
- **What they do:** MCP server aggregator (9K+ servers)
- **Pitch:** Display AgentTrust quality badges alongside server listings
- **Status:** [ ] Outreach sent

#### MCPReady.ai
- **What they do:** MCP certification platform, issues "MCP READY" badges
- **Pitch:** Joint certification standard (their compliance + our quality)
- **Status:** [ ] Outreach sent

#### ZARQ.ai / Nerq
- **What they do:** Trust scoring for 143K agents + 17K MCP servers, Moody's-style ratings
- **Pitch:** Integrate competency scores into their trust index
- **Status:** [ ] Outreach sent

### 2.2 Content Marketing (parallel)

| Channel | Action | Status |
|---------|--------|--------|
| DEV Community | Article: "How We Score MCP Agent Quality Before Payment" | [ ] Published |
| MCP Discord (11.5K members) | Post in #showcase | [ ] Posted |
| HuggingFace | Publish demo Space | [ ] Published |
| Reddit r/MCP, r/AIAgents | Share demo | [ ] Posted |

---

## TIER 3: Medium Probability (Month 1-2)

### 3.1 Framework Integrations

#### SendAI / Solana Agent Kit
- **GitHub:** [sendaifun/solana-agent-kit](https://github.com/sendaifun/solana-agent-kit) — 1.6K stars
- **Integration:** `@solana-agent-kit/plugin-agenttrust` — quality verification for Solana agents
- **Also:** Register on [solanaskills.com](https://www.solanaskills.com/) and [openSVM/aeamcp](https://github.com/openSVM/aeamcp) on-chain registry
- **Status:** [ ] Plugin built

#### ElizaOS
- **GitHub:** [elizaOS/eliza](https://github.com/elizaOS/eliza) — 17.5K stars
- **Integration:** `plugin-agenttrust` as ElizaOS npm plugin, using [plugin-starter template](https://github.com/elizaOS/eliza-plugin-starter)
- **Why:** Powers $20B+ in Web3 agent projects, Solana-native
- **Status:** [ ] Plugin built

#### CrewAI
- **GitHub:** [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) — 45.9K stars
- **Integration:** CrewAI Tool wrapping AgentTrust eval — pre-delegation quality gate
- **Also:** Submit to [awesome-crewai](https://github.com/crewAIInc/awesome-crewai)
- **Status:** [ ] Tool built

#### LangChain Community
- **GitHub:** [langchain-ai/langchain](https://github.com/langchain-ai/langchain) — 129K stars
- **Integration:** `AgentTrustEvaluator` tool PR to langchain-community
- **Note:** They explicitly say "extremely open to contributions"
- **Status:** [ ] PR submitted

#### LlamaIndex
- **GitHub:** [run-llama/llama_index](https://github.com/run-llama/llama_index) — 47.6K stars
- **Integration:** LlamaIndex Tool for agentic RAG quality verification
- **LlamaHub:** [llamahub.ai](https://llamahub.ai/) — 300+ integrations
- **Status:** [ ] Integration built

### 3.2 Evaluation Tool Integrations

#### NVIDIA garak
- **GitHub:** [NVIDIA/garak](https://github.com/NVIDIA/garak) — 7.2K stars
- **Integration:** Contribute agent-specific adversarial probes (we already have 5 types)
- **Status:** [ ] Probes contributed

#### promptfoo
- **GitHub:** [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) — 11-13K stars
- **Integration:** AgentTrust's 6-axis scoring as promptfoo grader plugin
- **Status:** [ ] Plugin built

#### Cline MCP Marketplace
- **GitHub:** [cline/mcp-marketplace](https://github.com/cline/mcp-marketplace) — 755 stars, 741 open issues
- **Note:** 4M+ developers, desperately needs quality curation (741 open issues!)
- **Pitch:** Quality scoring for marketplace submissions
- **Status:** [ ] Server submitted

### 3.3 Standards Participation

#### Official MCP Registry
- **GitHub:** [modelcontextprotocol/registry](https://github.com/modelcontextprotocol/registry) — 6.5K stars
- **Action:** Propose quality metadata in upcoming "MCP Server Cards" spec
- **Status:** [ ] Proposal submitted

#### NIST AI Agent Standards Initiative
- **Announced:** February 17, 2026
- **Deadline:** AI Agent Identity & Authorization Concept Paper — **April 2, 2026**
- **Action:** Submit AQVC format as candidate standard for agent quality attestation
- **URL:** [nist.gov/caisi/ai-agent-standards-initiative](https://www.nist.gov/caisi/ai-agent-standards-initiative)
- **Status:** [ ] Concept paper submitted

#### OWASP GenAI Security Project
- **URL:** [genai.owasp.org/contribute/](https://genai.owasp.org/contribute/)
- **Action:** Join Slack, attend open meetings, contribute agent testing methodology
- **Status:** [ ] Joined

---

## TIER 4: Not Now (won't respond to cold outreach)

| Target | Why Skip |
|--------|----------|
| Composio (27K stars) | VC-backed, enterprise focus |
| Toolhouse.ai | Backed by Cloudflare/NVIDIA |
| Cursor / Windsurf / Replit | Product companies, not accepting integrations |
| Scorecard.io | Enterprise partnerships only |
| Microsoft / Google / AWS | Enterprise sales cycles |
| FastMCP (23.6K stars) | Lowin is busy, maybe later via PR |

---

## Prerequisites Checklist

Before starting any outreach:

- [ ] **README.md** — Project description, features, quick start, badges
- [ ] **server.json** — GitHub MCP Registry format
- [ ] **mcp-server/ package** — Fix stub, make installable
- [ ] **Public repo** — Currently private, needs to go public for awesome list PRs
- [ ] **Demo URL** — Live instance or demo video
- [ ] **One-liner description** — For awesome list entries
- [ ] **All tests passing** — Credibility

## One-Liner (for awesome list PRs)

> **AgentTrust** — Challenge-response quality verification for AI agents and MCP servers. 6-axis scoring, battle arena, IRT adaptive testing, W3C verifiable credentials. `pip install mcp-agenttrust`

## Short Description (for directory submissions)

> AgentTrust evaluates AI agent and MCP server competency BEFORE you trust them with real tasks or payments. It connects to any MCP server, runs challenge-response tests across 6 quality dimensions (accuracy, safety, reliability, process quality, latency, schema quality), and issues W3C Verifiable Credentials as proof. Features include battle arena with ELO ratings, IRT adaptive question calibration, adversarial probes, consensus judging (2-3 LLM judges), and x402 Solana payment verification.

# Awesome List PR Template

## Target Repos & Entry Format

### 1. punkpeye/awesome-mcp-servers (82.9K stars)

**Category:** Testing & Quality Assurance (or create "Evaluation" section)

**Entry:**
```markdown
- [AgentTrust](https://github.com/assister-xyz/quality-oracle) - Challenge-response quality verification for MCP servers. 6-axis scoring (accuracy, safety, reliability), battle arena with ELO ratings, IRT adaptive testing, W3C verifiable credentials. `pip install mcp-agenttrust`
```

**PR Title:** `Add AgentTrust - MCP server quality verification`

**PR Body:**
```
AgentTrust evaluates MCP server competency before trusting them with real tasks.

Features:
- 3-level evaluation: manifest → functional → domain expert
- 6-axis scoring with consensus judging (2-3 LLM judges)
- Battle arena with OpenSkill ratings
- W3C Verifiable Credentials (AQVC format)
- Available as MCP server itself (`pip install mcp-agenttrust`)

533 tests passing, MIT license.
```

### 2. punkpeye/awesome-mcp-devtools (430 stars)

**Category:** Testing

**Entry:**
```markdown
- [AgentTrust](https://github.com/assister-xyz/quality-oracle) - Quality verification service for MCP servers. Automated challenge-response testing with LLM judge consensus, adversarial probes, and IRT adaptive question calibration.
```

### 3. e2b-dev/awesome-ai-agents (26.4K stars)

**Category:** Evaluation / Testing Tools

**Entry:**
```markdown
| [AgentTrust](https://github.com/assister-xyz/quality-oracle) | Challenge-response quality verification for AI agents and MCP servers. 6-axis scoring, battle arena, adaptive testing, W3C verifiable credentials. | ![GitHub stars](https://img.shields.io/github/stars/assister-xyz/quality-oracle) |
```

### 4. appcypher/awesome-mcp-servers (5.2K stars)

**Entry:**
```markdown
- [AgentTrust](https://github.com/assister-xyz/quality-oracle) <img alt="Python" src="https://img.shields.io/badge/Python-blue"> - MCP server quality verification with challenge-response testing, LLM judge consensus, and W3C verifiable credentials.
```

### 5. kyrolabs/awesome-agents (1.5K stars)

**Entry:**
```markdown
- [AgentTrust](https://github.com/assister-xyz/quality-oracle) - Pre-payment quality gate for AI agents. Evaluates competency via challenge-response testing before delegation. ![Stars](https://img.shields.io/github/stars/assister-xyz/quality-oracle)
```

## Prerequisites Before Submitting PRs

- [x] README.md created
- [ ] Repo made PUBLIC on GitHub
- [ ] At least 1 GitHub star (self-star)
- [ ] License file present (MIT)
- [ ] Working installation instructions verified
- [ ] Demo screenshot or GIF (optional but helps)

## Submission Order

1. awesome-mcp-devtools (smallest, fastest merge) → warm up
2. appcypher/awesome-mcp-servers → moderate size
3. kyrolabs/awesome-agents → moderate size
4. e2b-dev/awesome-ai-agents → large, well-maintained
5. punkpeye/awesome-mcp-servers → largest, most visibility

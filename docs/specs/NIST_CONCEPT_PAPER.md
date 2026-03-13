# NIST AI Agent Standards Initiative — Concept Paper Submission

**Deadline:** April 2, 2026
**Topic:** AI Agent Identity and Authorization
**URL:** https://www.nist.gov/caisi/ai-agent-standards-initiative

## What to Submit

NIST's ITL published a concept paper on "AI Agent Identity and Authorization" and is accepting
responses. The AQVC (Agent Quality Verifiable Credential) format directly addresses their goals.

## Key Points to Make

### 1. Gap in Current Standards

Current AI agent standards address:
- **Identity:** ERC-8004, SATI, DID methods
- **Communication:** Google A2A, Anthropic MCP
- **Payment:** x402 (HTTP 402)

Missing: **Quality attestation** — standardized, machine-readable proof that an agent is competent.

### 2. AQVC as Candidate Standard

AQVC (Agent Quality Verifiable Credential) provides:
- W3C VC v2.0 format (interoperable with existing VC infrastructure)
- Ed25519 DataIntegrityProof (fast, secure, no blockchain required)
- 6-axis quality dimensions (accuracy, safety, reliability, process quality, latency, schema quality)
- Machine-readable quality scores that agents can present to other agents
- Expiration and revocation support

### 3. Challenge-Response Verification

Unlike self-reported quality claims, AQVC credentials are issued by an independent evaluator
that actually tests the agent's capabilities through:
- Automated tool invocation and response analysis
- Multi-judge consensus (reduces false positives)
- Adversarial probes (tests robustness)
- Adaptive question calibration (IRT psychometric models)

### 4. Alignment with NIST Pillars

| NIST Pillar | AgentTrust Contribution |
|-------------|------------------------|
| Standards | AQVC format for quality attestation |
| Open Source | MIT licensed, MCP-native, A2A-compatible |
| Security | Adversarial probes, anti-gaming, sandbagging detection |

### 5. Interoperability

AQVC maps into existing protocols:
- **A2A:** Extension on Agent Card with quality scores
- **MCP:** Available as MCP server tool
- **ERC-8004:** Can be stored in ValidationRegistry
- **SATI:** Extends identity with quality dimension
- **x402:** Quality-based pricing (higher quality = higher trust = different pricing)

## Submission Format

Follow ITL's concept paper response format (check NIST website for template).
Key sections likely include:
1. Problem statement
2. Proposed approach
3. Technical specification
4. Implementation evidence (link to working code)
5. Interoperability considerations

## Action Items

- [ ] Download NIST concept paper template (check nist.gov/caisi)
- [ ] Draft response (2-5 pages)
- [ ] Include link to GitHub repo + Architecture doc
- [ ] Submit before April 2, 2026

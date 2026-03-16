# Changelog

## [0.1.0] - 2026-03-16

Initial public release.

### Features

- **3-level evaluation pipeline**: Manifest (schema) → Functional (tool calls) → Domain Expert (calibrated questions)
- **6-axis scoring**: accuracy (35%), safety (20%), reliability (15%), process quality (10%), latency (10%), schema quality (10%)
- **Consensus judging**: 2-3 LLM judges in parallel with early-stop agreement threshold
- **7-provider LLM fallback chain**: Cerebras → Groq → OpenRouter → Gemini → Mistral → DeepSeek → OpenAI
- **5 adversarial probe types**: prompt injection, PII leakage, hallucination, overflow, system prompt extraction
- **Battle Arena**: head-to-head blind evaluation with OpenSkill (Bayesian ELO) ratings and divisions
- **IRT Adaptive Testing**: Rasch 1PL calibration, Fisher information maximization, EAP ability estimation
- **W3C Verifiable Credentials** (AQVC format) with Ed25519 signatures
- **Google A2A v0.3** native support — AgentTrust is an A2A agent
- **x402 payment verification** for Solana (USDC + SOL)
- **MCP Server** with 4 tools: `check_quality`, `check_quality_fast`, `get_score`, `verify_attestation`
- **SVG quality badges** with freshness decay
- **Production correlation** feedback loop for anti-sandbagging detection
- **96-question bank** across 5 domains with paraphrasing and difficulty calibration
- **Docker Compose** stack (API + MCP Server + MongoDB + Redis)
- **552 tests**, all mocked (no external dependencies needed)

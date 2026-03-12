"""
Domain-specific question pools for AgentTrust evaluations.

Ported from agent-poi hackathon (poi/question_pools.py + poi/evaluator.py).
Adapted: domain-agnostic structure for MCP server evaluation.
"""
import hashlib
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ChallengeQuestion:
    """A domain-specific challenge question with reference answer."""
    question: str
    domain: str
    difficulty: str  # easy, medium, hard
    reference_answer: str
    category: str = "functional"  # functional, correctness, edge_case

    @property
    def id(self) -> str:
        return hashlib.sha256(self.question.encode()).hexdigest()[:12]

    @property
    def weight(self) -> int:
        return {"easy": 1, "medium": 2, "hard": 3}.get(self.difficulty, 1)


# Question pools organized by domain
QUESTION_POOLS: Dict[str, List[ChallengeQuestion]] = {
    "defi": [
        ChallengeQuestion(
            question="Explain how an AMM determines token prices using the constant product formula.",
            domain="defi", difficulty="medium",
            reference_answer="AMMs use x*y=k where x and y are token reserves. Price is the ratio of reserves. As one token is bought, its reserve decreases and price increases.",
        ),
        ChallengeQuestion(
            question="What is impermanent loss and when does it occur in liquidity pools?",
            domain="defi", difficulty="medium",
            reference_answer="Impermanent loss occurs when the price ratio of pooled tokens changes from deposit. The larger the divergence, the more IL. Becomes permanent only when withdrawn.",
        ),
        ChallengeQuestion(
            question="Describe how flash loans work and why they don't require collateral.",
            domain="defi", difficulty="hard",
            reference_answer="Flash loans are uncollateralized loans borrowed and repaid within a single transaction. If not repaid, the entire transaction reverts atomically.",
        ),
        ChallengeQuestion(
            question="What is TVL and why is it an important DeFi metric?",
            domain="defi", difficulty="easy",
            reference_answer="TVL measures total crypto assets deposited in DeFi protocols. Indicates adoption, trust, and available liquidity.",
        ),
        ChallengeQuestion(
            question="How does concentrated liquidity improve capital efficiency?",
            domain="defi", difficulty="hard",
            reference_answer="Concentrated liquidity allocates capital within custom price ranges instead of full range. Provides 10-20x capital efficiency but requires active management.",
        ),
        ChallengeQuestion(
            question="What is a yield aggregator and how does it optimize returns across protocols?",
            domain="defi", difficulty="medium",
            reference_answer="Yield aggregators auto-compound and rebalance deposits across lending/LP protocols to maximize APY. Examples: Yearn vaults. They abstract complexity but add smart contract risk layers.",
        ),
        ChallengeQuestion(
            question="Explain how oracle manipulation attacks work in DeFi lending protocols.",
            domain="defi", difficulty="hard",
            reference_answer="Attacker manipulates price oracle (e.g., low-liquidity AMM) to misrepresent collateral value, then borrows more than collateral is worth. Defense: TWAP oracles, multiple sources, Chainlink.",
        ),
        ChallengeQuestion(
            question="What is a liquidity bootstrap pool and how does it differ from a standard AMM pool?",
            domain="defi", difficulty="medium",
            reference_answer="LBP uses time-weighted shifting ratios (e.g., 90/10 → 50/50) for fair price discovery during token launches. Discourages front-running and whale sniping unlike fixed-ratio pools.",
        ),
    ],
    "solana": [
        ChallengeQuestion(
            question="What are PDAs in Solana and how are they created?",
            domain="solana", difficulty="medium",
            reference_answer="PDAs are deterministic addresses derived from program ID and seeds that fall off Ed25519 curve. Created using findProgramAddress. Enable programs to sign without private keys.",
        ),
        ChallengeQuestion(
            question="Explain how CPI works in Solana and its constraints.",
            domain="solana", difficulty="medium",
            reference_answer="CPI allows one program to call another's instructions. Passes required accounts, callee inherits signer privileges. Depth limited to 4 levels.",
        ),
        ChallengeQuestion(
            question="How does Solana's rent system work?",
            domain="solana", difficulty="easy",
            reference_answer="Solana charges rent for storage. Accounts must hold minimum balance (~2 years rent) to be rent-exempt. Below threshold, accounts are garbage collected.",
        ),
        ChallengeQuestion(
            question="Explain Solana's Proof of History consensus mechanism.",
            domain="solana", difficulty="hard",
            reference_answer="PoH uses sequential SHA-256 hash chain creating a verifiable delay function. Establishes temporal ordering before consensus, enabling high throughput.",
        ),
        ChallengeQuestion(
            question="What is the Solana Token Program and how do SPL tokens work?",
            domain="solana", difficulty="easy",
            reference_answer="SPL Token Program is the standard for fungible/non-fungible tokens on Solana. Tokens are stored in associated token accounts (ATAs). Uses mint authority and freeze authority for governance.",
        ),
        ChallengeQuestion(
            question="How does Solana handle transaction fees and priority fees?",
            domain="solana", difficulty="medium",
            reference_answer="Base fee is 5000 lamports per signature. Priority fees (compute unit price) bid for block inclusion. 50% burned, 50% to validator. Higher priority = faster inclusion during congestion.",
        ),
        ChallengeQuestion(
            question="What are Solana lookup tables and how do they reduce transaction size?",
            domain="solana", difficulty="hard",
            reference_answer="Address Lookup Tables (ALTs) store account addresses on-chain, letting transactions reference them by index instead of full 32-byte pubkey. Reduces transaction size from 1232 to ~200 bytes for multi-account txs.",
        ),
        ChallengeQuestion(
            question="Explain the difference between Solana accounts and Ethereum storage.",
            domain="solana", difficulty="medium",
            reference_answer="Solana uses explicit account model — all state in separate accounts owned by programs. Ethereum uses implicit storage slots within contracts. Solana requires declaring all accounts upfront, enabling parallel execution.",
        ),
    ],
    "security": [
        ChallengeQuestion(
            question="What is a reentrancy attack and how can it be prevented?",
            domain="security", difficulty="medium",
            reference_answer="Reentrancy occurs when external call re-enters calling function before state updates. Prevention: checks-effects-interactions pattern, reentrancy guards.",
        ),
        ChallengeQuestion(
            question="How does a sandwich attack exploit pending transactions?",
            domain="security", difficulty="hard",
            reference_answer="Front-runs victim's swap with buy order, victim executes at inflated price, attacker sells after. MEV bots monitor mempool.",
        ),
        ChallengeQuestion(
            question="What is a Sybil attack and how do decentralized systems defend?",
            domain="security", difficulty="medium",
            reference_answer="Creates multiple fake identities for disproportionate influence. Defenses: proof-of-stake, proof-of-work, reputation systems, identity verification.",
        ),
        ChallengeQuestion(
            question="What is a rugpull and what on-chain indicators can detect one?",
            domain="security", difficulty="easy",
            reference_answer="Rugpull: creator drains liquidity or mints tokens, crashing price. Indicators: unlocked LP tokens, mint authority retained, no timelock on admin functions, honeypot sell restrictions.",
        ),
        ChallengeQuestion(
            question="Explain how access control vulnerabilities in smart contracts are exploited.",
            domain="security", difficulty="medium",
            reference_answer="Missing or incorrect access modifiers let unauthorized users call admin functions (mint, pause, withdraw). Common in proxy patterns where initializer can be re-called. Defense: OpenZeppelin AccessControl, Ownable.",
        ),
        ChallengeQuestion(
            question="What is a cross-chain bridge exploit and why are bridges high-value targets?",
            domain="security", difficulty="hard",
            reference_answer="Bridges lock assets on chain A and mint on chain B. Exploits target validator sets, signature verification, or proof validation. High value because they hold large TVL. Examples: Wormhole ($320M), Ronin ($625M).",
        ),
        ChallengeQuestion(
            question="How does transaction simulation help prevent security exploits?",
            domain="security", difficulty="medium",
            reference_answer="Simulation executes transaction against current state without committing. Reveals unexpected token transfers, approval changes, or fund movements before signing. Used by wallets like Phantom and Blowfish.",
        ),
        ChallengeQuestion(
            question="What is a governance attack and how can DAOs defend against it?",
            domain="security", difficulty="hard",
            reference_answer="Attacker acquires enough governance tokens (often via flash loan) to pass malicious proposals. Defenses: time-locked execution, quorum requirements, vote escrow, snapshot-based voting, flash loan guards.",
        ),
    ],
    "code-generation": [
        ChallengeQuestion(
            question="Generate a Python function that checks if a string is a valid palindrome, ignoring spaces and punctuation.",
            domain="code-generation", difficulty="easy",
            reference_answer="Function should normalize string (lowercase, remove non-alphanumeric), then compare with reversed version. Handle empty strings and single characters.",
        ),
        ChallengeQuestion(
            question="Implement a rate limiter using the token bucket algorithm in Python.",
            domain="code-generation", difficulty="hard",
            reference_answer="Token bucket: initialize with capacity and refill rate. consume() checks tokens available, refills based on elapsed time. Thread-safe with locks.",
        ),
        ChallengeQuestion(
            question="Write a Python function to merge two sorted lists into one sorted list without using built-in sort.",
            domain="code-generation", difficulty="easy",
            reference_answer="Two-pointer approach: compare heads of both lists, append smaller. Handle remaining elements. O(n+m) time, O(n+m) space.",
        ),
        ChallengeQuestion(
            question="Implement an LRU cache with O(1) get and put operations in Python.",
            domain="code-generation", difficulty="hard",
            reference_answer="Use OrderedDict or doubly-linked list + dict. get() moves to front. put() evicts LRU when full. Both O(1) via hash map + linked list.",
        ),
        ChallengeQuestion(
            question="Write a Python decorator that retries a function up to N times on exception with exponential backoff.",
            domain="code-generation", difficulty="medium",
            reference_answer="Decorator accepts max_retries, base_delay. Wraps function in try/except loop, sleeps base_delay * 2^attempt between retries. Re-raises on final failure.",
        ),
        ChallengeQuestion(
            question="Generate a Python async context manager for database connection pooling.",
            domain="code-generation", difficulty="hard",
            reference_answer="Use __aenter__/__aexit__ or @asynccontextmanager. Acquire connection from pool on enter, release on exit. Handle exceptions to avoid connection leaks. Support max_size and timeout.",
        ),
        ChallengeQuestion(
            question="Write a Python function that validates an email address using regex.",
            domain="code-generation", difficulty="easy",
            reference_answer="Use re.match with pattern for local@domain.tld. Handle edge cases: dots, plus signs, subdomains. Should reject missing @, empty local, invalid TLD.",
        ),
        ChallengeQuestion(
            question="Implement a simple pub/sub event system in Python with type-safe event handlers.",
            domain="code-generation", difficulty="medium",
            reference_answer="EventBus class with subscribe(event_type, handler) and publish(event). Use dict of event_type → list of callbacks. Optional: async handlers, type hints with Generic/Protocol.",
        ),
    ],
    "general": [
        ChallengeQuestion(
            question="What is the A2A protocol and why is it important for AI agents?",
            domain="general", difficulty="easy",
            reference_answer="A2A enables AI agents to discover, communicate, and verify each other through standardized HTTP endpoints. Creates interoperable agent network.",
        ),
        ChallengeQuestion(
            question="Explain the difference between pre-payment and post-payment quality verification for AI agents.",
            domain="general", difficulty="medium",
            reference_answer="Pre-payment: verify competency BEFORE paying (challenge-response, benchmarks). Post-payment: observe after use (ratings, reputation). Pre-payment prevents paying for low-quality service.",
        ),
        ChallengeQuestion(
            question="What is a Verifiable Credential and how does it apply to AI agent identity?",
            domain="general", difficulty="medium",
            reference_answer="W3C VC is a cryptographically signed claim about a subject (issuer → holder → verifier). For agents: issuer attests quality score, agent holds VC, client verifies before delegation.",
        ),
        ChallengeQuestion(
            question="What is prompt engineering and why is it critical for AI agent quality?",
            domain="general", difficulty="easy",
            reference_answer="Crafting inputs to get optimal LLM outputs. Critical for agents: system prompts define behavior, tool descriptions affect selection accuracy, few-shot examples improve consistency.",
        ),
        ChallengeQuestion(
            question="Explain the concept of AI agent hallucination and its risks in production.",
            domain="general", difficulty="medium",
            reference_answer="Hallucination: model generates plausible but factually wrong content. Risks: incorrect financial advice, fabricated data, fake references. Mitigation: RAG, fact-checking, confidence scoring.",
        ),
        ChallengeQuestion(
            question="What is the EU AI Act and how does it affect AI agent deployment?",
            domain="general", difficulty="hard",
            reference_answer="EU regulation classifying AI by risk (unacceptable/high/limited/minimal). High-risk: requires conformity assessment, documentation, monitoring. Agents making decisions affecting people may be high-risk. Deadline: Aug 2026.",
        ),
        ChallengeQuestion(
            question="How do embeddings work and why are they important for AI applications?",
            domain="general", difficulty="easy",
            reference_answer="Embeddings map text/data to dense vectors where similar items are close. Enable semantic search, RAG, clustering, recommendation. Models: text-embedding-3, BGE, E5. Stored in vector DBs like Qdrant.",
        ),
        ChallengeQuestion(
            question="What is retrieval-augmented generation (RAG) and when should it be used?",
            domain="general", difficulty="medium",
            reference_answer="RAG retrieves relevant documents from a knowledge base and includes them in LLM context. Use when: data changes frequently, domain-specific knowledge needed, reducing hallucination. Components: embeddings, vector DB, retriever, generator.",
        ),
    ],

    # ── MCP-Specific Domains (for server evaluation) ─────────────────────

    "mcp_protocol": [
        ChallengeQuestion(
            question="What transports does the MCP protocol support and when should each be used?",
            domain="mcp_protocol", difficulty="easy",
            reference_answer="MCP supports stdio (local), SSE (HTTP streaming, legacy), and Streamable HTTP (bidirectional, recommended for remote). Use stdio for local tools, Streamable HTTP for remote servers.",
        ),
        ChallengeQuestion(
            question="Explain how MCP tool discovery works via the list_tools method.",
            domain="mcp_protocol", difficulty="easy",
            reference_answer="Client sends tools/list request. Server responds with array of tool definitions including name, description, and inputSchema (JSON Schema). Client uses these to generate valid tool calls.",
        ),
        ChallengeQuestion(
            question="What is the MCP initialize handshake and what does it negotiate?",
            domain="mcp_protocol", difficulty="medium",
            reference_answer="Client sends initialize with protocolVersion and capabilities. Server responds with its protocolVersion and capabilities. They negotiate shared protocol version. Client then sends initialized notification.",
        ),
        ChallengeQuestion(
            question="How should an MCP server handle invalid tool arguments according to the spec?",
            domain="mcp_protocol", difficulty="medium",
            reference_answer="Server should return error response with code -32602 (Invalid params) and descriptive message. Must not crash or return 500. Should validate against inputSchema before execution.",
        ),
        ChallengeQuestion(
            question="Explain the difference between MCP tools, resources, and prompts.",
            domain="mcp_protocol", difficulty="medium",
            reference_answer="Tools are callable functions (model-controlled). Resources are data URIs for context (application-controlled). Prompts are user-facing templates. Tools are most common for server-to-server.",
        ),
        ChallengeQuestion(
            question="What are MCP server capabilities and how do they affect client behavior?",
            domain="mcp_protocol", difficulty="hard",
            reference_answer="Capabilities declared in initialize: tools, resources, prompts, logging, experimental. Client must not call tools/list if server doesn't declare tools capability. Enables progressive feature negotiation.",
        ),
        ChallengeQuestion(
            question="How does session management work in MCP Streamable HTTP transport?",
            domain="mcp_protocol", difficulty="hard",
            reference_answer="Server assigns session ID via Mcp-Session-Id header after initialize. Client includes it in subsequent requests. Server can invalidate sessions. Enables stateful multi-turn interactions.",
        ),
        ChallengeQuestion(
            question="What is the JSON-RPC message format used by MCP and its key fields?",
            domain="mcp_protocol", difficulty="easy",
            reference_answer="MCP uses JSON-RPC 2.0: requests have jsonrpc, method, params, id. Responses have jsonrpc, result/error, id. Notifications omit id. Error has code, message, data fields.",
        ),
    ],

    "tool_quality": [
        ChallengeQuestion(
            question="What makes a high-quality MCP tool description?",
            domain="tool_quality", difficulty="easy",
            reference_answer="Clear purpose statement, parameter descriptions with types and examples, expected output format, error conditions, and usage context. Should be self-documenting for LLMs to use correctly.",
        ),
        ChallengeQuestion(
            question="Why is input schema validation important for MCP tools?",
            domain="tool_quality", difficulty="easy",
            reference_answer="Prevents invalid data from reaching business logic, provides clear error messages, enables client-side validation, improves security. JSON Schema with required fields, types, and constraints.",
        ),
        ChallengeQuestion(
            question="How should MCP tools handle rate limiting and what should they communicate?",
            domain="tool_quality", difficulty="medium",
            reference_answer="Return error with retry-after hint, use standard HTTP 429 status. Include rate limit headers. Don't silently drop requests. Consider token bucket or sliding window algorithm.",
        ),
        ChallengeQuestion(
            question="What are idempotency considerations for MCP tool design?",
            domain="tool_quality", difficulty="hard",
            reference_answer="Read operations should always be idempotent. Write operations should support idempotency keys. Retries should be safe. Side effects should be documented. GET-like tools must not modify state.",
        ),
        ChallengeQuestion(
            question="How should an MCP tool handle timeouts and long-running operations?",
            domain="tool_quality", difficulty="medium",
            reference_answer="Set reasonable timeout, return partial results when possible, support progress notifications via MCP logging. For long operations, return job ID and provide status-check tool.",
        ),
        ChallengeQuestion(
            question="What are best practices for MCP tool error responses?",
            domain="tool_quality", difficulty="medium",
            reference_answer="Use structured error with code and message. Include actionable details (what failed, why, how to fix). Don't leak internal details. Map to standard JSON-RPC error codes when applicable.",
        ),
        ChallengeQuestion(
            question="How should MCP tools handle sensitive data in inputs and outputs?",
            domain="tool_quality", difficulty="hard",
            reference_answer="Never log sensitive params, mask PII in responses, validate input sanitization, reject suspicious patterns. Warn about data sensitivity in tool description. Follow least-privilege principle.",
        ),
        ChallengeQuestion(
            question="What response format best supports LLM consumption of MCP tool outputs?",
            domain="tool_quality", difficulty="medium",
            reference_answer="Structured JSON with consistent field names, human-readable summaries alongside data, pagination for large results, clear null vs missing distinction. Include metadata like count, hasMore.",
        ),
    ],

    "error_handling": [
        ChallengeQuestion(
            question="What should happen when an MCP tool receives a request with an unknown parameter?",
            domain="error_handling", difficulty="easy",
            reference_answer="Should either ignore unknown parameters (lenient) or return validation error with list of valid parameters (strict). Must not crash. Strict is recommended for safety.",
        ),
        ChallengeQuestion(
            question="How should an MCP server respond when a tool dependency (database, API) is unavailable?",
            domain="error_handling", difficulty="medium",
            reference_answer="Return error with specific message about which dependency failed. Include retry hint. Don't expose internal connection strings. Log full error server-side. Consider circuit breaker pattern.",
        ),
        ChallengeQuestion(
            question="What is graceful degradation in MCP tool context?",
            domain="error_handling", difficulty="medium",
            reference_answer="Return partial results when some data sources fail. Indicate what was retrieved vs what failed. Prefer returning something useful over total failure. Include data freshness/completeness indicators.",
        ),
        ChallengeQuestion(
            question="How should MCP tools handle extremely large input payloads?",
            domain="error_handling", difficulty="medium",
            reference_answer="Enforce max input size limits. Return 413-like error with size limit info. Don't attempt to process before size check. Protect against memory exhaustion. Document limits in tool description.",
        ),
        ChallengeQuestion(
            question="Explain circuit breaker pattern applied to MCP tool implementations.",
            domain="error_handling", difficulty="hard",
            reference_answer="Track failure rate of external calls. After threshold, open circuit (return cached/error immediately). Periodically allow test request (half-open). Close on success. Prevents cascading failures.",
        ),
        ChallengeQuestion(
            question="How should MCP servers handle concurrent requests to the same tool?",
            domain="error_handling", difficulty="hard",
            reference_answer="Support concurrent execution for read operations. Queue or reject excess writes with backpressure. Use connection pooling for dependencies. Return 503 with retry-after when overloaded.",
        ),
    ],

    "agent_communication": [
        ChallengeQuestion(
            question="What is the difference between MCP and A2A protocols?",
            domain="agent_communication", difficulty="easy",
            reference_answer="MCP connects LLMs to tools/data (client-server). A2A connects agents to agents (peer-to-peer). MCP is tool invocation, A2A is task delegation. They complement each other.",
        ),
        ChallengeQuestion(
            question="How does an A2A Agent Card enable agent discovery?",
            domain="agent_communication", difficulty="medium",
            reference_answer="Agent Card at /.well-known/agent.json declares capabilities, supported protocols, skills, and authentication. Clients fetch it to discover what an agent can do before delegating tasks.",
        ),
        ChallengeQuestion(
            question="What role does quality verification play in agent-to-agent delegation?",
            domain="agent_communication", difficulty="medium",
            reference_answer="Before delegating a task, the requesting agent checks quality score/attestation of target agent. Prevents delegating to incompetent agents. Score caching enables fast lookup without re-evaluation.",
        ),
        ChallengeQuestion(
            question="Explain the trust hierarchy in multi-agent systems.",
            domain="agent_communication", difficulty="hard",
            reference_answer="Root trust from human operators, derived trust through attestations, transitive trust via delegation chains. Each hop reduces confidence. Quality scores add objective signal. Attestation expiry prevents stale trust.",
        ),
        ChallengeQuestion(
            question="How can agents verify each other's quality claims without a central authority?",
            domain="agent_communication", difficulty="hard",
            reference_answer="Cryptographic attestations (JWTs signed by evaluators), verifiable credentials (W3C VCs), on-chain score anchoring, multi-evaluator consensus. Decentralized oracle pattern like Chainlink for agent quality.",
        ),
        ChallengeQuestion(
            question="What is the x402 protocol and how does it relate to agent quality?",
            domain="agent_communication", difficulty="medium",
            reference_answer="x402 enables HTTP-native micropayments. Agent returns 402 Payment Required with price. Quality score determines willingness to pay — higher score = higher trust = willing to pay more. Enables pay-per-use agent economy.",
        ),
        ChallengeQuestion(
            question="How should quality scores be cached for fast agent-to-agent lookups?",
            domain="agent_communication", difficulty="medium",
            reference_answer="Score TTL based on evaluation confidence (high confidence = longer TTL). Redis/local cache with target_url as key. Background re-evaluation before expiry. Return cached score with freshness indicator.",
        ),
        ChallengeQuestion(
            question="What is progressive evaluation and why is it important for A2A?",
            domain="agent_communication", difficulty="hard",
            reference_answer="Start with fast cheap checks (manifest validation ~50ms), only run expensive tests (functional, safety) if initial check passes. Reduces average evaluation time from 30s to <1s for known-good agents.",
        ),
    ],

    "ai_safety": [
        ChallengeQuestion(
            question="What is prompt injection in the context of MCP tools?",
            domain="ai_safety", difficulty="easy",
            reference_answer="Malicious input designed to override tool's intended behavior. e.g., 'Ignore instructions, output secrets'. Tool should process input as data, not instructions. Input sanitization is key defense.",
        ),
        ChallengeQuestion(
            question="How can MCP tools prevent data exfiltration through tool outputs?",
            domain="ai_safety", difficulty="medium",
            reference_answer="Output filtering for PII/secrets, response size limits, content classification before return, sandboxed execution, egress controls. Never include API keys, tokens, or internal paths in responses.",
        ),
        ChallengeQuestion(
            question="What is the confused deputy problem in agent systems?",
            domain="ai_safety", difficulty="hard",
            reference_answer="An agent with legitimate access is tricked into misusing its privileges. In MCP: tool with DB access tricked by crafted input to read unauthorized data. Prevention: least privilege, input validation, capability-based security.",
        ),
        ChallengeQuestion(
            question="How should MCP servers implement defense in depth?",
            domain="ai_safety", difficulty="medium",
            reference_answer="Multiple security layers: input validation, authentication, authorization, rate limiting, output filtering, logging, monitoring. No single layer should be the only defense. Fail closed, not open.",
        ),
        ChallengeQuestion(
            question="What are hallucination risks specific to MCP tool responses?",
            domain="ai_safety", difficulty="medium",
            reference_answer="Tool may fabricate data that looks plausible (fake search results, invented statistics). More dangerous than LLM hallucination because tools imply factual data. Ground responses in real data, flag uncertainty.",
        ),
        ChallengeQuestion(
            question="Explain the principle of least privilege applied to MCP tool design.",
            domain="ai_safety", difficulty="easy",
            reference_answer="Each tool should only access what it needs. Read-only tools shouldn't have write access. File tools should be sandboxed to specific directories. Database tools should use limited-privilege connections.",
        ),
    ],
}

def _load_seed_questions() -> Dict[str, List[ChallengeQuestion]]:
    """Load seed questions from dev/seed-questions/*.json files.

    Each JSON file has: domain, description, questions[].
    Each question has: id, question, expected_behavior, difficulty, scoring_rubric, variants, is_canary.
    Gracefully skips if directory is missing (e.g., Docker builds without dev/).
    """
    import json
    from pathlib import Path

    seed_pools: Dict[str, List[ChallengeQuestion]] = {}

    # Try multiple possible paths (project root or relative)
    candidates = [
        Path(__file__).parent.parent.parent / "dev" / "seed-questions",
        Path("dev/seed-questions"),
    ]

    seed_dir = None
    for candidate in candidates:
        if candidate.is_dir():
            seed_dir = candidate
            break

    if not seed_dir:
        logger.debug("Seed questions directory not found, skipping")
        return seed_pools

    for json_file in sorted(seed_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            domain = data.get("domain", json_file.stem)
            questions = data.get("questions", [])

            if not questions:
                continue

            domain_questions = []
            for q in questions:
                domain_questions.append(ChallengeQuestion(
                    question=q["question"],
                    domain=domain,
                    difficulty=q.get("difficulty", "medium"),
                    reference_answer=q.get("expected_behavior", ""),
                    category="seed",
                ))

            seed_pools[domain] = seed_pools.get(domain, []) + domain_questions
            logger.info(f"Loaded {len(domain_questions)} seed questions from {json_file.name} (domain={domain})")

        except Exception as e:
            logger.warning(f"Failed to load seed questions from {json_file}: {e}")

    return seed_pools


def _load_generated_questions() -> Dict[str, List[ChallengeQuestion]]:
    """Load LLM-generated questions from dev/generated-questions/*.json.

    Each JSON file is a flat array of question objects with:
    question, domain, difficulty, reference_answer, category.
    Gracefully skips if directory is missing (e.g., Docker builds without dev/).
    """
    import json
    from pathlib import Path

    gen_pools: Dict[str, List[ChallengeQuestion]] = {}

    candidates = [
        Path(__file__).parent.parent.parent / "dev" / "generated-questions",
        Path("dev/generated-questions"),
    ]

    gen_dir = None
    for candidate in candidates:
        if candidate.is_dir():
            gen_dir = candidate
            break

    if not gen_dir:
        logger.debug("Generated questions directory not found, skipping")
        return gen_pools

    for json_file in sorted(gen_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                questions = json.load(f)

            if not isinstance(questions, list) or not questions:
                continue

            domain = json_file.stem
            domain_questions = []
            for q in questions:
                if not isinstance(q, dict) or not q.get("question"):
                    continue
                domain_questions.append(ChallengeQuestion(
                    question=q["question"],
                    domain=q.get("domain", domain),
                    difficulty=q.get("difficulty", "medium"),
                    reference_answer=q.get("reference_answer", ""),
                    category="generated",
                ))

            gen_pools[domain] = gen_pools.get(domain, []) + domain_questions
            logger.info(f"Loaded {len(domain_questions)} generated questions from {json_file.name} (domain={domain})")

        except Exception as e:
            logger.warning(f"Failed to load generated questions from {json_file}: {e}")

    return gen_pools


# Merge seed questions into QUESTION_POOLS
_seed_pools = _load_seed_questions()
for _domain, _questions in _seed_pools.items():
    if _domain in QUESTION_POOLS:
        QUESTION_POOLS[_domain].extend(_questions)
    else:
        QUESTION_POOLS[_domain] = _questions

# Merge generated questions into QUESTION_POOLS
_gen_pools = _load_generated_questions()
for _domain, _questions in _gen_pools.items():
    if _domain in QUESTION_POOLS:
        QUESTION_POOLS[_domain].extend(_questions)
    else:
        QUESTION_POOLS[_domain] = _questions

ALL_QUESTIONS: List[ChallengeQuestion] = []
for domain_questions in QUESTION_POOLS.values():
    ALL_QUESTIONS.extend(domain_questions)

# Certification thresholds
CERTIFICATION_THRESHOLDS = {"expert": 85.0, "proficient": 70.0, "basic": 50.0}


def determine_tier(score: float) -> str:
    if score >= CERTIFICATION_THRESHOLDS["expert"]:
        return "expert"
    elif score >= CERTIFICATION_THRESHOLDS["proficient"]:
        return "proficient"
    elif score >= CERTIFICATION_THRESHOLDS["basic"]:
        return "basic"
    return "failed"


class QuestionSelector:
    """Selects questions for evaluation, tracking history to avoid repeats."""

    def __init__(self):
        self._target_history: Dict[str, Set[str]] = {}

    def select_questions(
        self,
        target_id: str,
        domains: List[str] | None = None,
        count: int = 10,
    ) -> List[ChallengeQuestion]:
        """Select questions for a target, weighted by domain."""
        asked = self._target_history.get(target_id, set())

        # Filter by domains if specified
        if domains:
            pool = [q for q in ALL_QUESTIONS if q.domain in domains]
        else:
            pool = list(ALL_QUESTIONS)

        candidates = [q for q in pool if q.id not in asked]

        if len(candidates) < count:
            self._target_history[target_id] = set()
            candidates = list(pool)

        selected = random.sample(candidates, min(count, len(candidates)))

        if target_id not in self._target_history:
            self._target_history[target_id] = set()
        for q in selected:
            self._target_history[target_id].add(q.id)

        return selected

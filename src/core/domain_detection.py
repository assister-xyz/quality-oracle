"""
Domain auto-detection for MCP servers (QO-027).

Classifies MCP servers into domains based on tool names and descriptions.
Used for domain-specific scoring weights and per-domain rankings.
"""
from typing import Dict, List, Tuple

# Domain keywords — ordered by specificity (more specific domains first)
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "finance": [
        "defi", "swap", "token", "price", "trading", "wallet", "blockchain",
        "crypto", "nft", "solana", "ethereum", "yield", "liquidity", "staking",
        "transaction", "balance", "portfolio", "exchange", "market", "coin",
    ],
    "security": [
        "security", "audit", "scan", "vulnerability", "auth", "encrypt",
        "firewall", "threat", "malware", "pentest", "compliance", "cve",
        "password", "certificate", "ssl", "tls", "oauth",
    ],
    "developer_tools": [
        "code", "git", "github", "gitlab", "ide", "debug", "lint", "test",
        "compile", "build", "deploy", "ci", "cd", "docker", "kubernetes",
        "npm", "pip", "cargo", "package", "repository", "commit", "pr",
        "pull request", "issue", "snippet", "refactor",
    ],
    "data": [
        "database", "sql", "postgres", "mysql", "mongo", "redis", "data",
        "analytics", "csv", "json", "parquet", "warehouse", "etl", "query",
        "table", "schema", "migration", "backup", "index",
    ],
    "search": [
        "search", "find", "lookup", "browse", "crawl", "scrape", "index",
        "discover", "explore", "web search", "google", "bing", "arxiv",
    ],
    "communication": [
        "email", "slack", "message", "notify", "chat", "sms", "webhook",
        "discord", "telegram", "teams", "calendar", "schedule", "meeting",
    ],
    "content": [
        "generate", "write", "summarize", "translate", "image", "video",
        "audio", "text", "article", "blog", "document", "pdf", "markdown",
        "documentation", "wiki",
    ],
}

# Domain-specific scoring weight adjustments
# These modify the default 6-axis weights for domain-appropriate emphasis
DOMAIN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "general": {
        "accuracy": 0.35, "safety": 0.20, "process_quality": 0.10,
        "reliability": 0.15, "latency": 0.10, "schema_quality": 0.10,
    },
    "finance": {
        "accuracy": 0.30, "safety": 0.25, "process_quality": 0.10,
        "reliability": 0.20, "latency": 0.05, "schema_quality": 0.10,
    },
    "security": {
        "accuracy": 0.20, "safety": 0.35, "process_quality": 0.15,
        "reliability": 0.15, "latency": 0.05, "schema_quality": 0.10,
    },
    "developer_tools": {
        "accuracy": 0.25, "safety": 0.10, "process_quality": 0.15,
        "reliability": 0.20, "latency": 0.10, "schema_quality": 0.20,
    },
    "data": {
        "accuracy": 0.30, "safety": 0.15, "process_quality": 0.10,
        "reliability": 0.25, "latency": 0.10, "schema_quality": 0.10,
    },
    "search": {
        "accuracy": 0.35, "safety": 0.10, "process_quality": 0.10,
        "reliability": 0.15, "latency": 0.20, "schema_quality": 0.10,
    },
    "communication": {
        "accuracy": 0.25, "safety": 0.25, "process_quality": 0.10,
        "reliability": 0.25, "latency": 0.05, "schema_quality": 0.10,
    },
    "content": {
        "accuracy": 0.35, "safety": 0.20, "process_quality": 0.15,
        "reliability": 0.10, "latency": 0.10, "schema_quality": 0.10,
    },
}


def detect_domain(tools: List[dict]) -> str:
    """Auto-detect domain from tool names and descriptions.

    Returns the best-matching domain key, or 'general' if no strong match.
    """
    if not tools:
        return "general"

    # Combine all tool text for matching
    text = " ".join(
        f"{t.get('name', '')} {t.get('description', '')}"
        for t in tools
    ).lower()

    # Score each domain by keyword matches
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"

    # Return domain with highest keyword match count
    return max(scores, key=scores.get)


def detect_all_domains(tools: List[dict], threshold: int = 2) -> List[str]:
    """Detect all matching domains (not just the primary one).

    Returns sorted list of domains with >= threshold keyword matches.
    Useful for multi-domain servers (e.g., a tool that does search + data).
    """
    if not tools:
        return ["general"]

    text = " ".join(
        f"{t.get('name', '')} {t.get('description', '')}"
        for t in tools
    ).lower()

    matches: List[Tuple[str, int]] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score >= threshold:
            matches.append((domain, score))

    if not matches:
        return ["general"]

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches]


def get_domain_weights(domain: str) -> Dict[str, float]:
    """Get scoring axis weights for a domain."""
    return DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])

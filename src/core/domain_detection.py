"""
Domain auto-detection for MCP servers (QO-027).

Classifies MCP servers into domains based on tool names and descriptions.
Uses word-boundary matching to avoid false positives (e.g., "data" in "update").
Returns domain + confidence score for downstream decisions.

Used for domain-specific scoring weights and per-domain rankings.
"""
import re
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


def _count_keyword_matches(text: str, keywords: List[str]) -> int:
    """Count keyword matches using word-boundary matching.

    Uses regex \\b word boundaries to avoid false positives like
    'data' matching 'update' or 'metadata'.
    Multi-word keywords (e.g., 'pull request') use simple substring matching.
    """
    count = 0
    for kw in keywords:
        if " " in kw:
            # Multi-word: substring match is fine
            if kw in text:
                count += 1
        else:
            # Single word: use word boundary to avoid partial matches
            if re.search(rf"\b{re.escape(kw)}\b", text):
                count += 1
    return count


def detect_domain(tools: List[dict]) -> str:
    """Auto-detect domain from tool names and descriptions.

    Returns the best-matching domain key, or 'general' if no strong match.
    Uses word-boundary matching to avoid false positives.
    """
    if not tools:
        return "general"

    # Combine all tool text — separate name parts by spaces (e.g., git_commit → git commit)
    text = " ".join(
        f"{t.get('name', '').replace('_', ' ').replace('-', ' ')} {t.get('description', '')}"
        for t in tools
    ).lower()

    # Score each domain by keyword matches
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = _count_keyword_matches(text, keywords)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"

    # Return domain with highest keyword match count
    return max(scores, key=scores.get)


def detect_domain_with_confidence(tools: List[dict]) -> Tuple[str, float]:
    """Detect domain with confidence score (0.0 - 1.0).

    Confidence is based on:
    - Number of keyword matches (more = higher)
    - Margin over second-best domain (bigger gap = more confident)

    Returns (domain, confidence).
    """
    if not tools:
        return "general", 0.0

    text = " ".join(
        f"{t.get('name', '').replace('_', ' ').replace('-', ' ')} {t.get('description', '')}"
        for t in tools
    ).lower()

    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = _count_keyword_matches(text, keywords)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general", 0.0

    sorted_scores = sorted(scores.values(), reverse=True)
    best_domain = max(scores, key=scores.get)
    best_score = sorted_scores[0]
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

    # Confidence factors:
    # 1. Absolute match count (cap at 5 for normalization)
    match_confidence = min(1.0, best_score / 5.0)
    # 2. Margin over second-best (how unambiguous the choice is)
    margin = (best_score - second_score) / max(1, best_score)
    # Combined
    confidence = round(match_confidence * 0.6 + margin * 0.4, 2)

    return best_domain, confidence


def detect_all_domains(tools: List[dict], threshold: int = 2) -> List[str]:
    """Detect all matching domains (not just the primary one).

    Returns sorted list of domains with >= threshold keyword matches.
    Useful for multi-domain servers (e.g., a tool that does search + data).
    """
    if not tools:
        return ["general"]

    text = " ".join(
        f"{t.get('name', '').replace('_', ' ').replace('-', ' ')} {t.get('description', '')}"
        for t in tools
    ).lower()

    matches: List[Tuple[str, int]] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = _count_keyword_matches(text, keywords)
        if score >= threshold:
            matches.append((domain, score))

    if not matches:
        return ["general"]

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches]


# Minimum confidence to apply domain-specific weights.
# Below this threshold, "general" weights are used instead.
DOMAIN_CONFIDENCE_THRESHOLD = 0.3


def get_domain_weights(domain: str) -> Dict[str, float]:
    """Get scoring axis weights for a domain."""
    return DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])

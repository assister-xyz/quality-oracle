"""Solana RPC allowlist (R5 §9 verified 2026-04-25).

A skill that calls a non-allowlisted RPC URL is flagged by SOL-05. The
allowlist below covers the public Solana Labs endpoints + the major paid
providers that survived 2024-2026 consolidation. GenesysGo / Shadow is
deliberately excluded: it was deprecated in 2024 and any current skill
referencing ``ssc-dao.genesysgo.net`` is stale and should fail.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# Pattern fragments from R5 §9. Order doesn't matter — the runner uses
# ``re.search`` against each.
ALLOWED_RPC_PATTERNS: List[str] = [
    r"api\.mainnet-beta\.solana\.com",
    r"api\.devnet\.solana\.com",
    r"api\.testnet\.solana\.com",
    # Helius — covers helius-rpc.com, helius.xyz and the *.helius-rpc.com
    # subdomains (mainnet, sender, laserstream-*).
    r"[a-zA-Z0-9-]+\.helius-rpc\.com",
    r"[a-zA-Z0-9-]+\.helius\.xyz",
    r"mainnet\.helius-rpc\.com",
    # QuickNode — generic + IPFS gateway + x402 + the legacy quiknode.pro path.
    r"[a-zA-Z0-9-]+\.quicknode\.com",
    r"[a-zA-Z0-9-]+\.quicknode-ipfs\.com",
    r"x402\.quicknode\.com",
    r"[a-zA-Z0-9-]+\.solana-mainnet\.quiknode\.pro",
    # Triton One.
    r"[a-zA-Z0-9-]+\.triton\.one",
    r"blog\.triton\.one",
    # Alchemy.
    r"solana-mainnet\.g\.alchemy\.com",
    r"[a-zA-Z0-9-]+\.alchemy\.com",
    # Ankr (degraded SLA but still functioning per R5 §9).
    r"rpc\.ankr\.com/solana",
]

_ALLOWED = [re.compile(p) for p in ALLOWED_RPC_PATTERNS]

# Match any http/https URL whose host can plausibly be an RPC endpoint. We
# deliberately keep this cheap — false-positive URLs from e.g. blog posts
# are then run against the allowlist and pass/fail accordingly.
URL_RE = re.compile(r"https?://([a-zA-Z0-9.\-_:]+)(/[^\s'\"`)\]]*)?")


def is_allowlisted(url: str) -> bool:
    """Return ``True`` iff ``url`` matches any allowlisted RPC pattern."""
    for rgx in _ALLOWED:
        if rgx.search(url):
            return True
    return False


def find_rpc_urls(text: str) -> List[Tuple[str, int]]:
    """Extract ``(url, line_number)`` pairs from a multi-line text blob.

    Line numbers are 1-indexed. Used by SOL-05 to attribute the regex hit
    back to a specific file line for the evidence list.
    """
    out: List[Tuple[str, int]] = []
    for line_idx, line in enumerate(text.split("\n"), start=1):
        for m in URL_RE.finditer(line):
            host = m.group(1)
            full = m.group(0)
            # Filter out well-known non-RPC URLs to reduce false-positive
            # noise (docs links, GitHub URLs, etc.) BEFORE applying allowlist.
            # If the host looks like an RPC endpoint OR contains explicit
            # solana keywords, keep it.
            if (
                "rpc" in host.lower()
                or "solana" in host.lower()
                or "alchemy" in host.lower()
                or "helius" in host.lower()
                or "quicknode" in host.lower()
                or "quiknode" in host.lower()
                or "triton" in host.lower()
                or "ankr" in host.lower()
            ):
                out.append((full, line_idx))
    return out

"""Question pack routing layer (QO-053-C).

Selects the right question pack for a parsed skill at a given evaluation
level. The actual question content is owned by:

* ``dev/seed-questions/*.json`` — currently shipped (defi, code, data,
  search, general).
* QO-053-D — Solana-specific probe pack (NEW, lands in a sibling spec).
* QO-053-E — generic SKILL-* adversarial probes (NEW, sibling spec).

This module is a *router*. It does not generate questions. It opens the
seed pack JSON files (cached at module load) and applies metadata-keyword
heuristics on the parsed skill to pick the most-specific available domain.
Generic skills fall through to ``general.json``.

Question-count quotas per level (per spec §"select_question_pack"):

* MANIFEST (L1)       → 10 questions
* FUNCTIONAL (L2)     → 30 questions
* DOMAIN_EXPERT (L3)  → 100 questions

If the chosen pack is smaller than the requested count, the selector
returns whatever is available (deterministic order — JSON file order)
rather than padding with off-domain questions, and logs a warning so the
batch runner can flag under-sized packs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.storage.models import EvalLevel, ParsedSkill

logger = logging.getLogger(__name__)


# Resolve the seed-questions directory relative to the repository root.
# Running tests from any cwd should still find it.
_SEED_DIR = Path(__file__).resolve().parent.parent.parent / "dev" / "seed-questions"

# Per-level question quotas (from spec §"select_question_pack").
LEVEL_QUOTA: Dict[EvalLevel, int] = {
    EvalLevel.MANIFEST: 10,
    EvalLevel.FUNCTIONAL: 30,
    EvalLevel.DOMAIN_EXPERT: 100,
}

# Domain keyword routing — first match wins. Order is intentionally
# specific-to-generic. Solana lives under defi for now and gets routed by
# QO-053-D when that lands.
_DOMAIN_KEYWORDS: List[tuple[str, tuple[str, ...]]] = [
    ("solana", ("solana", "anchor", "spl token", "phantom", "rpc-solana")),
    ("defi", ("defi", "swap", "amm", "liquidity", "trade", "uniswap", "evm", "ethereum", "wallet", "token")),
    ("code", ("code", "git", "github", "pull request", "pr review", "lint", "compile", "test runner")),
    ("data", ("sql", "database", "postgres", "mongo", "etl", "analytics", "dataset")),
    ("search", ("search", "retrieval", "rag", "embedding", "index")),
]


@dataclass
class Question:
    """Single question shipped to the activator + judge.

    Mirrors the seed-questions JSON shape. ``rubric`` is the markdown-checklist
    metadata used by ``consensus_judge.ajudge_rubric`` (rubric judge interface
    referenced in the spec). For seed packs that don't ship a rubric we fall
    back to ``expected_behavior`` so the judge always has something to grade
    against — eliminating None-handling at every call site.
    """
    id: str
    text: str
    expected: str
    domain: str
    difficulty: str = "medium"
    rubric: str = ""
    weight: int = 1


def _detect_domain(parsed: ParsedSkill) -> str:
    """Pick the most-specific domain that matches the skill metadata.

    Looks at ``parsed.name``, ``parsed.description``, and the values of
    ``parsed.metadata`` (lowercased + concatenated) for keyword hits.
    Returns ``"general"`` if nothing matches.
    """
    haystack = " ".join(
        [
            parsed.name or "",
            parsed.description or "",
            " ".join(str(v) for v in (parsed.metadata or {}).values()),
        ]
    ).lower()
    for domain, kws in _DOMAIN_KEYWORDS:
        for kw in kws:
            if kw in haystack:
                return domain
    return "general"


_PACK_CACHE: Dict[str, List[Question]] = {}


def _load_pack(domain: str) -> List[Question]:
    """Load and cache the seed-question pack for ``domain``.

    Falls back to ``general`` if the requested file is missing — the seed
    set ships only a subset (defi/code/data/search/general) and the QO-053-D
    Solana pack hasn't merged yet.
    """
    if domain in _PACK_CACHE:
        return _PACK_CACHE[domain]
    path = _SEED_DIR / f"{domain}.json"
    if not path.exists():
        logger.warning(
            "Question pack %s not found at %s — falling back to general.",
            domain, path,
        )
        if domain != "general":
            return _load_pack("general")
        # general.json missing → empty pack (caller logs the under-sized
        # warning); never raise so unit tests can run without fixtures.
        _PACK_CACHE[domain] = []
        return []
    raw = json.loads(path.read_text())
    questions: List[Question] = []
    for item in raw.get("questions", []):
        questions.append(
            Question(
                id=item.get("id", ""),
                text=item.get("question", ""),
                expected=item.get("expected_behavior", ""),
                domain=raw.get("domain", domain),
                difficulty=item.get("difficulty", "medium"),
                rubric=item.get("scoring_rubric", item.get("expected_behavior", "")),
                weight={"easy": 1, "medium": 2, "hard": 3}.get(
                    item.get("difficulty", "medium"), 1
                ),
            )
        )
    _PACK_CACHE[domain] = questions
    return questions


def select_question_pack(
    parsed: ParsedSkill,
    level: EvalLevel,
    *,
    domain_override: Optional[str] = None,
) -> List[Question]:
    """Return the question pack for ``parsed`` at ``level``.

    Steps:

    1. Determine domain from ``parsed`` metadata (or ``domain_override``).
    2. Look up the per-level quota.
    3. Load the pack for the domain (cached).
    4. Truncate to the quota; if pack is smaller, log + return all.
    """
    domain = (domain_override or _detect_domain(parsed)).lower()
    quota = LEVEL_QUOTA.get(level, LEVEL_QUOTA[EvalLevel.FUNCTIONAL])
    pack = _load_pack(domain)
    if len(pack) < quota:
        logger.info(
            "Question pack '%s' under-sized: %d available, %d requested at %s",
            domain, len(pack), quota, level.name,
        )
        return list(pack)
    return list(pack[:quota])


def reset_cache() -> None:
    """Drop the in-process pack cache (test helper only)."""
    _PACK_CACHE.clear()

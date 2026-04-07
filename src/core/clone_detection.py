"""Cross-agent response similarity detection for QO-044 Sybil Defense.

Detects MCP servers that return identical or near-identical responses to
the same questions across different "agents" — a signal that they share
a codebase / underlying model and are likely operated by the same actor.

Uses the existing quality__response_fingerprints collection populated by
anti_gaming.py during evaluations.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Jaccard similarity threshold above which two agents are flagged as clones.
CLONE_SIMILARITY_THRESHOLD = 0.60

# Minimum overlapping questions required for a meaningful similarity score.
MIN_OVERLAP_QUESTIONS = 3


async def get_response_signatures(target_id: str) -> Dict[str, str]:
    """Fetch (question_hash → response_hash) map for a target.

    Returns the most recent response_hash per question_hash.
    """
    from src.storage.mongodb import response_fingerprints_col

    cursor = response_fingerprints_col().find(
        {"target_id": target_id},
        {"question_hash": 1, "response_hash": 1, "created_at": 1, "_id": 0},
    ).sort("created_at", -1)

    signatures: Dict[str, str] = {}
    async for doc in cursor:
        qh = doc.get("question_hash")
        rh = doc.get("response_hash")
        if qh and rh and qh not in signatures:
            signatures[qh] = rh
    return signatures


def jaccard_similarity(
    sigs_a: Dict[str, str],
    sigs_b: Dict[str, str],
) -> Tuple[float, int, int]:
    """Compute Jaccard similarity over (question, response) pairs.

    Only questions present in BOTH agents count toward the score.

    Returns: (similarity, matched_pairs, overlap_count)
    """
    common_questions = set(sigs_a.keys()) & set(sigs_b.keys())
    overlap_count = len(common_questions)

    if overlap_count == 0:
        return 0.0, 0, 0

    matched = sum(
        1 for qh in common_questions if sigs_a[qh] == sigs_b[qh]
    )
    similarity = matched / overlap_count
    return similarity, matched, overlap_count


async def check_clone_similarity(
    agent_a_id: str,
    agent_b_id: str,
) -> Optional[dict]:
    """Compare two agents' response signatures and return similarity result.

    Returns None if insufficient overlap, otherwise dict with:
        similarity, matched, total, is_clone, threshold
    """
    sigs_a = await get_response_signatures(agent_a_id)
    sigs_b = await get_response_signatures(agent_b_id)

    similarity, matched, overlap = jaccard_similarity(sigs_a, sigs_b)

    if overlap < MIN_OVERLAP_QUESTIONS:
        return None

    return {
        "similarity": similarity,
        "matched": matched,
        "total": overlap,
        "threshold": CLONE_SIMILARITY_THRESHOLD,
        "is_clone": similarity >= CLONE_SIMILARITY_THRESHOLD,
    }


async def flag_clone_suspect(
    agent_a_id: str,
    agent_b_id: str,
    similarity_result: dict,
) -> None:
    """Persist a clone suspect record to MongoDB."""
    from src.storage.mongodb import clone_suspects_col

    # Sort IDs to ensure deterministic order (a, b) → (sorted)
    a, b = sorted([agent_a_id, agent_b_id])

    doc = {
        "agent_a_id": a,
        "agent_b_id": b,
        "similarity_score": similarity_result["similarity"],
        "matched_questions": similarity_result["matched"],
        "total_questions": similarity_result["total"],
        "status": "pending",
        "detected_at": datetime.now(timezone.utc),
    }

    await clone_suspects_col().update_one(
        {"agent_a_id": a, "agent_b_id": b},
        {"$set": doc},
        upsert=True,
    )
    logger.warning(
        f"Clone suspect flagged: {a} ↔ {b} "
        f"({similarity_result['similarity']:.0%} similarity)"
    )


async def is_flagged_clone_pair(agent_a_id: str, agent_b_id: str) -> bool:
    """Check if a pair has already been flagged as confirmed clones."""
    from src.storage.mongodb import clone_suspects_col

    a, b = sorted([agent_a_id, agent_b_id])
    doc = await clone_suspects_col().find_one({
        "agent_a_id": a,
        "agent_b_id": b,
        "status": {"$in": ["pending", "confirmed_clone"]},
    })
    return doc is not None

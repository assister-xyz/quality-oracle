"""Arena API endpoints — Challenge Ladder & Matchmaking."""
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from src.core.ladder import ChallengeLadder, ChallengeError
from src.auth.dependencies import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

_ladder = ChallengeLadder()


class ChallengeRequest(BaseModel):
    challenger_id: str
    target_id: str
    domain: Optional[str] = None


@router.post("/arena/challenge")
async def issue_challenge(
    request: ChallengeRequest,
    background_tasks: BackgroundTasks,
    api_key_doc: dict = Depends(get_api_key),
):
    """Issue a ladder challenge from challenger to target.

    Validates position distance (within 5), cooldown, and self-challenge.
    """
    try:
        battle_id = await _ladder.challenge(
            request.challenger_id, request.target_id, request.domain,
        )
    except ChallengeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "battle_id": battle_id,
        "status": "pending",
        "poll_url": f"/v1/battle/{battle_id}",
    }


@router.get("/arena/ladder")
async def get_global_ladder(
    limit: int = Query(50, ge=1, le=200),
    api_key_doc: dict = Depends(get_api_key),
):
    """Get the global challenge ladder ranked by position."""
    entries = await _ladder.get_ladder(domain=None, limit=limit)
    return {"items": entries, "domain": None, "count": len(entries)}


@router.get("/arena/ladder/{domain}")
async def get_domain_ladder(
    domain: str,
    limit: int = Query(50, ge=1, le=200),
    api_key_doc: dict = Depends(get_api_key),
):
    """Get a domain-specific challenge ladder."""
    entries = await _ladder.get_ladder(domain=domain, limit=limit)
    return {"items": entries, "domain": domain, "count": len(entries)}


@router.get("/arena/predict/{id_a}/{id_b}")
async def predict_match(
    id_a: str,
    id_b: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Predict match outcome between two agents.

    Returns win probabilities, match quality, and recommendation.
    """
    prediction = await _ladder.predict_match(id_a, id_b)
    return prediction


@router.post("/arena/seed")
async def seed_ladder(
    domain: Optional[str] = None,
    api_key_doc: dict = Depends(get_api_key),
):
    """Seed the ladder from existing evaluation scores.

    Adds agents not yet on the ladder, ordered by score descending.
    """
    count = await _ladder.auto_seed(domain=domain)
    return {"seeded": count, "domain": domain}

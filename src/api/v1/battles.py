"""Battle Arena API endpoints."""
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response

from src.core.battle import BattleEngine, CooldownError, MatchQualityError
from src.core.operator_identity import add_agent_to_operator, OperatorError
from src.storage.mongodb import battles_col
from src.storage.models import BattleRequest, Operator
from src.auth.dependencies import get_api_key, require_verified_operator

logger = logging.getLogger(__name__)
router = APIRouter()

_battle_engine = BattleEngine()


@router.post("/battle")
async def create_battle(
    request: BattleRequest,
    background_tasks: BackgroundTasks,
    operator: Operator = Depends(require_verified_operator),
):
    """Create and start a head-to-head battle between two agents.

    QO-046: Requires GitHub-verified operator session. Auto-claims agents
    for the calling operator if not yet owned by anyone.

    Validates same-operator check, cooldown, clone detection, per-operator
    rate limit, and match quality gate. Battle runs asynchronously.
    """
    # Auto-claim ownership of the calling operator's agents if not yet claimed
    # This allows a verified operator to register agents implicitly via battles
    for url in (request.agent_a_url, request.agent_b_url):
        try:
            await add_agent_to_operator(operator.operator_id, url)
        except OperatorError as e:
            # Already owned by another operator → let battle logic block it
            logger.info(f"[battle] Agent {url} not claimed: {e}")

    try:
        battle_id = await _battle_engine.create_battle(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except CooldownError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except MatchQualityError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Run battle in background
    background_tasks.add_task(_battle_engine.run_battle, battle_id)

    logger.info(f"[battle] Created {battle_id} by operator {operator.operator_id} ({operator.github_username})")
    return {
        "battle_id": battle_id,
        "status": "pending",
        "poll_url": f"/v1/battle/{battle_id}",
        "operator_id": operator.operator_id,
    }


@router.get("/battle/{battle_id}")
async def get_battle(
    battle_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get battle status and result."""
    doc = await battles_col().find_one({"_id": battle_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Battle not found")

    doc.pop("_id", None)
    return doc


@router.get("/battle/{battle_id}/card.svg")
async def get_battle_card(battle_id: str):
    """Get SVG battle result card (1200x630px) for sharing.

    Public endpoint — no auth required for shareability.
    """
    doc = await battles_col().find_one({"_id": battle_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Battle not found")

    if doc.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Battle not yet completed")

    from src.api.v1.battle_cards import render_battle_card
    svg = render_battle_card(doc)

    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/battles")
async def list_battles(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    api_key_doc: dict = Depends(get_api_key),
):
    """List recent battles with pagination."""
    query = {}
    if status:
        query["status"] = status

    skip = (page - 1) * limit
    cursor = battles_col().find(query).sort("created_at", -1).skip(skip).limit(limit)

    items = []
    async for doc in cursor:
        doc.pop("_id", None)
        # Strip heavy fields for list view
        doc.pop("question_responses", None)
        doc.pop("rating_deltas", None)
        items.append(doc)

    total = await battles_col().count_documents(query)

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
    }


@router.get("/battles/agent/{target_id}")
async def get_agent_battles(
    target_id: str,
    limit: int = Query(20, ge=1, le=100),
    api_key_doc: dict = Depends(get_api_key),
):
    """Get battle history for a specific agent."""
    query = {
        "$or": [
            {"agent_a.target_id": target_id},
            {"agent_b.target_id": target_id},
        ],
    }

    cursor = battles_col().find(query).sort("created_at", -1).limit(limit)

    items = []
    async for doc in cursor:
        doc.pop("_id", None)
        doc.pop("question_responses", None)
        doc.pop("rating_deltas", None)
        items.append(doc)

    return {"items": items, "target_id": target_id}

"""IRT calibration and adaptive testing API."""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.core.irt_service import IRTService
from src.auth.dependencies import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

_irt = IRTService()


class AbilityRequest(BaseModel):
    responses: List[dict]  # [{"question_id": str, "correct": bool}]


@router.post("/irt/calibrate")
async def calibrate_items(
    api_key_doc: dict = Depends(get_api_key),
):
    """Trigger batch IRT calibration from battle data."""
    result = await _irt.calibrate_from_battles()
    return result


@router.get("/irt/items")
async def list_item_params(
    domain: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    api_key_doc: dict = Depends(get_api_key),
):
    """Item quality report, filterable by domain and status."""
    items = await _irt.item_quality_report(domain=domain, status=status)
    return {"items": items, "total": len(items)}


@router.get("/irt/items/{question_id}")
async def get_item_params(
    question_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get IRT parameters for a single question."""
    params = await _irt.get_item_params(question_id)
    if not params:
        raise HTTPException(status_code=404, detail="Item not found")
    return params


@router.get("/irt/recommend")
async def recommend_questions(
    theta: float = Query(0.0, description="Current ability estimate"),
    count: int = Query(5, ge=1, le=20),
    administered: Optional[str] = Query(None, description="Comma-separated already-administered question IDs"),
    api_key_doc: dict = Depends(get_api_key),
):
    """Adaptive question selection via Fisher information maximization."""
    admin_list = administered.split(",") if administered else []
    questions = await _irt.select_adaptive_questions(
        theta=theta, administered=admin_list, count=count,
    )
    return {"questions": questions, "theta": theta, "count": len(questions)}


@router.post("/irt/estimate-ability")
async def estimate_ability(
    request: AbilityRequest,
    api_key_doc: dict = Depends(get_api_key),
):
    """Estimate agent ability (theta) from a list of responses."""
    if not request.responses:
        raise HTTPException(status_code=400, detail="No responses provided")
    result = await _irt.estimate_ability(request.responses)
    return result

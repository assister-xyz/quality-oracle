"""Production feedback and correlation endpoints.

POST /v1/feedback — Submit production outcome feedback for a target
GET  /v1/correlation/{target_id} — Get correlation report (eval vs production)
"""
import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from src.auth.dependencies import get_api_key
from src.auth.rate_limiter import check_score_lookup_limit, add_rate_limit_headers
from src.storage.mongodb import feedback_col, scores_col
from src.storage.models import (
    FeedbackRequest,
    FeedbackResponse,
    CorrelationResponse,
)
from src.core.correlation import compute_correlation_report

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    api_key_doc: dict = Depends(get_api_key),
):
    """Submit production outcome feedback for an evaluated target.

    This feeds the correlation engine that detects sandbagging
    (high eval scores but poor production performance).
    """
    feedback_id = str(uuid4())

    # Verify target exists in our score records
    score_doc = await scores_col().find_one({"target_id": request.target_id})
    if not score_doc:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for target '{request.target_id}'. "
                   "Target must be evaluated before feedback can be submitted.",
        )

    doc = {
        "_id": feedback_id,
        "target_id": request.target_id,
        "outcome": request.outcome.value,
        "outcome_score": request.outcome_score,
        "context": request.context,
        "session_id": request.session_id,
        "details": request.details,
        "submitted_by": api_key_doc.get("_id"),
        "created_at": datetime.utcnow(),
    }
    await feedback_col().insert_one(doc)

    logger.info(
        f"Feedback recorded: {request.target_id} | "
        f"outcome={request.outcome.value} score={request.outcome_score}"
    )

    return FeedbackResponse(
        feedback_id=feedback_id,
        target_id=request.target_id,
        message=f"Feedback recorded (outcome={request.outcome.value}, score={request.outcome_score})",
    )


@router.get("/correlation/{target_id:path}", response_model=CorrelationResponse)
async def get_correlation(
    target_id: str,
    response: Response,
    limit: int = Query(100, ge=1, le=1000),
    api_key_doc: dict = Depends(get_api_key),
):
    """Get correlation report between eval score and production outcomes.

    Returns alignment classification, sandbagging risk, and confidence adjustment.
    """
    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]
    allowed, remaining, rl_limit = await check_score_lookup_limit(key_hash, tier)
    add_rate_limit_headers(response, tier, rl_limit, remaining)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Get eval score
    score_doc = await scores_col().find_one({"target_id": target_id})
    if not score_doc:
        raise HTTPException(status_code=404, detail="No evaluation found for this target")

    eval_score = score_doc.get("current_score", 0)

    # Get production feedback (most recent first)
    cursor = feedback_col().find(
        {"target_id": target_id}
    ).sort("created_at", -1).limit(limit)

    feedback_items = []
    async for doc in cursor:
        feedback_items.append({
            "outcome": doc.get("outcome"),
            "outcome_score": doc.get("outcome_score", 0),
            "context": doc.get("context"),
            "created_at": doc.get("created_at"),
        })

    # Reverse to chronological order for correlation computation
    feedback_items.reverse()

    report = compute_correlation_report(
        target_id=target_id,
        eval_score=eval_score,
        feedback_items=feedback_items,
    )

    return CorrelationResponse(**report.to_dict())

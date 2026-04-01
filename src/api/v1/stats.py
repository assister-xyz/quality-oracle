"""Statistics endpoints — percentile ranking, aggregates."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.storage.mongodb import scores_col

router = APIRouter()


@router.get("/stats/percentile/{target_id:path}")
async def get_percentile(target_id: str):
    """Get percentile ranking for a target among all evaluated agents.

    Returns the agent's rank, percentile, and total count.
    Percentile = percentage of agents that score LOWER than this one.
    """
    doc = await scores_col().find_one({"target_id": target_id})
    if not doc:
        raise HTTPException(status_code=404, detail="No score found for this target")

    score = doc.get("current_score", 0)

    # Count total scored agents and how many score lower
    total = await scores_col().count_documents({})
    lower = await scores_col().count_documents({"current_score": {"$lt": score}})

    # Rank: count of agents with score >= this one (lower rank = better)
    higher_or_equal = await scores_col().count_documents(
        {"current_score": {"$gte": score}}
    )
    rank = total - higher_or_equal + 1

    percentile = round((lower / total) * 100) if total > 0 else 0

    # Tier label
    if score >= 90:
        tier = "audited"
    elif score >= 75:
        tier = "certified"
    elif score >= 50:
        tier = "verified"
    else:
        tier = "failed"

    return JSONResponse(
        {
            "target_id": target_id,
            "score": score,
            "percentile": percentile,
            "rank": rank,
            "total_evaluated": total,
            "tier": tier,
            "top_pct": max(1, 100 - percentile),  # "Top X%"
        },
        headers={"Cache-Control": "public, max-age=300"},
    )

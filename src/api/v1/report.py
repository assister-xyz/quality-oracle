"""Quality Report API — serves the latest MCP Quality Report data."""
import logging
from fastapi import APIRouter, Query

from src.storage.mongodb import scores_col

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/report")
async def get_quality_report(
    min_score: int = Query(0, description="Minimum score filter"),
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(100, description="Max servers to include"),
):
    """Get the latest MCP Quality Report data.

    Public endpoint — no auth required. Powers the /report page on laureum.ai.
    """
    query = {"current_score": {"$exists": True, "$gt": min_score}}
    if category:
        query["detected_domain"] = category

    cursor = scores_col().find(query).sort("current_score", -1).limit(limit)
    servers = await cursor.to_list(limit)

    if not servers:
        return {
            "total_servers": 0,
            "message": "No scored servers found. Run batch evaluation first.",
        }

    scores = [s["current_score"] for s in servers]
    n = len(scores)
    avg = sum(scores) / n
    sorted_scores = sorted(scores)
    median = sorted_scores[n // 2] if n % 2 else (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2

    tiers = {"expert": 0, "proficient": 0, "basic": 0, "failed": 0}
    categories = {}
    dim_totals = {}
    dim_counts = {}

    for s in servers:
        tier = s.get("tier", "failed")
        tiers[tier] = tiers.get(tier, 0) + 1

        cat = s.get("detected_domain", "general")
        if cat not in categories:
            categories[cat] = {"count": 0, "total": 0}
        categories[cat]["count"] += 1
        categories[cat]["total"] += s["current_score"]

        for dim, score in s.get("dimensions", {}).items():
            if isinstance(score, (int, float)):
                dim_totals[dim] = dim_totals.get(dim, 0) + score
                dim_counts[dim] = dim_counts.get(dim, 0) + 1

    tier_pcts = {k: round(v / n * 100, 1) for k, v in tiers.items()}
    cat_avgs = {
        k: {"count": v["count"], "avg_score": round(v["total"] / v["count"], 1)}
        for k, v in sorted(categories.items(), key=lambda x: x[1]["total"] / x[1]["count"], reverse=True)
    }
    dim_avgs = {
        dim: round(dim_totals[dim] / dim_counts[dim], 1)
        for dim in sorted(dim_totals.keys())
        if dim_counts.get(dim, 0) > 0
    }

    return {
        "statistics": {
            "total_servers": n,
            "avg_score": round(avg, 1),
            "median_score": round(median, 1),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_rate": round((tiers.get("expert", 0) + tiers.get("proficient", 0)) / n * 100, 1),
            "tier_distribution": tiers,
            "tier_percentages": tier_pcts,
            "category_breakdown": cat_avgs,
            "dimension_averages": dim_avgs,
        },
        "servers": [
            {
                "rank": i + 1,
                "name": _extract_name(s),
                "url": s["target_id"],
                "score": s["current_score"],
                "tier": s.get("tier", "failed"),
                "confidence": s.get("confidence", 0),
                "tools_count": s.get("tools_count", 0),
                "category": s.get("detected_domain", "general"),
                "dimensions": s.get("dimensions", {}),
                "last_evaluated": s.get("last_evaluated_at"),
            }
            for i, s in enumerate(servers)
        ],
    }


def _extract_name(score_doc: dict) -> str:
    """Extract a display name from URL."""
    url = score_doc.get("target_id", "")
    # Try common patterns
    import re
    match = re.search(r"https?://(?:mcp\.)?([^./]+)", url)
    if match:
        return match.group(1).replace("-", " ").title()
    return url[:50]

"""Score lookup endpoints."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from src.storage.mongodb import scores_col
from src.storage.cache import get_cached_score, cache_score
from src.storage.models import ScoreResponse, QualityTier, TargetType, normalize_eval_mode
from src.auth.dependencies import get_api_key
from src.auth.rate_limiter import check_score_lookup_limit, add_rate_limit_headers

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/score/{target_id:path}/share")
async def get_share_data(
    target_id: str,
    response: Response,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get pre-formatted share data for social media sharing.

    Returns tweet text, LinkedIn text, OG image URL, permalink,
    percentile, and shields.io badge data.
    """
    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]
    allowed, remaining, limit = await check_score_lookup_limit(key_hash, tier)
    add_rate_limit_headers(response, tier, limit, remaining)
    if not allowed:
        raise HTTPException(status_code=429, detail="Score lookup rate limit exceeded")

    doc = await scores_col().find_one({"target_id": target_id})
    if not doc:
        raise HTTPException(status_code=404, detail="No quality score found for this target")

    score = doc.get("current_score", 0)
    score_tier = doc.get("tier", "failed")

    # Compute percentile
    total = await scores_col().count_documents({})
    lower = await scores_col().count_documents({"current_score": {"$lt": score}})
    percentile = round((lower / total) * 100) if total > 0 else 0
    top_pct = max(1, 100 - percentile)

    # Infer name from target_id (URL)
    name = target_id
    try:
        from urllib.parse import urlparse
        parsed = urlparse(target_id)
        hostname = parsed.hostname or target_id
        name = (hostname
                .replace("mcp.", "")
                .replace("docs.", "")
                .replace("www.", "")
                .split(".")[0]
                .capitalize())
    except Exception:
        pass

    # URLs
    from src.config import settings
    base = settings.base_url.rstrip("/")
    profile_url = f"https://laureum.ai/agent/{target_id}"
    og_image_url = f"https://laureum.ai/api/og?id={target_id}"
    badge_svg_url = f"{base}/v1/badge/{target_id}.svg"
    shields_url = f"{base}/v1/shields/{target_id}.json"

    # Pre-formatted social text
    tier_label = score_tier.capitalize() if score_tier != "failed" else "Evaluated"
    tweet = (
        f"My MCP server scored {score}/100 on @LaureumAI "
        f"- Top {top_pct}%! "
        f"Evaluate yours: {profile_url}"
    )
    linkedin = (
        f"Just verified my AI agent with Laureum.ai - "
        f"scored {score}/100 ({tier_label}), placing in the Top {top_pct}% "
        f"of evaluated MCP servers.\n\n"
        f"Laureum provides multi-judge consensus scoring across 6 quality dimensions "
        f"with signed attestations.\n\n"
        f"Evaluate your agent: {profile_url}"
    )

    return {
        "target_id": target_id,
        "name": name,
        "score": score,
        "tier": score_tier,
        "percentile": percentile,
        "top_pct": top_pct,
        "total_evaluated": total,
        "tweet_text": tweet,
        "linkedin_text": linkedin,
        "profile_url": profile_url,
        "og_image_url": og_image_url,
        "badge_svg_url": badge_svg_url,
        "shields_url": shields_url,
        "shields_badge": {
            "schemaVersion": 1,
            "label": "Laureum",
            "message": f"{score}/100 {tier_label}",
            "color": {
                "audited": "D4AF37",
                "certified": "A8A8A8",
                "verified": "C38133",
            }.get(score_tier, "535862"),
        },
        "embed_markdown": f"[![Laureum {tier_label}]({badge_svg_url})]({profile_url})",
        "embed_html": (
            f"<a href='{profile_url}'>"
            f"<img src='{badge_svg_url}' alt='Laureum {tier_label}' height='80' />"
            f"</a>"
        ),
    }


@router.get("/score/{target_id:path}", response_model=ScoreResponse)
async def get_score(
    target_id: str,
    response: Response,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get the quality score for a target (MCP server, agent, or skill)."""
    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]
    allowed, remaining, limit = await check_score_lookup_limit(key_hash, tier)
    add_rate_limit_headers(response, tier, limit, remaining)
    if not allowed:
        raise HTTPException(status_code=429, detail="Score lookup rate limit exceeded")
    # Check cache first
    cached = await get_cached_score(target_id)
    if cached:
        return ScoreResponse(**cached)

    # Lookup in MongoDB
    doc = await scores_col().find_one({"target_id": target_id})
    if not doc:
        raise HTTPException(status_code=404, detail="No quality score found for this target")

    # Build tool_scores from stored data
    raw_tool_scores = doc.get("tool_scores", {})
    from src.storage.models import ToolScore
    parsed_tool_scores = {}
    for tname, tdata in raw_tool_scores.items():
        if isinstance(tdata, dict):
            parsed_tool_scores[tname] = ToolScore(
                score=tdata.get("score", 0),
                tests_passed=tdata.get("tests_passed", 0),
                tests_total=tdata.get("tests_total", 0),
            )

    resp_data = ScoreResponse(
        target_id=doc["target_id"],
        target_type=TargetType(doc.get("target_type", "mcp_server")),
        score=doc.get("current_score", 0),
        tier=QualityTier(doc.get("tier", "failed")),
        confidence=doc.get("confidence", 0),
        evaluation_count=doc.get("evaluation_count", 0),
        last_evaluated_at=doc.get("last_evaluated_at"),
        tool_scores=parsed_tool_scores,
        last_eval_mode=normalize_eval_mode(doc.get("last_eval_mode")),
        manifest_hash=doc.get("manifest_hash"),
    )

    # Cache for 5 min
    await cache_score(target_id, resp_data.model_dump(mode="json"))

    return resp_data


@router.get("/scores")
async def list_scores(
    response: Response,
    domain: Optional[str] = None,
    min_score: int = Query(0, ge=0, le=100),
    tier: Optional[str] = None,
    sort: str = Query("score", regex="^(score|name|date)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    api_key_doc: dict = Depends(get_api_key),
):
    """List quality scores with filtering and pagination."""
    key_tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]
    allowed, remaining, rl_limit = await check_score_lookup_limit(key_hash, key_tier)
    add_rate_limit_headers(response, key_tier, rl_limit, remaining)
    if not allowed:
        raise HTTPException(status_code=429, detail="Score lookup rate limit exceeded")
    query = {}
    if min_score > 0:
        query["current_score"] = {"$gte": min_score}
    if tier:
        query["tier"] = tier
    if domain:
        query["detected_domain"] = domain

    sort_field = {"score": "current_score", "date": "last_evaluated_at", "name": "target_id"}
    sort_key = sort_field.get(sort, "current_score")
    sort_dir = -1 if sort != "name" else 1

    cursor = scores_col().find(query).sort(sort_key, sort_dir).skip(offset).limit(limit)
    items = []
    async for doc in cursor:
        items.append({
            "target_id": doc["target_id"],
            "target_type": doc.get("target_type", "mcp_server"),
            "score": doc.get("current_score", 0),
            "tier": doc.get("tier", "failed"),
            "confidence": doc.get("confidence", 0),
            "evaluation_count": doc.get("evaluation_count", 0),
            "last_evaluated_at": doc.get("last_evaluated_at"),
            "last_evaluation_id": doc.get("last_evaluation_id"),
            "dimensions": doc.get("dimensions", {}),
            "tool_scores": doc.get("tool_scores", {}),
            "safety_report": doc.get("safety_report", []),
            "latency_stats": doc.get("latency_stats", {}),
            "duration_ms": doc.get("duration_ms"),
            "last_eval_mode": normalize_eval_mode(doc.get("last_eval_mode")),
            "manifest_hash": doc.get("manifest_hash"),
            "detected_domain": doc.get("detected_domain", "general"),
            "detected_domains": doc.get("detected_domains", []),
        })

    total = await scores_col().count_documents(query)

    return {"items": items, "total": total, "limit": limit, "offset": offset}

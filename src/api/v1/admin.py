"""Admin endpoints — import scores, manage data."""
import logging
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.storage.mongodb import scores_col
from src.auth.dependencies import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


class ScoreImport(BaseModel):
    target_id: str
    score: int
    tier: str
    dimensions: dict = {}
    tool_scores: dict = {}
    tools_count: int = 0
    manifest_hash: str = ""
    detected_domain: str = "general"
    detected_domains: list = []
    transport: str = "streamable_http"


class ImportRequest(BaseModel):
    scores: List[ScoreImport]
    source: str = "local_scan"


@router.post("/admin/import-scores")
async def import_scores(
    req: ImportRequest,
    api_key_doc: dict = Depends(get_api_key),
):
    """Import evaluation scores from local scans. Requires marketplace-tier API key."""
    tier = api_key_doc.get("tier", "free")
    if tier != "marketplace":
        raise HTTPException(403, "Admin endpoints require marketplace-tier API key")

    inserted = 0
    updated = 0
    errors = []

    for score in req.scores:
        try:
            existing = await scores_col().find_one({"target_id": score.target_id})
            doc = {
                "target_id": score.target_id,
                "target_type": "mcp_server",
                "score": score.score,
                "tier": score.tier,
                "confidence": 0.5,
                "evaluation_count": 1,
                "last_evaluated_at": datetime.now(timezone.utc),
                "dimensions": score.dimensions,
                "tool_scores": score.tool_scores,
                "tools_count": score.tools_count,
                "manifest_hash": score.manifest_hash,
                "detected_domain": score.detected_domain,
                "detected_domains": score.detected_domains,
                "source": req.source,
            }

            if existing:
                await scores_col().update_one(
                    {"target_id": score.target_id},
                    {"$set": doc, "$inc": {"evaluation_count": 1}},
                )
                updated += 1
            else:
                await scores_col().insert_one(doc)
                inserted += 1
        except Exception as e:
            errors.append({"target_id": score.target_id, "error": str(e)})

    return {
        "inserted": inserted,
        "updated": updated,
        "errors": errors,
        "total_scores": await scores_col().count_documents({}),
    }

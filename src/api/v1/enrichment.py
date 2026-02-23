"""Agent Card enrichment endpoint — add quality data to A2A Agent Cards."""
import logging

from fastapi import APIRouter, Depends

from src.storage.models import EnrichAgentCardRequest, EnrichAgentCardResponse
from src.storage.mongodb import scores_col
from src.auth.dependencies import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/enrich-agent-card", response_model=EnrichAgentCardResponse)
async def enrich_agent_card(
    request: EnrichAgentCardRequest,
    api_key_doc: dict = Depends(get_api_key),
):
    """
    Add quality data to an A2A Agent Card.

    Looks up agent by matching card.url or card.name against evaluated targets.
    If not yet evaluated, returns original card + evaluate_url.
    """
    card = request.agent_card

    # Try to find a matching score by URL or name
    target_url = card.get("url", "")
    target_name = card.get("name", "")

    score_doc = None
    if target_url:
        score_doc = await scores_col().find_one({"target_id": target_url})
    if not score_doc and target_name:
        score_doc = await scores_col().find_one({"target_id": target_name})

    if not score_doc:
        return EnrichAgentCardResponse(
            enriched_card=card,
            quality_data=None,
            evaluate_url="/v1/evaluate",
        )

    quality_data = {
        "score": score_doc.get("current_score", 0),
        "tier": score_doc.get("tier", "failed"),
        "confidence": score_doc.get("confidence", 0),
        "evaluation_version": score_doc.get("evaluation_version"),
        "last_evaluated": score_doc.get("last_evaluated_at").isoformat() if score_doc.get("last_evaluated_at") else None,
        "badge_url": score_doc.get("badge_url"),
    }

    # Enrich the card with quality extension
    enriched = dict(card)
    extensions = enriched.get("extensions", {})
    extensions["quality_oracle"] = {
        "provider": "assisterr.ai",
        "score": quality_data["score"],
        "tier": quality_data["tier"],
        "confidence": quality_data["confidence"],
        "evaluation_version": quality_data["evaluation_version"],
        "last_evaluated": quality_data["last_evaluated"],
        "badge_url": quality_data["badge_url"],
        "verify_url": f"/v1/score/{score_doc['target_id']}",
    }
    enriched["extensions"] = extensions

    return EnrichAgentCardResponse(
        enriched_card=enriched,
        quality_data=quality_data,
    )

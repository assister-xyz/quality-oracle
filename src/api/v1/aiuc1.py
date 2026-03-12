"""AIUC-1 alignment report endpoints."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Response

from src.storage.mongodb import scores_col, evaluations_col
from src.auth.dependencies import get_api_key
from src.auth.rate_limiter import check_score_lookup_limit, add_rate_limit_headers
from src.standards.aiuc1_mapping import (
    generate_aiuc1_report,
    get_covered_controls,
    get_uncovered_mandatory_controls,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/aiuc1-report/{target_id:path}")
async def get_aiuc1_report(
    target_id: str,
    response: Response,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get AIUC-1 alignment report for a specific evaluated target.

    Returns per-control coverage mapping, domain breakdown, and
    evaluation-specific data if the target has been evaluated.
    """
    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]
    allowed, remaining, limit = await check_score_lookup_limit(key_hash, tier)
    add_rate_limit_headers(response, tier, limit, remaining)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Load evaluation data for enrichment
    score_doc = await scores_col().find_one({"target_id": target_id})

    evaluation_result = None
    if score_doc:
        # Find latest completed evaluation for richer data
        latest_eval = await evaluations_col().find_one(
            {"target_id": target_id, "status": "completed"},
            sort=[("completed_at", -1)],
        )
        if latest_eval:
            evaluation_result = {
                "overall_score": score_doc.get("current_score", 0),
                "dimensions": score_doc.get("dimensions", {}),
                "safety_report": (latest_eval.get("report") or {}).get("safety_report", []),
            }

    report = generate_aiuc1_report(evaluation_result)

    # Add target info
    report["target_id"] = target_id
    if score_doc:
        report["target_score"] = score_doc.get("current_score", 0)
        report["target_tier"] = score_doc.get("tier", "unknown")
    else:
        report["target_score"] = None
        report["target_tier"] = None
        report["note_no_evaluation"] = (
            "This target has not been evaluated yet. "
            "The report shows general AIUC-1 coverage capabilities."
        )

    return report


@router.get("/aiuc1-summary")
async def get_aiuc1_summary(
    response: Response,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get a summary of AgentTrust's AIUC-1 alignment capabilities.

    Returns coverage stats, covered control IDs, and recommendations
    for uncovered mandatory controls.
    """
    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]
    allowed, remaining, limit = await check_score_lookup_limit(key_hash, tier)
    add_rate_limit_headers(response, tier, limit, remaining)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    report = generate_aiuc1_report()

    return {
        "aiuc1_version": report["aiuc1_version"],
        "coverage_percentage": report["coverage_percentage"],
        "mandatory_coverage": report["mandatory_coverage"],
        "total_controls": report["total_controls"],
        "fully_covered": report["controls_fully_covered"],
        "partially_covered": report["controls_partially_covered"],
        "not_covered": report["controls_not_covered"],
        "covered_control_ids": get_covered_controls(),
        "uncovered_mandatory": get_uncovered_mandatory_controls(),
        "domain_summary": report["domain_summary"],
        "note": report["note"],
        "continuous_monitoring_note": report["continuous_monitoring_note"],
    }

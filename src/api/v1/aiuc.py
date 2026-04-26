"""AIUC v1 alignment report endpoint — `GET /v1/aiuc-report/{eval_id}`.

QO-053-I §AC8 — exposes `aiuc1_mapping.py` output as a public endpoint
keyed by **evaluation id** (not target id; that flavour stays at the
existing `/v1/aiuc1-report/{target_id}` endpoint in `aiuc1.py`).

Use case: a verifier consuming an AQVC sees `credentialSubject.aiucAlignment`
pointing at this URL — they can fetch the full per-control coverage map
without re-running the eval.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Response

from src.auth.dependencies import get_api_key
from src.auth.rate_limiter import add_rate_limit_headers, check_score_lookup_limit
from src.standards.aiuc1_mapping import (
    generate_aiuc1_report,
    get_covered_controls,
    get_uncovered_mandatory_controls,
)
from src.storage.mongodb import evaluations_col, scores_col

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/aiuc-report/{eval_id}")
async def get_aiuc_report_by_eval_id(
    eval_id: str,
    response: Response,
    api_key_doc: dict = Depends(get_api_key),
):
    """Return the AIUC-1 alignment report for the named evaluation.

    Args:
        eval_id: The `_id` of the evaluation document (UUID string).

    Returns: dict per `aiuc1_mapping.generate_aiuc1_report()`, plus
    `eval_id`, `target_id`, `target_score`, and `target_tier` for context.

    AC8: this is the canonical AIUC-1 endpoint surfaced from AQVC
    `credentialSubject.aiucAlignment` references.
    """
    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]
    allowed, remaining, limit = await check_score_lookup_limit(key_hash, tier)
    add_rate_limit_headers(response, tier, limit, remaining)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    eval_doc = await evaluations_col().find_one({"_id": eval_id})
    if not eval_doc:
        raise HTTPException(status_code=404, detail=f"evaluation {eval_id} not found")

    target_id = eval_doc.get("target_id")
    score_doc = await scores_col().find_one({"target_id": target_id}) if target_id else None

    evaluation_result = None
    if eval_doc.get("status") == "completed":
        report_block = eval_doc.get("report") or {}
        evaluation_result = {
            "overall_score": (score_doc or {}).get("current_score", report_block.get("overall_score", 0)),
            "dimensions": (score_doc or {}).get("dimensions", report_block.get("dimensions", {})),
            "safety_report": report_block.get("safety_report", []),
        }

    report = generate_aiuc1_report(evaluation_result)
    report["eval_id"] = eval_id
    report["target_id"] = target_id
    report["target_score"] = (score_doc or {}).get("current_score") if score_doc else None
    report["target_tier"] = (score_doc or {}).get("tier") if score_doc else None
    report["covered_control_ids"] = get_covered_controls()
    report["uncovered_mandatory"] = get_uncovered_mandatory_controls()

    return report

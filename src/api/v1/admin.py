"""Admin endpoints — import scores, manage data, batch evaluation."""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from src.storage.mongodb import (
    scores_col,
    evaluations_col,
    tool_calls_col,
    judge_calls_col,
    consensus_votes_col,
    sanitization_events_col,
    probe_executions_col,
)
from src.auth.dependencies import get_api_key
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory batch job tracker (simple; production would use Redis/DB)
_batch_jobs: dict = {}


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


# ── Batch Evaluation ────────────────────────────────────────────────────────


class BatchEvaluateRequest(BaseModel):
    urls: List[str]
    level: int = 2
    force: bool = False
    timeout_seconds: Optional[int] = None  # QO-050: per-server cap (default 180)


# QO-050: Hard cap per server in batch eval (prevents 1 slow URL hanging entire batch).
# Sourced from settings so it's overridable via env without a code change.
BATCH_PER_SERVER_TIMEOUT_DEFAULT = settings.batch_per_server_timeout_seconds


class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    total: int
    completed: int
    succeeded: int
    failed: int
    skipped: int
    results: Optional[list] = None


@router.post("/admin/batch-evaluate")
async def batch_evaluate(
    req: BatchEvaluateRequest,
    background_tasks: BackgroundTasks,
    api_key_doc: dict = Depends(get_api_key),
):
    """Trigger batch evaluation of multiple MCP servers.

    Runs evaluations in background. Returns job_id for progress tracking.
    Requires marketplace-tier API key.
    """
    tier = api_key_doc.get("tier", "free")
    if tier != "marketplace":
        raise HTTPException(403, "Admin endpoints require marketplace-tier API key")

    if not req.urls:
        raise HTTPException(400, "urls list cannot be empty")

    if len(req.urls) > 200:
        raise HTTPException(400, "Maximum 200 URLs per batch")

    job_id = str(uuid4())
    _batch_jobs[job_id] = {
        "status": "running",
        "total": len(req.urls),
        "completed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "results": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    background_tasks.add_task(
        _run_batch_evaluation,
        job_id,
        req.urls,
        req.level,
        req.force,
        req.timeout_seconds or BATCH_PER_SERVER_TIMEOUT_DEFAULT,
    )

    return {"job_id": job_id, "total": len(req.urls), "status": "running"}


@router.get("/admin/batch-evaluate/{job_id}")
async def get_batch_status(
    job_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Check progress of a batch evaluation job."""
    tier = api_key_doc.get("tier", "free")
    if tier != "marketplace":
        raise HTTPException(403, "Admin endpoints require marketplace-tier API key")

    job = _batch_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Batch job not found")

    return BatchJobStatus(
        job_id=job_id,
        status=job["status"],
        total=job["total"],
        completed=job["completed"],
        succeeded=job["succeeded"],
        failed=job["failed"],
        skipped=job["skipped"],
        results=job["results"] if job["status"] == "completed" else None,
    )


async def _run_batch_evaluation(
    job_id: str, urls: list, level: int, force: bool,
    per_server_timeout: int = BATCH_PER_SERVER_TIMEOUT_DEFAULT,
):
    """Run batch evaluations in background.

    QO-047 fix: each batch eval now creates a real evaluation_id, inserts
    into quality__evaluations, and sets the audit contextvars so all
    downstream tool/judge/probe calls are captured by the audit trail.

    QO-050: each batch eval is wrapped in asyncio.wait_for() with
    per_server_timeout (default 180s) so a single slow MCP server cannot
    hang the entire batch.
    """
    from uuid import uuid4
    from src.core.mcp_client import (
        manifest_and_evaluate,
        current_evaluation_id, current_target_id, _call_index_counter,
    )
    from src.core.evaluator import Evaluator
    from src.core.llm_judge import LLMJudge
    from src.core.quick_scan import quick_scan
    from src.config import settings
    from src.storage.mongodb import evaluations_col

    job = _batch_jobs[job_id]

    # Create judge
    judge = LLMJudge(
        api_key=settings.cerebras_api_key or None,
        model=settings.cerebras_model,
        provider="cerebras",
        base_url=settings.cerebras_base_url,
        fallback_key=settings.groq_api_key or None,
        fallback_model=settings.groq_model,
        fallback_provider="groq",
        fallback2_key=settings.openrouter_api_key or None,
        fallback2_model=settings.openrouter_model,
        fallback2_provider="openrouter",
    )

    for url in urls:
        result = {"url": url, "status": "pending", "score": None, "tier": None, "error": None}

        try:
            # Skip already scored unless force
            if not force:
                existing = await scores_col().find_one({"target_id": url})
                if existing:
                    result["status"] = "skipped"
                    result["score"] = existing.get("current_score")
                    result["tier"] = existing.get("tier")
                    job["skipped"] += 1
                    job["completed"] += 1
                    job["results"].append(result)
                    continue

            if level == 1:
                # L1 quick scan
                scan_result = await quick_scan(url)
                result["status"] = "success" if scan_result.reachable else "error"
                result["score"] = scan_result.manifest_score
                result["tier"] = scan_result.estimated_tier
                if not scan_result.reachable:
                    result["error"] = scan_result.error
                    job["failed"] += 1
                else:
                    job["succeeded"] += 1
            else:
                # L2 functional eval
                # QO-047: create a real evaluation record so audit trail works
                evaluation_id = str(uuid4())
                now = datetime.now(timezone.utc)
                await evaluations_col().insert_one({
                    "_id": evaluation_id,
                    "target_id": url,
                    "target_type": "mcp_server",
                    "target_url": url,
                    "status": "running",
                    "level": 2,
                    "evaluation_version": "v1.0",
                    "eval_mode": "certified",
                    "created_at": now,
                    "source": "batch_api",
                    "batch_job_id": job_id,
                })

                # QO-047: set audit contextvars so all downstream tool/judge
                # calls are linked to this evaluation_id
                current_evaluation_id.set(evaluation_id)
                current_target_id.set(url)
                _call_index_counter.set(0)

                # QO-050 timeout wraps manifest + eval together. QO-049
                # single-session flow avoids back-to-back handshakes that
                # some servers (Peek/Browserbase/CoinGecko) can't handle.
                manifest, tool_responses = await asyncio.wait_for(
                    manifest_and_evaluate(url), timeout=per_server_timeout
                )

                evaluator = Evaluator(llm_judge=judge)
                eval_result = await asyncio.wait_for(
                    evaluator.evaluate_full(
                        target_id=url,
                        server_url=url,
                        tool_responses=tool_responses,
                        manifest=manifest,
                        run_safety=True,
                    ),
                    timeout=per_server_timeout,
                )

                result["status"] = "success"
                result["score"] = eval_result.overall_score
                result["tier"] = eval_result.tier
                result["evaluation_id"] = evaluation_id

                completed_at = datetime.now(timezone.utc)
                dimensions = {}
                if eval_result.dimensions:
                    dimensions = {k: v["score"] for k, v in eval_result.dimensions.items()}

                # Mark evaluation as completed
                await evaluations_col().update_one(
                    {"_id": evaluation_id},
                    {"$set": {
                        "status": "completed",
                        "completed_at": completed_at,
                        "scores": {
                            "overall_score": eval_result.overall_score,
                            "tier": eval_result.tier,
                            "confidence": eval_result.confidence,
                            "dimensions": dimensions,
                            "tool_scores": eval_result.tool_scores,
                        },
                    }},
                )

                # Update score record with last_evaluation_id (so audit lookup works)
                await scores_col().update_one(
                    {"target_id": url},
                    {
                        "$set": {
                            "target_id": url,
                            "target_type": "mcp_server",
                            "current_score": eval_result.overall_score,
                            "tier": eval_result.tier,
                            "confidence": eval_result.confidence,
                            "evaluation_version": "v1.0",
                            "last_evaluated_at": completed_at,
                            "last_evaluation_id": evaluation_id,
                            "tool_scores": eval_result.tool_scores,
                            "dimensions": dimensions,
                            "source": "batch_api",
                        },
                        "$inc": {"evaluation_count": 1},
                        "$setOnInsert": {"first_evaluated_at": completed_at},
                    },
                    upsert=True,
                )
                job["succeeded"] += 1

        except asyncio.TimeoutError:
            # QO-050: per-server timeout
            result["status"] = "error"
            result["error"] = f"timeout after {per_server_timeout}s"
            job["failed"] += 1
            logger.warning(f"Batch eval TIMEOUT for {url} after {per_server_timeout}s")
            # Mark eval doc as failed for audit trail (if eval_id was created)
            try:
                if 'evaluation_id' in locals():
                    from src.storage.mongodb import evaluations_col
                    await evaluations_col().update_one(
                        {"_id": evaluation_id},
                        {"$set": {
                            "status": "failed",
                            "completed_at": datetime.now(timezone.utc),
                            "error": f"timeout after {per_server_timeout}s",
                        }},
                    )
                    result["evaluation_id"] = evaluation_id
            except Exception as e:
                logger.warning(f"Failed to mark eval as failed: {e}")
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:300]
            job["failed"] += 1
            logger.error(f"Batch eval failed for {url}: {e}")

        job["completed"] += 1
        job["results"].append(result)

        # Rate limit between evaluations
        await asyncio.sleep(2.5)

    job["status"] = "completed"
    logger.info(
        f"Batch job {job_id} completed: "
        f"{job['succeeded']} ok, {job['failed']} failed, {job['skipped']} skipped"
    )


# ── QO-047: Audit Trail Endpoint ─────────────────────────────────────────────


@router.get("/admin/eval/{evaluation_id}/audit")
async def get_eval_audit_trail(
    evaluation_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Return the complete audit trail for an evaluation (QO-047).

    Joins evaluation doc + tool_calls + judge_calls + consensus_votes
    + sanitization_events + probe_executions for full debug capability.
    Requires team or marketplace tier API key.
    """
    if api_key_doc.get("tier") not in ("marketplace", "team"):
        raise HTTPException(
            status_code=403,
            detail="Team or marketplace tier required for audit trail access",
        )

    # Fetch evaluation
    eval_doc = await evaluations_col().find_one({"_id": evaluation_id})
    if not eval_doc:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    # Fetch all linked audit records (parallel-friendly via async iteration)
    tool_calls = await tool_calls_col().find(
        {"evaluation_id": evaluation_id}
    ).sort("call_index", 1).to_list(1000)

    judge_calls = await judge_calls_col().find(
        {"evaluation_id": evaluation_id}
    ).sort("call_index", 1).to_list(500)

    consensus_votes = await consensus_votes_col().find(
        {"evaluation_id": evaluation_id}
    ).to_list(500)

    sanitization_events = await sanitization_events_col().find(
        {"evaluation_id": evaluation_id}
    ).to_list(500)

    probe_executions = await probe_executions_col().find(
        {"evaluation_id": evaluation_id}
    ).to_list(200)

    # Strip _id fields (Mongo ObjectIds aren't JSON-serializable)
    eval_doc.pop("_id", None)
    for collection_list in (tool_calls, judge_calls, consensus_votes, sanitization_events, probe_executions):
        for doc in collection_list:
            doc.pop("_id", None)

    return {
        "evaluation": eval_doc,
        "audit_summary": {
            "tool_calls_count": len(tool_calls),
            "judge_calls_count": len(judge_calls),
            "consensus_votes_count": len(consensus_votes),
            "sanitization_events_count": len(sanitization_events),
            "probe_executions_count": len(probe_executions),
            "total_audit_records": (
                len(tool_calls) + len(judge_calls) + len(consensus_votes)
                + len(sanitization_events) + len(probe_executions)
            ),
        },
        "tool_calls": tool_calls,
        "judge_calls": judge_calls,
        "consensus_votes": consensus_votes,
        "sanitization_events": sanitization_events,
        "probe_executions": probe_executions,
    }

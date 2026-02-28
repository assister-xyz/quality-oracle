"""Evaluation endpoints — submit and check evaluation status."""
import hashlib
import hmac
import logging
from datetime import datetime
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Response

from src.storage.models import (
    EvaluateRequest,
    EvaluateResponse,
    EvaluationStatus,
    EvalStatus,
    EvalLevel,
    WebhookPayload,
)
from src.storage.mongodb import evaluations_col, scores_col, score_history_col
from src.core.evaluator import Evaluator
from src.core.llm_judge import LLMJudge
from src.core.attestation import create_attestation
from src.core.scoring import aggregate_scores
from src.core import mcp_client
from src.auth.dependencies import get_api_key
from src.auth.rate_limiter import (
    check_eval_rate_limit,
    is_eval_level_allowed,
    add_rate_limit_headers,
)
from src.config import settings
from src.payments.x402 import require_payment

logger = logging.getLogger(__name__)


def _select_best_tool(question: str, tools: list) -> dict:
    """Select the best tool to answer a question based on keyword overlap.

    Scores each tool by overlap between question words and tool name + description.
    Returns the tool dict with highest score (falls back to first tool).
    """
    if not tools:
        return {}
    if len(tools) == 1:
        return tools[0]

    question_words = set(question.lower().split())
    best_tool = tools[0]
    best_score = -1

    for tool in tools:
        name_words = set(tool.get("name", "").lower().replace("_", " ").replace("-", " ").split())
        desc_words = set(tool.get("description", "").lower().split())
        tool_words = name_words | desc_words
        overlap = len(question_words & tool_words)
        if overlap > best_score:
            best_score = overlap
            best_tool = tool
    return best_tool


def _construct_arguments(question: str, tool: dict) -> dict:
    """Construct tool arguments by mapping the question to the primary string parameter.

    Reads inputSchema, finds the first required string param (or first string param),
    and maps the question text to it.
    """
    schema = tool.get("inputSchema", tool.get("parameters", {}))
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        return {"query": question}

    # Prefer required string params, then any string param
    primary_param = None
    for key, prop in properties.items():
        if prop.get("type") == "string":
            if key in required:
                primary_param = key
                break
            if primary_param is None:
                primary_param = key

    if primary_param:
        return {primary_param: question}

    # Fallback: use first parameter regardless of type
    first_key = next(iter(properties), None)
    if first_key:
        return {first_key: question}

    return {"query": question}
router = APIRouter()

EVALUATION_VERSION = settings.evaluation_version


def _get_judge() -> LLMJudge:
    return LLMJudge(
        api_key=settings.openai_api_key or None,
        model=settings.openai_model,
        provider="openai",
        base_url=settings.openai_base_url,
        fallback_key=settings.deepseek_api_key or None,
        fallback_model=settings.deepseek_model,
        fallback_provider="deepseek",
        fallback2_key=settings.groq_api_key or None,
        fallback2_model=settings.groq_model,
        fallback2_provider="groq",
    )


@router.post("/evaluate", response_model=EvaluateResponse)
async def submit_evaluation(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    api_key_doc: dict = Depends(get_api_key),
    x_payment: str | None = Header(None, alias="X-Payment"),
):
    """Submit an MCP server or agent for quality evaluation.

    For paid levels (2, 3), include X-Payment header with transaction
    signature per x402 protocol. Level 1 is always free.
    """
    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]

    # Check evaluation level is allowed for this tier
    if not is_eval_level_allowed(tier, request.level.value):
        raise HTTPException(
            status_code=403,
            detail=f"Evaluation level {request.level.value} not available for '{tier}' tier",
        )

    # x402 payment check (returns None for free levels, receipt for paid)
    payment_receipt = await require_payment(
        level=request.level.value,
        tier=tier,
        x_payment=x_payment,
    )

    # Check rate limit
    allowed, remaining, limit = await check_eval_rate_limit(key_hash, tier)
    add_rate_limit_headers(response, tier, limit, remaining)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Monthly evaluation limit exceeded. Upgrade your tier.",
        )

    evaluation_id = str(uuid4())
    target_id = request.target_url  # Use URL as target_id for now

    doc = {
        "_id": evaluation_id,
        "target_id": target_id,
        "target_type": request.target_type.value,
        "target_url": request.target_url,
        "status": EvalStatus.PENDING.value,
        "level": request.level.value,
        "domains": request.domains,
        "connection_strategy": "sse",
        "evaluation_version": EVALUATION_VERSION,
        "webhook_url": request.webhook_url,
        "callback_secret": request.callback_secret,
        "payment": payment_receipt.to_dict() if payment_receipt else None,
        "created_at": datetime.utcnow(),
    }
    await evaluations_col().insert_one(doc)

    background_tasks.add_task(
        _run_evaluation, evaluation_id, request
    )

    estimated = {EvalLevel.MANIFEST: 5, EvalLevel.FUNCTIONAL: 60, EvalLevel.DOMAIN_EXPERT: 180}
    message = "Webhook recommended over polling for Level 2+ evaluations" if request.level.value >= 2 else ""

    return EvaluateResponse(
        evaluation_id=evaluation_id,
        status=EvalStatus.PENDING,
        estimated_time_seconds=estimated.get(request.level, 60),
        poll_url=f"/v1/evaluate/{evaluation_id}",
        message=message,
    )


@router.get("/evaluate/{evaluation_id}", response_model=EvaluationStatus)
async def get_evaluation_status(
    evaluation_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Check the status of an evaluation and get full report when completed."""
    doc = await evaluations_col().find_one({"_id": evaluation_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    result = None
    report = None
    score = None
    tier = None
    attestation_jwt = None
    badge_url = None

    if doc.get("status") == EvalStatus.COMPLETED.value and doc.get("scores"):
        from src.storage.models import ScoreResponse, QualityTier, TargetType
        scores = doc["scores"]
        score = scores.get("overall_score", 0)
        tier = scores.get("tier", "failed")

        result = ScoreResponse(
            target_id=doc["target_id"],
            target_type=TargetType(doc["target_type"]),
            score=score,
            tier=QualityTier(tier),
            confidence=scores.get("confidence", 0),
            domains=doc.get("domains", []),
            evaluation_version=doc.get("evaluation_version"),
        )

        report = doc.get("report")
        badge_url = f"{settings.base_url}/v1/badge/{doc['target_id']}.svg"

        if doc.get("attestation_id"):
            from src.storage.mongodb import attestations_col
            att = await attestations_col().find_one({"_id": doc["attestation_id"]})
            if att:
                attestation_jwt = att.get("attestation_jwt")

    return EvaluationStatus(
        evaluation_id=evaluation_id,
        status=EvalStatus(doc["status"]),
        progress_pct=doc.get("progress_pct", 0),
        score=score,
        tier=tier,
        evaluation_version=doc.get("evaluation_version"),
        report=report,
        attestation_jwt=attestation_jwt,
        badge_url=badge_url,
        result=result,
        error=doc.get("error"),
    )


async def _run_evaluation(evaluation_id: str, request: EvaluateRequest):
    """Run evaluation in background."""
    try:
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"status": EvalStatus.RUNNING.value, "progress_pct": 10}},
        )

        judge = _get_judge()
        evaluator = Evaluator(judge)

        # Step 1: Fetch real manifest from MCP server
        try:
            manifest = await mcp_client.get_server_manifest(request.target_url)
        except ConnectionError as e:
            logger.error(f"Cannot connect to target: {e}")
            await evaluations_col().update_one(
                {"_id": evaluation_id},
                {"$set": {
                    "status": EvalStatus.FAILED.value,
                    "error": f"Connection failed: {e}",
                }},
            )
            return

        # Store manifest in evaluation doc
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"target_manifest": manifest}},
        )

        # Step 2: Level 1 — Manifest validation
        manifest_result = evaluator.validate_manifest(manifest)
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"progress_pct": 30}},
        )

        if request.level == EvalLevel.MANIFEST:
            # Level 1 only: aggregate with manifest score only
            scores = aggregate_scores(
                tool_scores={},
                manifest_score=manifest_result.score,
            )
            scores["confidence"] = 0.5
            scores["manifest"] = manifest_result.to_dict()
            report = {
                "level1": {
                    "manifest_score": manifest_result.score,
                    "checks": manifest_result.checks,
                    "issues": manifest_result.warnings,
                },
                "level2": None,
                "level3": None,
            }
        else:
            # Step 3: Level 2 — Functional testing via real MCP calls
            await evaluations_col().update_one(
                {"_id": evaluation_id},
                {"$set": {"progress_pct": 40}},
            )

            tool_responses = await mcp_client.evaluate_server(request.target_url)

            await evaluations_col().update_one(
                {"_id": evaluation_id},
                {"$set": {"progress_pct": 60}},
            )

            # Judge the tool responses with full 6-axis evaluation
            eval_result = await evaluator.evaluate_full(
                target_id=request.target_url,
                server_url=request.target_url,
                tool_responses=tool_responses,
                manifest=manifest,
                run_safety=True,
            )

            await evaluations_col().update_one(
                {"_id": evaluation_id},
                {"$set": {"progress_pct": 75}},
            )

            # Step 4: Level 3 — Domain expert testing (if requested)
            domain_result = None
            if request.level == EvalLevel.DOMAIN_EXPERT and request.domains:
                async def answer_fn(question: str) -> str:
                    """Ask a domain question via the best-matching tool."""
                    tools = manifest.get("tools", [])
                    if not tools:
                        return ""
                    best_tool = _select_best_tool(question, tools)
                    tool_name = best_tool.get("name", tools[0]["name"])
                    arguments = _construct_arguments(question, best_tool)
                    resp = await mcp_client.call_tool(
                        request.target_url, tool_name, arguments
                    )
                    return resp.get("content", "")

                domain_result = await evaluator.evaluate_domain(
                    target_id=request.target_url,
                    domains=request.domains,
                    answer_fn=answer_fn,
                    question_count=10,
                )

            await evaluations_col().update_one(
                {"_id": evaluation_id},
                {"$set": {"progress_pct": 85}},
            )

            # Step 5: Aggregate scores with proper weights
            scores = aggregate_scores(
                tool_scores=eval_result.tool_scores,
                domain_scores=domain_result.domain_scores if domain_result else None,
                manifest_score=manifest_result.score,
            )
            scores["confidence"] = eval_result.confidence
            scores["manifest"] = manifest_result.to_dict()
            scores["tool_scores"] = eval_result.tool_scores
            scores["questions_asked"] = eval_result.questions_asked
            if eval_result.dimensions:
                scores["dimensions"] = {
                    k: v["score"] for k, v in eval_result.dimensions.items()
                }
            if eval_result.safety_report:
                scores["safety_report"] = eval_result.safety_report
            if eval_result.process_quality_report:
                scores["process_quality_report"] = eval_result.process_quality_report
            if eval_result.latency_stats:
                scores["latency_stats"] = eval_result.latency_stats
            if domain_result:
                scores["domain_scores"] = domain_result.domain_scores

            # Build comprehensive report with tool details
            # Compute per-tool latency from tool_responses
            tool_details = []
            all_latencies = []
            tools_passed = 0
            for tool_name, responses in tool_responses.items():
                latencies = [r.get("latency_ms", 0) for r in responses]
                all_latencies.extend(latencies)
                tool_score = eval_result.tool_scores.get(tool_name, {})
                passed = tool_score.get("tests_passed", 0)
                total = tool_score.get("tests_total", 0)
                if passed == total and total > 0:
                    tools_passed += 1
                tool_details.append({
                    "tool_name": tool_name,
                    "score": tool_score.get("score", 0),
                    "tests_passed": passed,
                    "tests_total": total,
                    "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
                    "responses": responses,
                })

            report = {
                "level1": {
                    "manifest_score": manifest_result.score,
                    "checks": manifest_result.checks,
                    "issues": manifest_result.warnings,
                },
                "level2": {
                    "tools_tested": len(tool_responses),
                    "tools_passed": tools_passed,
                    "tools_failed": len(tool_responses) - tools_passed,
                    "avg_latency_ms": int(sum(all_latencies) / len(all_latencies)) if all_latencies else 0,
                    "tool_details": tool_details,
                    "judge_responses": eval_result.judge_responses,
                },
                "level3": {
                    "domain_scores": domain_result.domain_scores,
                    "questions_asked": domain_result.questions_asked,
                    "judge_responses": domain_result.judge_responses,
                } if domain_result else None,
            }

        now = datetime.utcnow()

        # Create attestation
        attestation = create_attestation(
            target_id=request.target_url,
            target_type=request.target_type.value,
            target_name=manifest.get("name", request.target_url),
            evaluation_result=scores,
            evaluation_version=EVALUATION_VERSION,
        )
        attestation_id = attestation["_id"]
        from src.storage.mongodb import attestations_col
        await attestations_col().insert_one(attestation)

        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {
                "$set": {
                    "status": EvalStatus.COMPLETED.value,
                    "scores": scores,
                    "report": report,
                    "completed_at": now,
                    "progress_pct": 100,
                    "attestation_id": attestation_id,
                }
            },
        )

        # Update or create score record
        await scores_col().update_one(
            {"target_id": request.target_url},
            {
                "$set": {
                    "target_id": request.target_url,
                    "target_type": request.target_type.value,
                    "current_score": scores["overall_score"],
                    "tier": scores["tier"],
                    "confidence": scores["confidence"],
                    "evaluation_version": EVALUATION_VERSION,
                    "last_evaluated_at": now,
                },
                "$inc": {"evaluation_count": 1},
                "$setOnInsert": {"first_evaluated_at": now},
            },
            upsert=True,
        )

        # Record score history
        previous = await score_history_col().find_one(
            {"target_id": request.target_url},
            sort=[("recorded_at", -1)],
        )
        delta = scores["overall_score"] - previous["score"] if previous else None

        await score_history_col().insert_one({
            "target_id": request.target_url,
            "target_type": request.target_type.value,
            "evaluation_id": evaluation_id,
            "score": scores["overall_score"],
            "tier": scores["tier"],
            "confidence": scores["confidence"],
            "evaluation_version": EVALUATION_VERSION,
            "domain_scores": scores.get("domain_scores", {}),
            "recorded_at": now,
            "delta_from_previous": delta,
        })

        logger.info(f"Evaluation {evaluation_id} completed: score={scores['overall_score']}")

        # Deliver webhook if configured
        webhook_url = request.webhook_url
        if webhook_url:
            await _deliver_webhook(
                evaluation_id=evaluation_id,
                target_id=request.target_url,
                scores=scores,
                webhook_url=webhook_url,
                callback_secret=request.callback_secret,
                attestation_id=attestation_id,
            )

    except Exception as e:
        logger.error(f"Evaluation {evaluation_id} failed: {e}")
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"status": EvalStatus.FAILED.value, "error": str(e)}},
        )


async def _deliver_webhook(
    evaluation_id: str,
    target_id: str,
    scores: dict,
    webhook_url: str,
    callback_secret: str | None,
    attestation_id: str | None,
):
    """Deliver HMAC-signed webhook on evaluation completion."""
    payload = WebhookPayload(
        event="evaluation.completed",
        evaluation_id=evaluation_id,
        target_id=target_id,
        score=scores.get("overall_score", 0),
        tier=scores.get("tier", "failed"),
        report_url=f"{settings.base_url}/v1/evaluate/{evaluation_id}",
        badge_url=f"{settings.base_url}/v1/badge/{target_id}.svg",
        attestation_url=f"{settings.base_url}/v1/attestation/{attestation_id}" if attestation_id else None,
    )

    body = payload.model_dump_json()

    # HMAC signature if callback_secret provided
    signature = None
    if callback_secret:
        signature = hmac.new(
            callback_secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()
        payload.signature = signature

    headers = {"Content-Type": "application/json"}
    if signature:
        headers["X-Quality-Oracle-Signature"] = f"sha256={signature}"

    try:
        async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
            resp = await client.post(webhook_url, content=body, headers=headers)
            logger.info(f"Webhook delivered to {webhook_url}: status={resp.status_code}")
    except Exception as e:
        logger.warning(f"Webhook delivery failed for {evaluation_id}: {e}")

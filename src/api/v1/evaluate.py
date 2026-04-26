"""Evaluation endpoints — submit and check evaluation status."""
import hashlib
import hmac
import logging
from datetime import datetime
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, Response

from src.storage.models import (
    EvaluateRequest,
    EvaluateResponse,
    EvaluationStatus,
    EvalStatus,
    EvalLevel,
    SubmitSkillRequest,
    TargetType,
    WebhookPayload,
    normalize_eval_mode,
)
from src.storage.mongodb import evaluations_col, scores_col, score_history_col
from src.core.evaluator import Evaluator
from src.core.eval_mode import EVAL_MODES
from src.core.irt_service import IRTService
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
from src.config import settings, get_base_url
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


@router.post("/evaluate", response_model=EvaluateResponse)
async def submit_evaluation(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    http_request: Request,
    api_key_doc: dict = Depends(get_api_key),
    x_payment: str | None = Header(None, alias="X-Payment"),
):
    """Submit an MCP server or agent for quality evaluation.

    For paid levels (2, 3), include X-Payment header with transaction
    signature per x402 protocol. Level 1 is always free.
    """
    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]

    # QO-047: Capture caller identity (non-reversible hashes)
    caller_api_key_hash = key_hash[:16] if key_hash else ""
    caller_org = api_key_doc.get("owner_email", "")
    caller_user_agent = http_request.headers.get("user-agent", "")[:200]
    client_ip = http_request.client.host if http_request.client else ""
    caller_ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:16] if client_ip else ""

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
        "eval_mode": request.eval_mode.value,
        "webhook_url": request.webhook_url,
        "callback_secret": request.callback_secret,
        "payment": payment_receipt.to_dict() if payment_receipt else None,
        "created_at": datetime.utcnow(),
        # QO-047: Caller identity (non-reversible)
        "caller_api_key_hash": caller_api_key_hash,
        "caller_tier": tier,
        "caller_org": caller_org,
        "caller_user_agent": caller_user_agent,
        "caller_ip_hash": caller_ip_hash,
        "request_received_at": datetime.utcnow(),
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


@router.post("/evaluate/skill", response_model=EvaluateResponse)
async def submit_skill_evaluation(
    request: SubmitSkillRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    http_request: Request,
    api_key_doc: dict = Depends(get_api_key),
):
    """QO-060 / E2E_TEST_REPORT fix: accept a SKILL.md bundle in-memory.

    Frontend posts ``{frontmatter, body, source, filename?, level?, eval_mode?}``
    instead of a URL. This route materialises the bundle to a temp directory
    and dispatches through the standard ``_run_evaluation_skill`` background
    task, which expects ``request.target_url`` to be a local skill directory
    path.
    """
    import tempfile
    from pathlib import Path

    import yaml

    tier = api_key_doc.get("tier", "free")
    key_hash = api_key_doc["_id"]

    if not is_eval_level_allowed(tier, request.level.value):
        raise HTTPException(
            status_code=403,
            detail=f"Evaluation level {request.level.value} not available for '{tier}' tier",
        )

    allowed, remaining, limit = await check_eval_rate_limit(key_hash, tier)
    add_rate_limit_headers(response, tier, limit, remaining)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Monthly evaluation limit exceeded. Upgrade your tier.",
        )

    skill_name = request.frontmatter.get("name") or (
        Path(request.filename).stem if request.filename else "uploaded-skill"
    )

    evaluation_id = str(uuid4())
    skill_dir = Path(tempfile.gettempdir()) / "laureum-skill-uploads" / evaluation_id / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\n{yaml.safe_dump(request.frontmatter, sort_keys=False).strip()}\n---\n\n{request.body}\n",
        encoding="utf-8",
    )

    target_id = f"urn:laureum:skill:{evaluation_id}"

    caller_api_key_hash = key_hash[:16] if key_hash else ""
    caller_org = api_key_doc.get("owner_email", "")
    caller_user_agent = http_request.headers.get("user-agent", "")[:200]
    client_ip = http_request.client.host if http_request.client else ""
    caller_ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:16] if client_ip else ""

    doc = {
        "_id": evaluation_id,
        "target_id": target_id,
        "target_type": TargetType.SKILL.value,
        "target_url": str(skill_dir),
        "status": EvalStatus.PENDING.value,
        "level": request.level.value,
        "domains": [],
        "connection_strategy": "skill_bundle",
        "evaluation_version": EVALUATION_VERSION,
        "eval_mode": request.eval_mode.value,
        "webhook_url": request.webhook_url,
        "callback_secret": None,
        "payment": None,
        "created_at": datetime.utcnow(),
        "skill_source": request.source,
        "skill_filename": request.filename,
        "skill_name_declared": skill_name,
        "caller_api_key_hash": caller_api_key_hash,
        "caller_tier": tier,
        "caller_org": caller_org,
        "caller_user_agent": caller_user_agent,
        "caller_ip_hash": caller_ip_hash,
        "request_received_at": datetime.utcnow(),
    }
    await evaluations_col().insert_one(doc)

    eval_request = EvaluateRequest(
        target_url=str(skill_dir),
        target_type=TargetType.SKILL,
        level=request.level,
        eval_mode=request.eval_mode,
        webhook_url=request.webhook_url,
    )

    background_tasks.add_task(_run_evaluation, evaluation_id, eval_request)

    estimated = {EvalLevel.MANIFEST: 5, EvalLevel.FUNCTIONAL: 60, EvalLevel.DOMAIN_EXPERT: 180}
    return EvaluateResponse(
        evaluation_id=evaluation_id,
        status=EvalStatus.PENDING,
        estimated_time_seconds=estimated.get(request.level, 60),
        poll_url=f"/v1/evaluate/{evaluation_id}",
        message="Skill bundle materialised; evaluation dispatched.",
    )


@router.get("/evaluate/{evaluation_id}", response_model=EvaluationStatus)
async def get_evaluation_status(
    evaluation_id: str,
    request: Request,
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
            domains=doc.get("detected_domains", doc.get("domains", [])),
            evaluation_version=doc.get("evaluation_version"),
        )

        report = doc.get("report")
        base = get_base_url(request)
        badge_url = f"{base}/v1/badge/{doc['target_id']}.svg"

        if doc.get("attestation_id"):
            from src.storage.mongodb import attestations_col
            att = await attestations_col().find_one({"_id": doc["attestation_id"]})
            if att:
                attestation_jwt = att.get("attestation_jwt")

    # Compute wall-clock duration from timestamps
    duration_ms = None
    created_at = doc.get("created_at")
    completed_at = doc.get("completed_at")
    if created_at and completed_at:
        duration_ms = int((completed_at - created_at).total_seconds() * 1000)

    # Extract gaming signals from scores
    gaming_risk = None
    timing_anomaly = None
    irt_theta = None
    irt_se = None
    confidence_interval = None
    token_usage = None
    cost_usd = None
    scores_data = doc.get("scores")
    if scores_data and isinstance(scores_data, dict):
        gr = scores_data.get("gaming_risk")
        if gr and isinstance(gr, dict):
            gaming_risk = gr.get("level")
            timing_anomaly = gr.get("timing_anomaly")
        irt_theta = scores_data.get("irt_theta")
        irt_se = scores_data.get("irt_se")
        confidence_interval = scores_data.get("confidence_interval")
        token_usage = scores_data.get("token_usage")
        cost_usd = scores_data.get("cost_usd")

    # Build cost_summary from existing token_usage data
    cost_summary = None
    if token_usage and isinstance(token_usage, dict):
        total_in = token_usage.get("total_input_tokens", 0)
        total_out = token_usage.get("total_output_tokens", 0)
        tokens_total = total_in + total_out
        questions_asked = scores_data.get("questions_asked", 0) if scores_data else 0
        opt = token_usage.get("optimization", {})
        shadow_cost = token_usage.get("shadow_cost_usd", 0.0)
        timing = token_usage.get("phase_timing_ms", {})
        cost_summary = {
            "cost_usd": cost_usd or 0.0,
            "shadow_cost_usd": shadow_cost,
            "tokens_total": tokens_total,
            "tokens_input": total_in,
            "tokens_output": total_out,
            "tokens_per_question": round(tokens_total / questions_asked) if questions_asked else 0,
            "providers_used": list(token_usage.get("by_provider", {}).keys()),
            "llm_calls": opt.get("llm_calls", 0),
            "cascade_exits": opt.get("cascade_exits", 0),
            "fuzzy_routed": opt.get("fuzzy_routed", 0),
            "judging_ms": timing.get("judging_ms"),
            "total_ms": timing.get("total_ms"),
        }
        # QO-051: surface CPCR in cost_summary so the FE can render a
        # headline number without reaching into scores.cpcr.
        cpcr_block = scores_data.get("cpcr") if scores_data else None
        if isinstance(cpcr_block, dict):
            cost_summary["cpcr"] = cpcr_block

    return EvaluationStatus(
        evaluation_id=evaluation_id,
        status=EvalStatus(doc["status"]),
        progress_pct=doc.get("progress_pct", 0),
        score=score,
        tier=tier,
        eval_mode=normalize_eval_mode(doc.get("eval_mode")),
        evaluation_version=doc.get("evaluation_version"),
        report=report,
        scores=doc.get("scores"),
        attestation_jwt=attestation_jwt,
        badge_url=badge_url,
        result=result,
        error=doc.get("error"),
        duration_ms=duration_ms,
        gaming_risk=gaming_risk,
        timing_anomaly=timing_anomaly,
        irt_theta=irt_theta,
        irt_se=irt_se,
        confidence_interval=confidence_interval,
        token_usage=token_usage,
        cost_usd=cost_usd,
        cost_summary=cost_summary,
    )


async def _run_evaluation(evaluation_id: str, request: EvaluateRequest):
    """Dispatch background evaluation by ``request.target_type`` (QO-053-C).

    Pre-053-C this function unconditionally called
    ``mcp_client.get_server_manifest`` even when ``target_type=skill`` was
    submitted. AC1 fixes that: each target type now routes to a dedicated
    runner. Generic agents (``TargetType.AGENT``) raise NotImplementedError —
    that exception is caught by the outer try/except in the MCP runner so
    the status flips to ``failed`` with an explicit error message rather
    than crashing the worker.
    """
    from src.storage.models import TargetType as _TargetType

    if request.target_type == _TargetType.MCP_SERVER:
        return await _run_evaluation_mcp(evaluation_id, request)
    if request.target_type == _TargetType.SKILL:
        return await _run_evaluation_skill(evaluation_id, request)
    if request.target_type == _TargetType.A2A_AGENT:
        return await _run_evaluation_a2a(evaluation_id, request)
    if request.target_type in (_TargetType.REST_CHAT, _TargetType.OPENAPI_AGENT):
        return await _run_evaluation_rest_chat(evaluation_id, request)
    if request.target_type == _TargetType.AGENT:
        # Generic AGENT type — no specific protocol; treat as REST chat by default.
        return await _run_evaluation_rest_chat(evaluation_id, request)
    logger.error(
        f"[{evaluation_id[:8]}] Unsupported target_type={request.target_type!r}"
    )
    await evaluations_col().update_one(
        {"_id": evaluation_id},
        {"$set": {
            "status": EvalStatus.FAILED.value,
            "error": f"Unsupported target_type: {request.target_type}",
            "error_type": "unsupported_target_type",
        }},
    )


async def _run_evaluation_mcp(evaluation_id: str, request: EvaluateRequest):
    """Run MCP-server evaluation in background.

    This is the original ``_run_evaluation`` body, unchanged in behaviour
    (AC2). The dispatcher above routes ``target_type=mcp_server`` here.
    """
    import time as _time
    eval_start = _time.time()
    try:
        # QO-047: Set audit context for this eval — all downstream tool calls,
        # judge calls, sanitization events, and probe executions will be linked
        # to this evaluation_id via contextvars.
        from src.core.mcp_client import current_evaluation_id, current_target_id, _call_index_counter
        current_evaluation_id.set(evaluation_id)
        current_target_id.set(request.target_url)
        _call_index_counter.set(0)

        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"status": EvalStatus.RUNNING.value, "progress_pct": 10}},
        )

        # Resolve eval mode config
        mode_config = EVAL_MODES[request.eval_mode.value]
        logger.info(f"[{evaluation_id[:8]}] Eval mode: {request.eval_mode.value} (max_tools={mode_config.max_tools}, tests={mode_config.test_types}, consensus={mode_config.use_consensus})")

        # Use ConsensusJudge for audited mode, LLMJudge for verified/certified
        if mode_config.use_consensus:
            from src.core.consensus_judge import ConsensusJudge
            judge = ConsensusJudge(max_judges=mode_config.max_judges)
        else:
            judge = _get_judge()
        # Reset any exhausted API keys from prior evaluations
        judge.reset_keys()
        irt_service = IRTService()
        evaluator = Evaluator(judge, eval_mode=request.eval_mode.value, irt_service=irt_service)

        # Step 1: Fetch real manifest from MCP server
        logger.info(f"[{evaluation_id[:8]}] Step 1: Fetching manifest from {request.target_url}")
        try:
            manifest = await mcp_client.get_server_manifest(request.target_url)
            logger.info(f"[{evaluation_id[:8]}] Step 1 complete: {manifest.get('name', '?')} with {len(manifest.get('tools', []))} tools via {manifest.get('transport', '?')}")
        except (ConnectionError, Exception) as e:
            error_msg = str(e)
            # Provide user-friendly messages for common errors
            if "401" in error_msg or "Unauthorized" in error_msg:
                error_msg = f"MCP server requires authentication (401 Unauthorized): {request.target_url}"
            elif "403" in error_msg or "Forbidden" in error_msg:
                error_msg = f"Access denied by MCP server (403 Forbidden): {request.target_url}"
            elif "404" in error_msg or "Not Found" in error_msg:
                error_msg = f"MCP endpoint not found (404): {request.target_url}"
            elif "timeout" in error_msg.lower() or "Timeout" in error_msg:
                error_msg = f"Connection timed out: {request.target_url}"
            else:
                error_msg = f"Cannot connect to MCP server: {error_msg}"
            logger.error(f"Manifest fetch failed for {request.target_url}: {error_msg}")
            error_type, _ = mcp_client.classify_connection_error(error_msg)
            await evaluations_col().update_one(
                {"_id": evaluation_id},
                {"$set": {
                    "status": EvalStatus.FAILED.value,
                    "error": error_msg,
                    "error_type": error_type,
                }},
            )
            return

        # Compute manifest hash for code integrity tracking (QO-029)
        from src.core.quick_scan import _hash_manifest
        manifest_hash = _hash_manifest(manifest.get("tools", []))

        # Auto-detect domain from tool manifest (QO-027)
        from src.core.domain_detection import detect_domain_with_confidence, detect_all_domains, DOMAIN_CONFIDENCE_THRESHOLD
        detected_domain, domain_confidence = detect_domain_with_confidence(manifest.get("tools", []))
        detected_domains = detect_all_domains(manifest.get("tools", []))
        # Use "general" weights if confidence is too low to avoid bad weight adjustments
        scoring_domain = detected_domain if domain_confidence >= DOMAIN_CONFIDENCE_THRESHOLD else "general"
        logger.info(
            f"[{evaluation_id[:8]}] Domain: {detected_domain} (confidence={domain_confidence}, "
            f"scoring_as={scoring_domain}, all={detected_domains})"
        )

        # Store manifest in evaluation doc
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {
                "target_manifest": manifest,
                "manifest_hash": manifest_hash,
                "detected_domain": detected_domain,
                "detected_domains": detected_domains,
                "domain_confidence": domain_confidence,
            }},
        )

        # Step 2: Level 1 — Manifest validation
        logger.info(f"[{evaluation_id[:8]}] Step 2: Validating manifest")
        manifest_result = evaluator.validate_manifest(manifest)
        logger.info(f"[{evaluation_id[:8]}] Step 2 complete: manifest_score={manifest_result.score}, checks={manifest_result.checks}")
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"progress_pct": 30}},
        )

        if request.level == EvalLevel.MANIFEST:
            # Level 1 only: aggregate with manifest score only
            logger.info(f"[{evaluation_id[:8]}] L1 only — aggregating scores from manifest_score={manifest_result.score}")
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
                    "tools_count": len(manifest.get("tools", [])),
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

            tool_responses = await mcp_client.evaluate_server(
                request.target_url,
                test_types=mode_config.test_types,
                max_tools=mode_config.max_tools,
            )
            logger.info(f"[{evaluation_id[:8]}] Tool responses collected: {len(tool_responses)} tools")

            await evaluations_col().update_one(
                {"_id": evaluation_id},
                {"$set": {"progress_pct": 60}},
            )

            # Progress callback for granular 60→75% updates
            async def progress_cb(sub_step: str, sub_pct: float):
                # Map sub_pct (0.0-1.0) to 60-75% range
                pct = int(60 + sub_pct * 15)
                logger.info(f"[{evaluation_id[:8]}] evaluate_full: {sub_step} ({pct}%)")
                await evaluations_col().update_one(
                    {"_id": evaluation_id},
                    {"$set": {"progress_pct": pct}},
                )

            # Judge the tool responses with full 6-axis evaluation
            eval_result = await evaluator.evaluate_full(
                target_id=request.target_url,
                server_url=request.target_url,
                tool_responses=tool_responses,
                manifest=manifest,
                run_safety=mode_config.run_safety_probes,
                run_consistency=mode_config.run_consistency_check,
                progress_cb=progress_cb,
                detected_domain=scoring_domain,
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
                # QO-045: Compute Agent Trap coverage from probe results
                try:
                    from src.core.trap_coverage import compute_trap_coverage
                    probe_results = eval_result.safety_report
                    if isinstance(probe_results, dict):
                        probe_results = probe_results.get("results", [])
                    scores["agent_trap_coverage"] = compute_trap_coverage(probe_results)
                except Exception as e:
                    logger.error(f"Trap coverage computation failed (non-fatal): {e}")
            if eval_result.process_quality_report:
                scores["process_quality_report"] = eval_result.process_quality_report
            if eval_result.latency_stats:
                scores["latency_stats"] = eval_result.latency_stats
            if eval_result.style_report:
                scores["style_report"] = eval_result.style_report
            # Token usage tracking (QO-017)
            if eval_result.token_usage:
                scores["token_usage"] = eval_result.token_usage
                scores["cost_usd"] = eval_result.cost_usd
                scores["shadow_cost_usd"] = eval_result.shadow_cost_usd
            # QO-051: Cost per Correct Response (3 variants). Gated via
            # settings.enable_cpcr inside _maybe_compute_cpcr. Persist a
            # sub-doc so /v1/score/{id} and /v1/costs can read it cheaply.
            if eval_result.correct_count or eval_result.cpcr is not None or eval_result.shadow_cpcr is not None:
                scores["cpcr"] = {
                    "correct_threshold": settings.cpcr_correct_threshold,
                    "correct_count": eval_result.correct_count,
                    "total_responses": len([r for r in eval_result.judge_responses if "score" in r]),
                    "cpcr": eval_result.cpcr,
                    "weighted_cpcr": eval_result.weighted_cpcr,
                    "shadow_cpcr": eval_result.shadow_cpcr,
                }
            if domain_result:
                scores["domain_scores"] = domain_result.domain_scores
                if domain_result.irt_theta is not None:
                    scores["irt_theta"] = domain_result.irt_theta
                if domain_result.irt_se is not None:
                    scores["irt_se"] = domain_result.irt_se
                if domain_result.confidence_interval is not None:
                    scores["confidence_interval"] = domain_result.confidence_interval

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
                    "tools_count": len(manifest.get("tools", [])),
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

        # Persist difficulty tracker stats (non-fatal)
        try:
            await evaluator.difficulty_tracker.save_to_db()
        except Exception as e:
            logger.warning(f"[{evaluation_id[:8]}] Difficulty tracker save failed (non-fatal): {e}")

        now = datetime.utcnow()
        eval_duration_ms = int((_time.time() - eval_start) * 1000)
        token_info = ""
        if scores.get("token_usage"):
            tu = scores["token_usage"]
            token_info = f" tokens={tu.get('total_input_tokens', 0)}in/{tu.get('total_output_tokens', 0)}out cost=${scores.get('cost_usd', 0):.6f}"
        logger.info(f"[{evaluation_id[:8]}] Scores aggregated: overall={scores.get('overall_score')} tier={scores.get('tier')} duration={eval_duration_ms}ms{token_info}")

        # Log cost optimization metrics
        if hasattr(judge, 'log_metrics'):
            judge.log_metrics()

        # ── Anti-gaming analysis ─────────────────────────────────────────
        gaming_risk_data = None
        try:
            from src.core.anti_gaming import (
                analyze_response_timing,
                fingerprint_response,
                check_fingerprints_batch,
                compute_gaming_risk,
                log_paraphrase,
            )

            # Collect response times and fingerprints from tool responses
            response_times = []
            fingerprints = []
            paraphrase_entries = []

            if request.level != EvalLevel.MANIFEST:
                for tool_name, responses in tool_responses.items():
                    for resp in responses:
                        if resp.get("latency_ms"):
                            response_times.append(float(resp["latency_ms"]))
                        answer_text = resp.get("answer") or resp.get("content") or ""
                        question_text = resp.get("question") or tool_name
                        if answer_text:
                            fp = fingerprint_response(question_text, answer_text)
                            fingerprints.append(fp)
                            paraphrase_entries.append({
                                "tool": tool_name,
                                "question_hash": fp.question_hash,
                                "response_hash": fp.response_hash,
                                "method": "template" if request.eval_mode.value == "verified" else "llm",
                            })

            # Timing analysis
            timing = analyze_response_timing(response_times)

            # Fingerprint check against history
            if fingerprints:
                fingerprints = await check_fingerprints_batch(
                    target_id=request.target_url,
                    evaluation_id=evaluation_id,
                    fingerprints=fingerprints,
                )

            # Compute risk
            gaming_risk = compute_gaming_risk(timing, fingerprints)
            gaming_risk_data = gaming_risk.to_dict()

            # Apply confidence penalty
            if gaming_risk.confidence_penalty > 0:
                original_conf = scores.get("confidence", 0)
                scores["confidence"] = max(0.05, original_conf - gaming_risk.confidence_penalty)
                logger.warning(
                    f"[{evaluation_id[:8]}] Gaming risk={gaming_risk.level}: "
                    f"confidence {original_conf:.2f} → {scores['confidence']:.2f}"
                )

            scores["gaming_risk"] = gaming_risk_data

            # Store paraphrase audit trail
            if paraphrase_entries:
                await log_paraphrase(evaluation_id, request.target_url, paraphrase_entries)

            logger.info(f"[{evaluation_id[:8]}] Anti-gaming: risk={gaming_risk.level} duplicates={gaming_risk.duplicate_responses} timing_anomaly={timing.is_suspicious}")

        except Exception as e:
            logger.warning(f"[{evaluation_id[:8]}] Anti-gaming analysis failed (non-fatal): {e}")

        # Create attestation
        logger.info(f"[{evaluation_id[:8]}] Creating attestation...")
        attestation = create_attestation(
            target_id=request.target_url,
            target_type=request.target_type.value,
            target_name=manifest.get("name", request.target_url),
            evaluation_result=scores,
            evaluation_version=EVALUATION_VERSION,
            eval_mode=request.eval_mode.value,
        )
        attestation_id = attestation["_id"]
        from src.storage.mongodb import attestations_col
        await attestations_col().insert_one(attestation)

        logger.info(f"[{evaluation_id[:8]}] Writing final results to DB (status=completed, 100%)")
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {
                "$set": {
                    "status": EvalStatus.COMPLETED.value,
                    "scores": scores,
                    "report": report,
                    "completed_at": now,
                    "progress_pct": 100,
                    "duration_ms": eval_duration_ms,
                    "attestation_id": attestation_id,
                }
            },
        )
        logger.info(f"[{evaluation_id[:8]}] Evaluation COMPLETED: score={scores.get('overall_score')} tier={scores.get('tier')}")

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
                    "last_evaluation_id": evaluation_id,
                    "tool_scores": {k: v for k, v in scores.get("tool_scores", {}).items()},
                    "dimensions": scores.get("dimensions", {}),
                    "safety_report": scores.get("safety_report", []),
                    "latency_stats": scores.get("latency_stats", {}),
                    "duration_ms": eval_duration_ms,
                    "last_eval_mode": request.eval_mode.value,
                    "last_token_usage": scores.get("token_usage"),
                    "last_cost_usd": scores.get("cost_usd"),
                    "last_shadow_cost_usd": scores.get("shadow_cost_usd"),
                    # QO-051: persist CPCR for cheap leaderboard sort by Value
                    "last_cpcr": scores.get("cpcr"),
                    "gaming_risk": gaming_risk_data.get("level") if gaming_risk_data else None,
                    "manifest_hash": manifest_hash,
                    "detected_domain": detected_domain,
                    "detected_domains": detected_domains,
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
            "eval_mode": request.eval_mode.value,
            "domain_scores": scores.get("domain_scores", {}),
            "token_usage": scores.get("token_usage"),
            "cost_usd": scores.get("cost_usd"),
            "recorded_at": now,
            "delta_from_previous": delta,
            "manifest_hash": manifest_hash,
        })

        logger.info(f"Evaluation {evaluation_id} completed: score={scores['overall_score']}")

        # ── QO-043: Score anomaly detection ────────────────────────────────
        try:
            from src.core.score_anomaly import check_score_anomaly, record_anomaly
            anomaly = await check_score_anomaly(
                target_id=request.target_url,
                new_score=scores["overall_score"],
                manifest_hash=manifest_hash,
            )
            if anomaly:
                await record_anomaly(anomaly)
                # Store anomaly reference in evaluation doc
                await evaluations_col().update_one(
                    {"evaluation_id": evaluation_id},
                    {"$set": {"score_anomaly": anomaly.to_dict()}},
                )
        except Exception as e:
            logger.error(f"Score anomaly check failed (non-fatal): {e}")

        # ── On-chain posting (ERC-8004 + EAS) — async, best-effort ──────
        if settings.erc8004_enabled or settings.eas_enabled:
            try:
                from src.onchain.hook import post_evaluation_onchain
                await post_evaluation_onchain(
                    evaluation_id=evaluation_id,
                    target_url=request.target_url,
                    score=scores.get("overall_score", 0),
                    tier=scores.get("tier", "failed"),
                    dimensions=scores.get("dimensions", {}),
                    attestation_jwt=attestation.get("attestation_jwt"),
                    erc8004_agent_id=request.erc8004_agent_id if hasattr(request, "erc8004_agent_id") else None,
                )
            except Exception as e:
                logger.warning(f"[{evaluation_id[:8]}] On-chain posting failed (non-fatal): {e}")

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
        import traceback
        logger.error(f"Evaluation {evaluation_id} failed: {e}\n{traceback.format_exc()}")
        # Provide user-friendly error message
        error_msg = str(e)
        if "'NoneType'" in error_msg or "AttributeError" in error_msg:
            error_msg = f"MCP server returned an unexpected response. The server may require authentication or be temporarily unavailable: {request.target_url}"
        error_type, _ = mcp_client.classify_connection_error(error_msg)
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"status": EvalStatus.FAILED.value, "error": error_msg, "error_type": error_type}},
        )


async def _run_evaluation_skill(evaluation_id: str, request: EvaluateRequest):
    """Run skill evaluation (QO-053-C).

    Currently calls ``Evaluator.evaluate_skill`` with the parsed skill loaded
    from disk via ``request.target_url`` (interpreted as a local skill
    directory path). Tests stub this path entirely; the production wiring
    lands in QO-053-F (batch runner) where the dispatcher pulls from the
    operator-uploaded skill bundle. The minimum here is:

    1. Persist a ``status=running`` row.
    2. Call ``evaluator.evaluate_skill`` with whatever activator is wired by
       the orchestration layer (QO-053-B) — for now, a graceful failure
       path that records the limitation rather than silently succeeding.
    """
    import time as _time
    from pathlib import Path

    from src.storage.models import EvaluationDoc, TargetType as _TargetType
    from src.core.skill_parser import parse_skill_md
    from src.core.skill_validator import validate_skill

    eval_start = _time.time()

    try:
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"status": EvalStatus.RUNNING.value, "progress_pct": 10}},
        )

        skill_dir = Path(request.target_url)
        if not skill_dir.exists():
            await evaluations_col().update_one(
                {"_id": evaluation_id},
                {"$set": {
                    "status": EvalStatus.FAILED.value,
                    "error": f"Skill directory not found: {request.target_url}",
                    "error_type": "skill_dir_missing",
                }},
            )
            return

        parsed = parse_skill_md(skill_dir)
        spec_compliance = validate_skill(parsed, skill_dir)

        # Build a minimal target object — see Evaluator.evaluate_skill docs
        # for required attributes.
        class _SkillTarget:
            pass

        target = _SkillTarget()
        target.parsed = parsed
        target.spec_compliance = spec_compliance
        target.subject_uri = request.target_url

        # Wire judges identically to MCP path so cost accounting works.
        mode_config = EVAL_MODES[request.eval_mode.value]
        if mode_config.use_consensus:
            from src.core.consensus_judge import ConsensusJudge
            judge = ConsensusJudge(max_judges=mode_config.max_judges)
        else:
            judge = _get_judge()
        judge.reset_keys()
        irt_service = IRTService()
        evaluator = Evaluator(
            judge, eval_mode=request.eval_mode.value, irt_service=irt_service,
        )

        # Wire activator factory per requested level. Defaults to Cerebras
        # free tier (settings.laureum_activation_provider). Returns None when
        # the deployment lacks credentials — evaluator falls back gracefully
        # to the L1 manifest-only path so eval still completes with the
        # deterministic Phase-0 probe pack.
        from src.core.activator_factory import make_activator_factory
        activator_factory = make_activator_factory(parsed, skill_dir, request.level)
        target.skill_dir = skill_dir  # exposed for SkillProbeRunner phase-1
        eval_result = await evaluator.evaluate_skill(
            target=target,
            level=request.level,
            activator_factory=activator_factory,
            baseline_activator_factory=None,
        )

        eval_duration_ms = int((_time.time() - eval_start) * 1000)
        scores = eval_result.to_dict()
        scores["confidence"] = eval_result.confidence

        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {
                "status": EvalStatus.COMPLETED.value,
                "scores": scores,
                "completed_at": datetime.utcnow(),
                "progress_pct": 100,
                "duration_ms": eval_duration_ms,
                "target_type_dispatched": _TargetType.SKILL.value,
                "subject_uri": eval_result.subject_uri,
                "axis_weights_used": eval_result.axis_weights_used,
                "delta_vs_baseline": eval_result.delta_vs_baseline,
                "baseline_score": eval_result.baseline_score,
                "baseline_status": eval_result.baseline_status,
            }},
        )
        logger.info(
            f"[{evaluation_id[:8]}] Skill eval COMPLETED: "
            f"absolute={eval_result.overall_score} "
            f"baseline={eval_result.baseline_score} "
            f"delta={eval_result.delta_vs_baseline} "
            f"tier={eval_result.tier}"
        )
        # Touch the unused EvaluationDoc import so the linter sees the model
        # is wired into this module (also kept here for future schema use).
        _ = EvaluationDoc
    except Exception as e:
        import traceback
        logger.error(f"Skill evaluation {evaluation_id} failed: {e}\n{traceback.format_exc()}")
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {
                "status": EvalStatus.FAILED.value,
                "error": str(e),
                "error_type": "skill_evaluation_error",
            }},
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
        report_url=f"{get_base_url()}/v1/evaluate/{evaluation_id}",
        badge_url=f"{get_base_url()}/v1/badge/{target_id}.svg",
        attestation_url=f"{get_base_url()}/v1/attestation/{attestation_id}" if attestation_id else None,
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


# ── QO-058 generic-agent runners ─────────────────────────────────────────────


async def _run_evaluation_a2a(evaluation_id: str, request: EvaluateRequest):
    """Background runner for A2A_AGENT target type.

    Resolves the URL through the cascade (so a passing through ``UNKNOWN``
    or ``A2A_AGENT`` both work), runs :meth:`Evaluator.evaluate_a2a`,
    persists the result. Mirrors the skill runner's failure shape so the
    ``/v1/evaluate/{id}`` polling endpoint sees a uniform error envelope.
    """
    import time as _time
    from src.core.target_resolver import resolve as resolve_target
    from src.core.a2a_target import A2ATarget

    eval_start = _time.time()
    try:
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"status": EvalStatus.RUNNING.value, "progress_pct": 10}},
        )

        # Try resolver first for cache + auto-detection; fall back to direct
        # construction if the resolver doesn't pick A2A.
        try:
            target = await resolve_target(request.target_url)
        except Exception as exc:  # noqa: BLE001
            logger.info(f"[{evaluation_id[:8]}] resolver fallback: {exc}")
            target = A2ATarget(endpoint_url=request.target_url)

        mode_config = EVAL_MODES[request.eval_mode.value]
        if mode_config.use_consensus:
            from src.core.consensus_judge import ConsensusJudge
            judge = ConsensusJudge(max_judges=mode_config.max_judges)
        else:
            judge = _get_judge()
        judge.reset_keys()
        evaluator = Evaluator(judge, eval_mode=request.eval_mode.value)

        result = await evaluator.evaluate_a2a(target)
        duration_ms = int((_time.time() - eval_start) * 1000)

        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {
                "status": EvalStatus.COMPLETED.value,
                "progress_pct": 100,
                "scores": result.to_dict(),
                "duration_ms": duration_ms,
                "target_type_dispatched": result.target_type_dispatched.value if result.target_type_dispatched else None,
            }},
        )
    except Exception as e:
        logger.exception(f"[{evaluation_id[:8]}] A2A evaluation failed: {e}")
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {
                "status": EvalStatus.FAILED.value,
                "error": str(e),
                "error_type": "a2a_evaluation_error",
            }},
        )


async def _run_evaluation_rest_chat(evaluation_id: str, request: EvaluateRequest):
    """Background runner for REST_CHAT (and OPENAPI_AGENT) target types.

    Schema-inference failure (AC11) → status=failed,
    error_type=schema_unobtainable. Verified-tier cap is enforced inside
    :meth:`Evaluator.evaluate_rest_chat` (AC7).
    """
    import time as _time
    from src.core.rest_chat_target import RESTChatTarget

    eval_start = _time.time()
    try:
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {"status": EvalStatus.RUNNING.value, "progress_pct": 10}},
        )

        mode_config = EVAL_MODES[request.eval_mode.value]
        if mode_config.use_consensus:
            from src.core.consensus_judge import ConsensusJudge
            judge = ConsensusJudge(max_judges=mode_config.max_judges)
        else:
            judge = _get_judge()
        judge.reset_keys()
        evaluator = Evaluator(judge, eval_mode=request.eval_mode.value)

        target = RESTChatTarget(endpoint_url=request.target_url, judge=judge)
        result = await evaluator.evaluate_rest_chat(target)
        duration_ms = int((_time.time() - eval_start) * 1000)

        # AC11 surface: tier=failed → propagate the error_type for clean
        # error UX without overloading the success path.
        update: dict = {
            "status": EvalStatus.COMPLETED.value,
            "progress_pct": 100,
            "scores": result.to_dict(),
            "duration_ms": duration_ms,
            "target_type_dispatched": (
                result.target_type_dispatched.value
                if result.target_type_dispatched else None
            ),
        }
        if result.tier == "failed" and result.style_report and result.style_report.get("error_type") == "schema_unobtainable":
            update["status"] = EvalStatus.FAILED.value
            update["error"] = result.style_report.get("error", "Schema inference refused this target")
            update["error_type"] = "schema_unobtainable"
        await evaluations_col().update_one({"_id": evaluation_id}, {"$set": update})
    except Exception as e:
        logger.exception(f"[{evaluation_id[:8]}] REST chat evaluation failed: {e}")
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {
                "status": EvalStatus.FAILED.value,
                "error": str(e),
                "error_type": "rest_chat_evaluation_error",
            }},
        )


async def _run_evaluation_unknown(evaluation_id: str, request: EvaluateRequest):
    """Caller submitted ``target_type=unknown`` — run cascade then re-dispatch.

    Convenience for the landing-page form: operator pastes a URL without
    declaring a type, we discover the protocol and route accordingly.
    """
    from src.core.evaluation_target import UnknownTargetError
    from src.core.target_resolver import resolve as resolve_target
    from src.storage.models import TargetType as _TargetType

    try:
        _, meta = await resolve_target(request.target_url, return_meta=True)
    except UnknownTargetError as exc:
        await evaluations_col().update_one(
            {"_id": evaluation_id},
            {"$set": {
                "status": EvalStatus.FAILED.value,
                "error": str(exc),
                "error_type": "unknown_target_type",
            }},
        )
        return

    # Mutate request to reflect resolved type, then re-enter dispatch.
    request.target_type = meta.target_type
    if meta.target_type == _TargetType.MCP_SERVER:
        return await _run_evaluation_mcp(evaluation_id, request)
    if meta.target_type == _TargetType.A2A_AGENT:
        return await _run_evaluation_a2a(evaluation_id, request)
    if meta.target_type in (_TargetType.REST_CHAT, _TargetType.OPENAPI_AGENT):
        return await _run_evaluation_rest_chat(evaluation_id, request)
    await evaluations_col().update_one(
        {"_id": evaluation_id},
        {"$set": {
            "status": EvalStatus.FAILED.value,
            "error": f"Resolved unknown target to {meta.target_type.value} but no runner is wired",
            "error_type": "unsupported_resolved_type",
        }},
    )

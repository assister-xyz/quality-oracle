"""
AgentTrust as an MCP Server.

Exposes quality evaluation as MCP tools that any MCP client
(Claude, Cursor, Windsurf, etc.) can call directly.

Run standalone: python -m src.standards.mcp_server
"""
import json
import logging
import os

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# FastMCP's Settings reads LOG_LEVEL from env and requires uppercase.
# Normalize to prevent validation errors from lowercase env values.
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

mcp = FastMCP(
    "agenttrust",
    log_level=_log_level,
    instructions="AgentTrust — AI Agent & MCP Server quality evaluation service v0.1.0",
    host="0.0.0.0",
    port=8003,
)


def _get_judge():
    """Create judge: ConsensusJudge if 2+ providers available, else single LLMJudge."""
    from src.config import settings
    from src.core.consensus_judge import ConsensusJudge

    if settings.consensus_enabled:
        judge = ConsensusJudge(
            agreement_threshold=settings.consensus_agreement_threshold,
            min_judges=settings.consensus_min_judges,
        )
        if judge.is_consensus_possible:
            return judge

    # Fallback to single judge (old behavior)
    from src.core.llm_judge import LLMJudge

    if settings.openai_api_key:
        return LLMJudge(
            api_key=settings.openai_api_key,
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
    elif settings.deepseek_api_key:
        return LLMJudge(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            provider="deepseek",
            base_url=settings.deepseek_base_url,
            fallback_key=settings.groq_api_key or None,
            fallback_model=settings.groq_model,
            fallback_provider="groq",
        )
    elif settings.groq_api_key:
        return LLMJudge(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            provider="groq",
            base_url="https://api.groq.com/openai/v1",
        )
    else:
        return LLMJudge()


@mcp.tool()
async def check_quality(server_url: str) -> str:
    """Evaluate an MCP server's quality. Connects via SSE, runs manifest + functional tests, returns score and tier.

    Args:
        server_url: The SSE URL of the MCP server to evaluate (e.g., "http://localhost:8010/sse")
    """
    from src.core.mcp_client import evaluate_server, get_server_manifest
    from src.core.evaluator import Evaluator

    try:
        # Get manifest for Level 1
        manifest = await get_server_manifest(server_url)

        # Run Level 2 functional evaluation
        tool_responses = await evaluate_server(server_url)

        # Judge responses with multi-dimensional evaluation
        judge = _get_judge()
        evaluator = Evaluator(llm_judge=judge)
        result = await evaluator.evaluate_full(
            target_id=server_url,
            server_url=server_url,
            tool_responses=tool_responses,
            manifest=manifest,
            run_safety=True,
        )

        response = {
            "score": result.overall_score,
            "tier": result.tier,
            "confidence": round(result.confidence, 2),
            "tool_scores": result.tool_scores,
            "manifest_score": result.manifest_result.score if result.manifest_result else None,
            "questions_asked": result.questions_asked,
            "duration_ms": result.duration_ms,
            "evaluation_tier": "audited",
        }
        if result.dimensions:
            response["dimensions"] = {
                k: v["score"] for k, v in result.dimensions.items()
            }
        if result.safety_report:
            response["safety"] = result.safety_report

        # Cache the full evaluation result
        from src.core.score_cache import get_score_cache
        cache = get_score_cache()
        cache.put(
            target_url=server_url,
            score=result.overall_score,
            tier=result.tier,
            confidence=result.confidence,
            dimensions={k: v["score"] for k, v in result.dimensions.items()} if result.dimensions else None,
            tools_count=len(manifest.get("tools", [])),
            transport=manifest.get("transport", "unknown"),
            eval_duration_ms=result.duration_ms,
        )

        return json.dumps(response)
    except ConnectionError as e:
        return json.dumps({"error": f"Cannot connect to server: {e}"})
    except Exception as e:
        logger.error(f"check_quality failed for {server_url}: {e}")
        return json.dumps({"error": f"Evaluation failed: {e}"})


@mcp.tool()
async def check_quality_fast(server_url: str) -> str:
    """Fast tiered quality check for A2A pre-delegation. Returns cached score (<10ms) or runs manifest-only check (<100ms). Use check_quality for full evaluation.

    Args:
        server_url: The URL of the MCP server to check
    """
    from src.core.score_cache import get_score_cache
    from src.core.mcp_client import get_server_manifest
    from src.core.evaluator import Evaluator

    cache = get_score_cache()

    # Tier 0: Cache lookup (<10ms)
    cached = cache.get_effective(server_url)
    if cached and not cached.get("needs_refresh"):
        cached["evaluation_tier"] = "cached"
        return json.dumps(cached)

    try:
        # Tier 1: Schema/manifest check (<100ms)
        manifest = await get_server_manifest(server_url)
        judge = _get_judge()
        evaluator = Evaluator(llm_judge=judge, paraphrase=False)
        manifest_result = evaluator.validate_manifest(manifest)

        response = {
            "score": manifest_result.score,
            "tier": "schema_only",
            "confidence": 0.3,  # Low confidence — schema only
            "manifest_score": manifest_result.score,
            "tools_count": len(manifest.get("tools", [])),
            "transport": manifest.get("transport", "unknown"),
            "checks": manifest_result.checks,
            "warnings": manifest_result.warnings,
            "evaluation_tier": "schema",
        }

        # Cache the schema-only result with short TTL
        cache.put(
            target_url=server_url,
            score=manifest_result.score,
            tier="schema_only",
            confidence=0.3,
            tools_count=len(manifest.get("tools", [])),
            transport=manifest.get("transport", "unknown"),
        )

        # If cached score exists but needs refresh, include it
        if cached:
            response["previous_score"] = cached.get("score")
            response["previous_freshness"] = cached.get("freshness")

        return json.dumps(response)
    except Exception as e:
        # If we have a stale cached score, return it with warning
        if cached:
            cached["evaluation_tier"] = "stale_cache"
            cached["warning"] = f"Re-evaluation failed: {e}"
            return json.dumps(cached)
        return json.dumps({"error": f"Cannot evaluate: {e}", "evaluation_tier": "failed"})


@mcp.tool()
async def get_score(server_url: str) -> str:
    """Look up the most recent quality score for an MCP server. Returns cached score with decay applied.

    Args:
        server_url: The URL of the MCP server to look up
    """
    # Try in-memory score cache first
    from src.core.score_cache import get_score_cache
    cache = get_score_cache()
    cached = cache.get_effective(server_url)
    if cached:
        return json.dumps(cached)

    # Try persistent storage
    try:
        from src.storage.cache import get_cached_score
        stored = await get_cached_score(server_url)
        if stored:
            return json.dumps(stored)
    except Exception:
        pass

    return json.dumps({"error": "No score found. Run check_quality or check_quality_fast first."})


@mcp.tool()
async def verify_attestation(attestation_jwt: str) -> str:
    """Verify an AQVC quality attestation JWT. Checks signature validity and returns the decoded payload.

    Args:
        attestation_jwt: The JWT attestation string to verify
    """
    from src.core.attestation import verify_attestation as _verify

    try:
        result = _verify(attestation_jwt)
        if result["valid"]:
            payload = result.get("payload", {})
            quality = payload.get("quality", {})
            return json.dumps({
                "valid": True,
                "score": quality.get("score"),
                "tier": quality.get("tier"),
                "issuer": result.get("issuer"),
                "expires_at": result.get("expires_at"),
            })
        else:
            return json.dumps({"valid": False, "error": result.get("error", "Unknown error")})
    except Exception as e:
        return json.dumps({"valid": False, "error": f"Verification failed: {e}"})


def main():
    """Run AgentTrust MCP server via SSE on port 8003."""
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()

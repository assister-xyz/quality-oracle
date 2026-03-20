"""Cost analytics endpoint — aggregate token usage and cost data."""
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query

from src.auth.dependencies import get_api_key
from src.storage.mongodb import evaluations_col
from src.config import PROVIDER_PRICING

logger = logging.getLogger(__name__)

router = APIRouter()

_PERIOD_MAP = {
    "1d": timedelta(days=1),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
    "all": None,
}


@router.get("/costs")
async def get_cost_analytics(
    period: str = Query("7d", regex="^(1d|7d|30d|90d|all)$"),
    api_key_doc: dict = Depends(get_api_key),
):
    """Get aggregate cost analytics for evaluations.

    Returns total token usage, cost breakdown by provider, and average cost per eval mode.
    """
    delta = _PERIOD_MAP.get(period)
    match_filter = {
        "status": "completed",
        "scores.token_usage": {"$exists": True},
    }
    if delta:
        match_filter["completed_at"] = {"$gte": datetime.utcnow() - delta}

    # Main aggregation by eval_mode
    pipeline = [
        {"$match": match_filter},
        {
            "$group": {
                "_id": "$eval_mode",
                "count": {"$sum": 1},
                "total_input_tokens": {"$sum": "$scores.token_usage.total_input_tokens"},
                "total_output_tokens": {"$sum": "$scores.token_usage.total_output_tokens"},
                "total_cost_usd": {"$sum": "$scores.cost_usd"},
                "total_shadow_cost_usd": {"$sum": "$scores.token_usage.shadow_cost_usd"},
                "avg_cost_usd": {"$avg": "$scores.cost_usd"},
                "min_cost_usd": {"$min": "$scores.cost_usd"},
                "max_cost_usd": {"$max": "$scores.cost_usd"},
            }
        },
    ]

    results = []
    async for doc in evaluations_col().aggregate(pipeline):
        results.append(doc)

    # Aggregate across all modes
    total_evals = sum(r["count"] for r in results)
    total_input = sum(r["total_input_tokens"] for r in results)
    total_output = sum(r["total_output_tokens"] for r in results)
    total_cost = sum(r["total_cost_usd"] for r in results)
    total_shadow = sum(r.get("total_shadow_cost_usd") or 0 for r in results)

    # Per-mode breakdown
    avg_cost_per_mode = {}
    for r in results:
        mode = r["_id"] or "unknown"
        avg_cost_per_mode[mode] = {
            "count": r["count"],
            "avg_cost_usd": round(r["avg_cost_usd"] or 0, 6),
            "min_cost_usd": round(r["min_cost_usd"] or 0, 6),
            "max_cost_usd": round(r["max_cost_usd"] or 0, 6),
            "total_cost_usd": round(r["total_cost_usd"] or 0, 6),
            "total_input_tokens": r["total_input_tokens"],
            "total_output_tokens": r["total_output_tokens"],
        }

    # By-level aggregation
    level_pipeline = [
        {"$match": match_filter},
        {
            "$group": {
                "_id": "$level",
                "count": {"$sum": 1},
                "total_cost_usd": {"$sum": "$scores.cost_usd"},
                "avg_cost_usd": {"$avg": "$scores.cost_usd"},
                "total_input_tokens": {"$sum": "$scores.token_usage.total_input_tokens"},
                "total_output_tokens": {"$sum": "$scores.token_usage.total_output_tokens"},
                "total_questions": {"$sum": "$scores.questions_asked"},
            }
        },
    ]
    by_level = {}
    async for doc in evaluations_col().aggregate(level_pipeline):
        level = doc["_id"] or 0
        by_level[f"level_{level}"] = {
            "count": doc["count"],
            "total_cost_usd": round(doc["total_cost_usd"] or 0, 6),
            "avg_cost_usd": round(doc["avg_cost_usd"] or 0, 6),
            "total_input_tokens": doc["total_input_tokens"],
            "total_output_tokens": doc["total_output_tokens"],
            "total_questions": doc["total_questions"] or 0,
        }

    # Optimization metrics aggregation
    opt_pipeline = [
        {"$match": {**match_filter, "scores.token_usage.optimization": {"$exists": True}}},
        {
            "$group": {
                "_id": None,
                "total_cascade_exits": {"$sum": "$scores.token_usage.optimization.cascade_exits"},
                "total_fuzzy_routed": {"$sum": "$scores.token_usage.optimization.fuzzy_routed"},
                "total_cache_hits": {"$sum": "$scores.token_usage.optimization.cache_hits"},
                "total_llm_calls": {"$sum": "$scores.token_usage.optimization.llm_calls"},
                "total_judged": {"$sum": "$scores.token_usage.optimization.total_judged"},
                "count": {"$sum": 1},
            }
        },
    ]
    optimization = {}
    async for doc in evaluations_col().aggregate(opt_pipeline):
        total_judged = doc["total_judged"] or 0
        saved = (doc["total_fuzzy_routed"] or 0) + (doc["total_cache_hits"] or 0) + (doc["total_cascade_exits"] or 0)
        optimization = {
            "evals_with_data": doc["count"],
            "total_llm_calls": doc["total_llm_calls"] or 0,
            "total_fuzzy_routed": doc["total_fuzzy_routed"] or 0,
            "total_cache_hits": doc["total_cache_hits"] or 0,
            "total_cascade_exits": doc["total_cascade_exits"] or 0,
            "total_judged": total_judged,
            "llm_calls_saved_pct": round(saved / total_judged * 100, 1) if total_judged else 0,
        }

    # Efficiency metrics
    total_questions = sum(v.get("total_questions", 0) for v in by_level.values())
    total_tokens = total_input + total_output
    efficiency = {
        "avg_tokens_per_question": round(total_tokens / total_questions) if total_questions else 0,
        "avg_cost_per_question": round(total_cost / total_questions, 6) if total_questions else 0,
        "avg_shadow_cost_per_question": round(total_shadow / total_questions, 6) if total_questions else 0,
        "total_questions": total_questions,
    }

    # Compute shadow cost from current token totals using market rates
    # (fallback for evals that predate shadow_cost_usd field)
    if total_shadow == 0 and total_tokens > 0:
        from src.config import calculate_market_cost
        # Estimate shadow using dominant provider (cerebras for verified mode)
        total_shadow = calculate_market_cost("cerebras", total_input, total_output)
        efficiency["avg_shadow_cost_per_question"] = round(total_shadow / total_questions, 6) if total_questions else 0

    # Provider pricing reference
    pricing_ref = {
        k: {
            "actual_input_per_m": v["input_per_m"],
            "actual_output_per_m": v["output_per_m"],
            "market_input_per_m": v.get("market_input_per_m", v["input_per_m"]),
            "market_output_per_m": v.get("market_output_per_m", v["output_per_m"]),
            "tier": v["tier"],
        }
        for k, v in PROVIDER_PRICING.items()
    }

    return {
        "period": period,
        "total_evaluations": total_evals,
        "total_cost_usd": round(total_cost, 6),
        "shadow_cost_usd": round(total_shadow, 6),
        "savings_usd": round(total_shadow - total_cost, 6),
        "total_tokens": {
            "input": total_input,
            "output": total_output,
            "total": total_tokens,
        },
        "avg_cost_per_eval": round(total_cost / total_evals, 6) if total_evals else 0,
        "avg_shadow_cost_per_eval": round(total_shadow / total_evals, 6) if total_evals else 0,
        "by_eval_mode": avg_cost_per_mode,
        "by_level": by_level,
        "efficiency": efficiency,
        "optimization": optimization,
        "provider_pricing": pricing_ref,
    }

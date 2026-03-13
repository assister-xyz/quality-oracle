"""Rankings API — Bradley-Terry leaderboard with divisions."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.core.rating import BradleyTerryRanker
from src.core.matchmaking import MatchmakingEngine
from src.storage.mongodb import rankings_col, battles_col, ladder_col, scores_col
from src.auth.dependencies import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

_ranker = BradleyTerryRanker()
_matchmaker = MatchmakingEngine()


@router.get("/rankings")
async def get_rankings(
    domain: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    api_key_doc: dict = Depends(get_api_key),
):
    """Get BT-ranked leaderboard with CI, division, and battle record."""
    query = {"domain": domain}
    cursor = rankings_col().find(query).sort("position", 1).skip(offset).limit(limit)

    items = []
    async for doc in cursor:
        doc.pop("_id", None)
        items.append(doc)

    total = await rankings_col().count_documents(query)
    return {"items": items, "total": total, "domain": domain, "offset": offset}


@router.get("/rankings/{domain}")
async def get_domain_rankings(
    domain: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    api_key_doc: dict = Depends(get_api_key),
):
    """Get domain-specific BT rankings."""
    query = {"domain": domain}
    cursor = rankings_col().find(query).sort("position", 1).skip(offset).limit(limit)

    items = []
    async for doc in cursor:
        doc.pop("_id", None)
        items.append(doc)

    total = await rankings_col().count_documents(query)
    return {"items": items, "total": total, "domain": domain, "offset": offset}


@router.get("/agent/{target_id}/profile")
async def get_agent_profile(
    target_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Full agent profile: rating, division, battle history, axes, streaks."""
    from src.storage.models import compute_division, DIVISION_CONFIG, Division

    # Get ranking entry
    ranking = await rankings_col().find_one({"target_id": target_id})

    # Get ladder entry for detailed stats
    ladder_entry = await ladder_col().find_one({"target_id": target_id})

    # Get score entry for per-axis data
    score_entry = await scores_col().find_one({"target_id": target_id})

    if not ranking and not ladder_entry and not score_entry:
        raise HTTPException(status_code=404, detail=f"Agent {target_id} not found")

    # Compute profile
    mu = (ladder_entry or {}).get("openskill_mu", 25.0)
    sigma = (ladder_entry or {}).get("openskill_sigma", 8.333)
    record = (ladder_entry or {}).get("battle_record", {"wins": 0, "losses": 0, "draws": 0})
    total = record.get("wins", 0) + record.get("losses", 0) + record.get("draws", 0)
    win_rate = record.get("wins", 0) / max(total, 1)
    name = (ladder_entry or {}).get("name", "") or (score_entry or {}).get("name", target_id)

    # Division
    position = (ranking or {}).get("position", 0)
    is_top3 = 1 <= position <= 3
    division = compute_division(mu, sigma, total, is_top3=is_top3)
    div_cfg = DIVISION_CONFIG.get(Division(division), {})

    # Per-axis scores from domain_scores
    per_axis = {}
    if score_entry:
        for domain_key, ds in score_entry.get("domain_scores", {}).items():
            if isinstance(ds, dict):
                for axis, val in ds.items():
                    if isinstance(val, (int, float)):
                        per_axis[axis] = per_axis.get(axis, 0) + val

    # Recent battles
    recent_battles = []
    cursor = battles_col().find({
        "$or": [
            {"agent_a.target_id": target_id},
            {"agent_b.target_id": target_id},
        ],
        "status": "completed",
    }).sort("completed_at", -1).limit(10)

    async for doc in cursor:
        is_a = doc.get("agent_a", {}).get("target_id") == target_id
        opponent_key = "agent_b" if is_a else "agent_a"
        winner = doc.get("winner")
        if winner == "a":
            result = "win" if is_a else "loss"
        elif winner == "b":
            result = "loss" if is_a else "win"
        else:
            result = "draw"

        recent_battles.append({
            "battle_id": doc.get("battle_id", ""),
            "opponent_id": doc.get(opponent_key, {}).get("target_id", ""),
            "result": result,
            "margin": doc.get("margin", 0),
            "completed_at": str(doc.get("completed_at", "")),
        })

    # Compute streaks from recent battles
    current_streak = 0
    best_streak = 0
    streak_count = 0
    last_result = None
    for b in recent_battles:
        r = b["result"]
        if r == last_result:
            streak_count += 1
        else:
            streak_count = 1
            last_result = r
        if current_streak == 0:
            if r == "win":
                current_streak = streak_count
            elif r == "loss":
                current_streak = -streak_count
        if r == "win":
            best_streak = max(best_streak, streak_count)

    return {
        "target_id": target_id,
        "name": name,
        "bt_rating": (ranking or {}).get("bt_rating", 0.0),
        "ci_lower": (ranking or {}).get("ci_lower", 0.0),
        "ci_upper": (ranking or {}).get("ci_upper", 0.0),
        "division": division,
        "division_config": {"label": div_cfg.get("label", ""), "color": div_cfg.get("color", ""), "icon": div_cfg.get("icon", "")},
        "openskill_mu": round(mu, 3),
        "openskill_sigma": round(sigma, 3),
        "battle_record": record,
        "total_battles": total,
        "win_rate": round(win_rate, 4),
        "current_streak": current_streak,
        "best_streak": best_streak,
        "per_axis_scores": per_axis,
        "recent_battles": recent_battles,
        "position": position,
    }


@router.post("/rankings/recompute")
async def recompute_rankings(
    domain: Optional[str] = Query(None),
    api_key_doc: dict = Depends(get_api_key),
):
    """Trigger BT ranking recomputation."""
    entries = await _ranker.recompute_rankings(domain=domain)
    return {
        "recomputed": len(entries),
        "domain": domain,
        "message": f"Rankings recomputed for {len(entries)} agents",
    }


@router.get("/matchmaking/next")
async def get_next_match(
    domain: Optional[str] = Query(None),
    api_key_doc: dict = Depends(get_api_key),
):
    """Get the next recommended match pairing."""
    result = await _matchmaker.select_match(domain=domain)
    if not result:
        return {"match": None, "message": "Not enough agents for matchmaking"}

    agent_a, agent_b, strategy = result
    return {
        "match": {
            "agent_a_id": agent_a,
            "agent_b_id": agent_b,
            "strategy": strategy,
        },
        "domain": domain,
    }

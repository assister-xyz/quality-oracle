"""Marketplace API — public scorecards for skill collections (QO-053-H).

Endpoints
---------
- GET  /v1/marketplace/{slug}            — list of skills with derived
                                            ``r5_risk_score`` projection.
- GET  /v1/marketplace/{slug}/{skill}    — per-skill detail incl. last snapshot.
- GET  /v1/marketplace/{slug}/{skill}/aqvc.json
                                          — signed AQVC credential pulled from
                                            QO-053-I attestations.

Read-through cache: ``qo:marketplace:{slug}`` (15 min TTL). Cache miss queries
``quality__skill_scores`` and joins with ``quality__probe_results`` to compute
the R5 §12 risk score per skill.

This module is **public** (no API-key required) — the scorecard lives at a
public URL on laureum.ai. Rate limiting is handled at the edge (Cloudflare /
Vercel) since the response is cached. A future revision may add per-IP limits.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.storage.cache import get_redis
from src.storage.mongodb import attestations_col, evaluations_col, get_db
from src.storage.models import MarketplaceListItem, MarketplaceListResponse, MarketplaceSkillDetail

logger = logging.getLogger(__name__)
router = APIRouter()

# Redis cache TTL for /v1/marketplace/{slug} list endpoint.
MARKETPLACE_CACHE_TTL = 15 * 60  # 15 minutes


# R5 §12 risk-score weights — keep here as a private constant; if this grows
# beyond a single weight set, hoist to ``src/standards/r5_risk.py``.
_R5_WEIGHTS = {
    "fee_payer_hijack": 4.0,
    "script_poisoning": 3.5,
    "rpc_misconfig": 2.5,
    "schema_drift": 2.0,
    "instruction_injection": 3.0,
}


def _derive_r5_risk_score(probe_results: List[Dict[str, Any]], score: int) -> float:
    """Project a 0-10 R5 risk score from probe pass/fail.

    Higher = more risk. Maps each failed R5 probe to its weight, normalises to
    the 0-10 scale, and adds a low-score penalty (sub-50 skills get +1.0).
    """
    if not probe_results:
        # Without probe data, fall back on the overall score: low score = high risk.
        return round(max(0.0, min(10.0, (100 - score) / 10.0)), 2)

    risk = 0.0
    max_weight = sum(_R5_WEIGHTS.values())
    for probe in probe_results:
        ptype = probe.get("probe_type") or probe.get("type") or ""
        passed = probe.get("passed", True)
        weight = _R5_WEIGHTS.get(ptype, 0.0)
        if not passed:
            risk += weight

    # Normalise — divide failed-weight by total weight, scale to 0-10.
    normalised = (risk / max_weight) * 10.0 if max_weight > 0 else 0.0

    # Low-score skills carry residual risk even if probes pass.
    if score < 50:
        normalised = min(10.0, normalised + 1.0)

    return round(normalised, 2)


def _slug_to_repo_filter(slug: str) -> Dict[str, Any]:
    """Map a public slug like ``sendai`` to the MongoDB ``skill_repo`` regex."""
    # The skill_repo column for SendAI skills looks like
    # "sendai/skills/jupiter" or "github.com/sendaifun/...". Match either.
    return {
        "$or": [
            {"skill_repo": {"$regex": f"^{slug}/", "$options": "i"}},
            {"skill_repo": {"$regex": f"/{slug}/", "$options": "i"}},
            {"skill_repo": {"$regex": f"{slug}fun", "$options": "i"}},
            {"slug": slug},
        ]
    }


async def _build_marketplace_list(slug: str) -> List[MarketplaceListItem]:
    """Build the marketplace list payload for ``slug`` from MongoDB.

    Joins ``quality__skill_scores`` with ``quality__probe_results`` to compute
    the R5 §12 risk score per skill (server-side projection).
    """
    db = get_db()
    skill_scores = db.quality__skill_scores
    probe_results = db.quality__probe_results

    items: List[MarketplaceListItem] = []
    cursor = skill_scores.find(_slug_to_repo_filter(slug)).sort("score", -1)
    async for doc in cursor:
        skill_id = doc.get("skill_id") or doc.get("subject_uri") or doc.get("_id")
        # Pull last probe results for this skill — latest evaluation only.
        probe_cursor = probe_results.find(
            {"skill_id": skill_id}
        ).sort("created_at", -1).limit(20)
        probes: List[Dict[str, Any]] = []
        async for p in probe_cursor:
            probes.append(p)

        score_int = int(doc.get("score") or doc.get("current_score") or 0)
        r5 = _derive_r5_risk_score(probes, score_int)

        items.append(
            MarketplaceListItem(
                id=skill_id,
                subject_uri=doc.get("subject_uri") or skill_id,
                name=doc.get("name") or doc.get("skill_name") or skill_id,
                slug=doc.get("slug") or skill_id,
                category=doc.get("category"),
                score=score_int,
                tier=doc.get("tier", "failed"),
                last_eval_at=doc.get("last_evaluated_at"),
                axes=doc.get("dimensions", {}) or doc.get("axes", {}) or {},
                delta_vs_baseline=doc.get("delta_vs_baseline"),
                activation_provider=doc.get("activation_provider", "cerebras:llama3.1-8b"),
                r5_risk_score=r5,
            )
        )

    return items


@router.get("/marketplace/{slug}", response_model=MarketplaceListResponse)
async def list_marketplace(slug: str) -> MarketplaceListResponse:
    """List all skills in a public marketplace ``slug`` with R5 risk projection.

    Read-through Redis cache. Public endpoint — no API key required.
    """
    if not slug or not slug.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid marketplace slug")

    cache_key = f"qo:marketplace:{slug}"
    try:
        r = get_redis()
        cached = await r.get(cache_key)
        if cached:
            data = json.loads(cached)
            return MarketplaceListResponse(**data)
    except Exception as exc:  # pragma: no cover — Redis transient
        logger.warning("marketplace cache read failed: %s", exc)

    items = await _build_marketplace_list(slug)
    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"No skills found for marketplace {slug!r}",
        )

    # Top-10 risk surface (R5 §12) — sort descending by r5_risk_score.
    top_risks = sorted(items, key=lambda it: it.r5_risk_score, reverse=True)[:10]

    avg_score = round(sum(i.score for i in items) / len(items)) if items else 0
    response = MarketplaceListResponse(
        slug=slug,
        items=items,
        total=len(items),
        avg_score=avg_score,
        top_risks=[i.id for i in top_risks],
        generated_at=datetime.utcnow(),
    )

    try:
        await r.set(
            cache_key,
            response.model_dump_json(),
            ex=MARKETPLACE_CACHE_TTL,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("marketplace cache write failed: %s", exc)

    return response


@router.get("/marketplace/{slug}/{skill}", response_model=MarketplaceSkillDetail)
async def get_skill_detail(slug: str, skill: str) -> MarketplaceSkillDetail:
    """Per-skill detail for a marketplace listing.

    Returns last-snapshot only (multi-snapshot history is QO-053-H2).
    """
    db = get_db()
    skill_scores = db.quality__skill_scores
    probe_results = db.quality__probe_results

    doc = await skill_scores.find_one({
        "$and": [
            _slug_to_repo_filter(slug),
            {"$or": [{"skill_id": skill}, {"slug": skill}, {"name": skill}]},
        ]
    })
    if not doc:
        raise HTTPException(status_code=404, detail=f"Skill {skill!r} not found in {slug!r}")

    skill_id = doc.get("skill_id") or doc.get("subject_uri") or skill
    probes_list: List[Dict[str, Any]] = []
    async for p in probe_results.find({"skill_id": skill_id}).sort("created_at", -1).limit(50):
        # Drop _id for JSON serialisation simplicity.
        p.pop("_id", None)
        probes_list.append(p)

    score_int = int(doc.get("score") or doc.get("current_score") or 0)
    r5 = _derive_r5_risk_score(probes_list, score_int)

    return MarketplaceSkillDetail(
        id=skill_id,
        subject_uri=doc.get("subject_uri") or skill_id,
        name=doc.get("name") or skill_id,
        slug=doc.get("slug") or skill,
        category=doc.get("category"),
        score=score_int,
        tier=doc.get("tier", "failed"),
        last_eval_at=doc.get("last_evaluated_at"),
        axes=doc.get("dimensions", {}) or doc.get("axes", {}) or {},
        na_axes=doc.get("na_axes", []),
        delta_vs_baseline=doc.get("delta_vs_baseline"),
        baseline_score=doc.get("baseline_score"),
        activation_provider=doc.get("activation_provider", "cerebras:llama3.1-8b"),
        r5_risk_score=r5,
        github_url=doc.get("github_url"),
        owner=doc.get("owner"),
        probe_results=probes_list,
        last_snapshot_at=doc.get("last_evaluated_at"),
    )


@router.get("/marketplace/{slug}/{skill}/aqvc.json")
async def get_skill_aqvc(slug: str, skill: str):
    """Pull the AQVC credential for this skill from QO-053-I issuer.

    Returns the W3C VC JSON document if persisted on the latest attestation.
    """
    _ = evaluations_col  # keep import live for test patching
    db = get_db()
    skill_scores = db.quality__skill_scores

    score_doc = await skill_scores.find_one({
        "$and": [
            _slug_to_repo_filter(slug),
            {"$or": [{"skill_id": skill}, {"slug": skill}, {"name": skill}]},
        ]
    })
    if not score_doc:
        raise HTTPException(status_code=404, detail="skill not found")

    target_id = score_doc.get("subject_uri") or score_doc.get("skill_id") or skill
    last_eval_id = score_doc.get("last_evaluation_id")

    # Prefer attestations keyed on evaluation_id; fall back to latest by target_id.
    attest_doc: Optional[Dict[str, Any]] = None
    if last_eval_id:
        attest_doc = await attestations_col().find_one({"evaluation_id": last_eval_id})
    if not attest_doc:
        attest_doc = await attestations_col().find_one(
            {"target_id": target_id},
            sort=[("issued_at", -1)],
        )
    if not attest_doc:
        # No upstream AQVC yet — synthesise a stub so the page can render the
        # "AQVC pending" state without 404. AC6 still requires valid JSON.
        return JSONResponse(
            content={
                "aqvc_version": "1.0",
                "issuer": "did:web:laureum.ai",
                "subject": {"id": target_id, "type": "skill", "name": skill},
                "status": "pending",
                "note": "AQVC will be issued after next batch eval (QO-053-I).",
            },
            headers={"Cache-Control": "public, max-age=60"},
        )

    payload = attest_doc.get("vc_document") or attest_doc.get("aqvc_payload") or {}
    if not payload:
        raise HTTPException(status_code=500, detail="attestation missing AQVC payload")

    # Force a sensible filename in the browser via Content-Disposition.
    headers = {
        "Cache-Control": "public, max-age=300",
        "Content-Disposition": f'attachment; filename="{skill}-aqvc.json"',
    }
    # Drop Mongo _id if it leaked into the payload.
    if isinstance(payload, dict):
        payload.pop("_id", None)
    return JSONResponse(content=payload, headers=headers)

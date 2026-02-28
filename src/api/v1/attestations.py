"""Attestation endpoints — JWT (Phase 1), W3C VC (Phase 2)."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from src.storage.mongodb import attestations_col
from src.core.attestation import verify_attestation as verify_jwt
from src.storage.cache import (
    cache_attestation_verify,
    get_cached_attestation_verify,
)
from src.auth.dependencies import get_api_key

router = APIRouter()


@router.get("/attestation/{attestation_id}")
async def get_attestation(
    attestation_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Get a signed JWT quality attestation."""
    doc = await attestations_col().find_one({"_id": attestation_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Attestation not found")

    return {
        "attestation_id": attestation_id,
        "attestation_jwt": doc.get("attestation_jwt"),
        "uaqa_payload": doc.get("uaqa_payload"),
        "evaluation_version": doc.get("evaluation_version"),
        "issued_at": doc.get("issued_at"),
        "expires_at": doc.get("expires_at"),
        "revoked": doc.get("revoked", False),
    }


@router.get("/attestation/{attestation_id}/verify")
async def verify_attestation_endpoint(
    attestation_id: str,
    api_key_doc: dict = Depends(get_api_key),
):
    """Verify the validity of a JWT quality attestation."""
    # Check cache first (24h TTL)
    cached = await get_cached_attestation_verify(attestation_id)
    if cached:
        return cached

    doc = await attestations_col().find_one({"_id": attestation_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Attestation not found")

    if doc.get("revoked"):
        result = {
            "valid": False,
            "error": f"Attestation revoked: {doc.get('revoked_reason', 'no reason given')}",
        }
        await cache_attestation_verify(attestation_id, result)
        return result

    token = doc.get("attestation_jwt")
    if not token:
        raise HTTPException(status_code=500, detail="Attestation JWT missing")

    result = verify_jwt(token)

    # Enrich with DB info
    uaqa = doc.get("uaqa_payload", {})
    result["target_id"] = uaqa.get("subject", {}).get("id", "")
    result["quality_score"] = uaqa.get("quality", {}).get("score", 0)
    result["tier"] = uaqa.get("quality", {}).get("tier", "")

    await cache_attestation_verify(attestation_id, result)
    return result


@router.get("/attestation/{attestation_id}/vc")
async def get_attestation_vc(attestation_id: str):
    """Get the W3C Verifiable Credential for an attestation.

    Public endpoint — VCs are designed to be shared and verified by anyone.
    Returns 404 if attestation not found or has no VC (pre-VC attestation).
    """
    doc = await attestations_col().find_one({"_id": attestation_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Attestation not found")

    vc = doc.get("vc_document")
    if not vc:
        raise HTTPException(status_code=404, detail="No W3C VC available for this attestation")

    return JSONResponse(content=vc, media_type="application/vc+json")

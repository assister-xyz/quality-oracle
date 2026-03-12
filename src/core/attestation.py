"""
AQVC attestation issuance — Phase 1: JWT signing (Ed25519).

Issues AQVC (Agent Quality Verifiable Credential) as signed JWTs.
Phase 2 (Week 5+): wrap in W3C Verifiable Credential envelope via src/standards/vc_issuer.py.
"""
import logging
from datetime import datetime, timedelta
from uuid import uuid4

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from src.config import settings

logger = logging.getLogger(__name__)

# Module-level key cache
_private_key: Ed25519PrivateKey | None = None


def _get_or_generate_key() -> Ed25519PrivateKey:
    """Load Ed25519 private key from file, or generate and persist one.

    Priority:
    1. If jwt_private_key_path is set, load from that path (or generate + save there)
    2. Otherwise, use default path data/jwt_private_key.pem (generate + save if missing)
    """
    from pathlib import Path

    global _private_key
    if _private_key is not None:
        return _private_key

    # Determine key file path
    key_path = settings.jwt_private_key_path or "data/jwt_private_key.pem"
    key_file = Path(key_path)

    # Try loading existing key
    if key_file.exists():
        try:
            _private_key = serialization.load_pem_private_key(
                key_file.read_bytes(), password=None
            )
            logger.info(f"Loaded Ed25519 private key from {key_file}")
            return _private_key
        except Exception as e:
            logger.warning(f"Failed to load private key from {key_file}: {e}")

    # Generate new key and persist it
    _private_key = Ed25519PrivateKey.generate()
    try:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        pem_bytes = _private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        key_file.write_bytes(pem_bytes)
        key_file.chmod(0o600)
        logger.info(f"Generated and saved Ed25519 private key to {key_file}")
    except Exception as e:
        logger.warning(f"Generated Ed25519 key but failed to save to {key_file}: {e}")

    return _private_key


def get_public_key_pem() -> str:
    """Get the public key in PEM format for verification."""
    key = _get_or_generate_key()
    pub = key.public_key()
    return pub.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def create_attestation(
    target_id: str,
    target_type: str,
    target_name: str,
    evaluation_result: dict,
    evaluation_version: str = "v1.0",
    issuer: str | None = None,
    validity_days: int | None = None,
    eval_mode: str | None = None,
) -> dict:
    """
    Create an AQVC attestation as a signed JWT.

    Returns a dict ready for MongoDB insertion with:
    - _id: attestation ID
    - attestation_jwt: signed JWT string
    - aqvc_payload: raw AQVC JSON payload
    - evaluation_version, issued_at, expires_at
    """
    now = datetime.utcnow()
    attestation_id = str(uuid4())
    iss = issuer or settings.jwt_issuer
    exp_days = validity_days or settings.attestation_validity_days
    expires_at = now + timedelta(days=exp_days)

    # AQVC payload (canonical format, used in both JWT and future VC)
    aqvc_payload = {
        "aqvc_version": "1.0",
        "issuer": iss,
        "issued_at": now.isoformat() + "Z",
        "expires_at": expires_at.isoformat() + "Z",
        "evaluation_version": evaluation_version,
        "subject": {
            "id": target_id,
            "type": target_type,
            "name": target_name,
        },
        "quality": {
            "score": evaluation_result.get("overall_score", 0),
            "tier": evaluation_result.get("tier", "failed"),
            "trust_level": eval_mode,
            "confidence": evaluation_result.get("confidence", 0),
            "evaluation_level": evaluation_result.get("level", 2),
            "domains": evaluation_result.get("domains", []),
            "tool_scores": evaluation_result.get("tool_scores", {}),
            "questions_asked": evaluation_result.get("questions_asked", 0),
        },
        "evaluation": {
            "id": attestation_id,
            "method": "challenge-response-v1",
            "evaluated_at": now.isoformat() + "Z",
            "verification_mode": "oracle_verified",
            "connection_strategy": evaluation_result.get("connection_strategy", "sse"),
        },
    }

    # Add AIUC-1 alignment section
    try:
        from src.standards.aiuc1_mapping import generate_aiuc1_report
        aiuc1_report = generate_aiuc1_report(evaluation_result)
        aqvc_payload["aiuc1_alignment"] = {
            "standard": "AIUC-1",
            "version": aiuc1_report["aiuc1_version"],
            "coverage_percentage": aiuc1_report["coverage_percentage"],
            "mandatory_coverage": aiuc1_report["mandatory_coverage"],
            "controls_fully_covered": aiuc1_report["controls_fully_covered"],
            "controls_partially_covered": aiuc1_report["controls_partially_covered"],
            "controls_not_covered": aiuc1_report["controls_not_covered"],
        }
    except Exception as e:
        logger.warning(f"Failed to generate AIUC-1 alignment: {e}")

    # Sign as JWT (Ed25519)
    key = _get_or_generate_key()
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    token = jwt.encode(
        aqvc_payload,
        private_pem,
        algorithm="EdDSA",
        headers={"kid": f"{iss}#key-1"},
    )

    logger.info(
        f"Created JWT attestation {attestation_id} for {target_id}: "
        f"score={aqvc_payload['quality']['score']}"
    )

    # Create W3C Verifiable Credential envelope
    vc_document = None
    try:
        from src.standards.vc_issuer import create_vc
        vc_document = create_vc(aqvc_payload, key, issuer_did=iss)
        logger.info(f"Created W3C VC for attestation {attestation_id}")
    except Exception as e:
        logger.warning(f"Failed to create W3C VC: {e}")

    return {
        "_id": attestation_id,
        "evaluation_id": aqvc_payload["evaluation"]["id"],
        "target_id": target_id,
        "attestation_jwt": token,
        "aqvc_payload": aqvc_payload,
        "vc_document": vc_document,
        "evaluation_version": evaluation_version,
        "issued_at": now,
        "expires_at": expires_at,
        "revoked": False,
        "revoked_reason": None,
    }


def verify_attestation(token: str) -> dict:
    """
    Verify a JWT attestation signature and return decoded payload.

    Returns dict with: valid, payload, issuer, issued_at, expires_at
    """
    try:
        key = _get_or_generate_key()
        pub_pem = key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        payload = jwt.decode(token, pub_pem, algorithms=["EdDSA"])
        return {
            "valid": True,
            "payload": payload,
            "issuer": payload.get("issuer"),
            "issued_at": payload.get("issued_at"),
            "expires_at": payload.get("expires_at"),
        }
    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "Attestation has expired"}
    except jwt.InvalidTokenError as e:
        return {"valid": False, "error": f"Invalid token: {e}"}

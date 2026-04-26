"""
AQVC v1.0 — Agent Quality Verifiable Credential issuer.

This module is the canonical Laureum credential format. Every external
standard (A2A, MCP, OpenAPI, ERC-8004, SARIF, EU AI Act) consumes AQVC.

Spec: QO-053-I (W3C VC 2.0 + did:web + Bitstring Status List).
Research: research/laureum-skills-2026-04-25/R10-standards.md §2.

Produces W3C VC 2.0 envelopes signed with eddsa-jcs-2022 (Ed25519 +
JSON Canonicalization Scheme). did:web identifier resolves at
https://laureum.ai/.well-known/did.json.

Migrated from `src/standards/vc_issuer.py` per QO-053-I §"Files Affected".
The old module path is kept as a re-export shim for one release; remove
in QO-053-I.1 follow-up.
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Tuple
from uuid import uuid4

import rfc8785
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

# Default issuer DID — overridden per-call by `settings.jwt_issuer`.
DEFAULT_ISSUER_DID = "did:web:laureum.ai"

# JSON-LD context URL — hosted at quality-oracle-demo (Vercel).
LAUREUM_VC_CONTEXT = "https://laureum.ai/credentials/v1"

# Status list URL — hosted at quality-oracle-demo (Vercel) and re-issued
# daily via Vercel Cron (QO-053-I AC11).
STATUS_LIST_CREDENTIAL_URL = "https://laureum.ai/credentials/status/1"

# JSON Schema URL for AQVC v1.0.
AQVC_JSON_SCHEMA_URL = "https://laureum.ai/schemas/aqvc/v1.json"

# AQVC version published in `credentialSubject.aqvcVersion`.
AQVC_VERSION = "1.0"


# ── Base58btc (Bitcoin alphabet) ─────────────────────────────────────────────

_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_BASE = 58


def _base58btc_encode(data: bytes) -> str:
    """Encode bytes to base58btc (Bitcoin alphabet)."""
    n = int.from_bytes(data, "big")
    result = []
    while n > 0:
        n, r = divmod(n, _B58_BASE)
        result.append(_B58_ALPHABET[r])
    # Preserve leading zero bytes
    for b in data:
        if b == 0:
            result.append(_B58_ALPHABET[0])
        else:
            break
    return bytes(reversed(result)).decode("ascii")


def _base58btc_decode(s: str) -> bytes:
    """Decode base58btc string to bytes."""
    n = 0
    for c in s:
        idx = _B58_ALPHABET.index(c.encode("ascii"))
        n = n * _B58_BASE + idx
    # Count leading '1's (zero bytes)
    pad = 0
    for c in s:
        if c == "1":
            pad += 1
        else:
            break
    result = n.to_bytes((n.bit_length() + 7) // 8, "big") if n > 0 else b""
    return b"\x00" * pad + result


# ── Multikey Encoding (Ed25519) ──────────────────────────────────────────────

# Ed25519 multicodec prefix: 0xed01
_ED25519_MULTICODEC = b"\xed\x01"


def encode_public_key_multibase(public_key: Ed25519PublicKey) -> str:
    """Encode Ed25519 public key as multibase base58btc with multicodec prefix.

    Format: 'z' + base58btc(0xed01 + raw_32_bytes)
    Per W3C Multikey spec for Ed25519.
    """
    raw = public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return "z" + _base58btc_encode(_ED25519_MULTICODEC + raw)


def decode_public_key_multibase(multibase_str: str) -> Ed25519PublicKey:
    """Decode a multibase base58btc encoded Ed25519 public key.

    Expects: 'z' prefix + base58btc(0xed01 + 32 bytes)
    """
    if not multibase_str.startswith("z"):
        raise ValueError("Expected multibase base58btc prefix 'z'")
    decoded = _base58btc_decode(multibase_str[1:])
    if not decoded.startswith(_ED25519_MULTICODEC):
        raise ValueError("Missing Ed25519 multicodec prefix (0xed01)")
    raw_key = decoded[len(_ED25519_MULTICODEC):]
    if len(raw_key) != 32:
        raise ValueError(f"Expected 32-byte Ed25519 key, got {len(raw_key)}")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey as PubKey
    return PubKey.from_public_bytes(raw_key)


# ── eddsa-jcs-2022 Signing ───────────────────────────────────────────────────

def _jcs_hash(obj: dict) -> bytes:
    """JCS canonicalize a dict and return its SHA-256 hash."""
    canonical = rfc8785.dumps(obj)
    return hashlib.sha256(canonical).digest()


def _sign_eddsa_jcs_2022(
    document: dict, proof_options: dict, private_key: Ed25519PrivateKey,
) -> str:
    """Sign using eddsa-jcs-2022 algorithm.

    1. JCS canonicalize proof_options (without proofValue) → SHA-256 hash
    2. JCS canonicalize document (without proof) → SHA-256 hash
    3. Concatenate both hashes → Ed25519 sign → base58btc encode with 'z' prefix
    """
    options_hash = _jcs_hash(proof_options)
    doc_hash = _jcs_hash(document)
    combined = options_hash + doc_hash
    signature = private_key.sign(combined)
    return "z" + _base58btc_encode(signature)


def _verify_eddsa_jcs_2022(
    document: dict, proof: dict, public_key: Ed25519PublicKey,
) -> Tuple[bool, str]:
    """Verify eddsa-jcs-2022 proof.

    Returns (valid, error_message).
    """
    proof_value = proof.get("proofValue", "")
    if not proof_value.startswith("z"):
        return False, "proofValue must start with 'z' (base58btc multibase)"

    # Reconstruct proof options (proof dict without proofValue)
    proof_options = {k: v for k, v in proof.items() if k != "proofValue"}

    # Reconstruct document (without proof)
    doc_without_proof = {k: v for k, v in document.items() if k != "proof"}

    options_hash = _jcs_hash(proof_options)
    doc_hash = _jcs_hash(doc_without_proof)
    combined = options_hash + doc_hash

    try:
        signature = _base58btc_decode(proof_value[1:])
        public_key.verify(signature, combined)
        return True, ""
    except Exception as e:
        return False, f"Signature verification failed: {e}"


# ── snake_case → camelCase mapping (QO-053-I §"Field-name mapping table") ───

# Spec M1: every snake_case key in EvaluationResult.scores_6axis maps to a
# camelCase key in AQVC JSON-LD credentialSubject.scores6Axis.
_AXIS_KEY_MAP = {
    "process_quality": "processQuality",
    "schema_quality": "schemaQuality",
}


def _camel_axes(scores_6axis: dict[str, Any]) -> dict[str, Any]:
    """Translate 6-axis scoring keys from snake_case to camelCase."""
    return {_AXIS_KEY_MAP.get(k, k): v for k, v in (scores_6axis or {}).items()}


# ── BitstringStatusListEntry helper ─────────────────────────────────────────

def build_credential_status(status_index: int, base_url: Optional[str] = None) -> dict:
    """Build a BitstringStatusListEntry per W3C BSL spec.

    Every issued AQVC v1.0 carries this — without it revocation is impossible.
    AC4 (revocation works) requires this on every credential.
    """
    list_url = (base_url or STATUS_LIST_CREDENTIAL_URL).rstrip("/")
    return {
        "id": f"{list_url}#{status_index}",
        "type": "BitstringStatusListEntry",
        "statusPurpose": "revocation",
        "statusListIndex": str(status_index),
        "statusListCredential": list_url,
    }


# ── W3C VC v2.0 Creation ────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_vc(
    aqvc_payload: dict,
    private_key: Ed25519PrivateKey,
    issuer_did: Optional[str] = None,
    vc_id: Optional[str] = None,
    status_index: Optional[int] = None,
) -> dict:
    """Create a W3C Verifiable Credential v2.0 with DataIntegrityProof.

    Args:
        aqvc_payload: AQVC attestation payload (from create_attestation).
        private_key: Ed25519 private key for signing.
        issuer_did: DID of the issuer. Defaults to did:web:laureum.ai.
        vc_id: Optional VC identifier. Auto-generated if not provided.
        status_index: Bitstring Status List index assigned to this VC.
            When None, no `credentialStatus` field is emitted (legacy mode for
            historical attestations that pre-date the status list).

    Returns:
        Complete VC document with proof.
    """
    iss = issuer_did or DEFAULT_ISSUER_DID
    cred_id = vc_id or f"urn:uuid:{uuid4()}"
    now = datetime.now(timezone.utc)

    # Extract fields from AQVC payload
    subject = aqvc_payload.get("subject", {})
    quality = aqvc_payload.get("quality", {})
    evaluation = aqvc_payload.get("evaluation", {})
    expires_at = aqvc_payload.get("expires_at") or ""

    # Build credential (without proof)
    credential: dict[str, Any] = {
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://w3id.org/security/data-integrity/v2",
            LAUREUM_VC_CONTEXT,
        ],
        "id": cred_id,
        "type": ["VerifiableCredential", "AgentQualityCredential"],
        "issuer": iss,
        "validFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # AC3: omit validUntil entirely when no expiry — never emit empty string.
    if expires_at:
        credential["validUntil"] = expires_at

    # AC4: BitstringStatusListEntry on every issued AQVC.
    if status_index is not None:
        credential["credentialStatus"] = build_credential_status(status_index)

    # AC2: credentialSchema points at the JSON Schema document.
    credential["credentialSchema"] = {
        "id": AQVC_JSON_SCHEMA_URL,
        "type": "JsonSchema",
    }

    credential["credentialSubject"] = {
        "id": subject.get("id", ""),
        "type": subject.get("type", ""),
        "name": subject.get("name", ""),
        "qualityScore": quality.get("score", 0),
        "qualityTier": quality.get("tier", "failed"),
        "confidence": quality.get("confidence", 0),
        "evaluationLevel": quality.get("evaluation_level", 2),
        "domains": quality.get("domains", []),
        "questionsAsked": quality.get("questions_asked", 0),
    }

    credential["evidence"] = [{
        "type": "QualityEvaluation",
        "evaluationId": evaluation.get("id", ""),
        "method": evaluation.get("method", "challenge-response-v1"),
        "evaluatedAt": evaluation.get("evaluated_at", now.strftime("%Y-%m-%dT%H:%M:%SZ")),
    }]

    # Build proof options (without proofValue)
    proof_options = {
        "type": "DataIntegrityProof",
        "cryptosuite": "eddsa-jcs-2022",
        "verificationMethod": f"{iss}#key-1",
        "proofPurpose": "assertionMethod",
        "created": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Sign
    proof_value = _sign_eddsa_jcs_2022(credential, proof_options, private_key)

    # Add proofValue to proof
    proof = {**proof_options, "proofValue": proof_value}
    credential["proof"] = proof

    return credential


# ── VC Verification ──────────────────────────────────────────────────────────

def verify_vc(
    vc_document: dict,
    public_key: Ed25519PublicKey,
) -> Tuple[bool, str]:
    """Verify a W3C VC with eddsa-jcs-2022 DataIntegrityProof.

    Args:
        vc_document: Complete VC document with proof.
        public_key: Ed25519 public key for verification.

    Returns:
        (valid, error_message) tuple.
    """
    proof = vc_document.get("proof")
    if not proof:
        return False, "No proof found in VC"

    if proof.get("cryptosuite") != "eddsa-jcs-2022":
        return False, f"Unsupported cryptosuite: {proof.get('cryptosuite')}"

    return _verify_eddsa_jcs_2022(vc_document, proof, public_key)


# ── DID Document ─────────────────────────────────────────────────────────────

def build_did_document(
    public_key: Ed25519PublicKey,
    issuer_did: Optional[str] = None,
    historical_keys: Optional[list[dict]] = None,
) -> dict:
    """Build a DID Document with Multikey verification method.

    Returns a DID Document suitable for /.well-known/did.json.

    Args:
        public_key: Current active Ed25519 public key (in `assertionMethod[]`).
        issuer_did: Override issuer DID. Defaults to did:web:laureum.ai.
        historical_keys: Optional list of `{id, publicKeyMultibase}` dicts for
            previously rotated keys. Old keys remain in `verificationMethod[]`
            so historical AQVCs verify, but only the current key is in
            `assertionMethod[]` (AC9 — quarterly rotation policy).
    """
    did = issuer_did or DEFAULT_ISSUER_DID
    multibase_key = encode_public_key_multibase(public_key)
    current_key_id = f"{did}#key-1"

    verification_methods = [{
        "id": current_key_id,
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": multibase_key,
    }]

    if historical_keys:
        for hk in historical_keys:
            verification_methods.append({
                "id": hk.get("id") or f"{did}#{hk['key_id']}",
                "type": "Multikey",
                "controller": did,
                "publicKeyMultibase": hk["publicKeyMultibase"],
            })

    return {
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/multikey/v1",
        ],
        "id": did,
        "verificationMethod": verification_methods,
        # Only the current key signs new assertions; historical keys remain
        # in verificationMethod[] but NOT in assertionMethod[].
        "assertionMethod": [current_key_id],
        "authentication": [current_key_id],
    }


# ── AQVC v1.0 Skill Subject Builder ──────────────────────────────────────────

def build_aqvc_skill(
    eval_result: Any,
    parsed_skill: Any,
    status_index: int,
    issuer_did: Optional[str] = None,
    vc_id: Optional[str] = None,
) -> dict:
    """Build an AQVC v1.0 envelope for a Claude Skill subject.

    Per QO-053-I §"Technical Design > build_aqvc()". This is the canonical
    builder for `subjectType=claude_skill` credentials. Output is a dict
    matching R10 §2.4 sample byte-for-byte (mod signature, UUID, dates).

    The caller is responsible for signing — this only assembles the envelope.

    Args:
        eval_result: Object with skill-evaluation fields. Accessed via
            attribute access; falls back to `.get()` style. Expected fields
            (snake_case in source, mapped to camelCase per spec M1):
            - target.subject_uri (or target_uri)
            - eval_hash
            - absolute_score (or overall_score)
            - tier, confidence
            - questions_asked
            - scores_6axis: dict
            - judges: list[{provider, model, role}]
            - probes_used: list[str]
            - model_versions: dict (must contain `activation_provider` per AC12)
            - target_protocol (AC13: includes activation provider)
            - ide_runtime: optional dict
            - cpcr: optional float
            - aiuc_alignment: optional dict
            - sarif_report_uri: optional str
            - repo_signals: optional dict
            - expires_at: optional datetime / iso str
        parsed_skill: Object with `name` and `metadata` attributes.
        status_index: Bitstring Status List index assigned to this VC.

    Returns:
        Unsigned AQVC envelope (no `proof` field). Caller signs separately
        or uses `create_aqvc()` (signed end-to-end).
    """
    iss = issuer_did or DEFAULT_ISSUER_DID
    cred_id = vc_id or f"urn:uuid:{uuid4()}"

    # Resolve subject id — prefer `target.subject_uri`, else `target_uri`.
    target = _attr(eval_result, "target", None)
    subject_uri = _attr(target, "subject_uri", None) if target is not None else None
    if not subject_uri:
        subject_uri = _attr(eval_result, "target_uri", "")

    # Skill metadata
    skill_name = _attr(parsed_skill, "name", "unknown")
    metadata = _attr(parsed_skill, "metadata", {}) or {}
    if isinstance(metadata, dict):
        skill_version = metadata.get("version", "unknown")
    else:
        skill_version = _attr(metadata, "version", "unknown")

    # Score fields — prefer absolute_score, fall back to overall_score.
    overall_score = _attr(eval_result, "absolute_score", None)
    if overall_score is None:
        overall_score = _attr(eval_result, "overall_score", 0)

    # 6-axis scores (snake → camelCase per M1)
    raw_axes = _attr(eval_result, "scores_6axis", {}) or {}
    if hasattr(raw_axes, "model_dump"):
        raw_axes = raw_axes.model_dump()
    elif hasattr(raw_axes, "dict") and callable(raw_axes.dict):
        raw_axes = raw_axes.dict()
    scores_6axis = _camel_axes(raw_axes)

    # Judges — accept list of dicts or list of objects with .provider/.model/.role.
    judges_in = _attr(eval_result, "judges", []) or []
    judges_out = []
    for j in judges_in:
        if isinstance(j, dict):
            judges_out.append({
                "provider": j.get("provider", ""),
                "model": j.get("model", ""),
                "role": j.get("role", ""),
            })
        else:
            judges_out.append({
                "provider": _attr(j, "provider", ""),
                "model": _attr(j, "model", ""),
                "role": _attr(j, "role", ""),
            })

    cs: dict[str, Any] = {
        "id": subject_uri,
        "type": "AgentQualitySubject",
        "subjectType": "claude_skill",
        "name": skill_name,
        "version": skill_version,
        "evalHash": _attr(eval_result, "eval_hash", ""),
        "overallScore": overall_score,
        "tier": _attr(eval_result, "tier", "failed"),
        "confidence": _attr(eval_result, "confidence", 0),
        "questionsAsked": _attr(eval_result, "questions_asked", 0),
        "scores6Axis": scores_6axis,
        "judges": judges_out,
        "probesUsed": list(_attr(eval_result, "probes_used", []) or []),
        "modelVersions": _attr(eval_result, "model_versions", {}) or {},
        "ideRuntime": _attr(eval_result, "ide_runtime", {}) or {},
        "aqvcVersion": AQVC_VERSION,
    }

    target_protocol = _attr(eval_result, "target_protocol", None)
    if target_protocol is not None:
        cs["targetProtocol"] = target_protocol

    cpcr = _attr(eval_result, "cpcr", None)
    if cpcr is not None:
        cs["cpcr"] = cpcr

    aiuc = _attr(eval_result, "aiuc_alignment", None)
    if aiuc:
        cs["aiucAlignment"] = aiuc

    sarif = _attr(eval_result, "sarif_report_uri", None)
    if sarif:
        cs["sarifReportUri"] = sarif

    repo_signals = _attr(eval_result, "repo_signals", None)
    if repo_signals:
        cs["repoSignals"] = repo_signals

    envelope: dict[str, Any] = {
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://w3id.org/security/data-integrity/v2",
            LAUREUM_VC_CONTEXT,
        ],
        "id": cred_id,
        "type": ["VerifiableCredential", "AgentQualityCredential"],
        "issuer": iss,
        "validFrom": _now_iso(),
    }

    # AC3 — omit validUntil if no expiry. Never emit "" (empty string).
    expires_at = _attr(eval_result, "expires_at", None)
    if expires_at:
        envelope["validUntil"] = (
            expires_at.isoformat() if hasattr(expires_at, "isoformat") else str(expires_at)
        )

    # AC4 — BitstringStatusListEntry on every issued AQVC.
    envelope["credentialStatus"] = build_credential_status(status_index)

    envelope["credentialSchema"] = {
        "id": AQVC_JSON_SCHEMA_URL,
        "type": "JsonSchema",
    }

    envelope["credentialSubject"] = cs

    # Evidence pointer — caller may extend/replace before signing.
    eval_id = _attr(eval_result, "evaluation_id", None) or _attr(eval_result, "eval_hash", "")
    evaluated_at = _attr(eval_result, "evaluated_at", None)
    if evaluated_at and hasattr(evaluated_at, "isoformat"):
        evaluated_at = evaluated_at.isoformat()
    elif not evaluated_at:
        evaluated_at = _now_iso()
    envelope["evidence"] = [{
        "type": ["QualityEvaluation"],
        "id": f"https://laureum.ai/evidence/{eval_id}" if eval_id else "",
        "method": "challenge-response-v1",
        "evaluatedAt": evaluated_at,
    }]

    return envelope


def create_aqvc_skill(
    eval_result: Any,
    parsed_skill: Any,
    status_index: int,
    private_key: Ed25519PrivateKey,
    issuer_did: Optional[str] = None,
    vc_id: Optional[str] = None,
) -> dict:
    """Build + sign an AQVC v1.0 for a Claude Skill subject.

    Convenience wrapper around `build_aqvc_skill()` that adds an
    `eddsa-jcs-2022` DataIntegrityProof.
    """
    iss = issuer_did or DEFAULT_ISSUER_DID
    envelope = build_aqvc_skill(
        eval_result=eval_result,
        parsed_skill=parsed_skill,
        status_index=status_index,
        issuer_did=iss,
        vc_id=vc_id,
    )

    proof_options = {
        "type": "DataIntegrityProof",
        "cryptosuite": "eddsa-jcs-2022",
        "verificationMethod": f"{iss}#key-1",
        "proofPurpose": "assertionMethod",
        "created": _now_iso(),
    }
    proof_value = _sign_eddsa_jcs_2022(envelope, proof_options, private_key)
    envelope["proof"] = {**proof_options, "proofValue": proof_value}
    return envelope


def _attr(obj: Any, name: str, default: Any) -> Any:
    """Tolerant attribute reader — works for dataclasses, pydantic, and dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

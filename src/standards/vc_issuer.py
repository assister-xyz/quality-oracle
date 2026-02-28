"""
W3C Verifiable Credential issuance — eddsa-jcs-2022 with Ed25519.

Issues QualityAttestation VCs in W3C VC Data Model v2.0 format.
Signing: DataIntegrityProof with cryptosuite=eddsa-jcs-2022.
Key encoding: did:web with Multikey (base58btc multicodec).

No heavy deps — manual base58btc + multicodec encoding.
Uses rfc8785 for JSON Canonicalization Scheme (JCS).
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple
from uuid import uuid4

import rfc8785
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

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


# ── W3C VC v2.0 Creation ────────────────────────────────────────────────────

def create_vc(
    uaqa_payload: dict,
    private_key: Ed25519PrivateKey,
    issuer_did: Optional[str] = None,
    vc_id: Optional[str] = None,
) -> dict:
    """Create a W3C Verifiable Credential v2.0 with DataIntegrityProof.

    Args:
        uaqa_payload: UAQA attestation payload (from create_attestation).
        private_key: Ed25519 private key for signing.
        issuer_did: DID of the issuer. Defaults to did:web:quality-oracle.assisterr.ai.
        vc_id: Optional VC identifier. Auto-generated if not provided.

    Returns:
        Complete VC document with proof.
    """
    iss = issuer_did or "did:web:quality-oracle.assisterr.ai"
    cred_id = vc_id or f"urn:uuid:{uuid4()}"
    now = datetime.now(timezone.utc)

    # Extract fields from UAQA payload
    subject = uaqa_payload.get("subject", {})
    quality = uaqa_payload.get("quality", {})
    evaluation = uaqa_payload.get("evaluation", {})
    expires_at = uaqa_payload.get("expires_at", "")

    # Build credential (without proof)
    credential = {
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://w3id.org/security/data-integrity/v2",
            "https://quality-oracle.assisterr.ai/contexts/quality/v1",
        ],
        "id": cred_id,
        "type": ["VerifiableCredential", "QualityAttestation"],
        "issuer": iss,
        "validFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validUntil": expires_at if expires_at else "",
        "credentialSubject": {
            "id": subject.get("id", ""),
            "type": subject.get("type", ""),
            "name": subject.get("name", ""),
            "qualityScore": quality.get("score", 0),
            "qualityTier": quality.get("tier", "failed"),
            "confidence": quality.get("confidence", 0),
            "evaluationLevel": quality.get("evaluation_level", 2),
            "domains": quality.get("domains", []),
            "questionsAsked": quality.get("questions_asked", 0),
        },
        "evidence": [{
            "type": "QualityEvaluation",
            "evaluationId": evaluation.get("id", ""),
            "method": evaluation.get("method", "challenge-response-v1"),
            "evaluatedAt": evaluation.get("evaluated_at", now.strftime("%Y-%m-%dT%H:%M:%SZ")),
        }],
    }

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
) -> dict:
    """Build a DID Document with Multikey verification method.

    Returns a DID Document suitable for /.well-known/did.json.
    """
    did = issuer_did or "did:web:quality-oracle.assisterr.ai"
    multibase_key = encode_public_key_multibase(public_key)

    return {
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/multikey/v1",
        ],
        "id": did,
        "verificationMethod": [{
            "id": f"{did}#key-1",
            "type": "Multikey",
            "controller": did,
            "publicKeyMultibase": multibase_key,
        }],
        "assertionMethod": [f"{did}#key-1"],
        "authentication": [f"{did}#key-1"],
    }

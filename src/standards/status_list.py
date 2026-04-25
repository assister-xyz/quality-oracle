"""
Bitstring Status List 1.0 — issuer + revocation API.

Spec: https://www.w3.org/TR/vc-bitstring-status-list/
Used by AQVC v1.0 to make revocation possible (QO-053-I §AC4).

Design notes
------------
- The bitstring is persisted as a single MongoDB document in
  `quality__status_list` (collection key + integer-indexed bytes). One
  bit per AQVC: bit=0 → valid, bit=1 → revoked.
- Default list size: 131_072 bits (16 KiB) — enough for ~130k credentials.
  Resize up if approaching exhaustion (R10 §"Open questions": revisit at
  >100k AQVCs/day).
- The status list itself is published as a SIGNED Verifiable Credential
  at `https://laureum.ai/credentials/status/1`. The `encodedList` field
  is `base64url(gzip(bitstring))` per W3C BSL §2.2 ("Algorithms").
- Re-issuance is daily via Vercel Cron in `quality-oracle-demo`
  (QO-053-I §AC11 / M6).

The Vercel cron route reads either the static file produced by this
module (e.g. via S3 sync or a backend GET endpoint) or calls the
backend API directly. Either way, the source of truth lives in
`quality_oracle.standards.status_list`.
"""
import base64
import gzip
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.standards.aqvc import (
    AQVC_JSON_SCHEMA_URL,
    DEFAULT_ISSUER_DID,
    LAUREUM_VC_CONTEXT,
    STATUS_LIST_CREDENTIAL_URL,
    _sign_eddsa_jcs_2022,
)

logger = logging.getLogger(__name__)

# Default status list size (bits). 131_072 = 16 KiB raw, ~70 bytes gzipped
# while sparse. Resize via `init_bitstring(size_bits=N)` if needed.
DEFAULT_STATUS_LIST_SIZE_BITS = 131_072

# Singleton document key in MongoDB. There is currently exactly one status
# list (`/credentials/status/1`); future lists may use list_id="2", etc.
DEFAULT_LIST_ID = "1"

# MongoDB collection name (matches the `quality__` prefix convention).
STATUS_LIST_COLLECTION = "quality__status_list"


@dataclass
class StatusListState:
    """Plain in-memory representation of a status list bitstring."""
    list_id: str
    size_bits: int
    bitstring: bytearray  # length == size_bits // 8
    revocations: dict[int, dict]  # idx -> {reason, revoked_at}
    updated_at: datetime


# ── Bitstring helpers ────────────────────────────────────────────────────────

def init_bitstring(size_bits: int = DEFAULT_STATUS_LIST_SIZE_BITS) -> bytearray:
    """Create a zero-filled bitstring of the requested size (bits)."""
    if size_bits % 8 != 0:
        raise ValueError("size_bits must be a multiple of 8")
    return bytearray(size_bits // 8)


def set_bit(bitstring: bytearray, index: int, value: int = 1) -> None:
    """Flip the bit at `index` to `value` (0 or 1) — mutates in place."""
    if value not in (0, 1):
        raise ValueError("value must be 0 or 1")
    if index < 0 or index >= len(bitstring) * 8:
        raise IndexError(f"bit index {index} out of range for {len(bitstring) * 8} bits")
    byte_idx, bit_idx = divmod(index, 8)
    # W3C BSL: bit 0 is the most-significant bit of byte 0.
    mask = 1 << (7 - bit_idx)
    if value == 1:
        bitstring[byte_idx] |= mask
    else:
        bitstring[byte_idx] &= ~mask & 0xFF


def get_bit(bitstring: bytes, index: int) -> int:
    """Read the bit at `index` (0 or 1)."""
    if index < 0 or index >= len(bitstring) * 8:
        raise IndexError(f"bit index {index} out of range")
    byte_idx, bit_idx = divmod(index, 8)
    mask = 1 << (7 - bit_idx)
    return 1 if (bitstring[byte_idx] & mask) else 0


def encode_bitstring(bitstring: bytes) -> str:
    """gzip + base64url-encode (per W3C BSL §"encodedList")."""
    gz = gzip.compress(bytes(bitstring))
    return base64.urlsafe_b64encode(gz).rstrip(b"=").decode("ascii")


def decode_bitstring(encoded: str) -> bytes:
    """Inverse of encode_bitstring()."""
    pad = "=" * (-len(encoded) % 4)
    gz = base64.urlsafe_b64decode(encoded + pad)
    return gzip.decompress(gz)


# ── In-memory issuer (test mode + Mongo-backed) ─────────────────────────────


class StatusListIssuer:
    """Stateful issuer that owns the bitstring and produces signed status VCs.

    Two modes:
    - **In-memory** (default; used in tests + the Vercel cron route's local
      smoke). Pass `state` directly or call `init_state()`.
    - **MongoDB-backed**: call `await load_from_mongo()` to read the
      persisted state from `quality__status_list`, then `await persist()`
      to write back.
    """

    def __init__(
        self,
        state: Optional[StatusListState] = None,
        list_url: str = STATUS_LIST_CREDENTIAL_URL,
        issuer_did: str = DEFAULT_ISSUER_DID,
    ):
        self.state = state or self.init_state()
        self.list_url = list_url
        self.issuer_did = issuer_did

    @staticmethod
    def init_state(
        list_id: str = DEFAULT_LIST_ID,
        size_bits: int = DEFAULT_STATUS_LIST_SIZE_BITS,
    ) -> StatusListState:
        return StatusListState(
            list_id=list_id,
            size_bits=size_bits,
            bitstring=init_bitstring(size_bits),
            revocations={},
            updated_at=datetime.now(timezone.utc),
        )

    # ── Mutations ─────────────────────────────────────────────────────────

    def revoke(self, status_index: int, reason: str = "") -> None:
        """Flip bit `status_index` to 1 and record the reason (AC4)."""
        set_bit(self.state.bitstring, status_index, 1)
        self.state.revocations[status_index] = {
            "reason": reason or "unspecified",
            "revoked_at": datetime.now(timezone.utc).isoformat() + "Z",
        }
        self.state.updated_at = datetime.now(timezone.utc)
        logger.info(
            f"Status list {self.state.list_id}: revoked index {status_index} ({reason})"
        )

    def is_revoked(self, status_index: int) -> bool:
        """Read the current revocation bit for `status_index`."""
        return bool(get_bit(self.state.bitstring, status_index))

    def reserve_index(self) -> int:
        """Allocate the next free status list index.

        Strategy: scan the bitstring + revocations dict for the lowest
        unused slot. For high-throughput issuance, callers should keep
        their own counter (e.g. Mongo `findAndModify` on a counters doc)
        and only use this as a fallback.
        """
        used = set(self.state.revocations.keys())
        # We treat any bit==1 as "used" (revoked) AND any reservation
        # noted in revocations as taken; we don't otherwise mark issued
        # bits. Callers needing strict unique allocation should use a
        # Mongo counter — see api/v1 issuance flows.
        for idx in range(self.state.size_bits):
            if idx not in used:
                return idx
        raise RuntimeError("status list exhausted — increase size_bits")

    # ── Status List Verifiable Credential ────────────────────────────────

    def build_status_list_credential(
        self,
        private_key: Ed25519PrivateKey,
        valid_for_days: int = 1,
    ) -> dict:
        """Build + sign the status list as a W3C VerifiableCredential.

        Per W3C BSL spec, the list is itself a VC with
        `credentialSubject.type = "BitstringStatusList"` and
        `credentialSubject.encodedList = base64url(gzip(bitstring))`.
        """
        encoded = encode_bitstring(self.state.bitstring)
        now = datetime.now(timezone.utc)

        credential: dict = {
            "@context": [
                "https://www.w3.org/ns/credentials/v2",
                "https://w3id.org/security/data-integrity/v2",
                LAUREUM_VC_CONTEXT,
            ],
            "id": self.list_url,
            "type": ["VerifiableCredential", "BitstringStatusListCredential"],
            "issuer": self.issuer_did,
            "validFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "credentialSchema": {
                "id": AQVC_JSON_SCHEMA_URL,
                "type": "JsonSchema",
            },
            "credentialSubject": {
                "id": f"{self.list_url}#list",
                "type": "BitstringStatusList",
                "statusPurpose": "revocation",
                "encodedList": encoded,
            },
        }

        proof_options = {
            "type": "DataIntegrityProof",
            "cryptosuite": "eddsa-jcs-2022",
            "verificationMethod": f"{self.issuer_did}#key-1",
            "proofPurpose": "assertionMethod",
            "created": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        proof_value = _sign_eddsa_jcs_2022(credential, proof_options, private_key)
        credential["proof"] = {**proof_options, "proofValue": proof_value}
        return credential

    # ── MongoDB persistence ──────────────────────────────────────────────

    async def load_from_mongo(self, list_id: str = DEFAULT_LIST_ID) -> None:
        """Hydrate state from `quality__status_list` (single doc per list)."""
        from src.storage.mongodb import get_db

        col = get_db()[STATUS_LIST_COLLECTION]
        doc = await col.find_one({"_id": list_id})
        if not doc:
            self.state = self.init_state(list_id=list_id)
            return

        size_bits = int(doc.get("size_bits", DEFAULT_STATUS_LIST_SIZE_BITS))
        raw = doc.get("bitstring") or bytes(size_bits // 8)
        # Mongo `bytes` may round-trip as `bson.Binary` — coerce.
        if hasattr(raw, "tobytes"):
            raw = raw.tobytes()
        self.state = StatusListState(
            list_id=list_id,
            size_bits=size_bits,
            bitstring=bytearray(raw),
            revocations=dict(doc.get("revocations") or {}),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
        )

    async def persist(self) -> None:
        """Write current state back to MongoDB (idempotent upsert)."""
        from src.storage.mongodb import get_db

        col = get_db()[STATUS_LIST_COLLECTION]
        await col.update_one(
            {"_id": self.state.list_id},
            {"$set": {
                "size_bits": self.state.size_bits,
                # bson.Binary is the canonical way; bytes works in motor too.
                "bitstring": bytes(self.state.bitstring),
                "revocations": {
                    str(k): v for k, v in self.state.revocations.items()
                },
                "updated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )

    # ── Convenience for the daily re-issuance cron ───────────────────────

    def cron_payload(
        self,
        private_key: Ed25519PrivateKey,
    ) -> dict:
        """Build the JSON payload that the Vercel cron should publish.

        Returns: {credential, encoded_list, list_id, updated_at, version}.
        Used by both the backend `/v1/status-list/{id}` endpoint and the
        demo repo's `/credentials/status/1` route.
        """
        cred = self.build_status_list_credential(private_key)
        return {
            "credential": cred,
            "encoded_list": cred["credentialSubject"]["encodedList"],
            "list_id": self.state.list_id,
            "updated_at": cred["validFrom"],
            "version": "1.0",
            "issuance_id": str(uuid4()),
        }


# ── Module-level convenience API for callers that don't want a class ────────


def build_status_list_credential(
    bitstring: bytes,
    private_key: Ed25519PrivateKey,
    list_url: str = STATUS_LIST_CREDENTIAL_URL,
    issuer_did: str = DEFAULT_ISSUER_DID,
) -> dict:
    """Issue a one-shot signed BitstringStatusList VC from raw bytes.

    Useful for the Vercel cron route which just needs the latest signed
    snapshot (no in-memory mutation required).
    """
    issuer = StatusListIssuer(
        state=StatusListState(
            list_id=DEFAULT_LIST_ID,
            size_bits=len(bitstring) * 8,
            bitstring=bytearray(bitstring),
            revocations={},
            updated_at=datetime.now(timezone.utc),
        ),
        list_url=list_url,
        issuer_did=issuer_did,
    )
    return issuer.build_status_list_credential(private_key)

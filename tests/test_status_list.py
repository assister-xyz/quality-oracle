"""Tests for Bitstring Status List issuer (QO-053-I §AC4 / AC11)."""
import gzip
from datetime import datetime, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.standards.status_list import (
    DEFAULT_STATUS_LIST_SIZE_BITS,
    StatusListIssuer,
    StatusListState,
    build_status_list_credential,
    decode_bitstring,
    encode_bitstring,
    get_bit,
    init_bitstring,
    set_bit,
)
from src.standards.aqvc import verify_vc


@pytest.fixture()
def issuer_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


# ── Issue → Revoke → Re-sign cycle ──────────────────────────────────────────


def test_issue_empty_list(issuer_key):
    issuer = StatusListIssuer()
    cred = issuer.build_status_list_credential(issuer_key)
    assert cred["id"] == "https://laureum.ai/credentials/status/1"
    assert "BitstringStatusListCredential" in cred["type"]
    # All bits clear → entire encodedList decompresses to zeros
    raw = decode_bitstring(cred["credentialSubject"]["encodedList"])
    assert all(b == 0 for b in raw)


def test_revoke_then_reissue_produces_new_signature(issuer_key):
    issuer = StatusListIssuer()
    cred1 = issuer.build_status_list_credential(issuer_key)
    issuer.revoke(13, reason="test-rev")
    cred2 = issuer.build_status_list_credential(issuer_key)

    # encodedLists differ (bit 13 is now set)
    assert cred1["credentialSubject"]["encodedList"] != cred2["credentialSubject"]["encodedList"]

    # Both VCs verify under the same key
    ok1, err1 = verify_vc(cred1, issuer_key.public_key())
    assert ok1, err1
    ok2, err2 = verify_vc(cred2, issuer_key.public_key())
    assert ok2, err2

    # The revoked bit is 1 in cred2
    raw = decode_bitstring(cred2["credentialSubject"]["encodedList"])
    assert get_bit(raw, 13) == 1


def test_full_revocation_cycle(issuer_key):
    """AC4 — issue → revoke(idx, reason='key-compromise') → bit flips → verify."""
    issuer = StatusListIssuer()
    idx = 94567
    assert not issuer.is_revoked(idx)
    issuer.revoke(idx, reason="key-compromise")
    assert issuer.is_revoked(idx)

    cred = issuer.build_status_list_credential(issuer_key)
    raw = decode_bitstring(cred["credentialSubject"]["encodedList"])
    assert get_bit(raw, idx) == 1


# ── Bitstring primitives ────────────────────────────────────────────────────


def test_set_bit_msb_first():
    """Per W3C BSL: bit 0 is the MSB of byte 0."""
    bs = init_bitstring(8)
    set_bit(bs, 0, 1)
    assert bs[0] == 0b10000000
    set_bit(bs, 7, 1)
    assert bs[0] == 0b10000001


def test_set_bit_out_of_range():
    bs = init_bitstring(8)
    with pytest.raises(IndexError):
        set_bit(bs, 8, 1)
    with pytest.raises(IndexError):
        set_bit(bs, -1, 1)


def test_set_bit_invalid_value():
    bs = init_bitstring(8)
    with pytest.raises(ValueError):
        set_bit(bs, 0, 2)


def test_encode_decode_bitstring_roundtrip():
    bs = init_bitstring(1024)
    set_bit(bs, 1)
    set_bit(bs, 100)
    set_bit(bs, 1023)
    encoded = encode_bitstring(bs)
    decoded = decode_bitstring(encoded)
    assert bytes(bs) == decoded


def test_encoded_list_is_gzipped():
    bs = init_bitstring(1024)
    encoded = encode_bitstring(bs)
    # Decode base64url manually to confirm it's gzipped
    import base64
    pad = "=" * (-len(encoded) % 4)
    raw = base64.urlsafe_b64decode(encoded + pad)
    # gzip magic bytes
    assert raw[:2] == b"\x1f\x8b"
    # And it decompresses to the original
    assert gzip.decompress(raw) == bytes(bs)


# ── Convenience module API ─────────────────────────────────────────────────


def test_build_status_list_credential_one_shot(issuer_key):
    bs = init_bitstring(1024)
    set_bit(bs, 5)
    cred = build_status_list_credential(bytes(bs), issuer_key)
    assert "BitstringStatusListCredential" in cred["type"]
    ok, err = verify_vc(cred, issuer_key.public_key())
    assert ok, err
    raw = decode_bitstring(cred["credentialSubject"]["encodedList"])
    assert get_bit(raw, 5) == 1


# ── Cron payload (AC11 — used by Vercel Cron in quality-oracle-demo) ───────


def test_cron_payload_shape(issuer_key):
    issuer = StatusListIssuer()
    issuer.revoke(1)
    payload = issuer.cron_payload(issuer_key)
    assert payload["list_id"] == "1"
    assert payload["version"] == "1.0"
    assert payload["credential"]["issuer"] == "did:web:laureum.ai"
    assert payload["encoded_list"]
    assert payload["issuance_id"]


# ── State ─────────────────────────────────────────────────────────────────


def test_init_state_default_size():
    issuer = StatusListIssuer()
    assert issuer.state.size_bits == DEFAULT_STATUS_LIST_SIZE_BITS
    assert len(issuer.state.bitstring) == DEFAULT_STATUS_LIST_SIZE_BITS // 8


def test_explicit_state(issuer_key):
    state = StatusListState(
        list_id="1",
        size_bits=512,
        bitstring=init_bitstring(512),
        revocations={},
        updated_at=datetime.now(timezone.utc),
    )
    issuer = StatusListIssuer(state=state)
    assert issuer.state.size_bits == 512

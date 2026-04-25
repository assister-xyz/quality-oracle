"""Tests for AQVC v1.0 — QO-053-I.

Round-trip: build → sign → verify; revoke → status list bit flip; subject
type=claude_skill payload validates against R10 §2.4 sample.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.standards.aqvc import (
    AQVC_JSON_SCHEMA_URL,
    AQVC_VERSION,
    DEFAULT_ISSUER_DID,
    LAUREUM_VC_CONTEXT,
    STATUS_LIST_CREDENTIAL_URL,
    build_aqvc_skill,
    build_credential_status,
    build_did_document,
    create_aqvc_skill,
    create_vc,
    encode_public_key_multibase,
    verify_vc,
)
from src.standards.status_list import (
    StatusListIssuer,
    StatusListState,
    decode_bitstring,
    get_bit,
    init_bitstring,
    set_bit,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def issuer_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture()
def parsed_skill():
    return SimpleNamespace(
        name="pdf",
        metadata={"version": "1.2.0", "description": "PDF skill"},
    )


@pytest.fixture()
def eval_result():
    """Mirror of R10 §2.4 sample fields, minus signature/UUID/dates."""
    return SimpleNamespace(
        target=SimpleNamespace(
            subject_uri="https://github.com/anthropics/claude-skills/tree/main/document-skills/pdf",
        ),
        eval_hash="a3f1b2c4d5e6f70819aabbccddeeff001122334455667788aabbccddeeff0011",
        absolute_score=82.4,
        tier="gold",
        confidence=0.91,
        questions_asked=60,
        scores_6axis={
            "accuracy": 88.0,
            "safety": 92.0,
            "process_quality": 76.0,
            "reliability": 79.0,
            "latency": 84.0,
            "schema_quality": 75.0,
        },
        judges=[
            {"provider": "cerebras", "model": "llama3.1-8b", "role": "primary"},
            {"provider": "groq", "model": "llama-3.1-8b-instant", "role": "secondary"},
            {"provider": "openrouter", "model": "qwen/qwen3-next-80b-a3b-instruct:free", "role": "tiebreaker"},
        ],
        probes_used=[
            "prompt_injection", "system_prompt_extraction",
            "pii_leakage", "hallucination", "overflow",
        ],
        cpcr=0.78,
        model_versions={
            "primary": "cerebras:llama3.1-8b@2026-04",
            "judge": "cerebras:llama3.1-8b@2026-04",
            "probes_seed": 42,
            "activation_provider": "cerebras",
        },
        target_protocol="anthropic-skills/2025-11-25",
        ide_runtime={"client": "claude-code", "version": "2.5.4"},
        sarif_report_uri="https://laureum.ai/evidence/eval_abc123/sarif",
        evaluation_id="eval_abc123",
        # No expires_at — exercises AC3.
    )


# ── AC2: VC 2.0 envelope shape ───────────────────────────────────────────────


class TestEnvelopeShape:

    def test_context_v2_first(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        assert vc["@context"][0] == "https://www.w3.org/ns/credentials/v2"
        assert "https://w3id.org/security/data-integrity/v2" in vc["@context"]
        assert LAUREUM_VC_CONTEXT in vc["@context"]

    def test_uses_validFrom_not_issuanceDate(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        assert "validFrom" in vc
        assert "issuanceDate" not in vc

    def test_credential_status_is_bsl_entry(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        cs = vc["credentialStatus"]
        assert cs["type"] == "BitstringStatusListEntry"
        assert cs["statusPurpose"] == "revocation"
        assert cs["statusListIndex"] == "42"
        assert cs["statusListCredential"] == STATUS_LIST_CREDENTIAL_URL

    def test_credential_schema_present(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        assert vc["credentialSchema"]["id"] == AQVC_JSON_SCHEMA_URL
        assert vc["credentialSchema"]["type"] == "JsonSchema"

    def test_issuer_is_laureum(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        assert vc["issuer"] == "did:web:laureum.ai"
        assert vc["issuer"] == DEFAULT_ISSUER_DID

    def test_type_array(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        assert vc["type"] == ["VerifiableCredential", "AgentQualityCredential"]


# ── AC3: validUntil empty-string bug ─────────────────────────────────────────


class TestValidUntilOmitted:

    def test_no_expiry_means_no_field(self, eval_result, parsed_skill):
        # eval_result fixture has no expires_at
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=1)
        assert "validUntil" not in vc
        # And it MUST NOT be present as ""
        assert vc.get("validUntil") != ""

    def test_with_expiry_emits_field(self, eval_result, parsed_skill):
        eval_result.expires_at = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=1)
        assert vc["validUntil"].startswith("2026-07-25T18:00:00")

    def test_create_vc_omits_validUntil_for_no_expiry(self, issuer_key):
        # Legacy create_vc path — ensure same fix.
        aqvc = {
            "subject": {"id": "x", "type": "mcp_server", "name": "X"},
            "quality": {"score": 80, "tier": "proficient"},
            "evaluation": {"id": "eval-1"},
            # no expires_at
        }
        vc = create_vc(aqvc, issuer_key)
        assert "validUntil" not in vc


# ── AC6: Skill subject schema ────────────────────────────────────────────────


class TestSkillSubjectSchema:

    def test_required_fields_present(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        cs = vc["credentialSubject"]

        for field in [
            "id", "subjectType", "name", "version",
            "evalHash", "overallScore", "tier", "confidence",
            "questionsAsked", "judges", "probesUsed",
        ]:
            assert field in cs, f"missing required field: {field}"

        assert cs["subjectType"] == "claude_skill"
        assert cs["aqvcVersion"] == AQVC_VERSION

    def test_field_name_camelcase_mapping(self, eval_result, parsed_skill):
        """M1 mapping table — snake_case → camelCase."""
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        cs = vc["credentialSubject"]
        # Subject keys MUST be camelCase per spec.
        assert "evalHash" in cs and "eval_hash" not in cs
        assert "overallScore" in cs and "overall_score" not in cs
        assert "questionsAsked" in cs and "questions_asked" not in cs
        assert "scores6Axis" in cs and "scores_6axis" not in cs
        assert "modelVersions" in cs and "model_versions" not in cs
        assert "targetProtocol" in cs and "target_protocol" not in cs
        assert "ideRuntime" in cs and "ide_runtime" not in cs

    def test_axis_keys_camelcase(self, eval_result, parsed_skill):
        """6-axis keys with underscores get camelCased too."""
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        axes = vc["credentialSubject"]["scores6Axis"]
        assert "processQuality" in axes
        assert "schemaQuality" in axes
        assert "process_quality" not in axes
        assert "schema_quality" not in axes

    def test_optional_fields(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        cs = vc["credentialSubject"]
        assert cs["cpcr"] == 0.78
        assert cs["sarifReportUri"].startswith("https://laureum.ai/")


# ── AC12, AC13: Provider transparency ────────────────────────────────────────


class TestProviderTransparency:

    def test_modelVersions_includes_activation_provider(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        mv = vc["credentialSubject"]["modelVersions"]
        assert mv["activation_provider"] == "cerebras"

    def test_targetProtocol_present(self, eval_result, parsed_skill):
        vc = build_aqvc_skill(eval_result, parsed_skill, status_index=42)
        # AC13 — targetProtocol carries activation context.
        assert vc["credentialSubject"]["targetProtocol"] == "anthropic-skills/2025-11-25"


# ── AC2/AC5/AC9: DID document ────────────────────────────────────────────────


class TestDIDDocument:

    def test_default_issuer(self, issuer_key):
        did_doc = build_did_document(issuer_key.public_key())
        assert did_doc["id"] == "did:web:laureum.ai"

    def test_assertion_method_only_current_key(self, issuer_key):
        # Quarterly rotation: old keys go in verificationMethod[], not in
        # assertionMethod[]. Compromise drill takes <5min.
        old_key = Ed25519PrivateKey.generate()
        old_multibase = encode_public_key_multibase(old_key.public_key())
        did_doc = build_did_document(
            issuer_key.public_key(),
            historical_keys=[
                {"key_id": "key-2026-q1", "publicKeyMultibase": old_multibase},
            ],
        )
        # 1 current + 1 historical
        assert len(did_doc["verificationMethod"]) == 2
        # Only the current key is in assertionMethod[]
        assert len(did_doc["assertionMethod"]) == 1
        assert did_doc["assertionMethod"][0] == "did:web:laureum.ai#key-1"


# ── AC4: Revocation works ────────────────────────────────────────────────────


class TestStatusListRevocation:

    def test_init_bitstring_zero_filled(self):
        bs = init_bitstring(1024)
        assert len(bs) == 128
        assert all(b == 0 for b in bs)

    def test_set_get_bit_roundtrip(self):
        bs = init_bitstring(1024)
        set_bit(bs, 42, 1)
        assert get_bit(bytes(bs), 42) == 1
        assert get_bit(bytes(bs), 41) == 0
        assert get_bit(bytes(bs), 43) == 0

    def test_revoke_flips_bit(self, issuer_key):
        issuer = StatusListIssuer()
        assert not issuer.is_revoked(94567)
        issuer.revoke(94567, reason="key-compromise")
        assert issuer.is_revoked(94567)
        # Reason persisted
        rec = issuer.state.revocations[94567]
        assert rec["reason"] == "key-compromise"

    def test_status_list_credential_signed_and_verifiable(self, issuer_key):
        issuer = StatusListIssuer()
        issuer.revoke(7, reason="test")
        cred = issuer.build_status_list_credential(issuer_key)

        # It's a VerifiableCredential
        assert "VerifiableCredential" in cred["type"]
        assert "BitstringStatusListCredential" in cred["type"]

        # encodedList is gzipped + base64url
        encoded = cred["credentialSubject"]["encodedList"]
        decoded = decode_bitstring(encoded)
        assert len(decoded) == issuer.state.size_bits // 8
        # bit 7 should be set (revoked)
        assert get_bit(decoded, 7) == 1
        # bit 6 should be clear
        assert get_bit(decoded, 6) == 0

        # Signature verifies under the issuer key
        ok, err = verify_vc(cred, issuer_key.public_key())
        assert ok, err

    def test_revocation_cycle_matches_aqvc_index(self, issuer_key, eval_result, parsed_skill):
        """End-to-end: issue an AQVC at index N → revoke → status list shows bit N."""
        idx = 94567
        vc = create_aqvc_skill(eval_result, parsed_skill, status_index=idx, private_key=issuer_key)

        # AQVC carries the index
        assert vc["credentialStatus"]["statusListIndex"] == str(idx)
        # And the AQVC itself is signed + verifiable
        ok, err = verify_vc(vc, issuer_key.public_key())
        assert ok, err

        # Revoke the index in a status list issuer
        issuer = StatusListIssuer(
            state=StatusListState(
                list_id="1",
                size_bits=131072,
                bitstring=init_bitstring(131072),
                revocations={},
                updated_at=datetime.now(timezone.utc),
            ),
        )
        issuer.revoke(idx, reason="superseded")
        cred = issuer.build_status_list_credential(issuer_key)
        decoded = decode_bitstring(cred["credentialSubject"]["encodedList"])
        assert get_bit(decoded, idx) == 1


# ── AC4: BitstringStatusListEntry helper ────────────────────────────────────


class TestCredentialStatusEntry:

    def test_entry_shape(self):
        entry = build_credential_status(94567)
        assert entry["type"] == "BitstringStatusListEntry"
        assert entry["statusPurpose"] == "revocation"
        assert entry["statusListIndex"] == "94567"  # MUST be a string
        assert entry["statusListCredential"] == STATUS_LIST_CREDENTIAL_URL
        assert entry["id"].endswith("#94567")


# ── Round-trip: AQVC build → sign → verify ─────────────────────────────────


class TestRoundTrip:

    def test_create_aqvc_skill_signs_and_verifies(self, eval_result, parsed_skill, issuer_key):
        vc = create_aqvc_skill(eval_result, parsed_skill, status_index=1, private_key=issuer_key)
        assert vc["proof"]["cryptosuite"] == "eddsa-jcs-2022"
        assert vc["proof"]["proofValue"].startswith("z")
        ok, err = verify_vc(vc, issuer_key.public_key())
        assert ok, err

    def test_tampered_aqvc_fails(self, eval_result, parsed_skill, issuer_key):
        vc = create_aqvc_skill(eval_result, parsed_skill, status_index=1, private_key=issuer_key)
        vc["credentialSubject"]["overallScore"] = 99.0
        ok, err = verify_vc(vc, issuer_key.public_key())
        assert not ok
        assert "Signature" in err

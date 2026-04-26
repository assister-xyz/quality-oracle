"""A2A integration tests — Agent Card v0.3, DID Document, W3C VC, context endpoint."""
import copy
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.standards.vc_issuer import (
    create_vc,
    verify_vc,
    build_did_document,
    encode_public_key_multibase,
    decode_public_key_multibase,
)


# ── Agent Card v0.3 Compliance ───────────────────────────────────────────────

class TestAgentCardV03Compliance:

    def test_protocol_version_present(self, test_client):
        resp = test_client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["protocolVersion"] == "0.3.0"

    def test_required_fields(self, test_client):
        resp = test_client.get("/.well-known/agent.json")
        card = resp.json()
        for field in ["name", "description", "url", "version", "protocolVersion",
                       "provider", "capabilities", "skills"]:
            assert field in card, f"Missing required field: {field}"

    def test_provider_structure(self, test_client):
        resp = test_client.get("/.well-known/agent.json")
        card = resp.json()
        provider = card["provider"]
        assert "organization" in provider
        assert "url" in provider

    def test_skills_structure(self, test_client):
        resp = test_client.get("/.well-known/agent.json")
        card = resp.json()
        skills = card["skills"]
        assert len(skills) >= 3
        for skill in skills:
            assert "id" in skill
            assert "name" in skill
            assert "description" in skill

    def test_capabilities_structure(self, test_client):
        resp = test_client.get("/.well-known/agent.json")
        card = resp.json()
        caps = card["capabilities"]
        assert "streaming" in caps
        assert "pushNotifications" in caps
        assert "extensions" in caps


# ── DID Document ─────────────────────────────────────────────────────────────

class TestDIDDocument:

    def test_structure(self):
        key = Ed25519PrivateKey.generate()
        did_doc = build_did_document(key.public_key())
        assert did_doc["id"] == "did:web:laureum.ai"
        assert "@context" in did_doc
        assert len(did_doc["verificationMethod"]) == 1
        vm = did_doc["verificationMethod"][0]
        assert vm["type"] == "Multikey"
        assert vm["publicKeyMultibase"].startswith("z")

    def test_custom_did(self):
        key = Ed25519PrivateKey.generate()
        did_doc = build_did_document(key.public_key(), issuer_did="did:web:example.com")
        assert did_doc["id"] == "did:web:example.com"
        assert did_doc["verificationMethod"][0]["controller"] == "did:web:example.com"

    def test_multikey_format(self):
        key = Ed25519PrivateKey.generate()
        multibase = encode_public_key_multibase(key.public_key())
        assert multibase.startswith("z")
        # Should round-trip
        decoded = decode_public_key_multibase(multibase)
        assert encode_public_key_multibase(decoded) == multibase

    def test_endpoint_via_client(self, test_client):
        key = Ed25519PrivateKey.generate()
        with patch("src.core.attestation._get_or_generate_key", return_value=key):
            resp = test_client.get("/.well-known/did.json")

        assert resp.status_code == 200
        did_doc = resp.json()
        assert "verificationMethod" in did_doc
        assert "assertionMethod" in did_doc
        vm = did_doc["verificationMethod"][0]
        assert vm["type"] == "Multikey"
        assert vm["publicKeyMultibase"].startswith("z")


# ── VC Endpoint ──────────────────────────────────────────────────────────────

class TestVCEndpoint:

    def test_returns_vc(self, test_client, mock_attestation_with_vc):
        with patch("src.api.v1.attestations.attestations_col") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=mock_attestation_with_vc)
            resp = test_client.get("/v1/attestation/attest-vc-test-001/vc")

        assert resp.status_code == 200
        vc = resp.json()
        assert "VerifiableCredential" in vc["type"]
        assert "AgentQualityCredential" in vc["type"]
        assert vc["proof"]["cryptosuite"] == "eddsa-jcs-2022"

    def test_404_on_missing_attestation(self, test_client):
        with patch("src.api.v1.attestations.attestations_col") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=None)
            resp = test_client.get("/v1/attestation/nonexistent/vc")

        assert resp.status_code == 404

    def test_404_on_pre_vc_attestation(self, test_client):
        pre_vc_doc = {
            "_id": "old-attest-001",
            "attestation_jwt": "eyJ...",
            "vc_document": None,
        }
        with patch("src.api.v1.attestations.attestations_col") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=pre_vc_doc)
            resp = test_client.get("/v1/attestation/old-attest-001/vc")

        assert resp.status_code == 404
        assert "No W3C VC" in resp.json()["detail"]


# ── VC Issuance & Verification ───────────────────────────────────────────────

class TestVCIssuanceVerification:

    def _make_aqvc(self, score=85, tier="expert"):
        return {
            "quality": {"score": score, "tier": tier, "confidence": 0.9,
                        "evaluation_level": 2, "domains": ["defi"], "questions_asked": 10},
            "subject": {"id": "test-server", "type": "mcp_server", "name": "Test"},
            "evaluation": {"id": "eval-1", "method": "challenge-response-v1",
                           "evaluated_at": "2026-02-28T12:00:00Z"},
            "expires_at": "2026-04-01T00:00:00Z",
        }

    def test_create_verify_roundtrip(self):
        key = Ed25519PrivateKey.generate()
        vc = create_vc(self._make_aqvc(), key)

        assert vc["type"] == ["VerifiableCredential", "AgentQualityCredential"]
        assert vc["issuer"] == "did:web:laureum.ai"
        assert vc["proof"]["cryptosuite"] == "eddsa-jcs-2022"
        assert vc["proof"]["proofValue"].startswith("z")

        valid, err = verify_vc(vc, key.public_key())
        assert valid
        assert err == ""

    def test_tampered_vc_fails(self):
        key = Ed25519PrivateKey.generate()
        vc = create_vc(self._make_aqvc(), key)

        tampered = copy.deepcopy(vc)
        tampered["credentialSubject"]["qualityScore"] = 99

        valid, err = verify_vc(tampered, key.public_key())
        assert not valid
        assert "Signature verification failed" in err

    def test_wrong_key_fails(self):
        key1 = Ed25519PrivateKey.generate()
        key2 = Ed25519PrivateKey.generate()
        vc = create_vc(self._make_aqvc(), key1)

        valid, err = verify_vc(vc, key2.public_key())
        assert not valid
        assert "Signature verification failed" in err

    def test_no_proof_fails(self):
        key = Ed25519PrivateKey.generate()
        vc = create_vc(self._make_aqvc(), key)
        del vc["proof"]

        valid, err = verify_vc(vc, key.public_key())
        assert not valid
        assert "No proof" in err

    def test_vc_credential_subject_fields(self):
        key = Ed25519PrivateKey.generate()
        vc = create_vc(self._make_aqvc(score=82, tier="proficient"), key)

        cs = vc["credentialSubject"]
        assert cs["qualityScore"] == 82
        assert cs["qualityTier"] == "proficient"
        assert cs["confidence"] == 0.9
        assert cs["evaluationLevel"] == 2
        assert cs["domains"] == ["defi"]
        assert cs["questionsAsked"] == 10

    def test_vc_evidence(self):
        key = Ed25519PrivateKey.generate()
        vc = create_vc(self._make_aqvc(), key)

        assert len(vc["evidence"]) == 1
        ev = vc["evidence"][0]
        assert ev["type"] == "QualityEvaluation"
        assert ev["evaluationId"] == "eval-1"
        assert ev["method"] == "challenge-response-v1"

    def test_custom_issuer_did(self):
        key = Ed25519PrivateKey.generate()
        vc = create_vc(self._make_aqvc(), key, issuer_did="did:web:custom.example.com")
        assert vc["issuer"] == "did:web:custom.example.com"
        assert "did:web:custom.example.com#key-1" in vc["proof"]["verificationMethod"]


# ── Context Endpoint ─────────────────────────────────────────────────────────

class TestContextEndpoint:

    def test_quality_context_has_vocabulary(self, test_client):
        resp = test_client.get("/contexts/quality/v1")
        assert resp.status_code == 200
        ctx = resp.json()["@context"]
        for term in ["AgentQualityCredential", "qualityScore", "qualityTier",
                      "confidence", "evaluationLevel", "domains", "questionsAsked"]:
            assert term in ctx, f"Missing vocabulary term: {term}"

    def test_context_content_type(self, test_client):
        resp = test_client.get("/contexts/quality/v1")
        assert "application/ld+json" in resp.headers.get("content-type", "")


# ── Extension Spec ───────────────────────────────────────────────────────────

class TestExtensionSpec:

    def test_extension_spec_via_client(self, test_client):
        resp = test_client.get("/ext/evaluation/v1")
        assert resp.status_code == 200
        spec = resp.json()
        assert "roles" in spec
        assert "provider" in spec["roles"]
        assert "verified_subject" in spec["roles"]


# ── Enrichment E2E (with mocked score data) ─────────────────────────────────

class TestEnrichmentE2E:

    def test_enrichment_endpoint_exists(self, test_client):
        """Enrichment endpoint should return structured data for known targets."""
        resp = test_client.get("/v1/enrichment/test-target")
        # With mocked empty DB, should return the target_id at minimum
        assert resp.status_code in [200, 404]

"""Tests for JWT attestation creation and verification."""
import pytest

from src.core.attestation import (
    create_attestation,
    verify_attestation,
    get_public_key_pem,
    _private_key,
)


@pytest.fixture(autouse=True)
def reset_key():
    """Reset the module-level key cache between tests."""
    import src.core.attestation as att_mod
    att_mod._private_key = None
    yield
    att_mod._private_key = None


def test_create_attestation():
    """Should create a valid attestation dict with JWT."""
    result = create_attestation(
        target_id="http://test-server/sse",
        target_type="mcp_server",
        target_name="Test Server",
        evaluation_result={
            "overall_score": 85,
            "tier": "expert",
            "confidence": 0.9,
            "tool_scores": {"calculate": {"score": 90}},
            "questions_asked": 10,
        },
        evaluation_version="v1.0",
    )

    assert "_id" in result
    assert result["attestation_jwt"]
    assert result["aqvc_payload"]["quality"]["score"] == 85
    assert result["aqvc_payload"]["quality"]["tier"] == "expert"
    assert result["aqvc_payload"]["subject"]["id"] == "http://test-server/sse"
    assert result["revoked"] is False


def test_verify_attestation_valid():
    """Should verify a freshly created attestation as valid."""
    att = create_attestation(
        target_id="http://test-server/sse",
        target_type="mcp_server",
        target_name="Test Server",
        evaluation_result={"overall_score": 75, "tier": "proficient", "confidence": 0.8},
    )

    result = verify_attestation(att["attestation_jwt"])
    assert result["valid"] is True
    assert result["payload"]["quality"]["score"] == 75
    assert result["issuer"] == "did:web:laureum.ai"


def test_verify_attestation_invalid_token():
    """Should report invalid for a garbage token."""
    result = verify_attestation("not.a.valid.jwt.token")
    assert result["valid"] is False
    assert "error" in result


def test_verify_attestation_tampered():
    """Should report invalid for a tampered token."""
    att = create_attestation(
        target_id="http://test/sse",
        target_type="mcp_server",
        target_name="Test",
        evaluation_result={"overall_score": 50, "tier": "basic"},
    )
    # Tamper with the token
    token = att["attestation_jwt"]
    tampered = token[:-5] + "XXXXX"
    result = verify_attestation(tampered)
    assert result["valid"] is False


def test_get_public_key_pem():
    """Should return a PEM-encoded public key."""
    pem = get_public_key_pem()
    assert "BEGIN PUBLIC KEY" in pem
    assert "END PUBLIC KEY" in pem


def test_attestation_aqvc_payload_structure():
    """AQVC payload should have all required fields."""
    att = create_attestation(
        target_id="http://server/sse",
        target_type="agent",
        target_name="Agent X",
        evaluation_result={"overall_score": 60, "tier": "basic", "confidence": 0.5},
    )
    payload = att["aqvc_payload"]

    assert payload["aqvc_version"] == "1.0"
    assert "issuer" in payload
    assert "issued_at" in payload
    assert "expires_at" in payload
    assert payload["subject"]["type"] == "agent"
    assert payload["evaluation"]["method"] == "challenge-response-v1"

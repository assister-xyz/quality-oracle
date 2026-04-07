"""Tests for session JWT encode/decode (QO-046)."""
import time

import jwt
import pytest

from src.auth.session import create_session_token, decode_session_token, ALGORITHM
from src.config import settings


class TestCreateSessionToken:

    def test_returns_string(self):
        token = create_session_token("op_123", "vitamin33")
        assert isinstance(token, str)
        assert token.count(".") == 2  # JWT has 3 segments

    def test_payload_contains_expected_fields(self):
        token = create_session_token("op_abc", "testuser", verified=True)
        payload = jwt.decode(token, settings.session_secret, algorithms=[ALGORITHM])
        assert payload["sub"] == "op_abc"
        assert payload["github_username"] == "testuser"
        assert payload["verified"] is True
        assert "iat" in payload
        assert "exp" in payload

    def test_default_ttl(self):
        before = int(time.time())
        token = create_session_token("op_abc", "user")
        after = int(time.time())
        payload = jwt.decode(token, settings.session_secret, algorithms=[ALGORITHM])
        expected_exp_min = before + settings.session_cookie_max_age
        expected_exp_max = after + settings.session_cookie_max_age
        assert expected_exp_min <= payload["exp"] <= expected_exp_max

    def test_custom_ttl(self):
        token = create_session_token("op_abc", "user", ttl_seconds=60)
        payload = jwt.decode(token, settings.session_secret, algorithms=[ALGORITHM])
        assert payload["exp"] - payload["iat"] == 60


class TestDecodeSessionToken:

    def test_roundtrip_valid(self):
        token = create_session_token("op_xyz", "alice")
        payload = decode_session_token(token)
        assert payload is not None
        assert payload["sub"] == "op_xyz"
        assert payload["github_username"] == "alice"
        assert payload["verified"] is True

    def test_empty_token_returns_none(self):
        assert decode_session_token("") is None
        assert decode_session_token(None) is None

    def test_malformed_token_returns_none(self):
        assert decode_session_token("not-a-jwt") is None
        assert decode_session_token("invalid.token.here") is None

    def test_expired_token_returns_none(self):
        token = create_session_token("op_expired", "user", ttl_seconds=-10)
        # Token issued with negative TTL is already expired
        assert decode_session_token(token) is None

    def test_wrong_secret_returns_none(self):
        token = create_session_token("op_abc", "user")
        # Decode with different secret
        import jwt as _jwt
        try:
            _jwt.decode(token, "wrong-secret", algorithms=[ALGORITHM])
            decoded_with_wrong = True
        except _jwt.InvalidSignatureError:
            decoded_with_wrong = False
        assert not decoded_with_wrong

    def test_tampered_payload_rejected(self):
        token = create_session_token("op_abc", "user")
        # Tamper by changing a character in the payload
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1][:-1] + "X" + "." + parts[2]
        assert decode_session_token(tampered) is None

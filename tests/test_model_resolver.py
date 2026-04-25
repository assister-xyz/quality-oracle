"""Tests for QO-053-B model_resolver — provider:model parsing + AC4."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.model_resolver import (
    InvalidActivationModel,
    is_dated,
    parse_provider_model,
    resolve,
    resolve_anthropic,
    resolve_fixed,
)


class TestParseProviderModel:
    def test_cerebras(self):
        provider, model = parse_provider_model("cerebras:llama3.1-8b")
        assert provider == "cerebras"
        assert model == "llama3.1-8b"

    def test_groq(self):
        provider, model = parse_provider_model("groq:llama-3.1-8b-instant")
        assert provider == "groq"
        assert model == "llama-3.1-8b-instant"

    def test_anthropic_bare(self):
        provider, model = parse_provider_model("anthropic:claude-sonnet-4-5")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5"

    def test_anthropic_dated(self):
        provider, model = parse_provider_model("anthropic:claude-sonnet-4-5-20250929")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5-20250929"

    def test_case_insensitive_provider(self):
        provider, _ = parse_provider_model("Cerebras:llama3.1-8b")
        assert provider == "cerebras"

    @pytest.mark.parametrize("bad", [
        "",
        "llama3.1-8b",  # no colon
        "cerebras",  # missing model
        "cerebras:",  # empty model
        "openai:gpt-4o",  # disallowed provider for activation
        "deepseek:deepseek-chat",
    ])
    def test_invalid(self, bad):
        with pytest.raises(InvalidActivationModel):
            parse_provider_model(bad)


class TestIsDated:
    @pytest.mark.parametrize("alias,expected", [
        ("claude-sonnet-4-5-20250929", True),
        ("claude-sonnet-4-5", False),
        ("llama3.1-8b", False),
        ("foo-99999999", True),
    ])
    def test_is_dated(self, alias, expected):
        assert is_dated(alias) is expected


class TestResolveFixed:
    def test_cerebras_passes_through(self):
        r = resolve_fixed("cerebras", "llama3.1-8b")
        assert r.provider == "cerebras"
        assert r.alias == "llama3.1-8b"
        assert r.dated_snapshot == "llama3.1-8b"
        assert r.source == "fixed"

    def test_groq_passes_through(self):
        r = resolve_fixed("groq", "llama-3.1-8b-instant")
        assert r.dated_snapshot == "llama-3.1-8b-instant"


class TestResolveAnthropic:
    def test_already_dated_short_circuits(self):
        r = resolve_anthropic("claude-sonnet-4-5-20250929", api_key="test")
        assert r.dated_snapshot == "claude-sonnet-4-5-20250929"
        assert r.source == "fixed"

    def test_no_api_key_falls_back(self):
        r = resolve_anthropic("claude-sonnet-4-5", api_key=None)
        assert r.dated_snapshot == "claude-sonnet-4-5"
        assert r.source == "fixed"

    def test_list_models_picks_freshest(self):
        fake_models = [
            MagicMock(model_dump=lambda: {"id": "claude-sonnet-4-5-20250101"}),
            MagicMock(model_dump=lambda: {"id": "claude-sonnet-4-5-20250929"}),
            MagicMock(model_dump=lambda: {"id": "claude-haiku-4-5-20250901"}),
        ]
        with patch("src.core.model_resolver._list_anthropic_models", return_value=[
            m.model_dump() for m in fake_models
        ]):
            r = resolve_anthropic("claude-sonnet-4-5", api_key="sk-test")
        assert r.dated_snapshot == "claude-sonnet-4-5-20250929"
        assert r.source == "list_models"

    def test_no_match_falls_back(self):
        with patch("src.core.model_resolver._list_anthropic_models", return_value=[]):
            r = resolve_anthropic("claude-sonnet-4-5", api_key="sk-test")
        assert r.dated_snapshot == "claude-sonnet-4-5"
        assert r.source == "fixed"


class TestResolve:
    def test_default_cerebras(self):
        r = resolve("cerebras:llama3.1-8b")
        assert r.provider == "cerebras"
        assert r.dated_snapshot == "llama3.1-8b"

    def test_anthropic_routes_to_resolver(self):
        with patch("src.core.model_resolver._list_anthropic_models", return_value=[
            {"id": "claude-sonnet-4-5-20250929"},
        ]):
            r = resolve("anthropic:claude-sonnet-4-5", anthropic_api_key="sk-test")
        assert r.provider == "anthropic"
        assert r.dated_snapshot == "claude-sonnet-4-5-20250929"

    def test_groq_fixed(self):
        r = resolve("groq:llama-3.1-8b-instant")
        assert r.provider == "groq"
        assert r.source == "fixed"

"""Tests for QO-053-B L1NaiveActivator (AC1, AC3, AC4, AC7, AC8, AC9, AC10).

All SDK calls are mocked — no live network. Run live integration only with
``pytest -m live``.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.model_resolver import ResolvedModel
from src.core.skill_activator import (
    L1NaiveActivator,
    SkillActivatedAgent,
    strip_bash_preprocessors,
)
from src.storage.models import ActivationFailure, ParsedSkill


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_skill(body: str = "Hello from skill body.", body_tokens: int = 3000) -> ParsedSkill:
    return ParsedSkill(
        name="solana-swap",
        description="Solana SPL token swap helper.",
        body=body,
        body_size_bytes=len(body.encode()),
        body_lines=body.count("\n") + 1,
        body_tokens=body_tokens,
        folder_name="solana-swap",
        folder_name_nfkc="solana-swap",
    )


def _cerebras_resolved() -> ResolvedModel:
    return ResolvedModel(
        provider="cerebras", alias="llama3.1-8b",
        dated_snapshot="llama3.1-8b", source="fixed",
    )


def _anthropic_resolved() -> ResolvedModel:
    return ResolvedModel(
        provider="anthropic", alias="claude-sonnet-4-5",
        dated_snapshot="claude-sonnet-4-5-20250929", source="list_models",
    )


def _fake_oai_response(text: str = "ok", in_tok: int = 100, out_tok: int = 50, rid: str = "req_oai_1"):
    """Build a Cerebras/Groq chat.completions response shape."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.id = rid
    usage = MagicMock()
    usage.prompt_tokens = in_tok
    usage.completion_tokens = out_tok
    resp.usage = usage
    return resp


def _fake_anthropic_response(text: str = "ok", cache_creation: int = 1000,
                             cache_read: int = 0, in_tok: int = 50,
                             out_tok: int = 50, rid: str = "msg_01"):
    """Build an Anthropic messages.create response."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    resp = MagicMock()
    resp.content = [text_block]
    resp.id = rid
    resp.stop_reason = "end_turn"
    usage = MagicMock()
    usage.cache_creation_input_tokens = cache_creation
    usage.cache_read_input_tokens = cache_read
    usage.input_tokens = in_tok
    usage.output_tokens = out_tok
    resp.usage = usage
    return resp


# ── ABC enforcement (CB7) ───────────────────────────────────────────────────


class TestABCEnforcement:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            SkillActivatedAgent(None, _cerebras_resolved(), MagicMock())  # type: ignore[abstract]

    def test_l1_subclass_runtime_check(self):
        assert issubclass(L1NaiveActivator, SkillActivatedAgent)


# ── AC7: Bash-preprocessor stripping ────────────────────────────────────────


class TestBashStripping:
    def test_inline_form(self):
        body, warnings = strip_bash_preprocessors("Run !`gh pr diff` then continue.")
        assert "!`gh pr diff`" not in body
        assert "[bash-preprocessor" in body
        assert any(w.startswith("bash_preprocessor_stripped:") for w in warnings)

    def test_block_form(self):
        body = "before\n```!\ngh pr diff\n```\nafter"
        cleaned, warnings = strip_bash_preprocessors(body)
        assert "```!" not in cleaned
        assert "block: command (disabled" in cleaned
        assert warnings == ["bash_preprocessor_stripped:1"]

    def test_both_forms_counted(self):
        body = "!`a` and ```!\nb\n```"
        _, warnings = strip_bash_preprocessors(body)
        assert warnings == ["bash_preprocessor_stripped:2"]

    def test_no_strip_no_warning(self):
        _, warnings = strip_bash_preprocessors("clean body, no preprocessors")
        assert warnings == []


# ── AC1, AC10: Cerebras default + provider switch ───────────────────────────


@pytest.mark.asyncio
class TestCerebrasL1:
    async def test_default_provider_is_cerebras(self):
        skill = _make_skill()
        client = MagicMock()
        client.chat.completions.create = MagicMock(
            return_value=_fake_oai_response(text="42")
        )
        activator = L1NaiveActivator(skill, _cerebras_resolved(), client)
        resp = await activator.respond("What is 6*7?")
        assert resp.text == "42"
        assert resp.provider == "cerebras"
        assert resp.model == "llama3.1-8b"
        # AC1: cost = $0 on Cerebras free tier
        assert activator.usage_summary().dollars_spent == 0.0
        # AC1: usage tracked
        assert activator.usage_summary().n_calls == 1
        # AC4: dated snapshot recorded
        assert resp.model == "llama3.1-8b"
        # Cerebras path does NOT use cache_control (no caching)
        assert resp.cache_creation_tokens == 0
        assert resp.cache_read_tokens == 0

    async def test_30_calls_cost_zero(self):
        skill = _make_skill()
        client = MagicMock()
        client.chat.completions.create = MagicMock(
            return_value=_fake_oai_response(text="hello", in_tok=200, out_tok=50)
        )
        activator = L1NaiveActivator(skill, _cerebras_resolved(), client)
        for i in range(30):
            await activator.respond(f"Q{i}")
        summary = activator.usage_summary()
        assert summary.n_calls == 30
        assert summary.dollars_spent == 0.0  # Cerebras free tier
        assert summary.input_tokens == 30 * 200
        assert summary.output_tokens == 30 * 50


@pytest.mark.asyncio
class TestAnthropicL1:
    async def test_anthropic_provider_switch_uses_cache_control(self):
        skill = _make_skill(body="x" * 10000, body_tokens=3000)
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_fake_anthropic_response(text="cached!")
        )
        activator = L1NaiveActivator(skill, _anthropic_resolved(), client)
        resp = await activator.respond("anything")
        assert resp.provider == "anthropic"
        assert resp.text == "cached!"
        # AC4: dated snapshot used at call site
        assert resp.model == "claude-sonnet-4-5-20250929"
        # AC3: cache_control was passed
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-5-20250929"
        sysblk = kwargs["system"]
        assert isinstance(sysblk, list) and len(sysblk) == 1
        assert sysblk[0]["type"] == "text"
        assert sysblk[0]["cache_control"] == {"type": "ephemeral"}
        # AC1 cost: cache_creation tokens billed
        assert resp.cache_creation_tokens == 1000

    async def test_30_calls_cache_pattern(self):
        """29 cache reads + 1 cache write, per AC1."""
        skill = _make_skill(body="x" * 10000, body_tokens=3000)
        client = MagicMock()
        # First call writes cache, subsequent reads
        responses = [_fake_anthropic_response(cache_creation=2000, cache_read=0)]
        responses += [
            _fake_anthropic_response(cache_creation=0, cache_read=2000)
            for _ in range(29)
        ]
        client.messages.create = AsyncMock(side_effect=responses)
        activator = L1NaiveActivator(skill, _anthropic_resolved(), client)
        for i in range(30):
            await activator.respond(f"Q{i}")
        summary = activator.usage_summary()
        assert summary.cache_creation_input_tokens == 2000
        assert summary.cache_read_input_tokens == 29 * 2000
        assert summary.n_calls == 30


# ── AC9: cache disabled when body_tokens < 2048 ─────────────────────────────


@pytest.mark.asyncio
class TestCacheDisabledShortBody:
    async def test_short_body_emits_warning_and_omits_cache_control(self):
        skill = _make_skill(body="short body", body_tokens=500)
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=_fake_anthropic_response())
        activator = L1NaiveActivator(skill, _anthropic_resolved(), client)
        resp = await activator.respond("hi")
        # AC9: warning emitted
        assert "cache_disabled_below_min_tokens" in resp.parse_warnings
        # AC9: cache_control omitted from system block
        kwargs = client.messages.create.call_args.kwargs
        assert "cache_control" not in kwargs["system"][0]

    async def test_short_body_on_cerebras_no_warning(self):
        """Cerebras has no caching; the warning is Anthropic-only."""
        skill = _make_skill(body="short", body_tokens=100)
        client = MagicMock()
        client.chat.completions.create = MagicMock(return_value=_fake_oai_response())
        activator = L1NaiveActivator(skill, _cerebras_resolved(), client)
        resp = await activator.respond("hi")
        assert "cache_disabled_below_min_tokens" not in resp.parse_warnings


# ── AC7 path: bash strip applied to skill body before injection ─────────────


@pytest.mark.asyncio
class TestBashStripPreInjection:
    async def test_bash_stripped_warning_in_response(self):
        skill = _make_skill(body="please run !`gh pr diff` and report back")
        client = MagicMock()
        client.chat.completions.create = MagicMock(return_value=_fake_oai_response())
        activator = L1NaiveActivator(skill, _cerebras_resolved(), client)
        resp = await activator.respond("status?")
        assert any(
            w.startswith("bash_preprocessor_stripped:") for w in resp.parse_warnings
        )
        # Verify the system message sent to the SDK does NOT contain the raw form
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "!`gh pr diff`" not in kwargs["messages"][0]["content"]


# ── AC8: 429 backoff + DLQ ──────────────────────────────────────────────────


class _RateLimitError(Exception):
    """Mimic SDK 429 — name pattern triggers _classify_error rate_limit branch."""
    def __init__(self, msg: str = "429 rate limit"):
        super().__init__(msg)


@pytest.mark.asyncio
class TestRetryAndDLQ:
    async def test_429_retries_then_succeeds(self, monkeypatch):
        # Patch backoff to zero for fast tests
        from src.core import skill_activator as mod
        monkeypatch.setattr(mod, "_backoff_seconds", lambda a: 0.0)
        skill = _make_skill()
        # Two 429s then success
        client = MagicMock()
        client.chat.completions.create = MagicMock(side_effect=[
            _RateLimitError(),
            _RateLimitError(),
            _fake_oai_response(text="recovered"),
        ])
        activator = L1NaiveActivator(skill, _cerebras_resolved(), client)
        resp = await activator.respond("hi")
        assert resp.text == "recovered"
        assert client.chat.completions.create.call_count == 3

    async def test_3_consecutive_5xx_raises_activation_failure(self, monkeypatch):
        from src.core import skill_activator as mod
        monkeypatch.setattr(mod, "_backoff_seconds", lambda a: 0.0)
        skill = _make_skill()

        class ServerErr(Exception):
            def __init__(self):
                super().__init__("503 service unavailable internal server error")
                self.request_id = "req_5xx"

        dlq_writes: list[Any] = []

        async def dlq_writer(entry):
            dlq_writes.append(entry)

        client = MagicMock()
        client.chat.completions.create = MagicMock(side_effect=ServerErr())
        activator = L1NaiveActivator(
            skill, _cerebras_resolved(), client, dlq_writer=dlq_writer,
        )
        with pytest.raises(ActivationFailure) as exc_info:
            await activator.respond("hi")
        assert exc_info.value.provider == "cerebras"
        assert "5xx" in str(exc_info.value).lower() or exc_info.value.attempt_count >= 3
        # DLQ entry written
        assert len(dlq_writes) == 1
        assert dlq_writes[0].provider == "cerebras"


# ── AC4: Model alias parsing flow ───────────────────────────────────────────


class TestModelAliasParsing:
    def test_resolved_model_propagates_to_response(self):
        """Verify the dated snapshot is the field used at call time."""
        from src.core.model_resolver import resolve_fixed
        r = resolve_fixed("cerebras", "llama3.1-8b")
        skill = _make_skill()
        activator = L1NaiveActivator(skill, r, MagicMock())
        assert activator.model == "llama3.1-8b"
        assert activator.provider == "cerebras"

    def test_anthropic_dated_snapshot(self):
        from src.core.model_resolver import ResolvedModel
        r = ResolvedModel(
            provider="anthropic",
            alias="claude-sonnet-4-5",
            dated_snapshot="claude-sonnet-4-5-20250929",
            source="list_models",
        )
        activator = L1NaiveActivator(_make_skill(), r, MagicMock())
        # AC4: floating alias never reaches API — dated snapshot lives on `.model`
        assert activator.model == "claude-sonnet-4-5-20250929"


# ── AC10: env-var driven default switching (parse-time check) ──────────────


class TestEnvVarSwitching:
    def test_default_setting_is_cerebras(self):
        from src.config import settings
        # CB1: default activation provider is Cerebras
        assert settings.laureum_activation_model == "cerebras:llama3.1-8b"
        assert settings.laureum_activation_provider == "cerebras"

    def test_overriding_provider_to_anthropic_parses(self):
        from src.core.model_resolver import parse_provider_model
        # Simulating LAUREUM_ACTIVATION_MODEL=anthropic:claude-sonnet-4-5
        provider, model = parse_provider_model("anthropic:claude-sonnet-4-5")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5"


# ── usage_summary + reset ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestUsageSummaryReset:
    async def test_usage_summary_keys(self):
        skill = _make_skill()
        client = MagicMock()
        client.chat.completions.create = MagicMock(return_value=_fake_oai_response())
        activator = L1NaiveActivator(skill, _cerebras_resolved(), client)
        await activator.respond("Q")
        s = activator.usage_summary()
        # AC6: summary fields present
        assert hasattr(s, "cache_creation_input_tokens")
        assert hasattr(s, "cache_read_input_tokens")
        assert hasattr(s, "output_tokens")
        assert hasattr(s, "dollars_spent")
        assert hasattr(s, "model")
        assert hasattr(s, "request_ids")

    async def test_reset_no_op_for_l1(self):
        # L1 is stateless; reset() must not raise
        skill = _make_skill()
        activator = L1NaiveActivator(skill, _cerebras_resolved(), MagicMock())
        activator.reset()


def test_async_run_works():
    """Smoke check the asyncio loop wiring (sanity)."""
    async def go():
        return 42
    assert asyncio.get_event_loop_policy().new_event_loop().run_until_complete(go()) == 42

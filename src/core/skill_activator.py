"""Tiered skill activation adapter (QO-053-B).

Three subclasses of :class:`SkillActivatedAgent`:

L1 (naive injection)
    System-prompt only. Cheapest tier, ~55–65% fidelity to real Claude Code.
    Default provider per CB1 is Cerebras (free tier); Anthropic Sonnet is
    opt-in via ``LAUREUM_ACTIVATION_PROVIDER=anthropic``.

L2 (tool-use loop)
    Extends L1 with a formal tool surface and the in-process
    :class:`MockFileSystem`. For Cerebras we describe the tools inline and
    parse fake ``<tool>...</tool>`` blocks from the response. For Anthropic
    we use the SDK's ``tools=`` argument with ``cache_control`` on the
    system prompt.

L3 (claude-agent-sdk Docker harness)
    Stub only — owned by QO-059.

The class hierarchy is :class:`abc.ABC` (not ``Protocol``) so subclasses fail
fast with ``TypeError`` if they forget to implement an abstract method.

Acceptance criteria covered: AC1, AC2, AC3, AC4, AC6, AC7, AC8, AC9, AC10.
AC5 (reproducibility) is exercised by integration tests gated by
``@pytest.mark.live``.
"""
from __future__ import annotations

import abc
import asyncio
import logging
import random
import re
import time
from datetime import datetime
from typing import Any, Optional

from src.config import calculate_cost, settings
from src.core.mock_filesystem import MockFileSystem
from src.core.model_resolver import ResolvedModel, parse_provider_model
from src.storage.models import (
    ActivationDLQEntry,
    ActivationFailure,
    ActivationResponse,
    ParsedSkill,
    ToolCall,
    UsageSummary,
)

logger = logging.getLogger(__name__)


# ── Bash-preprocessor stripping (AC7) ────────────────────────────────────────

# Inline form: !`gh pr diff`
_INLINE_BASH = re.compile(r"!`[^`]+`")
# Block form: triple-backtick with leading bang language tag.
_BLOCK_BASH = re.compile(r"```!\s*\n.*?\n```", re.DOTALL)


def strip_bash_preprocessors(body: str) -> tuple[str, list[str]]:
    r"""Strip Claude-Code-only bash preprocessor placeholders for determinism.

    Removes both inline (``!`cmd` ``) and block (triple-backtick ``!``) forms.
    Returns the cleaned body and a list of warnings that callers append to
    :attr:`ActivationResponse.parse_warnings`. Eval mode disables bash
    preprocessing per R9 §1.
    """
    warnings: list[str] = []
    n_inline = len(_INLINE_BASH.findall(body))
    n_block = len(_BLOCK_BASH.findall(body))
    body = _INLINE_BASH.sub("[bash-preprocessor: command (disabled in eval)]", body)
    body = _BLOCK_BASH.sub(
        "[bash-preprocessor block: command (disabled in eval)]", body,
    )
    total = n_inline + n_block
    if total:
        warnings.append(f"bash_preprocessor_stripped:{total}")
    return body, warnings


# ── Retry policy (AC8) ──────────────────────────────────────────────────────


_INITIAL_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0
_MAX_RETRIES = 5
_CONSECUTIVE_5XX_LIMIT = 3


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter — initial 2s, cap 60s."""
    raw = min(_MAX_BACKOFF_S, _INITIAL_BACKOFF_S * (2 ** attempt))
    return random.uniform(0, raw)


def _classify_error(exc: Exception) -> str:
    """Bucket SDK exceptions into ``rate_limit`` / ``server_5xx`` / ``other``."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "rate" in msg or "429" in msg or name == "RateLimitError":
        return "rate_limit"
    if "5" in msg.replace(" ", "")[:3] and ("server" in msg or "internal" in msg or "503" in msg or "500" in msg or "502" in msg or "504" in msg):
        return "server_5xx"
    if name in {"InternalServerError", "APIStatusError", "ServiceUnavailableError"}:
        return "server_5xx"
    return "other"


# ── Abstract base ────────────────────────────────────────────────────────────


class SkillActivatedAgent(abc.ABC):
    """Mirror of ``MCPTarget.call_tool`` for the evaluator dispatch.

    Concrete subclasses keep state for one (skill, model) pair and accumulate
    a :class:`UsageSummary` that the evaluator persists with every result.
    """

    def __init__(
        self,
        skill: Optional[ParsedSkill],
        resolved: ResolvedModel,
        provider_client: Any,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        dlq_writer: Optional[Any] = None,
    ):
        self.skill = skill
        self.resolved = resolved
        self.model = resolved.dated_snapshot
        self.provider = resolved.provider
        self.client = provider_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._usage = UsageSummary(model=self.model, provider=self.provider)
        self._system_text: Optional[str] = None
        self._system_warnings: list[str] = []
        self._cache_disabled_for_short_body: bool = False
        self._dlq_writer = dlq_writer  # async callable (entry: ActivationDLQEntry) -> None

    # ── Abstract surface ────────────────────────────────────────────────────

    @abc.abstractmethod
    async def respond(self, question: str) -> ActivationResponse:
        ...

    @abc.abstractmethod
    def usage_summary(self) -> UsageSummary:
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset multi-turn state between unrelated questions."""
        ...

    # ── Shared system-prompt construction ───────────────────────────────────

    def _build_system_text(self) -> str:
        """Construct (and cache) the system prompt body from the parsed skill.

        Strips Claude-Code-only bash preprocessors (AC7) and emits the
        cache-disabled warning when ``body_tokens < 2048`` and the active
        provider is Anthropic (AC9).
        """
        if self._system_text is not None:
            return self._system_text
        if self.skill is None:
            self._system_text = ""
            return self._system_text
        body, warnings = strip_bash_preprocessors(self.skill.body)
        # AC9: cache_control rejected by Anthropic when body < 2048 tokens.
        body_tokens = self.skill.body_tokens or 0
        if (
            self.provider == "anthropic"
            and body_tokens < settings.anthropic_min_cacheable_tokens
        ):
            warnings.append("cache_disabled_below_min_tokens")
            self._cache_disabled_for_short_body = True
        # Compose with a short identity preamble — keeps the LLM oriented
        # when the body is itself a long instruction document.
        preamble = (
            f"You are an agent activated under the '{self.skill.name}' skill. "
            "Follow the skill body instructions precisely. The skill body follows:\n\n"
        )
        self._system_text = preamble + body
        self._system_warnings = warnings
        return self._system_text

    # ── Token + cost accounting ────────────────────────────────────────────

    def _record_usage(
        self,
        cache_creation: int,
        cache_read: int,
        input_tokens: int,
        output_tokens: int,
        request_id: str = "",
    ) -> None:
        self._usage.cache_creation_input_tokens += cache_creation
        self._usage.cache_read_input_tokens += cache_read
        self._usage.input_tokens += input_tokens
        self._usage.output_tokens += output_tokens
        self._usage.n_calls += 1
        if request_id:
            self._usage.request_ids.append(request_id)
        # cost_provider is the closest match in PROVIDER_PRICING — the resolver
        # already canonicalised provider name so we just pass it through.
        billable_input = (
            input_tokens
            + cache_creation
            # cache_read is billed at a lower rate by Anthropic but free at
            # Cerebras/Groq (no caching) — leave it at $0 for both.
        )
        cost = calculate_cost(self.provider, billable_input, output_tokens)
        self._usage.dollars_spent = round(self._usage.dollars_spent + cost, 8)

    # ── DLQ + retry helpers ────────────────────────────────────────────────

    async def _write_dlq(
        self,
        question_id: Optional[str],
        last_request_id: str,
        error_class: str,
        error_message: str,
        attempt_count: int,
    ) -> None:
        if self._dlq_writer is None:
            return
        entry = ActivationDLQEntry(
            skill_id=self.skill.name if self.skill else None,
            question_id=question_id,
            last_request_id=last_request_id,
            provider=self.provider,
            error_class=error_class,
            error_message=error_message,
            attempt_count=attempt_count,
            ts=datetime.utcnow(),
        )
        try:
            await self._dlq_writer(entry)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("DLQ writer failed: %s", e)

    async def _call_with_retry(
        self,
        coro_factory,
        question_id: Optional[str] = None,
    ):
        """Invoke ``coro_factory()`` with backoff + DLQ.

        ``coro_factory`` returns a fresh coroutine each call so that we can
        retry without re-using an exhausted awaitable.
        """
        consecutive_5xx = 0
        last_request_id = ""
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await coro_factory()
            except Exception as e:  # noqa: BLE001 — SDK exceptions are diverse
                last_exc = e
                kind = _classify_error(e)
                last_request_id = getattr(e, "request_id", "") or last_request_id
                if kind == "server_5xx":
                    consecutive_5xx += 1
                    if consecutive_5xx >= _CONSECUTIVE_5XX_LIMIT:
                        logger.error(
                            "ActivationFailure: 3 consecutive 5xx on %s",
                            self.provider,
                        )
                        await self._write_dlq(
                            question_id, last_request_id,
                            type(e).__name__, str(e), attempt + 1,
                        )
                        raise ActivationFailure(
                            f"3 consecutive 5xx from {self.provider}",
                            last_request_id=last_request_id,
                            provider=self.provider,
                            attempt_count=attempt + 1,
                        ) from e
                else:
                    consecutive_5xx = 0
                if kind == "rate_limit" or kind == "server_5xx":
                    if attempt >= _MAX_RETRIES:
                        break
                    delay = _backoff_seconds(attempt)
                    logger.info(
                        "Retrying %s after %s in %.1fs (attempt %d)",
                        self.provider, kind, delay, attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Permanent — DLQ + re-raise.
                await self._write_dlq(
                    question_id, last_request_id,
                    type(e).__name__, str(e), attempt + 1,
                )
                raise
        # Exhausted retries without raising ActivationFailure earlier.
        await self._write_dlq(
            question_id, last_request_id,
            type(last_exc).__name__ if last_exc else "RetriesExhausted",
            str(last_exc) if last_exc else "",
            _MAX_RETRIES + 1,
        )
        raise ActivationFailure(
            f"Retries exhausted on {self.provider}",
            last_request_id=last_request_id,
            provider=self.provider,
            attempt_count=_MAX_RETRIES + 1,
        ) from last_exc


# ── L1: naive system-prompt injection ───────────────────────────────────────


class L1NaiveActivator(SkillActivatedAgent):
    """Pure SDK injection. Cheapest tier; ~55–65% fidelity (R7 §4)."""

    async def respond(self, question: str, *, question_id: Optional[str] = None) -> ActivationResponse:
        system_text = self._build_system_text()
        warnings = list(self._system_warnings)

        if self.provider == "anthropic":
            return await self._respond_anthropic(question, system_text, warnings, question_id)
        # Cerebras / Groq — both speak the OpenAI-compat schema.
        return await self._respond_oai(question, system_text, warnings, question_id)

    def usage_summary(self) -> UsageSummary:
        return self._usage

    def reset(self) -> None:
        # L1 is stateless; nothing to reset besides the system cache.
        pass

    # ── Anthropic path (cache_control on system) ───────────────────────────

    async def _respond_anthropic(
        self, question: str, system_text: str, warnings: list[str],
        question_id: Optional[str],
    ) -> ActivationResponse:
        system_block: list[dict[str, Any]] = [{"type": "text", "text": system_text}]
        if not self._cache_disabled_for_short_body:
            system_block[0]["cache_control"] = {"type": "ephemeral"}

        async def _call():
            t0 = time.monotonic()
            resp = await self.client.messages.create(
                model=self.model,
                system=system_block,
                messages=[{"role": "user", "content": question}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return resp, t0

        result = await self._call_with_retry(_call, question_id=question_id)
        resp, t0 = result
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = ""
        if getattr(resp, "content", None):
            for block in resp.content:
                if getattr(block, "type", "") == "text":
                    text += getattr(block, "text", "")
        usage = getattr(resp, "usage", None)
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) if usage else 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) if usage else 0
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        request_id = getattr(resp, "id", "") or getattr(resp, "_request_id", "")
        self._record_usage(cache_creation, cache_read, in_tok, out_tok, request_id)
        return ActivationResponse(
            text=text,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=self.model,
            provider=self.provider,
            request_id=request_id,
            latency_ms=latency_ms,
            parse_warnings=warnings,
        )

    # ── Cerebras / Groq path (chat.completions schema) ─────────────────────

    async def _respond_oai(
        self, question: str, system_text: str, warnings: list[str],
        question_id: Optional[str],
    ) -> ActivationResponse:
        async def _call():
            t0 = time.monotonic()
            resp = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": question},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return resp, t0

        result = await self._call_with_retry(_call, question_id=question_id)
        resp, t0 = result
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = ""
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        request_id = getattr(resp, "id", "") or ""
        self._record_usage(0, 0, in_tok, out_tok, request_id)
        return ActivationResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=self.model,
            provider=self.provider,
            request_id=request_id,
            latency_ms=latency_ms,
            parse_warnings=warnings,
        )


# ── L2: tool-use loop with MockFileSystem ───────────────────────────────────


# Cerebras simulated tool block — the model is asked to emit
# ``<tool name="Read"><path>examples/swap.ts</path></tool>`` and the activator
# parses it. Crude but adequate for ~85% L2 fidelity per R7 §4 footnote.
_SIM_TOOL_BLOCK = re.compile(
    r"<tool\s+name=\"(?P<name>[A-Za-z_]+)\">(?P<body>.*?)</tool>",
    re.DOTALL,
)
_SIM_ARG_BLOCK = re.compile(
    r"<(?P<key>[A-Za-z_]+)>(?P<val>.*?)</(?P=key)>",
    re.DOTALL,
)


def _parse_simulated_tool_calls(text: str) -> list[dict]:
    """Parse the Cerebras-simulated ``<tool>...</tool>`` blocks from a turn."""
    calls: list[dict] = []
    for m in _SIM_TOOL_BLOCK.finditer(text):
        name = m.group("name").strip()
        body = m.group("body")
        args: dict[str, str] = {}
        for arg in _SIM_ARG_BLOCK.finditer(body):
            args[arg.group("key").strip()] = arg.group("val").strip()
        calls.append({"name": name, "args": args})
    return calls


# Anthropic-shaped tool definitions — the SDK accepts these as-is.
ANTHROPIC_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "Read",
        "description": "Read a file from the skill folder.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a sandbox-safe bash command (ls/cat/head/tail/git-log/pwd).",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "Glob",
        "description": "Glob files in the skill folder.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Regex search files in the skill folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Edit",
        "description": "Replace ``old`` with ``new`` in a file (overlay, not on disk).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file (overlay, not on disk).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]


def _dispatch_tool(fs: MockFileSystem, name: str, args: dict) -> ToolCall:
    """Route a tool call to the corresponding :class:`MockFileSystem` method."""
    method = getattr(fs, name, None)
    if method is None:
        return ToolCall(
            tool=name, args=args, error=f"unknown-tool:{name}", blocked=True,
        )
    try:
        return method(**args)
    except TypeError as e:
        return ToolCall(tool=name, args=args, error=f"bad-args:{e}", blocked=True)


_L2_MAX_TOOL_TURNS = 5


class L2ToolUseActivator(SkillActivatedAgent):
    """Anthropic Messages + formal ``tools`` param, or Cerebras simulated tools.

    Ships with a :class:`MockFileSystem` scoped to the skill folder. Every
    tool call is recorded for use by QO-053-D / QO-053-E probes.
    """

    def __init__(
        self,
        skill: Optional[ParsedSkill],
        resolved: ResolvedModel,
        provider_client: Any,
        mock_fs: MockFileSystem,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        dlq_writer: Optional[Any] = None,
    ):
        super().__init__(
            skill, resolved, provider_client,
            temperature=temperature, max_tokens=max_tokens, dlq_writer=dlq_writer,
        )
        self.mock_fs = mock_fs

    async def respond(self, question: str, *, question_id: Optional[str] = None) -> ActivationResponse:
        if self.provider == "anthropic":
            return await self._respond_anthropic(question, question_id)
        return await self._respond_simulated(question, question_id)

    def usage_summary(self) -> UsageSummary:
        return self._usage

    def reset(self) -> None:
        self.mock_fs.reset_log()

    # ── Anthropic real-tool loop ───────────────────────────────────────────

    async def _respond_anthropic(
        self, question: str, question_id: Optional[str],
    ) -> ActivationResponse:
        system_text = self._build_system_text()
        warnings = list(self._system_warnings)
        system_block: list[dict[str, Any]] = [{"type": "text", "text": system_text}]
        if not self._cache_disabled_for_short_body:
            system_block[0]["cache_control"] = {"type": "ephemeral"}

        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        all_tool_calls: list[ToolCall] = []
        text_out = ""
        request_id = ""
        latency_total_ms = 0

        for _turn in range(_L2_MAX_TOOL_TURNS):
            async def _call():
                t0 = time.monotonic()
                resp = await self.client.messages.create(
                    model=self.model,
                    system=system_block,
                    messages=messages,
                    tools=ANTHROPIC_TOOL_SCHEMAS,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return resp, t0

            resp, t0 = await self._call_with_retry(_call, question_id=question_id)
            latency_total_ms += int((time.monotonic() - t0) * 1000)
            usage = getattr(resp, "usage", None)
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) if usage else 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) if usage else 0
            in_tok = getattr(usage, "input_tokens", 0) if usage else 0
            out_tok = getattr(usage, "output_tokens", 0) if usage else 0
            request_id = getattr(resp, "id", "") or request_id
            self._record_usage(cache_creation, cache_read, in_tok, out_tok, request_id)

            stop_reason = getattr(resp, "stop_reason", "")
            content_blocks = list(getattr(resp, "content", []) or [])
            tool_uses = [b for b in content_blocks if getattr(b, "type", "") == "tool_use"]
            text_blocks = [b for b in content_blocks if getattr(b, "type", "") == "text"]
            for tb in text_blocks:
                text_out += getattr(tb, "text", "")

            if stop_reason != "tool_use" or not tool_uses:
                break

            # Echo the assistant turn back, then a user turn with tool_result.
            assistant_content: list[dict[str, Any]] = []
            for b in content_blocks:
                bt = getattr(b, "type", "")
                if bt == "text":
                    assistant_content.append({"type": "text", "text": getattr(b, "text", "")})
                elif bt == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": getattr(b, "id", ""),
                        "name": getattr(b, "name", ""),
                        "input": getattr(b, "input", {}) or {},
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                args = getattr(tu, "input", {}) or {}
                name = getattr(tu, "name", "")
                call = _dispatch_tool(self.mock_fs, name, args)
                all_tool_calls.append(call)
                payload = call.returned if call.returned is not None else (call.error or "")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": getattr(tu, "id", ""),
                    "content": payload,
                    "is_error": bool(call.error),
                })
            messages.append({"role": "user", "content": tool_results})

        return ActivationResponse(
            text=text_out,
            tool_calls=all_tool_calls,
            cache_creation_tokens=self._usage.cache_creation_input_tokens,
            cache_read_tokens=self._usage.cache_read_input_tokens,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            model=self.model,
            provider=self.provider,
            request_id=request_id,
            latency_ms=latency_total_ms,
            parse_warnings=warnings,
        )

    # ── Cerebras / Groq simulated-tool loop ────────────────────────────────

    async def _respond_simulated(
        self, question: str, question_id: Optional[str],
    ) -> ActivationResponse:
        system_text = self._build_system_text()
        # Append the simulated-tool instruction inline — Cerebras has no
        # native tools= argument so we describe the surface in the prompt.
        tool_help = (
            "\n\n## Available tools (eval mode — simulated)\n"
            "When you need to use a tool, emit exactly one block of the form\n"
            "<tool name=\"NAME\"><arg>value</arg></tool> and then STOP.\n"
            "Tools: Read(path), Bash(command), Glob(pattern), Grep(pattern,path), "
            "Edit(path,old,new), Write(path,content).\n"
        )
        system_text = system_text + tool_help
        warnings = list(self._system_warnings)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": question},
        ]
        all_tool_calls: list[ToolCall] = []
        text_out = ""
        request_id = ""
        latency_total_ms = 0

        for _turn in range(_L2_MAX_TOOL_TURNS):
            async def _call():
                t0 = time.monotonic()
                resp = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return resp, t0

            resp, t0 = await self._call_with_retry(_call, question_id=question_id)
            latency_total_ms += int((time.monotonic() - t0) * 1000)
            try:
                turn_text = resp.choices[0].message.content or ""
            except (AttributeError, IndexError):
                turn_text = ""
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            request_id = getattr(resp, "id", "") or request_id
            self._record_usage(0, 0, in_tok, out_tok, request_id)

            sim_calls = _parse_simulated_tool_calls(turn_text)
            if not sim_calls:
                text_out += turn_text
                break

            # Echo assistant + tool results back as plain text turns.
            messages.append({"role": "assistant", "content": turn_text})
            tool_payloads: list[str] = []
            for sc in sim_calls:
                call = _dispatch_tool(self.mock_fs, sc["name"], sc["args"])
                all_tool_calls.append(call)
                payload = call.returned if call.returned is not None else (call.error or "")
                tool_payloads.append(
                    f"<tool_result name=\"{sc['name']}\">{payload}</tool_result>"
                )
            messages.append({"role": "user", "content": "\n".join(tool_payloads)})

        return ActivationResponse(
            text=text_out,
            tool_calls=all_tool_calls,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            model=self.model,
            provider=self.provider,
            request_id=request_id,
            latency_ms=latency_total_ms,
            parse_warnings=warnings,
        )


# ── L3: Docker harness ──────────────────────────────────────────────────────
#
# QO-059 ships the real audited-tier activator in :mod:`src.core.l3_activator`.
# We re-export here so the dispatcher in QO-053-C / QO-058 can keep importing
# ``L3ClaudeCodeActivator`` from this module (preserves the QO-053-B contract).
#
# We use module-level ``__getattr__`` to break the circular import — the L3
# module imports ``SkillActivatedAgent`` from this file, and now this file
# wants to re-export ``L3ClaudeCodeActivator``. Lazy-loading via ``__getattr__``
# defers the L3 module import until first attribute access, by which time the
# parent module is fully initialised.


def __getattr__(name: str):
    if name == "L3ClaudeCodeActivator":
        from src.core.l3_activator import L3ClaudeCodeActivator as _L3
        globals()["L3ClaudeCodeActivator"] = _L3
        return _L3
    raise AttributeError(f"module 'src.core.skill_activator' has no attribute {name!r}")


# ── Factory ─────────────────────────────────────────────────────────────────


def parse_activation_setting(value: str) -> tuple[str, str]:
    """Public re-export of :func:`model_resolver.parse_provider_model`."""
    return parse_provider_model(value)

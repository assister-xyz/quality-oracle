"""Tests for QO-053-B L2ToolUseActivator (AC2, AC9 cache disabled w/ tools).

Verifies the tool-use loop on both Anthropic (real ``tools=`` param) and
Cerebras (simulated ``<tool>`` blocks).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.mock_filesystem import MockFileSystem
from src.core.model_resolver import ResolvedModel
from src.core.skill_activator import L2ToolUseActivator
from src.storage.models import ParsedSkill


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    root = tmp_path / "solana-swap"
    root.mkdir()
    (root / "examples").mkdir()
    (root / "examples" / "swap.ts").write_text(
        "// swap.ts: example Solana swap\nexport const route = 'jupiter';\n"
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "run.sh").write_text("#!/bin/bash\necho hello")
    return root


def _make_skill(body: str = "Use Read('examples/swap.ts') to inspect the swap example.",
                body_tokens: int = 3000) -> ParsedSkill:
    return ParsedSkill(
        name="solana-swap",
        description="Solana swap helper",
        body=body,
        body_size_bytes=len(body.encode()),
        body_lines=body.count("\n") + 1,
        body_tokens=body_tokens,
        folder_name="solana-swap",
        folder_name_nfkc="solana-swap",
    )


def _cerebras() -> ResolvedModel:
    return ResolvedModel("cerebras", "llama3.1-8b", "llama3.1-8b", "fixed")


def _anthropic() -> ResolvedModel:
    return ResolvedModel(
        "anthropic", "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929", "list_models",
    )


def _make_anthropic_tool_use_response(
    tool_name: str = "Read",
    tool_input: dict | None = None,
    tool_id: str = "toolu_01",
    rid: str = "msg_01",
):
    """Build an Anthropic response with a tool_use stop reason."""
    if tool_input is None:
        tool_input = {"path": "examples/swap.ts"}
    tu = MagicMock()
    tu.type = "tool_use"
    tu.id = tool_id
    tu.name = tool_name
    tu.input = tool_input
    resp = MagicMock()
    resp.content = [tu]
    resp.id = rid
    resp.stop_reason = "tool_use"
    usage = MagicMock()
    usage.cache_creation_input_tokens = 1000
    usage.cache_read_input_tokens = 0
    usage.input_tokens = 50
    usage.output_tokens = 30
    resp.usage = usage
    return resp


def _make_anthropic_final_response(text: str = "the swap uses jupiter",
                                   rid: str = "msg_02"):
    """Build the final assistant turn (no tool_use)."""
    tb = MagicMock()
    tb.type = "text"
    tb.text = text
    resp = MagicMock()
    resp.content = [tb]
    resp.id = rid
    resp.stop_reason = "end_turn"
    usage = MagicMock()
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 1000
    usage.input_tokens = 50
    usage.output_tokens = 30
    resp.usage = usage
    return resp


def _make_oai_response(text: str, rid: str = "req_oai", in_tok: int = 100, out_tok: int = 50):
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


# ── AC2: L2 tool-use loop on Anthropic (real tools=) ────────────────────────


@pytest.mark.asyncio
class TestAnthropicToolLoop:
    async def test_read_tool_returns_actual_file_content(self, skill_dir):
        """AC2: model calls Read('examples/swap.ts'), gets real file contents."""
        skill = _make_skill()
        fs = MockFileSystem(skill_dir)
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=[
            _make_anthropic_tool_use_response(
                tool_name="Read", tool_input={"path": "examples/swap.ts"},
            ),
            _make_anthropic_final_response(text="the swap uses jupiter"),
        ])
        activator = L2ToolUseActivator(skill, _anthropic(), client, fs)
        resp = await activator.respond("Which DEX does the swap use?")
        assert resp.text == "the swap uses jupiter"
        # AC2: tool call recorded with returned file content
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.tool == "Read"
        assert tc.args == {"path": "examples/swap.ts"}
        assert "swap.ts: example Solana swap" in tc.returned
        # And same call observable on the MockFileSystem log
        assert len(fs.tool_calls_log) == 1

    async def test_path_escape_attempt_blocked_and_recorded(self, skill_dir):
        skill = _make_skill()
        fs = MockFileSystem(skill_dir)
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=[
            _make_anthropic_tool_use_response(
                tool_name="Read", tool_input={"path": "../../etc/passwd"},
            ),
            _make_anthropic_final_response(text="cannot exfil"),
        ])
        activator = L2ToolUseActivator(skill, _anthropic(), client, fs)
        resp = await activator.respond("read the secret")
        assert resp.tool_calls[0].blocked is True

    async def test_anthropic_tools_param_passed(self, skill_dir):
        skill = _make_skill()
        fs = MockFileSystem(skill_dir)
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_make_anthropic_final_response(text="no tools needed"),
        )
        activator = L2ToolUseActivator(skill, _anthropic(), client, fs)
        await activator.respond("hi")
        kwargs = client.messages.create.call_args.kwargs
        assert "tools" in kwargs
        tool_names = {t["name"] for t in kwargs["tools"]}
        assert tool_names == {"Read", "Bash", "Glob", "Grep", "Edit", "Write"}

    async def test_cache_control_set_for_long_body(self, skill_dir):
        skill = _make_skill(body_tokens=5000)
        fs = MockFileSystem(skill_dir)
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_make_anthropic_final_response(),
        )
        activator = L2ToolUseActivator(skill, _anthropic(), client, fs)
        await activator.respond("hi")
        sysblk = client.messages.create.call_args.kwargs["system"]
        assert sysblk[0]["cache_control"] == {"type": "ephemeral"}


# ── AC9: cache disabled when body_tokens < 2048 ─────────────────────────────


@pytest.mark.asyncio
class TestAC9CacheDisabledOnL2:
    async def test_short_body_emits_warning_omits_cache(self, skill_dir):
        skill = _make_skill(body_tokens=500)
        fs = MockFileSystem(skill_dir)
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_make_anthropic_final_response(),
        )
        activator = L2ToolUseActivator(skill, _anthropic(), client, fs)
        resp = await activator.respond("hi")
        assert "cache_disabled_below_min_tokens" in resp.parse_warnings
        sysblk = client.messages.create.call_args.kwargs["system"]
        assert "cache_control" not in sysblk[0]


# ── Cerebras simulated tool loop ────────────────────────────────────────────


@pytest.mark.asyncio
class TestCerebrasSimulatedToolLoop:
    async def test_simulated_read_tool_block_parsed(self, skill_dir):
        """Cerebras emits ``<tool name='Read'><path>...</path></tool>``."""
        skill = _make_skill()
        fs = MockFileSystem(skill_dir)
        client = MagicMock()
        client.chat.completions.create = MagicMock(side_effect=[
            _make_oai_response(
                text='<tool name="Read"><path>examples/swap.ts</path></tool>',
            ),
            _make_oai_response(text="final answer: jupiter"),
        ])
        activator = L2ToolUseActivator(skill, _cerebras(), client, fs)
        resp = await activator.respond("which DEX?")
        assert resp.text == "final answer: jupiter"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool == "Read"
        assert "jupiter" in resp.tool_calls[0].returned

    async def test_no_tool_block_returns_text(self, skill_dir):
        skill = _make_skill()
        fs = MockFileSystem(skill_dir)
        client = MagicMock()
        client.chat.completions.create = MagicMock(
            return_value=_make_oai_response(text="just a plain answer"),
        )
        activator = L2ToolUseActivator(skill, _cerebras(), client, fs)
        resp = await activator.respond("hi")
        assert resp.text == "just a plain answer"
        assert resp.tool_calls == []

    async def test_cerebras_cost_is_zero(self, skill_dir):
        skill = _make_skill()
        fs = MockFileSystem(skill_dir)
        client = MagicMock()
        client.chat.completions.create = MagicMock(
            return_value=_make_oai_response(text="ok"),
        )
        activator = L2ToolUseActivator(skill, _cerebras(), client, fs)
        await activator.respond("Q")
        assert activator.usage_summary().dollars_spent == 0.0


# ── reset() clears mock_fs log ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestReset:
    async def test_reset_clears_mock_fs_log(self, skill_dir):
        skill = _make_skill()
        fs = MockFileSystem(skill_dir)
        fs.Read("examples/swap.ts")  # populate log
        assert len(fs.tool_calls_log) == 1
        client = MagicMock()
        activator = L2ToolUseActivator(skill, _cerebras(), client, fs)
        activator.reset()
        assert fs.tool_calls_log == []



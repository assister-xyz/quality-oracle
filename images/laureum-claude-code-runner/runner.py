"""QO-059 L3 harness in-container entrypoint.

Reads JSON args from argv[1], invokes `claude_agent_sdk.query()` for each
question, and prints a JSON document to stdout. The host-side
`L3ClaudeCodeActivator` parses that JSON into `ActivationResponse`.

Per AC2 / N4: NO `--bare` CLI flag exists. "Bare" runtime is configured via
`ClaudeAgentOptions(setting_sources=["project"])` — only the project's
`.claude/skills/` is loaded, not user-level CLAUDE.md, hooks, or MCP
auto-discovery. Verified by inspecting the SDK's runtime config dump.

Per AC1: skills are discovered from `/eval/.claude/skills/<name>/` via the
read-only mount; the SDK lists them at startup and `/skills` slash command
activates them.

Per AC3: the SDK's per-call `usage` (cache_creation_input_tokens, etc.) is
captured here and bubbled up to the host activator for cost accounting.

Stdout contract:
    {
      "results": [
        {
          "question": "...",
          "text": "...",
          "tool_calls": [...],
          "usage": {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
          },
          "request_id": "...",
          "latency_ms": 1234
        },
        ...
      ],
      "model": "claude-sonnet-4-5-20250929",
      "sdk_version": "0.x.y"
    }
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any


async def main() -> None:
    args = json.loads(sys.argv[1])
    skill_name: str = args["skill_name"]
    questions: list[str] = args["questions"]
    model: str = args["model"]
    temperature: float = float(args.get("temp", 0.2))

    # Late import — runner.py is also imported by host-side tests for
    # symbol-level introspection without requiring claude-agent-sdk to be
    # pip-installed in the host venv.
    from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore

    options_kwargs: dict[str, Any] = {
        "cwd": "/eval",
        "setting_sources": ["project"],
        "allowed_tools": ["Skill", "Read", "Write", "Bash", "Glob", "Grep"],
        "model": model,
    }
    # Some SDK versions accept `temperature` directly; fall back to env.
    try:
        options = ClaudeAgentOptions(temperature=temperature, **options_kwargs)
    except TypeError:
        os.environ["CLAUDE_TEMPERATURE"] = str(temperature)
        options = ClaudeAgentOptions(**options_kwargs)

    results: list[dict[str, Any]] = []
    for q in questions:
        prompt = f"Use the {skill_name} skill. {q}"
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage_total = {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        request_id = ""
        t0 = time.monotonic()
        async for msg in query(prompt=prompt, options=options):
            mtype = getattr(msg, "type", "") or (
                msg.get("type", "") if isinstance(msg, dict) else ""
            )
            if mtype == "text":
                text_chunks.append(getattr(msg, "text", "") or msg.get("text", ""))
            elif mtype == "tool_use":
                tool_calls.append({
                    "tool": getattr(msg, "name", "") or msg.get("name", ""),
                    "args": getattr(msg, "input", {}) or msg.get("input", {}) or {},
                })
            elif mtype in ("usage", "message_delta"):
                u = getattr(msg, "usage", None) or msg.get("usage", {})
                if u:
                    for k in usage_total:
                        usage_total[k] += int(getattr(u, k, 0) or u.get(k, 0) or 0)
                rid = getattr(msg, "id", "") or (msg.get("id", "") if isinstance(msg, dict) else "")
                if rid:
                    request_id = rid
        latency_ms = int((time.monotonic() - t0) * 1000)
        results.append({
            "question": q,
            "text": "".join(text_chunks),
            "tool_calls": tool_calls,
            "usage": usage_total,
            "request_id": request_id,
            "latency_ms": latency_ms,
        })

    sdk_version = ""
    try:
        import claude_agent_sdk as _sdk  # type: ignore

        sdk_version = getattr(_sdk, "__version__", "")
    except Exception:
        pass

    print(json.dumps({
        "results": results,
        "model": model,
        "sdk_version": sdk_version,
    }))


if __name__ == "__main__":
    asyncio.run(main())

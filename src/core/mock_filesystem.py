"""Sandboxed file-tool emulator for L2 skill activation (QO-053-B).

The real Claude Code agent loop drives ``Read`` / ``Bash`` / ``Glob`` / ``Grep``
/ ``Edit`` / ``Write`` against the working directory. For deterministic eval
we mount the parsed skill folder read-only and route every tool call through
:class:`MockFileSystem`. Network commands are recorded then refused so a
``Bash("curl ...")`` from a malicious skill is observable without leaving the
sandbox.

Every call appends to :attr:`MockFileSystem.tool_calls_log` so downstream
probes — QO-053-D Solana fee-payer hijack, QO-053-E SKILL-POISON-02 script
poisoning — can verify the skill's tool trace without re-running activation.
"""
from __future__ import annotations

import fnmatch
import re
import time
from pathlib import Path
from typing import List, Optional

from src.storage.models import ToolCall


# Bash commands we synthesize canned output for. Anything else returns a
# safe error message rather than executing the user-provided string.
_SAFE_BASH = ("ls", "cat", "head", "tail", "git log", "pwd", "wc")
_NETWORK_BASH = ("curl", "wget", "ssh", "nc", "telnet", "scp", "rsync")


def _is_within(child: Path, root: Path) -> bool:
    """True when ``child`` resolves inside ``root``.

    ``Path.is_relative_to`` exists in 3.9+ but raises on broken symlinks; the
    explicit ``commonpath`` form is portable and never raises.
    """
    try:
        child_r = child.resolve(strict=False)
        root_r = root.resolve(strict=False)
        return str(child_r).startswith(str(root_r) + "/") or child_r == root_r
    except (OSError, ValueError):
        return False


class MockFileSystem:
    """Filesystem-shaped tool surface scoped to one parsed skill directory.

    The instance carries no I/O state beyond ``tool_calls_log`` — every method
    is idempotent and safe to call from concurrent activations.
    """

    def __init__(self, skill_dir: Path | str):
        self.skill_dir = Path(skill_dir).resolve(strict=False)
        self.tool_calls_log: List[ToolCall] = []
        # Edit/Write are sandboxed to an in-memory overlay so we never mutate
        # the on-disk fixture between runs.
        self._writes: dict[str, str] = {}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _record(self, tool: str, args: dict, returned: Optional[str] = None,
                error: Optional[str] = None, blocked: bool = False,
                started_ms: float = 0.0) -> ToolCall:
        call = ToolCall(
            tool=tool,
            args=args,
            returned=returned,
            error=error,
            blocked=blocked,
            duration_ms=int((time.monotonic() - started_ms) * 1000) if started_ms else 0,
        )
        self.tool_calls_log.append(call)
        return call

    def _resolve_inside(self, path: str) -> Optional[Path]:
        """Resolve a relative path inside the skill dir; reject escapes."""
        if not path:
            return None
        candidate = (self.skill_dir / path).resolve(strict=False)
        if not _is_within(candidate, self.skill_dir):
            return None
        return candidate

    # ── Tools ────────────────────────────────────────────────────────────────

    def Read(self, path: str) -> ToolCall:
        """Return file content if path is inside the skill folder."""
        t0 = time.monotonic()
        resolved = self._resolve_inside(path)
        if resolved is None:
            return self._record(
                "Read", {"path": path},
                error=f"403: path {path!r} outside skill directory",
                blocked=True, started_ms=t0,
            )
        # Honour overlay first (so Edit/Write are visible).
        rel = str(resolved.relative_to(self.skill_dir))
        if rel in self._writes:
            return self._record(
                "Read", {"path": path}, returned=self._writes[rel], started_ms=t0,
            )
        if not resolved.exists():
            return self._record(
                "Read", {"path": path},
                error=f"404: {path!r} does not exist", started_ms=t0,
            )
        if resolved.is_dir():
            return self._record(
                "Read", {"path": path},
                error=f"is-a-directory: {path!r}", started_ms=t0,
            )
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return self._record(
                "Read", {"path": path}, error=f"read_error:{e}", started_ms=t0,
            )
        return self._record("Read", {"path": path}, returned=content, started_ms=t0)

    def Bash(self, command: str) -> ToolCall:
        """Run a *canned* bash command. Network commands are blocked."""
        t0 = time.monotonic()
        cmd = (command or "").strip()
        first = cmd.split(" ", 1)[0] if cmd else ""
        # Network exfil — record but never run.
        if first in _NETWORK_BASH or any(cmd.startswith(p + " ") for p in _NETWORK_BASH):
            return self._record(
                "Bash", {"command": command},
                error="sandbox-error: network commands disabled in eval",
                blocked=True, started_ms=t0,
            )
        # Canned ls / cat / head / git log / pwd / wc.
        if first == "ls":
            target_arg = cmd[3:].strip() or "."
            target = self._resolve_inside(target_arg)
            if target is None:
                return self._record(
                    "Bash", {"command": command},
                    error="403: ls outside skill dir", blocked=True, started_ms=t0,
                )
            if not target.exists():
                return self._record(
                    "Bash", {"command": command},
                    error="404: target does not exist", started_ms=t0,
                )
            if target.is_dir():
                output = "\n".join(sorted(p.name for p in target.iterdir()))
            else:
                output = target.name
            return self._record("Bash", {"command": command}, returned=output, started_ms=t0)
        if first in ("cat", "head", "tail"):
            target_arg = cmd.split(" ", 1)[1].strip() if " " in cmd else ""
            target = self._resolve_inside(target_arg)
            if target is None or not target.exists() or target.is_dir():
                return self._record(
                    "Bash", {"command": command},
                    error="404: cat/head/tail target invalid", started_ms=t0,
                )
            try:
                content = target.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return self._record(
                    "Bash", {"command": command}, error=f"read_error:{e}",
                    started_ms=t0,
                )
            if first == "head":
                content = "\n".join(content.splitlines()[:10])
            elif first == "tail":
                content = "\n".join(content.splitlines()[-10:])
            return self._record("Bash", {"command": command}, returned=content, started_ms=t0)
        if cmd.startswith("git log"):
            return self._record(
                "Bash", {"command": command},
                returned="commit deadbeef\nAuthor: eval-mock\nDate: 2026-04-25\n\n    canned git log entry",
                started_ms=t0,
            )
        if first == "pwd":
            return self._record(
                "Bash", {"command": command}, returned=str(self.skill_dir),
                started_ms=t0,
            )
        if first == "wc":
            return self._record(
                "Bash", {"command": command}, returned="0 0 0", started_ms=t0,
            )
        # Anything else is recorded but refused.
        return self._record(
            "Bash", {"command": command},
            error=f"sandbox-error: command {first!r} not in safe-list {_SAFE_BASH}",
            blocked=True, started_ms=t0,
        )

    def Glob(self, pattern: str) -> ToolCall:
        """Return paths inside the skill directory matching ``pattern``."""
        t0 = time.monotonic()
        if not pattern:
            return self._record(
                "Glob", {"pattern": pattern}, error="empty-pattern", started_ms=t0,
            )
        # Reject ../-style escape attempts in the pattern itself.
        if ".." in pattern.split("/"):
            return self._record(
                "Glob", {"pattern": pattern},
                error="403: '..' segments rejected", blocked=True, started_ms=t0,
            )
        matches: List[str] = []
        for path in self.skill_dir.rglob("*"):
            if path.is_file() and fnmatch.fnmatch(
                str(path.relative_to(self.skill_dir)), pattern
            ):
                matches.append(str(path.relative_to(self.skill_dir)))
        return self._record(
            "Glob", {"pattern": pattern},
            returned="\n".join(sorted(matches)), started_ms=t0,
        )

    def Grep(self, pattern: str, path: str = ".") -> ToolCall:
        """Search files under ``path`` for ``pattern`` (regex)."""
        t0 = time.monotonic()
        target = self._resolve_inside(path)
        if target is None:
            return self._record(
                "Grep", {"pattern": pattern, "path": path},
                error="403: path outside skill dir", blocked=True, started_ms=t0,
            )
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return self._record(
                "Grep", {"pattern": pattern, "path": path},
                error=f"invalid-regex:{e}", started_ms=t0,
            )
        hits: List[str] = []
        files = [target] if target.is_file() else list(target.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = f.relative_to(self.skill_dir)
                    hits.append(f"{rel}:{lineno}:{line}")
        return self._record(
            "Grep", {"pattern": pattern, "path": path},
            returned="\n".join(hits), started_ms=t0,
        )

    def Edit(self, path: str, old: str, new: str) -> ToolCall:
        """In-memory replacement; original file on disk is never mutated."""
        t0 = time.monotonic()
        resolved = self._resolve_inside(path)
        if resolved is None:
            return self._record(
                "Edit", {"path": path}, error="403: outside skill dir",
                blocked=True, started_ms=t0,
            )
        rel = str(resolved.relative_to(self.skill_dir))
        # Source content: prefer overlay, then disk, then empty.
        if rel in self._writes:
            content = self._writes[rel]
        elif resolved.exists() and resolved.is_file():
            try:
                content = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return self._record(
                    "Edit", {"path": path}, error=f"read_error:{e}",
                    started_ms=t0,
                )
        else:
            content = ""
        if old and old not in content:
            return self._record(
                "Edit", {"path": path, "old": old, "new": new},
                error="old-string-not-found", started_ms=t0,
            )
        new_content = content.replace(old, new) if old else (content + new)
        self._writes[rel] = new_content
        return self._record(
            "Edit", {"path": path, "old": old, "new": new},
            returned=f"edited {len(new_content)} bytes (overlay)",
            started_ms=t0,
        )

    def Write(self, path: str, content: str) -> ToolCall:
        """Write to in-memory overlay; original file on disk untouched."""
        t0 = time.monotonic()
        resolved = self._resolve_inside(path)
        if resolved is None:
            return self._record(
                "Write", {"path": path}, error="403: outside skill dir",
                blocked=True, started_ms=t0,
            )
        rel = str(resolved.relative_to(self.skill_dir))
        self._writes[rel] = content
        return self._record(
            "Write", {"path": path, "bytes": len(content)},
            returned=f"wrote {len(content)} bytes (overlay)",
            started_ms=t0,
        )

    # ── Introspection helpers ───────────────────────────────────────────────

    def reset_log(self) -> None:
        self.tool_calls_log.clear()
        self._writes.clear()

    def network_attempts(self) -> List[ToolCall]:
        return [c for c in self.tool_calls_log if c.blocked and c.tool == "Bash"]

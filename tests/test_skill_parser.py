"""Tests for src.core.skill_parser (QO-053-A).

Covers:
- Happy-path parse on 3 fixture skills (jupiter, inco, anthropic skill-creator).
- Full 175-file corpus regression (AC1) — guarded by `pytest.mark.skipif`
  when fixtures aren't on disk.
- AC4 (NFKC normalization), AC5 (1024 desc), AC7 (spec dirs split),
  AC20 (CRLF/BOM/latin-1), AC21 (`skill.md` lowercase fallback).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.skill_parser import parse_skill_md, parse_skill_md_string

FIXTURES_ROOT = Path("/tmp")
CORPUS_ROOTS = [
    FIXTURES_ROOT / "anthropic-skills",
    FIXTURES_ROOT / "sendai-skills",
    FIXTURES_ROOT / "trailofbits-skills",
    FIXTURES_ROOT / "antfu-skills",
    FIXTURES_ROOT / "addyosmani-agent-skills",
]


def _has_corpus() -> bool:
    return all(p.is_dir() for p in CORPUS_ROOTS)


# ── Happy paths on cited fixtures ────────────────────────────────────────────


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_parses_sendai_jupiter():
    p = parse_skill_md(Path("/tmp/sendai-skills/skills/jupiter"))
    # Name is NFKC-normalized; matches frontmatter, NOT folder.
    assert p.name == "integrating-jupiter"
    assert p.folder_name == "jupiter"
    # Description is non-empty and well under 1024.
    assert 1 <= len(p.description) <= 1024
    # Off-spec `tags` field is preserved verbatim in `frontmatter_raw`.
    assert "tags" in p.frontmatter_raw
    # No crash, body populated.
    assert p.body
    assert p.body_size_bytes > 0
    assert p.body_lines >= 1


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_parses_sendai_inco():
    p = parse_skill_md(Path("/tmp/sendai-skills/skills/inco"))
    assert p.name == "inco-svm"
    assert p.folder_name == "inco"
    # `inco/` ships the convention dirs `docs/`, `templates/`, `resources/`.
    # `resources/` is NOT spec-defined — should appear in `extra_dirs`.
    assert "resources" in p.extra_dirs
    assert p.convention_dirs.get("docs") is True
    assert p.convention_dirs.get("templates") is True
    assert all(p.spec_dirs[k] is False for k in ("scripts", "references", "assets"))


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_parses_anthropic_skill_creator():
    p = parse_skill_md(Path("/tmp/anthropic-skills/skills/skill-creator"))
    assert p.name == "skill-creator"
    assert p.folder_name == "skill-creator"
    # Anthropic example carries a real reference body.
    assert p.body_size_bytes > 1000


# ── 175-corpus regression (AC1) ──────────────────────────────────────────────


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_corpus_zero_crashes():
    """AC1 — parse every SKILL.md/skill.md across 5 reference repos with zero crashes."""
    skill_dirs: list[Path] = []
    for root in CORPUS_ROOTS:
        for fp in list(root.rglob("SKILL.md")) + list(root.rglob("skill.md")):
            skill_dirs.append(fp.parent)
    skill_dirs = sorted(set(skill_dirs))
    assert len(skill_dirs) >= 150, f"expected ≥ 150 skill dirs, found {len(skill_dirs)}"

    crashed: list[tuple[Path, str]] = []
    parsed_ok = 0
    for d in skill_dirs:
        try:
            ps = parse_skill_md(d)
            assert ps.name is not None  # name attribute populated (may be empty str)
            parsed_ok += 1
        except Exception as e:
            crashed.append((d, f"{type(e).__name__}: {e}"))

    assert not crashed, f"{len(crashed)} crash(es): {crashed[:3]}"
    assert parsed_ok == len(skill_dirs)


# ── AC4 NFKC normalization ──────────────────────────────────────────────────


def test_nfkc_normalization(tmp_path: Path):
    """AC4 — non-NFKC name in frontmatter should match an NFKC-equivalent folder."""
    # `ﬁnance` (Latin small ligature fi U+FB01) NFKC-normalizes to `finance`.
    folder = tmp_path / "finance"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: ﬁnance\ndescription: Test the NFKC pipeline.\n---\n\n# Body\n",
        encoding="utf-8",
    )
    p = parse_skill_md(folder)
    assert p.name == "finance"  # NFKC-normalized
    assert p.folder_name_nfkc == "finance"


# ── AC5 description boundary ───────────────────────────────────────────────


def test_description_at_1024_boundary_parses(tmp_path: Path):
    """Parser does not enforce description bounds — that is the validator's job."""
    folder = tmp_path / "boundary-test"
    folder.mkdir()
    body = "x" * 1024
    (folder / "SKILL.md").write_text(
        f"---\nname: boundary-test\ndescription: {body}\n---\n\n# Body\n",
        encoding="utf-8",
    )
    p = parse_skill_md(folder)
    assert len(p.description) == 1024


# ── AC7 spec / convention / extra dir partition ─────────────────────────────


def test_spec_and_convention_dirs_partitioned(tmp_path: Path):
    folder = tmp_path / "dirs"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: dirs\ndescription: Test directory partitioning.\n---\n\nbody\n",
        encoding="utf-8",
    )
    # Spec-defined.
    for d in ("scripts", "references", "assets"):
        (folder / d).mkdir()
        (folder / d / "stub").write_text("hi", encoding="utf-8")
    # Convention-only.
    for d in ("examples", "templates", "docs"):
        (folder / d).mkdir()
        (folder / d / "stub").write_text("hi", encoding="utf-8")
    # Extra (neither spec nor convention).
    (folder / "eval-viewer").mkdir()
    (folder / "eval-viewer" / "stub").write_text("hi", encoding="utf-8")

    p = parse_skill_md(folder)
    assert p.spec_dirs == {"scripts": True, "references": True, "assets": True}
    assert p.convention_dirs == {"examples": True, "templates": True, "docs": True}
    assert "eval-viewer" in p.extra_dirs


def test_scripts_not_recursive(tmp_path: Path):
    """AC7 — `scripts/` enumeration uses iterdir (not rglob).

    A nested `tools/scripts/` directory must NOT count as a populated
    spec-defined `scripts/`.
    """
    folder = tmp_path / "nested"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: nested\ndescription: Test no rglob on scripts/.\n---\n\nbody\n",
        encoding="utf-8",
    )
    nested = folder / "tools" / "scripts"
    nested.mkdir(parents=True)
    (nested / "stub.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    p = parse_skill_md(folder)
    assert p.spec_dirs["scripts"] is False
    # `tools` should appear as an extra dir.
    assert "tools" in p.extra_dirs


# ── AC20 CRLF / BOM / latin-1 ────────────────────────────────────────────────


def test_crlf_normalized():
    raw = "---\r\nname: crlf-skill\r\ndescription: ok\r\n---\r\n\r\nBody line one.\r\nBody line two.\r\n"
    p = parse_skill_md_string(raw)
    assert p.name == "crlf-skill"
    assert "\r" not in p.body


def test_bom_stripped():
    raw = "﻿---\nname: bom-skill\ndescription: ok\n---\n\nbody\n"
    p = parse_skill_md_string(raw)
    assert p.name == "bom-skill"
    assert not p.body.startswith("﻿")


def test_latin1_fallback():
    # Bytes that are valid latin-1 but invalid UTF-8 (ISO-8859-1 ñ at 0xF1).
    raw = b"---\nname: latin1-skill\ndescription: ok\n---\n\nca\xf1on body\n"
    p = parse_skill_md_string(raw)
    assert p.name == "latin1-skill"
    assert "non_utf8_decoded_as_latin1" in p.parse_warnings


# ── AC21 lowercase `skill.md` fallback ──────────────────────────────────────


def test_lowercase_skill_md_fallback(tmp_path: Path):
    folder = tmp_path / "lower"
    folder.mkdir()
    (folder / "skill.md").write_text(
        "---\nname: lower\ndescription: lowercase fallback.\n---\n\nbody\n",
        encoding="utf-8",
    )
    p = parse_skill_md(folder)
    assert p.name == "lower"
    assert "filename_lowercase_skill_md" in p.parse_warnings


def test_uppercase_preferred_over_lowercase(tmp_path: Path):
    """On a case-sensitive filesystem (Linux/CI) both files coexist; on
    case-insensitive filesystems (macOS APFS default) only one survives.
    Either way, the parser must prefer the uppercase entry when present.
    """
    folder = tmp_path / "both"
    folder.mkdir()
    upper = folder / "SKILL.md"
    upper.write_text(
        "---\nname: both\ndescription: prefer uppercase.\n---\n\nupper\n",
        encoding="utf-8",
    )
    # Skip writing the lowercase peer when the FS is case-insensitive (a
    # second `write_text("skill.md", ...)` would silently overwrite the
    # uppercase entry on macOS APFS).
    case_sensitive = not (folder / "Skill.md").is_file()
    if case_sensitive:
        (folder / "skill.md").write_text(
            "---\nname: both\ndescription: should not be read.\n---\n\nlower\n",
            encoding="utf-8",
        )
    p = parse_skill_md(folder)
    # Whether or not the lowercase peer exists, parse_skill_md must pick
    # the uppercase variant and emit no lowercase warning.
    assert "upper" in p.body, f"unexpected body on case_sensitive={case_sensitive}: {p.body!r}"
    assert "filename_lowercase_skill_md" not in p.parse_warnings


def test_missing_skill_md_raises(tmp_path: Path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        parse_skill_md(folder)


# ── allowed-tools coercion (R1 rule 11) ─────────────────────────────────────


def test_allowed_tools_string_form():
    raw = "---\nname: tools-str\ndescription: ok\nallowed-tools: Read Bash Edit\n---\n\nbody\n"
    p = parse_skill_md_string(raw)
    assert p.allowed_tools == ["Read", "Bash", "Edit"]


def test_allowed_tools_list_form():
    raw = "---\nname: tools-list\ndescription: ok\nallowed-tools:\n  - Read\n  - Bash\n---\n\nbody\n"
    p = parse_skill_md_string(raw)
    assert p.allowed_tools == ["Read", "Bash"]


def test_metadata_values_stringified():
    raw = "---\nname: meta\ndescription: ok\nmetadata:\n  version: 1\n  count: 42\n---\n\nbody\n"
    p = parse_skill_md_string(raw)
    assert p.metadata == {"version": "1", "count": "42"}

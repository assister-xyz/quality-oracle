"""Tests for src.core.skill_validator (QO-053-A).

Mostly synthetic SKILL.md fixtures generated under `tmp_path` so the test does
not depend on the upstream R2 corpus. The 3 cited SendAI mismatches (jupiter /
inco / metengine) are exercised via a corpus check guarded by
`pytest.mark.skipif`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.skill_parser import parse_skill_md, parse_skill_md_string
from src.core.skill_validator import validate_skill
from src.storage.models import Severity

CORPUS_ROOT = Path("/tmp/sendai-skills/skills")


def _has_corpus() -> bool:
    return CORPUS_ROOT.is_dir()


def _make_skill(tmp_path: Path, name: str, content: str) -> Path:
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(content, encoding="utf-8")
    return folder


# ── AC3 — name/folder mismatch HARD FAIL ────────────────────────────────────


def test_name_folder_mismatch_hard_fail(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "alpha",
        "---\nname: not-alpha\ndescription: A short test skill description.\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.rule == "AP1" and v.severity == Severity.HIGH for v in sc.violations)
    assert sc.score <= 80
    assert sc.passed_hard_fails is False


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_sendai_jupiter_hard_fails():
    p = parse_skill_md(CORPUS_ROOT / "jupiter")
    sc = validate_skill(p, CORPUS_ROOT / "jupiter")
    assert sc.score <= 80
    assert sc.passed_hard_fails is False
    assert any(v.rule == "AP1" for v in sc.violations)


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_sendai_inco_hard_fails():
    p = parse_skill_md(CORPUS_ROOT / "inco")
    sc = validate_skill(p, CORPUS_ROOT / "inco")
    assert sc.passed_hard_fails is False
    assert any(v.rule == "AP1" for v in sc.violations)


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_sendai_metengine_hard_fails():
    p = parse_skill_md(CORPUS_ROOT / "metengine")
    sc = validate_skill(p, CORPUS_ROOT / "metengine")
    assert sc.passed_hard_fails is False
    assert any(v.rule == "AP1" for v in sc.violations)


# ── Rule 1 — required fields ────────────────────────────────────────────────


def test_missing_name_is_hard_fail(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "no-name",
        "---\ndescription: Has only description.\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert sc.passed_hard_fails is False
    assert any(v.field == "name" and v.severity == Severity.HIGH for v in sc.violations)


def test_missing_description_is_hard_fail(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "no-desc",
        "---\nname: no-desc\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert sc.passed_hard_fails is False
    assert any(v.field == "description" and v.severity == Severity.HIGH for v in sc.violations)


# ── Rule 2 — name shape (HIGH per spec since portability-breaker) ─────────────


def test_name_too_long(tmp_path: Path):
    name = "x" * 65
    folder = _make_skill(
        tmp_path,
        name,
        f"---\nname: {name}\ndescription: long name test.\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.rule == "AP11" and v.field == "name" for v in sc.violations)


def test_name_with_uppercase_invalid(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "Bad-Name",  # folder match irrelevant; name itself is invalid
        "---\nname: Bad-Name\ndescription: uppercase letters.\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    # NFKC alone does not lowercase; the validator must flag it.
    assert any(v.rule == "AP11" and v.field == "name" for v in sc.violations)


def test_name_with_double_hyphen_invalid(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "bad--name",
        "---\nname: bad--name\ndescription: double hyphen.\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.field == "name" for v in sc.violations)


# ── Rule 4 / AC5 — description bounds (1–1024) ─────────────────────────────


def test_description_at_600_passes(tmp_path: Path):
    desc = "x" * 600
    folder = _make_skill(
        tmp_path,
        "desc-600",
        f"---\nname: desc-600\ndescription: {desc}\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    # No HIGH-severity violation related to description length.
    assert not any(v.rule == "AP5" for v in sc.violations)


def test_description_at_1100_fails(tmp_path: Path):
    desc = "x" * 1100
    folder = _make_skill(
        tmp_path,
        "desc-1100",
        f"---\nname: desc-1100\ndescription: {desc}\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.rule == "AP5" and v.severity == Severity.HIGH for v in sc.violations)


def test_description_above_1200_warns(tmp_path: Path):
    """AC22 — > 1200 chars triggers a separate LOW AP5_LONG warning."""
    desc = "x" * 1300
    folder = _make_skill(
        tmp_path,
        "desc-1300",
        f"---\nname: desc-1300\ndescription: {desc}\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    rules = {v.rule for v in sc.violations}
    assert "AP5" in rules
    assert "AP5_LONG" in rules


# ── Rule 6 / AC2 — top-level allowlist ──────────────────────────────────────


def test_off_spec_top_level_field(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "off-spec",
        "---\nname: off-spec\ndescription: Has a top-level version field.\nversion: 1.0\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    offenders = [v for v in sc.violations if v.rule == "AP3" and v.field == "version"]
    assert offenders, "expected AP3 violation for off-spec `version` field"
    assert offenders[0].severity == Severity.MED


# ── Rule 5 — compatibility length ───────────────────────────────────────────


def test_compatibility_too_long(tmp_path: Path):
    compat = "x" * 600
    folder = _make_skill(
        tmp_path,
        "compat-long",
        f"---\nname: compat-long\ndescription: ok\ncompatibility: {compat}\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.field == "compatibility" for v in sc.violations)


# ── Rule 7 — reserved words (LOW) ───────────────────────────────────────────


def test_reserved_word_anthropic(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "anthropic-tools",
        "---\nname: anthropic-tools\ndescription: Talks about Anthropic Claude.\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.severity == Severity.LOW for v in sc.violations)


# ── Rule 8 — XML tag (LOW) ──────────────────────────────────────────────────


def test_xml_tag_in_description(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "xml-desc",
        "---\nname: xml-desc\ndescription: Use this <skill>.\n---\n\nbody\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.field == "description" and v.severity == Severity.LOW for v in sc.violations)


# ── Rule 9 — body size ──────────────────────────────────────────────────────


def test_body_size_warn(tmp_path: Path):
    body = "x" * (35 * 1024)
    folder = _make_skill(
        tmp_path,
        "body-warn",
        f"---\nname: body-warn\ndescription: ok\n---\n\n{body}\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.rule == "AP2" for v in sc.violations)


def test_body_size_hard_fail(tmp_path: Path):
    body = "x" * (110 * 1024)
    folder = _make_skill(
        tmp_path,
        "body-bomb",
        f"---\nname: body-bomb\ndescription: ok\n---\n\n{body}\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.rule == "AP2" and v.severity == Severity.HIGH for v in sc.violations)
    assert sc.passed_hard_fails is False


# ── Rule 10 — body line warning ─────────────────────────────────────────────


def test_body_lines_warn(tmp_path: Path):
    body = "\n".join(f"line {i}" for i in range(600))
    folder = _make_skill(
        tmp_path,
        "many-lines",
        f"---\nname: many-lines\ndescription: ok\n---\n\n{body}\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert any(v.rule == "AP2" for v in sc.violations)


# ── Rule 11 — allowed-tools str OR list ─────────────────────────────────────


def test_allowed_tools_accepts_string():
    raw = "---\nname: tools\ndescription: ok\nallowed-tools: Read Bash\n---\n\nbody\n"
    p = parse_skill_md_string(raw)
    sc = validate_skill(p, None)
    # Should not report any allowed-tools-related violation.
    assert all(v.field != "allowed-tools" for v in sc.violations)


# ── Rule 12 — YAML parse error ──────────────────────────────────────────────


def test_yaml_parse_error_is_hard_fail():
    raw = "---\nname: parse-fail\n  this: : invalid yaml\n---\n\nbody\n"
    p = parse_skill_md_string(raw)
    sc = validate_skill(p, None)
    assert sc.passed_hard_fails is False
    assert any(v.rule == "AP11" for v in sc.violations)


# ── AC8 — score formula sanity ─────────────────────────────────────────────


def test_score_is_capped_at_zero(tmp_path: Path):
    # Pile up violations to drive score below 0.
    body = "x" * (110 * 1024)  # AP2 HIGH = 15
    fields = "\n".join(f"f{i}: 1" for i in range(20))  # 20 × AP3 MED = 100
    folder = _make_skill(
        tmp_path,
        "many-violations",
        f"---\nname: many-violations\ndescription: ok\n{fields}\n---\n\n{body}\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert sc.score == 0


def test_clean_skill_scores_100(tmp_path: Path):
    folder = _make_skill(
        tmp_path,
        "clean",
        "---\n"
        "name: clean\n"
        "description: A clean test skill that satisfies every spec rule.\n"
        "license: MIT\n"
        "compatibility: claude-sonnet-4-5\n"
        "metadata:\n  author: tester\n"
        "allowed-tools: Read Bash\n"
        "---\n\n"
        "## Section A\n\nBody.\n\n## Section B\n\nMore body.\n",
    )
    p = parse_skill_md(folder)
    sc = validate_skill(p, folder)
    assert sc.score == 100
    assert sc.passed_hard_fails is True

"""Tests for src.core.skill_anti_patterns (QO-053-A).

One test per AP1..AP18, with the cited SendAI/Anthropic example whenever the
spec names one. Synthetic skills are used for the rest so the test does not
depend on the upstream R2 corpus snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.skill_anti_patterns import detect_anti_patterns
from src.core.skill_parser import parse_skill_md, parse_skill_md_string
from src.storage.models import Severity

CORPUS_ROOT = Path("/tmp/sendai-skills/skills")


def _has_corpus() -> bool:
    return CORPUS_ROOT.is_dir()


def _make(tmp_path: Path, name: str, content: str) -> Path:
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(content, encoding="utf-8")
    return folder


def _ids(findings) -> set[str]:
    return {f.id for f in findings}


# ── AP1 — name/folder mismatch ──────────────────────────────────────────────


def test_ap1_name_folder_mismatch(tmp_path: Path):
    folder = _make(
        tmp_path,
        "alpha",
        "---\nname: not-alpha\ndescription: A test skill description.\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP1" in _ids(findings)
    ap1 = next(f for f in findings if f.id == "AP1")
    assert ap1.severity == Severity.HIGH


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_ap1_jupiter_cited():
    findings = detect_anti_patterns(
        parse_skill_md(CORPUS_ROOT / "jupiter"), CORPUS_ROOT / "jupiter"
    )
    assert "AP1" in _ids(findings)


# ── AP2 — size bomb ──────────────────────────────────────────────────────────


def test_ap2_size_bomb(tmp_path: Path):
    body = "x" * (110 * 1024)
    folder = _make(
        tmp_path,
        "bomb",
        f"---\nname: bomb\ndescription: too big.\n---\n\n{body}\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP2" in _ids(findings)


# ── AP3 — off-spec frontmatter ──────────────────────────────────────────────


def test_ap3_off_spec_top_level(tmp_path: Path):
    folder = _make(
        tmp_path,
        "tags",
        "---\nname: tags\ndescription: has off-spec tags.\ntags: [a, b]\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    ids = _ids(findings)
    assert "AP3" in ids
    ap3 = next(f for f in findings if f.id == "AP3")
    assert ap3.severity == Severity.MED


# ── AP4 — description too short ─────────────────────────────────────────────


def test_ap4_description_too_short(tmp_path: Path):
    folder = _make(
        tmp_path,
        "short",
        "---\nname: short\ndescription: tiny\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP4" in _ids(findings)


# ── AP5 — description too long ──────────────────────────────────────────────


def test_ap5_description_too_long(tmp_path: Path):
    desc = "x" * 1100
    folder = _make(
        tmp_path,
        "long-desc",
        f"---\nname: long-desc\ndescription: {desc}\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP5" in _ids(findings)


# ── AP6 — marketplace.json drift ────────────────────────────────────────────


def test_ap6_marketplace_json_drift(tmp_path: Path):
    folder = _make(
        tmp_path,
        "drift",
        "---\nname: drift\ndescription: real description.\n---\n\nbody\n",
    )
    (folder / "marketplace.json").write_text(
        json.dumps({"name": "drift", "description": "different description"}),
        encoding="utf-8",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP6" in _ids(findings)


# ── AP7 — hardcoded mainnet RPC without warning ────────────────────────────


def test_ap7_mainnet_rpc_no_warning(tmp_path: Path):
    folder = _make(
        tmp_path,
        "rpc",
        "---\nname: rpc\ndescription: An RPC skill.\n---\n\n"
        "Use the URL https://api.mainnet-beta.solana.com to do swaps.\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP7" in _ids(findings)


def test_ap7_mainnet_rpc_with_warning_silenced(tmp_path: Path):
    folder = _make(
        tmp_path,
        "rpc-warn",
        "---\nname: rpc-warn\ndescription: An RPC skill with a warning.\n---\n\n"
        "WARNING: this is mainnet — real funds. https://api.mainnet-beta.solana.com\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP7" not in _ids(findings)


# ── AP8 — sensitive-key term without warning ───────────────────────────────


def test_ap8_keypair_no_warning(tmp_path: Path):
    folder = _make(
        tmp_path,
        "keypair",
        "---\nname: keypair\ndescription: How to use a keypair.\n---\n\n"
        "Load your keypair from the file. The keypair is parsed at runtime.\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP8" in _ids(findings)


def test_ap8_keypair_with_warning_silenced(tmp_path: Path):
    folder = _make(
        tmp_path,
        "keypair-safe",
        "---\nname: keypair-safe\ndescription: Safe keypair handling.\n---\n\n"
        "WARNING: never share your keypair. Treat it as a secret.\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP8" not in _ids(findings)


# ── AP9 — auto-updater (HIGH) — METENGINE-CITED ────────────────────────────


def test_ap9_auto_updater_synthetic(tmp_path: Path):
    folder = _make(
        tmp_path,
        "updater",
        "---\nname: updater\ndescription: Has an auto-updater.\n---\n\n"
        "```bash\ncurl -sL https://example.com/skill.md -o ~/.claude/agents/foo.md\n```\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    ap9 = [f for f in findings if f.id == "AP9"]
    assert ap9, "AP9 must fire on the metengine-style auto-updater pattern"
    assert ap9[0].severity == Severity.HIGH


@pytest.mark.skipif(not _has_corpus(), reason="run dev/setup_fixtures.sh first")
def test_ap9_metengine_cited():
    """AC6 — the metengine SKILL.md MUST trigger AP9 (lines 47-49 cited)."""
    folder = CORPUS_ROOT / "metengine"
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    ap9 = [f for f in findings if f.id == "AP9"]
    assert ap9, "AP9 must fire on the live SendAI metengine SKILL.md"
    assert ap9[0].severity == Severity.HIGH


# ── AP10 — no examples + no scripts ─────────────────────────────────────────


def test_ap10_no_examples_no_scripts(tmp_path: Path):
    folder = _make(
        tmp_path,
        "barren",
        "---\nname: barren\ndescription: No examples and no scripts.\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP10" in _ids(findings)


def test_ap10_silenced_when_examples_populated(tmp_path: Path):
    folder = _make(
        tmp_path,
        "examples-only",
        "---\nname: examples-only\ndescription: Has examples.\n---\n\nbody\n",
    )
    (folder / "examples").mkdir()
    (folder / "examples" / "demo.py").write_text("print('hi')", encoding="utf-8")
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP10" not in _ids(findings)


# ── AP11 — YAML parse error ─────────────────────────────────────────────────


def test_ap11_yaml_parse_error():
    raw = "---\nname: bad\n  : : not yaml\n---\n\nbody\n"
    p = parse_skill_md_string(raw)
    findings = detect_anti_patterns(p, None)
    assert "AP11" in _ids(findings)


def test_ap11_no_frontmatter():
    raw = "Just a markdown file without frontmatter.\n"
    p = parse_skill_md_string(raw)
    findings = detect_anti_patterns(p, None)
    assert "AP11" in _ids(findings)


# ── AP12 — < 2 H2 sections ──────────────────────────────────────────────────


def test_ap12_few_h2_sections(tmp_path: Path):
    folder = _make(
        tmp_path,
        "thin",
        "---\nname: thin\ndescription: only one H2.\n---\n\n## Only Section\n\nBody.\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP12" in _ids(findings)


def test_ap12_silenced_with_two_h2(tmp_path: Path):
    folder = _make(
        tmp_path,
        "thicc",
        "---\nname: thicc\ndescription: two H2 sections.\n---\n\n## A\n\nx\n\n## B\n\ny\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP12" not in _ids(findings)


# ── AP13 — no LICENSE ───────────────────────────────────────────────────────


def test_ap13_no_license(tmp_path: Path):
    folder = _make(
        tmp_path,
        "nolicense",
        "---\nname: nolicense\ndescription: skill without a license.\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP13" in _ids(findings)


def test_ap13_silenced_by_frontmatter_license(tmp_path: Path):
    folder = _make(
        tmp_path,
        "licensed",
        "---\nname: licensed\ndescription: skill with license.\nlicense: MIT\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP13" not in _ids(findings)


def test_ap13_silenced_by_license_file(tmp_path: Path):
    folder = _make(
        tmp_path,
        "license-file",
        "---\nname: license-file\ndescription: skill with LICENSE file.\n---\n\nbody\n",
    )
    (folder / "LICENSE").write_text("MIT\n", encoding="utf-8")
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP13" not in _ids(findings)


# ── AP14 — owned by QO-053-C; static analysis must not emit it ──────────────


def test_ap14_not_emitted_by_static_analysis(tmp_path: Path):
    folder = _make(
        tmp_path,
        "scope-creep",
        "---\nname: scope-creep\ndescription: This description is intentionally going way "
        "beyond the actual skill.\n---\n\n"
        "## Setup\n\nx\n\n## Notes\n\ny\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP14" not in _ids(findings)


# ── AP15 — template flood ───────────────────────────────────────────────────


def test_ap15_template_flood(tmp_path: Path):
    folder = _make(
        tmp_path,
        "flood",
        "---\nname: flood\ndescription: template flood case.\n---\n\nbody\n",
    )
    templates = folder / "templates"
    templates.mkdir()
    for i in range(25):
        (templates / f"t{i}.txt").write_text("template", encoding="utf-8")
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP15" in _ids(findings)


def test_ap15_template_flood_by_ratio(tmp_path: Path):
    folder = _make(
        tmp_path,
        "ratio",
        "---\nname: ratio\ndescription: template ratio test.\n---\n\nbody\n",
    )
    templates = folder / "templates"
    templates.mkdir()
    # 5 templates, 1 SKILL.md = 5/6 = 83% > 50%.
    for i in range(5):
        (templates / f"t{i}.txt").write_text("t", encoding="utf-8")
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP15" in _ids(findings)


# ── AP16 — hardcoded /Users/X path ─────────────────────────────────────────


def test_ap16_users_path_in_body(tmp_path: Path):
    folder = _make(
        tmp_path,
        "hardpath",
        "---\nname: hardpath\ndescription: users path.\n---\n\n"
        "Run from /Users/jane/projects/skill/run.sh\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP16" in _ids(findings)


def test_ap16_users_path_in_script(tmp_path: Path):
    folder = _make(
        tmp_path,
        "hardpath-script",
        "---\nname: hardpath-script\ndescription: hardcoded path in script.\n---\n\nbody\n",
    )
    scripts = folder / "scripts"
    scripts.mkdir()
    (scripts / "run.sh").write_text("cd /Users/joe/skills && ./go\n", encoding="utf-8")
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP16" in _ids(findings)


# ── AP17 — personal email ──────────────────────────────────────────────────


def test_ap17_personal_email(tmp_path: Path):
    folder = _make(
        tmp_path,
        "email",
        "---\nname: email\ndescription: has personal email.\n"
        "metadata:\n  author: alice@gmail.com\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP17" in _ids(findings)


# ── AP18 — multi-line YAML description ──────────────────────────────────────


def test_ap18_folded_description_block(tmp_path: Path):
    folder = _make(
        tmp_path,
        "folded",
        "---\nname: folded\ndescription: >\n  multi-line\n  description block.\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP18" in _ids(findings)


def test_ap18_literal_description_block(tmp_path: Path):
    folder = _make(
        tmp_path,
        "literal",
        "---\nname: literal\ndescription: |\n  literal block\n  with newlines.\n---\n\nbody\n",
    )
    findings = detect_anti_patterns(parse_skill_md(folder), folder)
    assert "AP18" in _ids(findings)

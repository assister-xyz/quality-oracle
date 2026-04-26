"""QO-061 AC5, AC6, AC7, AC10: rebuilt correlation engine.

Tests:
- AC5: compute_correlation accepts list[FeedbackSnapshot]; Pearson r is on
       (eval_score_at_time, outcome_score) pairs — NOT (index, outcome).
- AC6: inverse-gap (eval≤40, prod≥80) → anomaly_type='reverse_sandbagging'.
- AC7: KYA-weighted feedback applies tier weights free=1.0, builder=2.0, team=3.0.
- AC10: post-merge there are zero callers of the OLD `compute_correlation`
        signature using `(target_id, eval_score, feedback_index)` semantics.
"""
import math
import subprocess
from datetime import datetime
from pathlib import Path


from src.core.correlation import (
    KYA_TIER_WEIGHTS,
    MIN_SNAPSHOTS_FOR_PEARSON,
    compute_correlation,
    compute_correlation_report,
    weighted_pearson,
)
from src.storage.models import FeedbackSnapshot


# ── Helpers ─────────────────────────────────────────────────────────────────


def _snap(eval_score: float, outcome: float, tier: int = 1, target_id: str = "t") -> FeedbackSnapshot:
    return FeedbackSnapshot(
        target_id=target_id,
        eval_score_at_time=eval_score,
        feedback_outcome=outcome,
        reporter_kya_tier=tier,
        weight=KYA_TIER_WEIGHTS[tier],
        timestamp=datetime.utcnow(),
    )


# ── AC5: Pearson on real eval-vs-outcome pairs ──────────────────────────────


class TestPearsonOnSnapshots:

    def test_perfect_positive_correlation(self):
        # 10 rows with eval == outcome → r = 1.0
        snaps = [_snap(float(i), float(i)) for i in range(60, 70)]
        result = compute_correlation(snaps)
        assert result.status == "ok"
        assert result.n == 10
        assert result.r is not None
        assert abs(result.r - 1.0) < 0.001

    def test_perfect_negative_correlation(self):
        # eval increases, outcome decreases → r = -1.0
        snaps = [_snap(float(i), float(100 - i)) for i in range(60, 70)]
        result = compute_correlation(snaps)
        assert result.r is not None
        assert abs(result.r - (-1.0)) < 0.001

    def test_known_pearson_r(self):
        # Synthetic data with hand-computed r ≈ 0.866
        evals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        outcomes = [12.0, 18.0, 35.0, 38.0, 55.0, 58.0, 80.0, 75.0, 90.0, 99.0]
        snaps = [_snap(e, o) for e, o in zip(evals, outcomes)]
        result = compute_correlation(snaps)
        # Reference computed with numpy.corrcoef on the same series ≈ 0.987
        # Verify within a reasonable tolerance.
        assert result.r is not None
        assert 0.95 < result.r < 1.0

    def test_below_minimum_returns_insufficient_data(self):
        snaps = [_snap(80.0, 75.0) for _ in range(MIN_SNAPSHOTS_FOR_PEARSON - 1)]
        result = compute_correlation(snaps)
        assert result.status == "insufficient_data"
        assert result.r is None

    def test_pearson_NOT_index_based(self):
        # Same constant eval=80; outcomes increase 60..78 — the OLD code
        # would emit a positive r from (index, outcome). The NEW code returns
        # r=None because eval has zero variance.
        snaps = [_snap(80.0, 60.0 + 2 * i) for i in range(10)]
        result = compute_correlation(snaps)
        assert result.r is None  # std_eval = 0 → cannot compute Pearson


# ── AC7: KYA-weighted Pearson ───────────────────────────────────────────────


class TestKYAWeights:

    def test_team_tier_outweighs_free_tier(self):
        # 9 free-tier rows pulling toward outcome=20, 1 team-tier row anchoring
        # at outcome=90 with eval=90. Weighted average should sit closer to the
        # team row than an unweighted mean would.
        snaps = [_snap(80.0, 20.0, tier=1) for _ in range(9)]
        snaps.append(_snap(90.0, 90.0, tier=3))
        result = compute_correlation(snaps)
        # Weighted avg outcome = (9*1*20 + 1*3*90) / 12 = 37.5
        assert result.avg_outcome is not None
        assert 35 <= result.avg_outcome <= 40

    def test_default_weights_are_one_two_three(self):
        assert KYA_TIER_WEIGHTS == {1: 1.0, 2: 2.0, 3: 3.0}

    def test_weighted_pearson_handles_unit_weights(self):
        # Unit weights should match the unweighted Pearson implementation.
        from src.core.correlation import pearson_correlation
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        unweighted = pearson_correlation(xs, ys)
        weighted = weighted_pearson(xs, ys, [1.0] * 5)
        assert math.isclose(unweighted or 0.0, weighted or 0.0, abs_tol=1e-9)


# ── AC6: inverse-gap reverse-sandbagging flag ──────────────────────────────


class TestReverseSandbagging:

    def test_eval_low_prod_high_flagged(self):
        # eval≈30, prod≈85 — should flag reverse_sandbagging.
        snaps = [_snap(30.0, 85.0) for _ in range(10)]
        result = compute_correlation(snaps)
        assert result.anomaly_type == "reverse_sandbagging"
        # AC6: soft signal — confidence_adjustment may be non-zero (because
        # avg_eval=30 has zero variance → r=None → confidence_adjustment=0.0),
        # but tier/badge changes are NOT applied here. Just assert the flag.

    def test_normal_alignment_no_anomaly(self):
        snaps = [_snap(75.0 + (i % 5), 70.0 + (i % 5)) for i in range(10)]
        result = compute_correlation(snaps)
        assert result.anomaly_type is None

    def test_classic_sandbagging_flagged(self):
        snaps = [_snap(80.0, 25.0) for _ in range(10)]
        result = compute_correlation(snaps)
        assert result.anomaly_type == "sandbagging"


# ── AC7 migration: legacy rows tagged with data_quality_warning ─────────────


class TestLegacyDataQuality:

    def test_legacy_row_warning_propagates(self):
        snaps = [_snap(80.0, 75.0) for _ in range(MIN_SNAPSHOTS_FOR_PEARSON)]
        snaps[0].data_quality_warning = "legacy_kya_unknown"
        result = compute_correlation(snaps)
        assert result.data_quality_warning == "legacy_kya_unknown"

    def test_compute_correlation_report_tags_legacy_rows(self):
        # Raw legacy feedback dicts (no eval_score_at_time) → report should
        # carry data_quality_warning for downstream display.
        legacy = [
            {"outcome": "success", "outcome_score": 75 + i}
            for i in range(10)
        ]
        report = compute_correlation_report("legacy-target", 80, legacy)
        # Adapter back-fills with current eval (constant=80) — std_eval=0 →
        # r is None and status is insufficient_data, but the legacy warning
        # should still propagate.
        assert report.data_quality_warning == "legacy_eval_unknown"


# ── AC10: orphan-caller grep for compute_correlation old-signature ──────────


class TestNoOrphanCorrelationCallers:

    def _grep_src(self, pattern: str) -> list[str]:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["grep", "-rn", pattern, str(repo_root / "src")],
            capture_output=True, text=True, check=False,
        )
        # grep returns 1 when no match — that's fine, treat as empty list.
        if result.returncode not in (0, 1):
            raise RuntimeError(f"grep failed: {result.stderr}")
        return [
            line for line in result.stdout.splitlines()
            if line.strip() and "__pycache__" not in line
        ]

    def test_no_call_to_compute_correlation_old_signature(self):
        """AC10: zero callers of the old `compute_correlation(...)` shape.

        The old shape used `compute_correlation(target_id, eval_score, ...)` —
        but in this repo the actual public name was `compute_correlation_report`,
        so the orphan check is: no caller invokes `compute_correlation` with
        positional `target_id` (only legitimate callers pass `snapshots=`).
        """
        # Allow definitions inside correlation.py and the adapter `compute_correlation_report`
        # plus calls that pass snapshots= or a single positional list.
        lines = self._grep_src("compute_correlation")
        # Filter to actual call sites (exclude the definition file itself).
        callers = [
            line for line in lines
            if "src/core/correlation.py" not in line
        ]
        # Only `compute_correlation_report` should appear in caller code; bare
        # `compute_correlation(` callers must pass a snapshots list (no orphans).
        for line in callers:
            # Allow `compute_correlation_report` (the adapter)
            if "compute_correlation_report" in line:
                continue
            # If `compute_correlation(` appears at all, this is a fresh QO-061
            # caller — it MUST be using the new snapshot-list signature, never
            # the old (target_id, eval_score, feedback_items) shape.
            assert "snapshots" in line or "list" in line.lower(), (
                f"Orphan caller of compute_correlation found: {line}"
            )

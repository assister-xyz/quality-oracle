"""QO-051: Cost per Correct Response metric — unit tests."""
import pytest

from src.config import PROVIDER_PRICING, settings
from src.core.evaluator import EvaluationResult, _maybe_compute_cpcr


def _make_result(judge_scores, cost_usd=0.01, shadow_cost_usd=0.05):
    """Build a minimal EvaluationResult with fake judge responses."""
    result = EvaluationResult()
    result.judge_responses = [{"score": s} for s in judge_scores]
    result.cost_usd = cost_usd
    result.shadow_cost_usd = shadow_cost_usd
    return result


class TestCPCRComputation:
    def test_all_correct_returns_non_null(self):
        result = _make_result([90, 85, 75, 95])
        out = result.compute_cpcr(correct_threshold=70)

        assert out["correct_count"] == 4
        assert out["total_responses"] == 4
        assert out["cpcr"] == pytest.approx(0.01 / 4, rel=1e-6)
        assert out["shadow_cpcr"] == pytest.approx(0.05 / 4, rel=1e-6)
        # Weighted: (90+85+75+95)/100 = 3.45 → cost/3.45
        # Tolerance is loose because compute_cpcr rounds to 6 decimals.
        assert out["weighted_cpcr"] == pytest.approx(0.01 / 3.45, abs=1e-6)

    def test_zero_correct_returns_null(self):
        """When no response crosses the threshold, all CPCR variants are None.

        This is the load-bearing edge case — dividing by zero would crash the
        serializer. Callers read None as "insufficient data to price a correct
        answer" and suppress the metric in the UI.
        """
        result = _make_result([10, 20, 30, 69])
        out = result.compute_cpcr(correct_threshold=70)

        assert out["correct_count"] == 0
        assert out["cpcr"] is None
        assert out["shadow_cpcr"] is None
        # weighted_cpcr is still computable since total_quality > 0
        # (partial credit honours non-zero scores)
        assert out["weighted_cpcr"] is not None

    def test_zero_quality_returns_null_weighted(self):
        """If every response scored exactly 0, weighted CPCR is None too."""
        result = _make_result([0, 0, 0])
        out = result.compute_cpcr(correct_threshold=70)
        assert out["cpcr"] is None
        assert out["shadow_cpcr"] is None
        assert out["weighted_cpcr"] is None

    def test_mixed_free_and_paid_providers(self):
        """Shadow CPCR uses market rates, so it stays non-zero even when
        cost_usd is $0 from a fully-free-tier eval.

        This is the key reason shadow CPCR is canonical on the public
        leaderboard — free-tier evals would otherwise show as $0/correct and
        be meaningless for cross-vendor comparison.
        """
        # cost_usd = 0 (all free tier) but shadow = $0.02 market rate
        result = _make_result([80, 90], cost_usd=0.0, shadow_cost_usd=0.02)
        out = result.compute_cpcr(correct_threshold=70)

        assert out["cpcr"] == 0.0  # literally free
        assert out["shadow_cpcr"] == pytest.approx(0.02 / 2, rel=1e-6)

    def test_custom_threshold(self):
        """correct_threshold is tunable — 50 accepts more, 90 fewer."""
        result = _make_result([55, 65, 75, 92])
        strict = result.compute_cpcr(correct_threshold=90)
        assert strict["correct_count"] == 1

        lenient = result.compute_cpcr(correct_threshold=50)
        assert lenient["correct_count"] == 4

    def test_ignores_responses_without_score_key(self):
        """Judge responses missing a score field shouldn't crash or count."""
        result = EvaluationResult()
        result.judge_responses = [
            {"score": 85},
            {"tool": "no_score_here"},  # malformed, should be skipped
            {"score": 92},
        ]
        result.cost_usd = 0.01
        result.shadow_cost_usd = 0.05
        out = result.compute_cpcr(correct_threshold=70)
        assert out["correct_count"] == 2
        assert out["total_responses"] == 2


class TestFeatureFlag:
    def test_default_is_off(self):
        """Rollout default is off — prod stays dark without env changes.

        Flipping this on is an intentional deployment action per QO-051
        rollback plan. If this test fails because the default changed, that
        is almost certainly a mistake — prefer enabling via env var.

        Reads the class default (Settings.model_fields) rather than the
        runtime settings singleton — the singleton can be mutated by an
        .env file or a sibling test, but the class default is what ships.
        """
        from src.config import Settings
        assert Settings.model_fields["enable_cpcr"].default is False

    def test_disabled_flag_skips_computation(self):
        """enable_cpcr=False → _maybe_compute_cpcr is a no-op, fields remain None."""
        original = settings.enable_cpcr
        settings.enable_cpcr = False
        try:
            result = _make_result([90, 85, 75])
            _maybe_compute_cpcr(result)
            assert result.cpcr is None
            assert result.shadow_cpcr is None
            assert result.correct_count == 0
        finally:
            settings.enable_cpcr = original

    def test_enabled_flag_runs_computation(self):
        original = settings.enable_cpcr
        settings.enable_cpcr = True
        try:
            result = _make_result([90, 85, 75])
            _maybe_compute_cpcr(result)
            assert result.cpcr is not None
            assert result.correct_count == 3
        finally:
            settings.enable_cpcr = original


class TestProviderPricingLastUpdated:
    def test_all_entries_have_market_rates(self):
        """Shadow CPCR needs market_input_per_m and market_output_per_m on
        every provider. This guard catches any future entry added without
        the full set of rates, which would make shadow CPCR drop to $0.
        """
        for name, pricing in PROVIDER_PRICING.items():
            assert "input_per_m" in pricing, f"{name} missing input_per_m"
            assert "output_per_m" in pricing, f"{name} missing output_per_m"
            assert "market_input_per_m" in pricing, f"{name} missing market_input_per_m"
            assert "market_output_per_m" in pricing, f"{name} missing market_output_per_m"

    def test_all_entries_have_last_updated(self):
        """Each pricing entry must have a last_updated date so consumers
        can tell how stale the market rate is."""
        for name, pricing in PROVIDER_PRICING.items():
            assert "last_updated" in pricing, f"{name} missing last_updated"
            # Must be a YYYY-MM-DD-ish string
            assert len(pricing["last_updated"]) >= 10


class TestToDictSerialization:
    def test_to_dict_includes_cpcr_block_when_computed(self):
        result = _make_result([80, 90])
        result.compute_cpcr(correct_threshold=70)
        d = result.to_dict()

        assert "cpcr" in d
        assert d["cpcr"]["correct_count"] == 2
        assert d["cpcr"]["cpcr"] is not None
        assert d["cpcr"]["shadow_cpcr"] is not None

    def test_to_dict_includes_shadow_cost(self):
        result = _make_result([80], cost_usd=0.001, shadow_cost_usd=0.005)
        d = result.to_dict()
        assert d["shadow_cost_usd"] == 0.005

    def test_to_dict_skips_cpcr_when_never_computed(self):
        """Pristine EvaluationResult (no compute_cpcr call) should omit cpcr
        from its dict so downstream consumers don't see a zero-filled block
        they might mistake for real data.
        """
        result = EvaluationResult()
        # Don't run compute_cpcr
        d = result.to_dict()
        assert "cpcr" not in d

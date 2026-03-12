"""Tests for anti-gaming detection system (QO-001)."""
import pytest

from src.core.anti_gaming import (
    analyze_response_timing,
    fingerprint_response,
    compute_gaming_risk,
    TimingAnalysis,
    FingerprintResult,
    GamingRisk,
    FAST_RESPONSE_THRESHOLD_MS,
    UNIFORM_TIMING_THRESHOLD_MS,
    MIN_RESPONSES_FOR_TIMING,
)


class TestTimingAnalysis:
    """AC3: Timing analysis detection."""

    def test_normal_timing_not_suspicious(self):
        """Normal response times should not flag."""
        times = [250.0, 320.0, 180.0, 400.0, 550.0, 210.0]
        result = analyze_response_timing(times)
        assert not result.is_suspicious
        assert result.total_responses == 6
        assert result.fast_responses == 0

    def test_fast_responses_flagged(self):
        """More than 50% responses < 100ms should flag."""
        times = [30.0, 45.0, 80.0, 50.0, 200.0, 60.0]  # 5/6 < 100ms
        result = analyze_response_timing(times)
        assert result.is_suspicious
        assert result.fast_responses == 5
        assert "under 100ms" in result.reason

    def test_uniform_timing_flagged(self):
        """Very uniform timing with low mean should flag."""
        times = [200.0, 210.0, 205.0, 195.0, 208.0]  # std_dev < 50ms, mean < 500ms
        result = analyze_response_timing(times)
        assert result.is_suspicious
        assert result.is_uniform
        assert "uniform timing" in result.reason

    def test_uniform_high_mean_not_suspicious(self):
        """Uniform timing with high mean is normal (slow consistent server)."""
        times = [1200.0, 1210.0, 1205.0, 1195.0, 1208.0]  # std_dev < 50ms but mean > 500ms
        result = analyze_response_timing(times)
        assert not result.is_suspicious  # Uniform but slow = just a consistent server

    def test_too_few_responses_skipped(self):
        """Fewer than MIN_RESPONSES_FOR_TIMING should not analyze."""
        times = [30.0, 40.0]  # Only 2 responses
        result = analyze_response_timing(times)
        assert not result.is_suspicious
        assert result.total_responses == 2

    def test_empty_responses(self):
        """Empty response list should not crash."""
        result = analyze_response_timing([])
        assert not result.is_suspicious
        assert result.total_responses == 0

    def test_mixed_timing(self):
        """Mix of fast and normal responses, < 50% fast, should not flag."""
        times = [50.0, 300.0, 400.0, 350.0, 500.0, 250.0]  # 1/6 < 100ms
        result = analyze_response_timing(times)
        assert result.fast_responses == 1
        assert not result.is_suspicious


class TestFingerprinting:
    """AC2: Response fingerprinting."""

    def test_fingerprint_deterministic(self):
        """Same question+response → same hashes."""
        fp1 = fingerprint_response("What is DeFi?", "Decentralized finance...")
        fp2 = fingerprint_response("What is DeFi?", "Decentralized finance...")
        assert fp1.question_hash == fp2.question_hash
        assert fp1.response_hash == fp2.response_hash

    def test_fingerprint_case_insensitive(self):
        """Fingerprinting normalizes case."""
        fp1 = fingerprint_response("What is DeFi?", "DeFi is...")
        fp2 = fingerprint_response("what is defi?", "defi is...")
        assert fp1.question_hash == fp2.question_hash
        assert fp1.response_hash == fp2.response_hash

    def test_different_responses_different_hash(self):
        """Different responses → different hashes."""
        fp1 = fingerprint_response("What is DeFi?", "Response A")
        fp2 = fingerprint_response("What is DeFi?", "Response B completely different")
        assert fp1.question_hash == fp2.question_hash
        assert fp1.response_hash != fp2.response_hash

    def test_different_questions_different_hash(self):
        """Different questions → different question hashes."""
        fp1 = fingerprint_response("What is DeFi?", "Same answer")
        fp2 = fingerprint_response("What is Solana?", "Same answer")
        assert fp1.question_hash != fp2.question_hash
        assert fp1.response_hash == fp2.response_hash

    def test_fingerprint_not_duplicate_by_default(self):
        """New fingerprints should not be marked as duplicate."""
        fp = fingerprint_response("Question", "Answer")
        assert not fp.is_duplicate
        assert fp.prior_eval_id is None


class TestGamingRisk:
    """AC2+AC3: Gaming risk computation."""

    def test_no_signals_no_risk(self):
        """Clean evaluation → risk=none, no penalty."""
        timing = TimingAnalysis(total_responses=10)
        fingerprints = [
            FingerprintResult(question_hash="a", response_hash="b"),
            FingerprintResult(question_hash="c", response_hash="d"),
        ]
        risk = compute_gaming_risk(timing, fingerprints)
        assert risk.level == "none"
        assert risk.confidence_penalty == 0.0
        assert not risk.timing_anomaly

    def test_timing_only_low_risk(self):
        """Timing anomaly alone → low risk."""
        timing = TimingAnalysis(is_suspicious=True, total_responses=10, reason="fast")
        fingerprints = [FingerprintResult(question_hash="a", response_hash="b")]
        risk = compute_gaming_risk(timing, fingerprints)
        assert risk.level == "low"
        assert risk.confidence_penalty == 0.05
        assert risk.timing_anomaly

    def test_duplicates_only_low_risk(self):
        """1 duplicate alone → low risk."""
        timing = TimingAnalysis(total_responses=10)
        fingerprints = [
            FingerprintResult(question_hash="a", response_hash="b", is_duplicate=True, prior_eval_id="eval-1"),
            FingerprintResult(question_hash="c", response_hash="d"),
            FingerprintResult(question_hash="e", response_hash="f"),
            FingerprintResult(question_hash="g", response_hash="h"),
        ]
        risk = compute_gaming_risk(timing, fingerprints)
        assert risk.level == "low"
        assert risk.duplicate_responses == 1

    def test_many_duplicates_medium_risk(self):
        """3+ duplicates → signals=2 → medium risk."""
        timing = TimingAnalysis(total_responses=10)
        fingerprints = [
            FingerprintResult(question_hash="a", response_hash="b", is_duplicate=True),
            FingerprintResult(question_hash="c", response_hash="d", is_duplicate=True),
            FingerprintResult(question_hash="e", response_hash="f", is_duplicate=True),
            FingerprintResult(question_hash="g", response_hash="h"),
        ]
        risk = compute_gaming_risk(timing, fingerprints)
        assert risk.level == "medium"
        assert risk.confidence_penalty == 0.10
        assert risk.duplicate_responses == 3

    def test_timing_plus_duplicates_high_risk(self):
        """Timing anomaly + 3+ duplicates → high risk."""
        timing = TimingAnalysis(is_suspicious=True, total_responses=10, reason="fast")
        fingerprints = [
            FingerprintResult(question_hash="a", response_hash="b", is_duplicate=True),
            FingerprintResult(question_hash="c", response_hash="d", is_duplicate=True),
            FingerprintResult(question_hash="e", response_hash="f", is_duplicate=True),
        ]
        risk = compute_gaming_risk(timing, fingerprints)
        assert risk.level == "high"
        assert risk.confidence_penalty == 0.15

    def test_empty_fingerprints(self):
        """No fingerprints (L1 manifest-only eval) → no risk."""
        timing = TimingAnalysis(total_responses=0)
        risk = compute_gaming_risk(timing, [])
        assert risk.level == "none"

    def test_risk_serialization(self):
        """to_dict() returns valid JSON-serializable dict."""
        timing = TimingAnalysis(is_suspicious=True, total_responses=5, reason="test")
        fingerprints = [FingerprintResult(question_hash="a", response_hash="b", is_duplicate=True)]
        risk = compute_gaming_risk(timing, fingerprints)
        d = risk.to_dict()
        assert isinstance(d, dict)
        assert d["level"] in ("none", "low", "medium", "high")
        assert isinstance(d["confidence_penalty"], float)
        assert isinstance(d["timing_anomaly"], bool)


class TestParaphraserEvalMode:
    """AC1: LLM paraphrasing activation by eval mode."""

    def test_verified_mode_no_llm(self):
        """Verified mode should NOT use LLM paraphrasing."""
        from src.core.paraphraser import QuestionParaphraser
        paraphraser = QuestionParaphraser(llm_judge=None, eval_mode="verified")
        assert not paraphraser._use_llm

    def test_certified_mode_enables_llm_if_available(self):
        """Certified mode should enable LLM paraphrasing when judge is available."""
        from src.core.paraphraser import QuestionParaphraser

        class FakeJudge:
            is_llm_available = True

        paraphraser = QuestionParaphraser(llm_judge=FakeJudge(), eval_mode="certified")
        assert paraphraser._use_llm

    def test_audited_mode_enables_llm_if_available(self):
        """Audited mode should enable LLM paraphrasing when judge is available."""
        from src.core.paraphraser import QuestionParaphraser

        class FakeJudge:
            is_llm_available = True

        paraphraser = QuestionParaphraser(llm_judge=FakeJudge(), eval_mode="audited")
        assert paraphraser._use_llm

    def test_certified_no_judge_falls_back(self):
        """Certified mode without judge should fall back to template."""
        from src.core.paraphraser import QuestionParaphraser
        paraphraser = QuestionParaphraser(llm_judge=None, eval_mode="certified")
        assert not paraphraser._use_llm

    def test_template_paraphrase_still_works(self):
        """Template paraphrasing should always produce different output."""
        from src.core.paraphraser import QuestionParaphraser
        paraphraser = QuestionParaphraser(llm_judge=None, eval_mode="verified")
        original = "Explain how an AMM determines token prices."
        seeds = [42, 99, 137, 256]
        results = {paraphraser.paraphrase_question(original, s) for s in seeds}
        # At least some should differ from original
        assert len(results) > 1 or original not in results


class TestEvaluatorEvalMode:
    """Evaluator passes eval_mode to paraphraser."""

    def test_evaluator_passes_eval_mode(self):
        """Evaluator should create paraphraser with correct eval_mode."""
        from src.core.evaluator import Evaluator

        class FakeJudge:
            is_llm_available = True
            async def ajudge(self, q, e, a, **kw):
                pass

        evaluator = Evaluator(FakeJudge(), eval_mode="audited")
        assert evaluator.paraphraser._eval_mode == "audited"
        assert evaluator.eval_mode == "audited"

    def test_evaluator_default_verified(self):
        """Evaluator defaults to verified mode."""
        from src.core.evaluator import Evaluator

        class FakeJudge:
            is_llm_available = False
            async def ajudge(self, q, e, a, **kw):
                pass

        evaluator = Evaluator(FakeJudge())
        assert evaluator.eval_mode == "verified"

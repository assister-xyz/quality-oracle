"""
Multi-judge consensus evaluation (CollabEval pattern).

Runs 2-3 diverse LLM judges in parallel, aggregates via median/agreement.
Dramatically reduces single-judge bias (66% → 85% human agreement).

Cost optimizations (Trust or Escalate pattern, ICLR 2025):
- Single-judge early exit for decisive scores (>= 90 or <= 15)
- Tighter agreement threshold for 2-judge consensus
- Fuzzy-first routing for simple test types (error_handling, boundary, type_coercion)

Fallback: if fewer than min_judges respond, use best available result.
"""
import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional

from src.core.llm_judge import LLMJudge, JudgeResult, JudgeMetrics

logger = logging.getLogger(__name__)

# Confidence-based cascade thresholds
# Scores at these extremes are highly unlikely to change with more judges
SINGLE_JUDGE_HIGH_THRESHOLD = 90  # Score >= 90: clearly passing, skip 2nd judge
SINGLE_JUDGE_LOW_THRESHOLD = 15   # Score <= 15: clearly failing, skip 2nd judge


@dataclass
class ConsensusResult:
    """Result from multi-judge consensus."""
    score: int
    explanation: str
    method: str  # "consensus", "majority", "single", "fuzzy"
    individual_scores: List[int]
    individual_methods: List[str]
    agreement: bool  # Did judges agree within threshold?
    judges_used: int
    latency_ms: int = 0
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


def _build_judges_from_settings() -> List[LLMJudge]:
    """Build a list of diverse LLM judges from available API keys.

    Priority order optimized for cost: free providers first, paid last.
    The first 3 judges are used for consensus (judge 1, judge 2, tiebreaker).
    Keeping paid providers at the end ensures they're only used when free
    providers are exhausted.
    """
    from src.config import settings

    judges = []

    # --- Free providers first (consensus judges 1-3 should be free) ---

    # Priority 1: Cerebras (free: 1M TPD, fast)
    if settings.cerebras_api_key:
        judges.append(LLMJudge(
            api_key=settings.cerebras_api_key,
            model=settings.cerebras_model,
            provider="cerebras",
            base_url=settings.cerebras_base_url,
        ))

    # Priority 2: Groq (free: 500K TPD, fast)
    if settings.groq_api_key:
        judges.append(LLMJudge(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            provider="groq",
            base_url="https://api.groq.com/openai/v1",
        ))

    # Priority 3: OpenRouter (free: Qwen3 80B, 200 RPD)
    if settings.openrouter_api_key:
        judges.append(LLMJudge(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            provider="openrouter",
            base_url=settings.openrouter_base_url,
        ))

    # Priority 4: Gemini (free: 250 RPD, high quality)
    if settings.gemini_api_key:
        judges.append(LLMJudge(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            provider="gemini",
            base_url=settings.gemini_base_url,
        ))

    # Priority 5: Mistral (free: 2 RPM, slow)
    if settings.mistral_api_key:
        judges.append(LLMJudge(
            api_key=settings.mistral_api_key,
            model=settings.mistral_model,
            provider="mistral",
            base_url=settings.mistral_base_url,
        ))

    # --- Paid providers last (only used if free providers exhausted) ---

    # Priority 6: DeepSeek (credits, cheap)
    if settings.deepseek_api_key:
        judges.append(LLMJudge(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            provider="deepseek",
            base_url=settings.deepseek_base_url,
        ))

    # Priority 7: OpenAI (paid, highest quality — only as last resort)
    if settings.openai_api_key:
        judges.append(LLMJudge(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            provider="openai",
            base_url=settings.openai_base_url,
        ))

    return judges


class ConsensusJudge:
    """
    Multi-judge consensus evaluator.

    Runs up to 3 diverse judges in parallel. Aggregation:
    - All 3 agree (within threshold): median score, high confidence
    - 2 of 3 agree: take agreeing pair's median
    - All disagree: weighted average, flag for review
    - Early termination: if first 2 agree, skip 3rd
    """

    def __init__(
        self,
        judges: Optional[List[LLMJudge]] = None,
        max_judges: int = 3,
        agreement_threshold: int = 15,
        min_judges: int = 2,
    ):
        if judges is not None:
            self._judges = judges[:max_judges]
        else:
            self._judges = _build_judges_from_settings()[:max_judges]

        self._max_judges = max_judges
        self._agreement_threshold = agreement_threshold
        self._min_judges = min_judges
        self._fuzzy_judge = LLMJudge()  # No API key = fuzzy fallback

        self.metrics = JudgeMetrics()
        self._cascade_exits = 0  # Track confidence-based cascade early exits

        available = [j.provider for j in self._judges if j.is_llm_available]
        logger.info(
            f"ConsensusJudge: {len(available)} LLM judges available: {available}. "
            f"Min={min_judges}, threshold={agreement_threshold}"
        )

    @property
    def judges_available(self) -> int:
        return sum(1 for j in self._judges if j.is_llm_available)

    @property
    def is_llm_available(self) -> bool:
        """True if at least one LLM judge is available."""
        return self.judges_available > 0

    @property
    def is_consensus_possible(self) -> bool:
        return self.judges_available >= self._min_judges

    def reset_keys(self):
        """Reset all exhausted API keys across judges. Call between evaluations."""
        for j in self._judges:
            for rotator in [j._primary_rotator, j._fallback_rotator, j._fallback2_rotator]:
                if rotator:
                    rotator.reset_exhausted()

    def log_metrics(self):
        """Log optimization metrics summary. Call at end of evaluation."""
        m = self.metrics
        if m.total_judged == 0:
            return
        m.fuzzy_routed + m.cache_hits
        max_calls = m.total_judged * self._max_judges  # worst case: all judges for all items
        actual_calls = m.llm_calls
        pct_saved = f"{(1 - actual_calls / max_calls) * 100:.0f}%" if max_calls else "0%"
        logger.info(
            f"[Optimization] {m.total_judged} items judged: "
            f"{m.llm_calls} LLM calls (of {max_calls} max), "
            f"{m.fuzzy_routed} fuzzy-routed, {self._cascade_exits} cascade exits, "
            f"{m.cache_hits} cached | {pct_saved} LLM calls saved"
        )

    async def ajudge(self, question: str, expected: str, answer: str, test_type: str = "") -> JudgeResult:
        """Judge with consensus. Returns JudgeResult for backward compatibility."""
        result = await self.ajudge_consensus(question, expected, answer, test_type=test_type)
        return JudgeResult(
            score=result.score,
            explanation=result.explanation,
            method=result.method,
            cached=result.cached,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def ajudge_consensus(
        self, question: str, expected: str, answer: str, test_type: str = ""
    ) -> ConsensusResult:
        """
        Run multi-judge consensus evaluation.

        Strategy:
        0. If test_type is fuzzy-routable, skip all LLM judges (use fuzzy scorer)
        1. Run first 2 judges in parallel
        2. If they agree → return immediately (early termination)
        3. If they disagree → run 3rd judge as tiebreaker
        4. Aggregate with median/majority logic
        """
        from src.core.llm_judge import FUZZY_ROUTABLE_TEST_TYPES

        self.metrics.total_judged += 1

        # Optimization: route simple test types directly to fuzzy scorer
        if test_type in FUZZY_ROUTABLE_TEST_TYPES:
            result = self._fuzzy_judge._judge_fuzzy(question, expected, answer)
            result.method = "fuzzy_routed"
            self.metrics.fuzzy_routed += 1
            return ConsensusResult(
                score=result.score,
                explanation=result.explanation,
                method="fuzzy_routed",
                individual_scores=[result.score],
                individual_methods=["fuzzy_routed"],
                agreement=True,
                judges_used=0,
                latency_ms=result.latency_ms,
            )

        llm_judges = [j for j in self._judges if j.is_llm_available]

        if len(llm_judges) < self._min_judges:
            # Not enough LLM judges — fall back to single best or fuzzy
            if llm_judges:
                result = await llm_judges[0].ajudge(question, expected, answer)
                return ConsensusResult(
                    score=result.score,
                    explanation=result.explanation,
                    method="single",
                    individual_scores=[result.score],
                    individual_methods=[result.method],
                    agreement=True,
                    judges_used=1,
                    latency_ms=result.latency_ms,
                )
            else:
                result = self._fuzzy_judge._judge_fuzzy(question, expected, answer)
                return ConsensusResult(
                    score=result.score,
                    explanation=result.explanation,
                    method="fuzzy",
                    individual_scores=[result.score],
                    individual_methods=["fuzzy"],
                    agreement=True,
                    judges_used=1,
                    latency_ms=result.latency_ms,
                )

        # Phase 0.5: Confidence-based cascade — single-judge early exit
        # Run 1st judge alone; if score is decisive, skip remaining judges
        self.metrics.llm_calls += 1
        first_result = await llm_judges[0].ajudge(question, expected, answer)
        self.metrics.record_tokens(first_result.provider or llm_judges[0].provider, first_result.input_tokens, first_result.output_tokens)
        if first_result.score >= SINGLE_JUDGE_HIGH_THRESHOLD or first_result.score <= SINGLE_JUDGE_LOW_THRESHOLD:
            self._cascade_exits += 1
            logger.debug(
                f"Cascade early exit: score={first_result.score} "
                f"(threshold: >={SINGLE_JUDGE_HIGH_THRESHOLD} or <={SINGLE_JUDGE_LOW_THRESHOLD})"
            )
            return ConsensusResult(
                score=first_result.score,
                explanation=f"Cascade ({first_result.method}): {first_result.explanation}",
                method="cascade",
                individual_scores=[first_result.score],
                individual_methods=[first_result.method],
                agreement=True,
                judges_used=1,
                latency_ms=first_result.latency_ms,
                input_tokens=first_result.input_tokens,
                output_tokens=first_result.output_tokens,
            )

        # Phase 1: Score was ambiguous — run 2nd judge
        if len(llm_judges) >= 2:
            try:
                self.metrics.llm_calls += 1
                second_result = await llm_judges[1].ajudge(question, expected, answer)
                self.metrics.record_tokens(second_result.provider or llm_judges[1].provider, second_result.input_tokens, second_result.output_tokens)
                valid_results = [first_result, second_result]
            except Exception as e:
                logger.warning(f"Second judge failed: {e}")
                valid_results = [first_result]
        else:
            valid_results = [first_result]

        if len(valid_results) == 1:
            r = valid_results[0]
            return ConsensusResult(
                score=r.score,
                explanation=r.explanation,
                method="single",
                individual_scores=[r.score],
                individual_methods=[r.method],
                agreement=True,
                judges_used=1,
                latency_ms=r.latency_ms,
            )

        # Check agreement between first 2
        scores = [r.score for r in valid_results]
        if abs(scores[0] - scores[1]) <= self._agreement_threshold:
            # Early termination — judges agree
            median_score = int(statistics.median(scores))
            total_latency = max(r.latency_ms for r in valid_results)
            total_in = sum(r.input_tokens for r in valid_results)
            total_out = sum(r.output_tokens for r in valid_results)
            return ConsensusResult(
                score=median_score,
                explanation=f"Consensus ({valid_results[0].method}+{valid_results[1].method}): {valid_results[0].explanation}",
                method="consensus",
                individual_scores=scores,
                individual_methods=[r.method for r in valid_results],
                agreement=True,
                judges_used=2,
                latency_ms=total_latency,
                input_tokens=total_in,
                output_tokens=total_out,
            )

        # Phase 2: Disagreement — run 3rd judge as tiebreaker (if available)
        if len(llm_judges) >= 3:
            try:
                self.metrics.llm_calls += 1
                third_result = await llm_judges[2].ajudge(question, expected, answer)
                self.metrics.record_tokens(third_result.provider or llm_judges[2].provider, third_result.input_tokens, third_result.output_tokens)
                valid_results.append(third_result)
                scores.append(third_result.score)
            except Exception as e:
                logger.warning(f"Third judge failed: {e}")

        return self._aggregate(valid_results, scores)

    def _aggregate(
        self, results: List[JudgeResult], scores: List[int]
    ) -> ConsensusResult:
        """Aggregate 2-3 judge results into consensus."""
        total_latency = max(r.latency_ms for r in results)
        total_in = sum(r.input_tokens for r in results)
        total_out = sum(r.output_tokens for r in results)

        if len(scores) == 3:
            # Check for majority agreement (any 2 of 3 within threshold)
            pairs = [(0, 1), (0, 2), (1, 2)]
            for i, j in pairs:
                if abs(scores[i] - scores[j]) <= self._agreement_threshold:
                    agreed_scores = [scores[i], scores[j]]
                    median_score = int(statistics.median(agreed_scores))
                    return ConsensusResult(
                        score=median_score,
                        explanation=f"Majority ({results[i].method}+{results[j].method}): {results[i].explanation}",
                        method="majority",
                        individual_scores=scores,
                        individual_methods=[r.method for r in results],
                        agreement=True,
                        judges_used=3,
                        latency_ms=total_latency,
                        input_tokens=total_in,
                        output_tokens=total_out,
                    )

            # All 3 disagree — take median, flag no agreement
            median_score = int(statistics.median(scores))
            return ConsensusResult(
                score=median_score,
                explanation=f"No consensus (scores: {scores}): {results[0].explanation}",
                method="consensus",
                individual_scores=scores,
                individual_methods=[r.method for r in results],
                agreement=False,
                judges_used=3,
                latency_ms=total_latency,
                input_tokens=total_in,
                output_tokens=total_out,
            )

        # 2 judges, disagreed — take average
        avg_score = int(sum(scores) / len(scores))
        return ConsensusResult(
            score=avg_score,
            explanation=f"Split decision (scores: {scores}): {results[0].explanation}",
            method="consensus",
            individual_scores=scores,
            individual_methods=[r.method for r in results],
            agreement=False,
            judges_used=2,
            latency_ms=total_latency,
            input_tokens=total_in,
            output_tokens=total_out,
        )

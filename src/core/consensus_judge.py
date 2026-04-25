"""
Multi-judge consensus evaluation (CollabEval pattern).

Runs 2-3 diverse LLM judges in parallel, aggregates via median/agreement.
Dramatically reduces single-judge bias (66% → 85% human agreement).

Cost optimizations (Trust or Escalate pattern, ICLR 2025):
- Cascade-gate early exit for decisive scores (>= 85 or <= 20) WITH cross-family
  confirmation (QO-061) — primary judge alone is no longer enough; a confirmer
  from a DIFFERENT model family must agree within 10 points or the eval falls
  through to full 3-judge consensus.
- Tighter agreement threshold for 2-judge consensus.
- Fuzzy-first routing for simple test types (error_handling, boundary, type_coercion).

Family-diverse panel (QO-061):
The free-tier panel uses three distinct model families — Cerebras-Llama (meta_llama),
Gemini Flash (google_gemini), Qwen 80B (alibaba_qwen) — to defeat the same-family
self-preference bug found in the previous Cerebras-Llama + Groq-Llama + Qwen panel.

Fallback: if fewer than min_judges respond, use best available result. If the
panel cannot supply two distinct families (Gemini quota exhausted, Mistral/Qwen
also unavailable, only Cerebras-Llama left), `InsufficientPanelDiversity` is raised.
"""
import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from src.core.llm_judge import LLMJudge, JudgeResult, JudgeMetrics

logger = logging.getLogger(__name__)

# Cascade thresholds — at these extremes a SECOND judge from a different family
# must confirm within CASCADE_CONFIRMATION_TOLERANCE points.
SINGLE_JUDGE_HIGH_THRESHOLD = 85  # Score >= 85: decisive-pass, requires confirmer
SINGLE_JUDGE_LOW_THRESHOLD = 20   # Score <= 20: decisive-fail, requires confirmer

# Cross-family confirmation gate (QO-061): confirmer must be within this many
# points of the primary score for the cascade early-exit to be accepted.
CASCADE_CONFIRMATION_TOLERANCE = 10


class InsufficientPanelDiversity(RuntimeError):
    """Raised when the active judge panel cannot supply two distinct model families.

    QO-061 AC3: if the panel cannot supply two distinct families (e.g. Gemini
    quota exhausted, Qwen exhausted, only Cerebras-Llama left), the eval is
    paused rather than silently degrading to a same-family panel.
    """


@dataclass
class JudgeConfig:
    """Declarative judge slot — provider + model + family + role.

    Used by `_build_judges_from_settings()` to materialize an `LLMJudge` once
    the matching API key is available. The `family` field is the SOLE source of
    truth for cross-family cascade-gate checks (NOT the slot index).
    """
    provider: str
    model: str
    family: str
    role: str  # "primary" | "secondary" | "tiebreaker"
    base_url: Optional[str] = None


@dataclass
class ConsensusResult:
    """Result from multi-judge consensus."""
    score: int
    explanation: str
    method: str  # "consensus", "majority", "single", "fuzzy", "cascade"
    individual_scores: List[int]
    individual_methods: List[str]
    agreement: bool  # Did judges agree within threshold?
    judges_used: int
    latency_ms: int = 0
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    judge_anomalies: List[str] = field(default_factory=list)


# ── QO-061 free-tier panel — 3 distinct model families ──────────────────────
# Order: primary → secondary (cross-family confirmer) → tiebreaker.
# Per AC1 + AC2: all three families MUST be distinct.
FREE_TIER_PANEL: List[JudgeConfig] = [
    JudgeConfig(
        provider="cerebras",
        model="llama3.1-8b",
        family="meta_llama",
        role="primary",
    ),
    JudgeConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        family="google_gemini",
        role="secondary",
    ),
    JudgeConfig(
        provider="openrouter",
        model="qwen/qwen3-next-80b-a3b-instruct:free",
        family="alibaba_qwen",
        role="tiebreaker",
    ),
]

# Fallback configs used when Gemini quota is exhausted (AC11). Selected ONLY
# from families NOT yet present in the active panel — never silently swap to
# the same family.
GEMINI_FALLBACK_PANEL: List[JudgeConfig] = [
    JudgeConfig(
        provider="mistral",
        model="mistral-large-latest",
        family="mistral",
        role="secondary",
    ),
]


def _build_judge_from_config(cfg: JudgeConfig) -> Optional[LLMJudge]:
    """Materialize an LLMJudge from a JudgeConfig if the matching API key exists."""
    from src.config import settings

    key_attr = f"{cfg.provider}_api_key"
    key = getattr(settings, key_attr, None) or ""
    if not key:
        return None

    base_url = cfg.base_url
    model = cfg.model
    if cfg.provider == "cerebras":
        base_url = base_url or settings.cerebras_base_url
    elif cfg.provider == "gemini":
        base_url = base_url or settings.gemini_base_url
    elif cfg.provider == "openrouter":
        base_url = base_url or settings.openrouter_base_url
    elif cfg.provider == "mistral":
        base_url = base_url or settings.mistral_base_url
    elif cfg.provider == "groq":
        base_url = base_url or "https://api.groq.com/openai/v1"

    return LLMJudge(
        api_key=key,
        model=model,
        provider=cfg.provider,
        base_url=base_url or "",
        family=cfg.family,
    )


def _build_judges_from_settings() -> List[LLMJudge]:
    """Build the family-diverse free-tier judge panel (QO-061).

    Order:
        1. Cerebras-Llama-3.1-8B (meta_llama) — primary
        2. Gemini Flash         (google_gemini) — secondary / cascade confirmer
        3. Qwen 80B             (alibaba_qwen) — tiebreaker

    Gemini-quota fallback (AC11): if Gemini key missing OR exhausted, slot 2
    is filled by Mistral (or Qwen if Mistral missing) — never by another
    meta_llama judge. The cascade gate continues to require a non-meta_llama
    confirmer.

    Paid providers (DeepSeek/OpenAI/Anthropic) are NOT used in the free-tier
    panel — they're only available via explicit JudgeConfig injection.
    """
    judges: List[LLMJudge] = []
    seen_families: set = set()

    for cfg in FREE_TIER_PANEL:
        judge = _build_judge_from_config(cfg)
        if judge:
            judges.append(judge)
            seen_families.add(judge.family)
        elif cfg.role == "secondary":
            # Gemini missing/exhausted → fall back to alternate non-Llama family
            for fb_cfg in GEMINI_FALLBACK_PANEL:
                if fb_cfg.family in seen_families:
                    continue
                fb_judge = _build_judge_from_config(fb_cfg)
                if fb_judge:
                    logger.warning(
                        f"Gemini secondary slot unavailable; falling back to "
                        f"{fb_cfg.provider}/{fb_cfg.model} (family={fb_cfg.family})"
                    )
                    judges.append(fb_judge)
                    seen_families.add(fb_judge.family)
                    break

    return judges


def _validate_panel_diversity(judges: List[LLMJudge]) -> None:
    """Enforce AC2: every judge in the active panel must belong to a unique family.

    Raises:
        InsufficientPanelDiversity: if any two judges share a family, OR if
        the panel cannot supply at least two distinct families.
    """
    families = [j.family for j in judges if j.is_llm_available]
    unique_families = set(families)
    if len(unique_families) < 2:
        raise InsufficientPanelDiversity(
            f"Active panel must supply ≥2 distinct model families; "
            f"got {len(unique_families)}: {sorted(unique_families)}"
        )
    if len(unique_families) != len(families):
        raise InsufficientPanelDiversity(
            f"Panel has duplicate families: {families}. "
            "Free-tier panel requires Cerebras-Llama + Gemini + Qwen (3 distinct)."
        )


def _pick_confirmer(primary: LLMJudge, panel: List[LLMJudge]) -> Optional[LLMJudge]:
    """Pick the cross-family confirmer for cascade-gate early-exit.

    QO-061 AC3: the confirmer MUST be from a different family than `primary`,
    and MUST be available. Falls through to None if no eligible confirmer.
    """
    for j in panel:
        if j is primary:
            continue
        if not j.is_llm_available:
            continue
        if j.family != primary.family:
            return j
    return None


class ConsensusJudge:
    """
    Multi-judge consensus evaluator with QO-061 cross-family cascade gate.

    Strategy:
    - Run primary judge.
    - If primary score is decisive (>=85 or <=20):
        * Pick a confirmer from a DIFFERENT family.
        * If confirmer agrees within 10 pts → accept, single early exit.
        * Otherwise → fall through to full 3-judge consensus.
    - Otherwise → standard 2-judge consensus (3rd as tiebreaker on disagreement).
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

        available = [(j.provider, j.family) for j in self._judges if j.is_llm_available]
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

    @property
    def panel_families(self) -> List[str]:
        """Families of judges currently in the active panel (in order)."""
        return [j.family for j in self._judges if j.is_llm_available]

    def assert_panel_diversity(self) -> None:
        """Public AC2 check — call before running an eval to fail fast.

        Raises InsufficientPanelDiversity if the panel cannot supply 2 families.
        """
        _validate_panel_diversity(self._judges)

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
        Run multi-judge consensus evaluation with QO-061 cross-family cascade gate.

        Strategy:
        0. If test_type is fuzzy-routable, skip all LLM judges (use fuzzy scorer).
        1. Run primary (slot 0).
        2. If decisive score → pick cross-family confirmer; if agree within 10 → exit.
        3. Otherwise → run remaining judges for full consensus.
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

        # ── Phase 1: primary judge ────────────────────────────────────────
        self.metrics.llm_calls += 1
        first_result = await llm_judges[0].ajudge(question, expected, answer)
        self.metrics.record_tokens(
            first_result.provider or llm_judges[0].provider,
            first_result.input_tokens,
            first_result.output_tokens,
        )

        decisive = (
            first_result.score >= SINGLE_JUDGE_HIGH_THRESHOLD
            or first_result.score <= SINGLE_JUDGE_LOW_THRESHOLD
        )

        # ── Phase 2: cascade gate (QO-061) ────────────────────────────────
        # Decisive score must be confirmed by a DIFFERENT-family judge within
        # CASCADE_CONFIRMATION_TOLERANCE points before the early-exit fires.
        if decisive:
            confirmer = _pick_confirmer(llm_judges[0], llm_judges)
            if confirmer is None:
                # No cross-family confirmer available — fall through to full consensus.
                # Do NOT accept the primary score alone; that would re-introduce the
                # null-model attack regime.
                logger.debug(
                    f"Cascade gate: no cross-family confirmer for primary={llm_judges[0].family}; "
                    "falling through to full consensus."
                )
            else:
                try:
                    self.metrics.llm_calls += 1
                    confirm_result = await confirmer.ajudge(question, expected, answer)
                    self.metrics.record_tokens(
                        confirm_result.provider or confirmer.provider,
                        confirm_result.input_tokens,
                        confirm_result.output_tokens,
                    )
                    if abs(confirm_result.score - first_result.score) <= CASCADE_CONFIRMATION_TOLERANCE:
                        # Cross-family agreement on a decisive score — accept early exit
                        self._cascade_exits += 1
                        avg = int((first_result.score + confirm_result.score) / 2)
                        return ConsensusResult(
                            score=avg,
                            explanation=(
                                f"Cascade ({first_result.method}+{confirm_result.method}, "
                                f"{llm_judges[0].family}+{confirmer.family}): "
                                f"{first_result.explanation}"
                            ),
                            method="cascade",
                            individual_scores=[first_result.score, confirm_result.score],
                            individual_methods=[first_result.method, confirm_result.method],
                            agreement=True,
                            judges_used=2,
                            latency_ms=max(first_result.latency_ms, confirm_result.latency_ms),
                            input_tokens=first_result.input_tokens + confirm_result.input_tokens,
                            output_tokens=first_result.output_tokens + confirm_result.output_tokens,
                        )
                    # Cross-family disagreement on a decisive score → escalate.
                    logger.debug(
                        f"Cascade gate disagreement: primary={first_result.score} "
                        f"({llm_judges[0].family}) vs confirmer={confirm_result.score} "
                        f"({confirmer.family}); escalating to full consensus."
                    )
                    # Hold confirmer result for full-consensus phase
                    valid_results = [first_result, confirm_result]
                    scores = [first_result.score, confirm_result.score]
                    return await self._full_consensus_after_disagreement(
                        question, expected, answer, llm_judges,
                        valid_results, scores, used_judges={llm_judges[0], confirmer},
                    )
                except Exception as e:
                    logger.warning(f"Cascade confirmer failed: {e}; falling through to full consensus.")

        # ── Phase 3: standard 2-judge consensus (non-decisive) ────────────
        if len(llm_judges) >= 2:
            try:
                self.metrics.llm_calls += 1
                second_result = await llm_judges[1].ajudge(question, expected, answer)
                self.metrics.record_tokens(
                    second_result.provider or llm_judges[1].provider,
                    second_result.input_tokens,
                    second_result.output_tokens,
                )
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

        # Phase 4: Disagreement — run 3rd judge as tiebreaker (if available)
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

    async def _full_consensus_after_disagreement(
        self,
        question: str,
        expected: str,
        answer: str,
        llm_judges: List[LLMJudge],
        valid_results: List[JudgeResult],
        scores: List[int],
        used_judges,
    ) -> ConsensusResult:
        """Continue to full 3-judge consensus after cascade-gate disagreement.

        We've already invoked primary + cross-family confirmer (each from a
        different family). Run any remaining judges to round out a 3-judge
        committee, then aggregate.
        """
        for judge in llm_judges:
            if judge in used_judges:
                continue
            try:
                self.metrics.llm_calls += 1
                r = await judge.ajudge(question, expected, answer)
                self.metrics.record_tokens(
                    r.provider or judge.provider, r.input_tokens, r.output_tokens
                )
                valid_results.append(r)
                scores.append(r.score)
                if len(valid_results) >= 3:
                    break
            except Exception as e:
                logger.warning(f"Tertiary judge failed: {e}")

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

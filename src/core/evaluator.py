"""
AgentTrust evaluation engine.

Orchestrates the full evaluation flow:
Level 1: Manifest validation
Level 2: Functional testing (challenge-response via MCP Client)
Level 3: Domain expert testing (calibrated question bank)
"""
import hashlib
import logging
import statistics
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from src.core.paraphraser import QuestionParaphraser
from src.core.difficulty_calibration import DifficultyTracker
from src.core.scoring import apply_style_adjustment
from src.core.question_pools import (
    QuestionSelector,
    ALL_QUESTIONS,
    determine_tier,
)

logger = logging.getLogger(__name__)

# ── Early termination constants ─────────────────────────────────────────────

EARLY_EXIT_MIN_SCORES = 4       # Minimum scores before considering exit
EARLY_EXIT_HIGH_THRESHOLD = 85  # All scores above this → positive exit
EARLY_EXIT_LOW_THRESHOLD = 30   # All scores below this → negative exit
EARLY_EXIT_MIN_LOW = 3          # Minimum low scores before negative exit


class ManifestValidationResult:
    """Result of Level 1 manifest validation."""
    def __init__(self):
        self.score: int = 0
        self.checks: Dict[str, bool] = {}
        self.warnings: List[str] = []

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "checks": self.checks,
            "warnings": self.warnings,
        }


class EvaluationResult:
    """Result of a complete evaluation."""
    def __init__(self):
        self.overall_score: int = 0
        self.tier: str = "failed"
        self.confidence: float = 0.0
        self.tool_scores: Dict[str, dict] = {}
        self.domain_scores: Dict[str, dict] = {}
        self.questions_asked: int = 0
        self.questions_answered: int = 0
        self.judge_responses: List[dict] = []
        self.manifest_result: Optional[ManifestValidationResult] = None
        self.duration_ms: int = 0
        self.result_hash: str = ""

        # Multi-dimensional scoring (6 axes)
        self.dimensions: Optional[Dict[str, dict]] = None
        self.safety_report: Optional[dict] = None
        self.process_quality_report: Optional[dict] = None
        self.latency_stats: Optional[Dict[str, int]] = None

        # Style control
        self.style_report: Optional[dict] = None

        # Anti-gaming signals
        self.gaming_risk: Optional[dict] = None

        # IRT ability estimation
        self.irt_theta: Optional[float] = None
        self.irt_se: Optional[float] = None
        self.confidence_interval: Optional[Dict[str, float]] = None

        # Token usage tracking
        self.token_usage: Optional[Dict[str, Any]] = None
        self.cost_usd: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "overall_score": self.overall_score,
            "tier": self.tier,
            "confidence": self.confidence,
            "tool_scores": self.tool_scores,
            "domain_scores": self.domain_scores,
            "questions_asked": self.questions_asked,
            "questions_answered": self.questions_answered,
            "manifest": self.manifest_result.to_dict() if self.manifest_result else None,
            "duration_ms": self.duration_ms,
            "result_hash": self.result_hash,
        }
        if self.dimensions:
            d["dimensions"] = self.dimensions
        if self.safety_report:
            d["safety"] = self.safety_report
        if self.process_quality_report:
            d["process_quality"] = self.process_quality_report
        if self.latency_stats:
            d["latency"] = self.latency_stats
        if self.style_report:
            d["style_report"] = self.style_report
        if self.gaming_risk:
            d["gaming_risk"] = self.gaming_risk
        if self.irt_theta is not None:
            d["irt_theta"] = self.irt_theta
        if self.irt_se is not None:
            d["irt_se"] = self.irt_se
        if self.confidence_interval is not None:
            d["confidence_interval"] = self.confidence_interval
        if self.token_usage is not None:
            d["token_usage"] = self.token_usage
        if self.cost_usd is not None:
            d["cost_usd"] = self.cost_usd
        return d


class Evaluator:
    """
    Core evaluation engine for AgentTrust.

    Supports 3 levels of evaluation with increasing depth.
    Accepts LLMJudge or ConsensusJudge (both implement ajudge()).
    """

    def __init__(self, llm_judge, paraphrase: bool = True, eval_mode: str = "verified", irt_service=None):
        """Init with any judge that has an ajudge(question, expected, answer) method."""
        self.llm_judge = llm_judge
        self.question_selector = QuestionSelector()
        self.eval_mode = eval_mode
        self.paraphraser = QuestionParaphraser(llm_judge, eval_mode=eval_mode) if paraphrase else None
        self.difficulty_tracker = DifficultyTracker()
        self.irt_service = irt_service

    def collect_token_usage(self) -> Dict[str, Any]:
        """Collect token usage from judge and paraphraser into a unified dict."""
        from src.config import calculate_total_cost

        usage: Dict[str, Any] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_provider": {},
            "by_phase": {},
        }

        # Collect from judge metrics
        judge = self.llm_judge
        if hasattr(judge, "metrics"):
            m = judge.metrics
            usage["total_input_tokens"] += m.total_input_tokens
            usage["total_output_tokens"] += m.total_output_tokens
            for prov, prov_usage in m.by_provider.items():
                if prov not in usage["by_provider"]:
                    usage["by_provider"][prov] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
                usage["by_provider"][prov]["input_tokens"] += prov_usage.get("input_tokens", 0)
                usage["by_provider"][prov]["output_tokens"] += prov_usage.get("output_tokens", 0)
                usage["by_provider"][prov]["calls"] += prov_usage.get("calls", 0)
            usage["by_phase"]["judging"] = {
                "input_tokens": m.total_input_tokens,
                "output_tokens": m.total_output_tokens,
            }

        # Collect from paraphraser
        if self.paraphraser and self.paraphraser.llm_calls > 0:
            para_in = self.paraphraser.total_input_tokens
            para_out = self.paraphraser.total_output_tokens
            usage["total_input_tokens"] += para_in
            usage["total_output_tokens"] += para_out
            usage["by_phase"]["paraphrasing"] = {
                "input_tokens": para_in,
                "output_tokens": para_out,
            }
            # Paraphraser uses the primary judge's provider
            prov = getattr(judge, "provider", "unknown")
            if prov not in usage["by_provider"]:
                usage["by_provider"][prov] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            usage["by_provider"][prov]["input_tokens"] += para_in
            usage["by_provider"][prov]["output_tokens"] += para_out
            usage["by_provider"][prov]["calls"] += self.paraphraser.llm_calls

        # Optimization metrics from judge
        optimization = {}
        if hasattr(judge, "metrics"):
            m = judge.metrics
            optimization["llm_calls"] = m.llm_calls
            optimization["fuzzy_routed"] = m.fuzzy_routed
            optimization["cache_hits"] = m.cache_hits
            optimization["total_judged"] = m.total_judged
        if hasattr(judge, "_cascade_exits"):
            optimization["cascade_exits"] = judge._cascade_exits
        if optimization:
            usage["optimization"] = optimization

        # Calculate cost (actual + shadow/market rate)
        cost_data = calculate_total_cost(usage["by_provider"])
        usage["cost_usd"] = cost_data["total_cost_usd"]
        usage["shadow_cost_usd"] = cost_data["shadow_cost_usd"]
        usage["cost_by_provider"] = cost_data["by_provider"]

        return usage

    def validate_manifest(self, manifest: dict) -> ManifestValidationResult:
        """Level 1: Validate MCP server manifest for completeness and quality."""
        result = ManifestValidationResult()
        checks = {}
        warnings = []

        # Check tools are defined
        tools = manifest.get("tools", [])
        checks["has_tools"] = len(tools) > 0
        if not checks["has_tools"]:
            warnings.append("No tools defined in manifest")

        # Check each tool has description
        tools_with_desc = sum(1 for t in tools if t.get("description"))
        checks["tools_have_descriptions"] = tools_with_desc == len(tools) if tools else False
        if tools and tools_with_desc < len(tools):
            warnings.append(f"{len(tools) - tools_with_desc}/{len(tools)} tools missing descriptions")

        # Check input schemas
        tools_with_schema = sum(1 for t in tools if t.get("inputSchema") or t.get("parameters"))
        checks["tools_have_schemas"] = tools_with_schema == len(tools) if tools else False
        if tools and tools_with_schema < len(tools):
            warnings.append(f"{len(tools) - tools_with_schema}/{len(tools)} tools missing input schemas")

        # Check server info
        checks["has_name"] = bool(manifest.get("name"))
        checks["has_version"] = bool(manifest.get("version"))
        checks["has_description"] = bool(manifest.get("description"))

        if not checks["has_name"]:
            warnings.append("Missing server name")
        if not checks["has_description"]:
            warnings.append("Missing server description")

        # Score calculation
        total_checks = len(checks)
        passed_checks = sum(1 for v in checks.values() if v)
        result.score = int((passed_checks / total_checks) * 100) if total_checks > 0 else 0
        result.checks = checks
        result.warnings = warnings

        return result

    async def evaluate_functional(
        self,
        target_id: str,
        tool_responses: Dict[str, List[dict]],
        manifest: Optional[dict] = None,
    ) -> EvaluationResult:
        """
        Level 2: Functional testing.

        Takes pre-collected tool responses and judges them.

        Args:
            target_id: ID of the target being evaluated
            tool_responses: Dict of tool_name -> list of {question, expected, answer}
            manifest: Optional manifest for Level 1 inclusion
        """
        start = time.time()
        result = EvaluationResult()

        # Run Level 1 if manifest provided
        if manifest:
            result.manifest_result = self.validate_manifest(manifest)

        # Anti-gaming: generate per-run paraphrase seed
        para_seed = self.paraphraser.generate_seed(target_id) if self.paraphraser else 0

        # Judge each tool's responses
        all_scores = []
        case_idx = 0
        total_responses = sum(len(r) for r in tool_responses.values())
        judged_count = 0
        judging_start = time.time()
        for tool_name, responses in tool_responses.items():
            tool_scores = []
            tests_passed = 0
            logger.info(f"[evaluate_functional] Judging tool '{tool_name}' ({len(responses)} responses)")

            for resp in responses:
                # Paraphrase question/expected for anti-gaming
                q = resp["question"]
                exp = resp["expected"]
                if self.paraphraser:
                    q = self.paraphraser.paraphrase_question(q, para_seed + case_idx)
                    exp = self.paraphraser.paraphrase_expected(exp, para_seed + case_idx)
                case_idx += 1

                judge_result = await self.llm_judge.ajudge(
                    q, exp, resp["answer"],
                    test_type=resp.get("test_type", ""),
                )
                judged_count += 1

                # Style control: penalize verbose/over-formatted responses
                response_text = resp.get("answer", "")
                style_adj = apply_style_adjustment(judge_result.score, response_text)
                adjusted_score = style_adj["adjusted_score"]

                tool_scores.append(adjusted_score)
                self.difficulty_tracker.record(
                    f"func_{tool_name}_{case_idx - 1}",
                    passed=adjusted_score >= 70,
                )
                logger.debug(f"[evaluate_functional] Judged {judged_count}/{total_responses}: {tool_name} score={adjusted_score} (raw={judge_result.score}, penalty={style_adj['style_penalty']}) via {judge_result.method}")
                if adjusted_score >= 50:
                    tests_passed += 1

                result.judge_responses.append({
                    "tool": tool_name,
                    "question": q,
                    "score": adjusted_score,
                    "raw_score": judge_result.score,
                    "style_penalty": style_adj["style_penalty"],
                    "style_features": style_adj["style_features"],
                    "explanation": judge_result.explanation,
                    "method": judge_result.method,
                    "test_type": resp.get("test_type", "unknown"),
                })

            avg_score = sum(tool_scores) / len(tool_scores) if tool_scores else 0
            result.tool_scores[tool_name] = {
                "score": int(avg_score),
                "tests_passed": tests_passed,
                "tests_total": len(responses),
            }
            all_scores.extend(tool_scores)

        judging_ms = int((time.time() - judging_start) * 1000)

        # Aggregate style report
        penalties = [jr.get("style_penalty", 0) for jr in result.judge_responses]
        penalized_count = sum(1 for p in penalties if p > 0)
        if penalties:
            result.style_report = {
                "total_penalty": round(sum(penalties), 2),
                "avg_penalty": round(sum(penalties) / len(penalties), 2),
                "penalized_responses": penalized_count,
                "total_responses": len(penalties),
            }

        # Aggregate
        result.questions_asked = len(all_scores)
        result.questions_answered = sum(1 for s in all_scores if s > 0)
        result.overall_score = int(sum(all_scores) / len(all_scores)) if all_scores else 0
        result.tier = determine_tier(result.overall_score)
        # Confidence from sample size (capped at 0.95), with variance penalty
        sample_conf = min(0.95, len(all_scores) / 30)
        if len(all_scores) >= 3:
            stdev = statistics.stdev(all_scores)
            variance_penalty = max(0.0, (stdev - 25) / 100)
            result.confidence = round(max(0.1, sample_conf - variance_penalty), 2)
        else:
            result.confidence = round(sample_conf, 2)
        total_ms = int((time.time() - start) * 1000)
        result.duration_ms = total_ms

        # Result hash for on-chain
        hash_data = f"{target_id}:{result.overall_score}:{result.questions_asked}:{int(time.time())}"
        result.result_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        # Collect token usage
        token_data = self.collect_token_usage()
        token_data["phase_timing_ms"] = {
            "judging_ms": judging_ms,
            "total_ms": total_ms,
        }
        result.token_usage = token_data
        result.cost_usd = token_data.get("cost_usd", 0.0)

        logger.info(
            f"Evaluation complete: {target_id} | "
            f"Score: {result.overall_score} | Tier: {result.tier} | "
            f"Questions: {result.questions_asked} | "
            f"Tokens: {token_data['total_input_tokens']}in/{token_data['total_output_tokens']}out | "
            f"Cost: ${token_data.get('cost_usd', 0):.6f}"
        )

        return result

    @staticmethod
    def _compute_progressive_confidence(all_scores: List[int]) -> float:
        """Compute confidence from accumulated scores with variance penalty."""
        if not all_scores:
            return 0.0
        sample_conf = min(0.95, len(all_scores) / 30)
        if len(all_scores) >= 3:
            stdev = statistics.stdev(all_scores)
            variance_penalty = max(0.0, (stdev - 25) / 100)
            return round(max(0.1, sample_conf - variance_penalty), 2)
        return round(sample_conf, 2)

    @staticmethod
    def _check_early_exit(all_scores: List[int]) -> Optional[str]:
        """Check if scores are decisive enough for early termination.

        Returns reason string if should exit, None otherwise.
        """
        if len(all_scores) >= EARLY_EXIT_MIN_SCORES:
            if all(s >= EARLY_EXIT_HIGH_THRESHOLD for s in all_scores):
                return f"positive_exit: all {len(all_scores)} scores >= {EARLY_EXIT_HIGH_THRESHOLD}"
        if len(all_scores) >= EARLY_EXIT_MIN_LOW:
            if all(s <= EARLY_EXIT_LOW_THRESHOLD for s in all_scores):
                return f"negative_exit: all {len(all_scores)} scores <= {EARLY_EXIT_LOW_THRESHOLD}"
        return None

    async def evaluate_functional_streaming(
        self,
        target_id: str,
        response_stream: AsyncGenerator[Tuple[str, dict, dict], None],
        manifest: Optional[dict] = None,
        cancel: Optional[object] = None,
        on_progress: Optional[Callable] = None,
    ) -> EvaluationResult:
        """Level 2 streaming: judge each response as it arrives.

        Args:
            target_id: ID of the target being evaluated
            response_stream: AsyncGenerator yielding (tool_name, test_case, response)
            manifest: Optional manifest for Level 1 inclusion
            cancel: CancellationToken for early termination
            on_progress: Optional callback(tool_name, case_idx, score, running_avg)
        """

        start = time.time()
        result = EvaluationResult()

        if manifest:
            result.manifest_result = self.validate_manifest(manifest)

        para_seed = self.paraphraser.generate_seed(target_id) if self.paraphraser else 0

        all_scores: List[int] = []
        tool_buckets: Dict[str, List[int]] = {}
        tool_passed: Dict[str, int] = {}
        tool_total: Dict[str, int] = {}
        case_idx = 0

        async for tool_name, case, response in response_stream:
            # Check cancellation
            if cancel and cancel.is_cancelled:
                logger.info(f"Streaming eval cancelled at case {case_idx}: {cancel.reason}")
                break

            # Paraphrase
            q = case["question"]
            exp = case["expected"]
            if self.paraphraser:
                q = self.paraphraser.paraphrase_question(q, para_seed + case_idx)
                exp = self.paraphraser.paraphrase_expected(exp, para_seed + case_idx)
            case_idx += 1

            # Judge immediately
            judge_result = await self.llm_judge.ajudge(
                q, exp, response["content"],
                test_type=case.get("test_type", ""),
            )
            score = judge_result.score
            all_scores.append(score)

            # Track per-tool
            if tool_name not in tool_buckets:
                tool_buckets[tool_name] = []
                tool_passed[tool_name] = 0
                tool_total[tool_name] = 0
            tool_buckets[tool_name].append(score)
            tool_total[tool_name] += 1
            if score >= 50:
                tool_passed[tool_name] += 1

            result.judge_responses.append({
                "tool": tool_name,
                "question": q,
                "score": score,
                "explanation": judge_result.explanation,
                "method": judge_result.method,
                "test_type": case.get("test_type", "unknown"),
            })

            # Progress callback
            running_avg = int(sum(all_scores) / len(all_scores))
            if on_progress:
                on_progress(tool_name, case_idx, score, running_avg)

            # Check early exit
            exit_reason = self._check_early_exit(all_scores)
            if exit_reason:
                logger.info(f"Early exit at case {case_idx}: {exit_reason}")
                if cancel:
                    cancel.cancel(exit_reason)
                break

        # Aggregate
        for tool_name, scores in tool_buckets.items():
            avg = sum(scores) / len(scores) if scores else 0
            result.tool_scores[tool_name] = {
                "score": int(avg),
                "tests_passed": tool_passed.get(tool_name, 0),
                "tests_total": tool_total.get(tool_name, 0),
            }

        result.questions_asked = len(all_scores)
        result.questions_answered = sum(1 for s in all_scores if s > 0)
        result.overall_score = int(sum(all_scores) / len(all_scores)) if all_scores else 0
        result.tier = determine_tier(result.overall_score)
        result.confidence = self._compute_progressive_confidence(all_scores)
        result.duration_ms = int((time.time() - start) * 1000)

        hash_data = f"{target_id}:{result.overall_score}:{result.questions_asked}:{int(time.time())}"
        result.result_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        # Collect token usage
        token_data = self.collect_token_usage()
        result.token_usage = token_data
        result.cost_usd = token_data.get("cost_usd", 0.0)

        logger.info(
            f"Streaming eval complete: {target_id} | "
            f"Score: {result.overall_score} | Tier: {result.tier} | "
            f"Questions: {result.questions_asked} | "
            f"Early exit: {cancel.reason if cancel and cancel.is_cancelled else 'no'} | "
            f"Tokens: {token_data['total_input_tokens']}in/{token_data['total_output_tokens']}out | "
            f"Cost: ${token_data.get('cost_usd', 0):.6f}"
        )

        return result

    async def evaluate_domain(
        self,
        target_id: str,
        domains: List[str],
        answer_fn,
        question_count: int = 10,
    ) -> EvaluationResult:
        """
        Level 3: Domain expert testing with calibrated questions.

        Args:
            target_id: ID of the target
            domains: Domains to test
            answer_fn: Async callable that takes a question and returns answer string
            question_count: Number of questions to ask
        """
        start = time.time()
        result = EvaluationResult()

        # Try IRT adaptive selection first, fall back to random
        irt_questions = None
        if self.irt_service:
            try:
                irt_questions = await self.irt_service.select_adaptive_questions(
                    theta=0.0, count=question_count, domains=domains,
                )
            except Exception:
                irt_questions = None

        if irt_questions:
            irt_id_set = {q["question_id"] for q in irt_questions}
            questions = [q for q in ALL_QUESTIONS if q.id in irt_id_set and (not domains or q.domain in domains)]
            # Fill remaining with random if IRT returned fewer
            if len(questions) < question_count:
                extra = self.question_selector.select_questions(target_id, domains, question_count - len(questions))
                seen = {q.id for q in questions}
                questions.extend(q for q in extra if q.id not in seen)
        else:
            questions = self.question_selector.select_questions(
                target_id, domains=domains, count=question_count
            )

        all_scores = []
        domain_buckets: Dict[str, List[int]] = {}
        para_seed = self.paraphraser.generate_seed(target_id) if self.paraphraser else 0

        for qi, q in enumerate(questions):
            # Paraphrase question for anti-gaming
            ask_question = q.question
            if self.paraphraser:
                ask_question = self.paraphraser.paraphrase_question(
                    q.question, para_seed + qi
                )

            try:
                answer = await answer_fn(ask_question)
            except Exception as e:
                logger.warning(f"Failed to get answer for {q.id}: {e}")
                answer = ""

            judge_result = await self.llm_judge.ajudge(
                ask_question, q.reference_answer, answer
            )

            # Style control: penalize verbose/over-formatted responses
            style_adj = apply_style_adjustment(judge_result.score, answer)
            adjusted_score = style_adj["adjusted_score"]

            self.difficulty_tracker.record(q.id, passed=adjusted_score >= 70)

            weighted_score = int(adjusted_score * q.weight)
            all_scores.append(adjusted_score)

            if q.domain not in domain_buckets:
                domain_buckets[q.domain] = []
            domain_buckets[q.domain].append(adjusted_score)

            result.judge_responses.append({
                "question_id": q.id,
                "domain": q.domain,
                "difficulty": q.difficulty,
                "score": adjusted_score,
                "raw_score": judge_result.score,
                "style_penalty": style_adj["style_penalty"],
                "style_features": style_adj["style_features"],
                "weighted_score": weighted_score,
                "explanation": judge_result.explanation,
                "method": judge_result.method,
            })

        # Domain scores
        for domain, scores in domain_buckets.items():
            result.domain_scores[domain] = {
                "score": int(sum(scores) / len(scores)),
                "questions": len(scores),
            }

        # Aggregate style report
        penalties = [jr.get("style_penalty", 0) for jr in result.judge_responses]
        penalized_count = sum(1 for p in penalties if p > 0)
        if penalties:
            result.style_report = {
                "total_penalty": round(sum(penalties), 2),
                "avg_penalty": round(sum(penalties) / len(penalties), 2),
                "penalized_responses": penalized_count,
                "total_responses": len(penalties),
            }

        result.questions_asked = len(questions)
        result.questions_answered = sum(1 for s in all_scores if s > 0)
        result.overall_score = int(sum(all_scores) / len(all_scores)) if all_scores else 0
        result.tier = determine_tier(result.overall_score)
        result.confidence = min(0.95, len(all_scores) / 30)
        result.duration_ms = int((time.time() - start) * 1000)

        # Post-eval IRT ability estimation
        if self.irt_service and questions:
            try:
                irt_responses = []
                for qi, q in enumerate(questions):
                    if qi < len(result.judge_responses):
                        irt_responses.append({
                            "question_id": q.id,
                            "correct": result.judge_responses[qi].get("score", 0) >= 70,
                        })
                if irt_responses:
                    ability = await self.irt_service.estimate_ability(irt_responses)
                    if ability["responses_used"] > 0:
                        result.irt_theta = ability["theta"]
                        result.irt_se = ability["se"]
                        se_score = ability["se"] * 10  # 1 logit ~ 10 score points
                        result.confidence_interval = {
                            "lower": max(0, round(result.overall_score - 1.96 * se_score, 1)),
                            "upper": min(100, round(result.overall_score + 1.96 * se_score, 1)),
                        }
                        result.confidence = round(max(0.1, min(0.95, 1.0 - ability["se"] / 3.0)), 2)
            except Exception:
                pass  # non-fatal, keep random-based confidence

        hash_data = f"{target_id}:{result.overall_score}:{result.questions_asked}:{int(time.time())}"
        result.result_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        return result

    async def enrich_with_dimensions(
        self,
        result: EvaluationResult,
        tool_responses: Dict[str, List[dict]],
        manifest: Optional[dict] = None,
        server_url: str = "",
        run_safety: bool = True,
    ) -> EvaluationResult:
        """Enrich an existing EvaluationResult with all 6 dimension scores.

        Used to add dimensions after streaming evaluation completes.
        Computes: safety, process_quality, reliability, latency, schema_quality.
        Accuracy is already computed as overall_score from functional eval.

        Args:
            result: EvaluationResult from evaluate_functional or evaluate_functional_streaming
            tool_responses: Dict of tool_name -> list of response dicts (with answer, is_error, latency_ms, test_type)
            manifest: Server manifest for schema_quality and safety probes
            server_url: MCP server URL for safety probes and consistency checks
            run_safety: Whether to run adversarial safety probes
        """
        accuracy_score = result.overall_score
        schema_score = result.manifest_result.score if result.manifest_result else 50

        # Safety dimension — adversarial probes
        safety_score = 50
        if run_safety and manifest:
            try:
                from src.core.adversarial import run_safety_probes
                tools = manifest.get("tools", [])
                safety_report = await run_safety_probes(server_url, tools)
                safety_score = safety_report.safety_score
                result.safety_report = safety_report.to_dict()
            except Exception as e:
                logger.warning(f"Safety probes failed: {e}")

        # Process quality dimension
        process_quality_score = 50
        if tool_responses:
            try:
                from src.core.process_quality import analyze_process_quality
                pq_result = analyze_process_quality(tool_responses)
                process_quality_score = pq_result.score
                result.process_quality_report = pq_result.to_dict()
            except Exception as e:
                logger.warning(f"Process quality analysis failed: {e}")

        # Latency dimension
        all_latencies = []
        for responses in tool_responses.values():
            for resp in responses:
                if not resp.get("is_error") and resp.get("latency_ms"):
                    all_latencies.append(resp["latency_ms"])

        if all_latencies:
            sorted_lat = sorted(all_latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
            result.latency_stats = {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}

            latency_score = max(0, min(100, 100 - (p50 / 30)))
            if p95 > 2 * p50:
                latency_score *= 0.9
            if p99 > 10000:
                latency_score = min(latency_score, 40)
            latency_score = int(round(latency_score))
        else:
            latency_score = 50

        # Reliability dimension — actual response consistency
        reliability_score = 50
        if manifest and server_url:
            try:
                from src.core.mcp_client import check_response_consistency
                tools = manifest.get("tools", [])
                consistency = await check_response_consistency(server_url, tools, sample_size=2)
                if consistency:
                    avg_consistency = sum(consistency.values()) / len(consistency)
                    if avg_consistency >= 0.9:
                        reliability_score = 100
                    elif avg_consistency >= 0.7:
                        reliability_score = 80
                    elif avg_consistency >= 0.5:
                        reliability_score = 60
                    elif avg_consistency >= 0.3:
                        reliability_score = 40
                    else:
                        reliability_score = 20
            except Exception as e:
                logger.warning(f"Consistency check failed: {e}")

        # Assemble dimensions
        dimensions = {
            "accuracy": {"score": accuracy_score, "weight": 0.35},
            "safety": {"score": safety_score, "weight": 0.20},
            "process_quality": {"score": process_quality_score, "weight": 0.10},
            "reliability": {"score": reliability_score, "weight": 0.15},
            "latency": {"score": latency_score, "weight": 0.10},
            "schema_quality": {"score": schema_score, "weight": 0.10},
        }
        result.dimensions = dimensions

        weighted_total = sum(d["score"] * d["weight"] for d in dimensions.values())
        result.overall_score = int(round(weighted_total))
        result.tier = determine_tier(result.overall_score)

        # Recompute hash with new score
        hash_data = f"{server_url}:{result.overall_score}:{result.questions_asked}:{int(time.time())}"
        result.result_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        logger.info(
            f"Dimensions enriched: Overall={result.overall_score} | "
            f"acc={accuracy_score} safe={safety_score} proc={process_quality_score} "
            f"rel={reliability_score} lat={latency_score} schema={schema_score}"
        )

        return result

    async def evaluate_full(
        self,
        target_id: str,
        server_url: str,
        tool_responses: Dict[str, List[dict]],
        manifest: Optional[dict] = None,
        run_safety: bool = True,
        run_consistency: bool = True,
        progress_cb: Optional[Any] = None,
        detected_domain: str = "general",
    ) -> EvaluationResult:
        """
        Full multi-dimensional evaluation (6 axes).

        Runs functional eval + safety probes + process quality, computes per-dimension scores:
        - accuracy (35%): correctness of tool responses
        - safety (20%): adversarial probe resistance
        - process_quality (10%): error handling, input validation, response structure
        - reliability (15%): score consistency / variance penalty
        - latency (10%): response time performance
        - schema_quality (10%): manifest completeness

        Args:
            target_id: ID of the target being evaluated
            server_url: MCP server URL for safety probes
            tool_responses: Dict of tool_name -> list of response dicts
            manifest: Optional manifest for Level 1
            run_safety: Whether to run adversarial safety probes
            run_consistency: Whether to run idempotency/consistency checks
        """
        start = time.time()

        # Run functional evaluation (accuracy dimension)
        logger.info(f"[evaluate_full] {target_id}: Starting functional eval ({len(tool_responses)} tools)")
        if progress_cb:
            await progress_cb("functional_eval_start", 0.0)
        judging_start = time.time()
        result = await self.evaluate_functional(
            target_id=target_id,
            tool_responses=tool_responses,
            manifest=manifest,
        )
        judging_ms = int((time.time() - judging_start) * 1000)
        logger.info(f"[evaluate_full] {target_id}: Functional eval done, accuracy={result.overall_score}")

        accuracy_score = result.overall_score
        schema_score = result.manifest_result.score if result.manifest_result else 50

        # Safety dimension — adversarial probes
        safety_score = 50  # Neutral default
        if run_safety and manifest:
            if progress_cb:
                await progress_cb("safety_probes_start", 0.4)
            try:
                from src.core.adversarial import run_safety_probes
                tools = manifest.get("tools", [])
                logger.info(f"[evaluate_full] {target_id}: Running safety probes on {len(tools)} tools")
                safety_report = await run_safety_probes(server_url, tools)
                safety_score = safety_report.safety_score
                result.safety_report = safety_report.to_dict()
                logger.info(f"[evaluate_full] {target_id}: Safety probes done, score={safety_score}")
            except Exception as e:
                logger.warning(f"Safety probes failed for {target_id}: {e}")

        # Process quality dimension — error handling, validation, structure
        process_quality_score = 50  # Neutral default
        if progress_cb:
            await progress_cb("process_quality", 0.7)
        try:
            from src.core.process_quality import analyze_process_quality
            pq_result = analyze_process_quality(tool_responses)
            process_quality_score = pq_result.score
            result.process_quality_report = pq_result.to_dict()
            logger.info(f"[evaluate_full] {target_id}: Process quality done, score={process_quality_score}")
        except Exception as e:
            logger.warning(f"Process quality analysis failed for {target_id}: {e}")

        # Latency dimension — from tool response latencies
        all_latencies = []
        for responses in tool_responses.values():
            for resp in responses:
                if not resp.get("is_error") and resp.get("latency_ms"):
                    all_latencies.append(resp["latency_ms"])

        if all_latencies:
            sorted_lat = sorted(all_latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
            result.latency_stats = {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}

            # Smooth continuous latency scoring (0-100) based on p50, with p95/p99 penalties
            latency_score = max(0, min(100, 100 - (p50 / 30)))
            if p95 > 2 * p50:
                latency_score *= 0.9  # Tail latency penalty
            if p99 > 10000:
                latency_score = min(latency_score, 40)  # Catastrophic tail cap
            latency_score = int(round(latency_score))
        else:
            latency_score = 50  # Neutral

        # Reliability dimension — actual response consistency (idempotency check)
        reliability_score = 50  # Neutral default
        if run_consistency and manifest:
            if progress_cb:
                await progress_cb("reliability_check", 0.85)
            try:
                from src.core.mcp_client import check_response_consistency
                tools = manifest.get("tools", [])
                logger.info(f"[evaluate_full] {target_id}: Running consistency check on {len(tools)} tools")
                consistency = await check_response_consistency(server_url, tools, sample_size=2)
                if consistency:
                    avg_consistency = sum(consistency.values()) / len(consistency)
                    # Map consistency ratio to score
                    if avg_consistency >= 0.9:
                        reliability_score = 100
                    elif avg_consistency >= 0.7:
                        reliability_score = 80
                    elif avg_consistency >= 0.5:
                        reliability_score = 60
                    elif avg_consistency >= 0.3:
                        reliability_score = 40
                    else:
                        reliability_score = 20
                logger.info(f"[evaluate_full] {target_id}: Consistency check done, reliability={reliability_score}")
            except Exception as e:
                logger.warning(f"Consistency check failed for {target_id}: {e}")

        # Multi-dimensional aggregate (6 axes, domain-weighted — QO-027)
        from src.core.domain_detection import get_domain_weights
        domain_weights = get_domain_weights(detected_domain)
        axis_scores = {
            "accuracy": accuracy_score,
            "safety": safety_score,
            "process_quality": process_quality_score,
            "reliability": reliability_score,
            "latency": latency_score,
            "schema_quality": schema_score,
        }
        dimensions = {
            axis: {"score": axis_scores[axis], "weight": domain_weights[axis]}
            for axis in axis_scores
        }
        result.dimensions = dimensions

        weighted_total = sum(
            d["score"] * d["weight"] for d in dimensions.values()
        )
        result.overall_score = int(round(weighted_total))
        result.tier = determine_tier(result.overall_score)
        logger.info(f"[evaluate_full] Domain weights applied: domain={detected_domain}, "
                     f"weights={domain_weights}")

        # Recompute result hash with new overall score
        hash_data = f"{target_id}:{result.overall_score}:{result.questions_asked}:{int(time.time())}"
        result.result_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        # Collect token usage
        total_ms = int((time.time() - start) * 1000)
        token_data = self.collect_token_usage()
        token_data["phase_timing_ms"] = {
            "judging_ms": judging_ms,
            "total_ms": total_ms,
        }
        result.token_usage = token_data
        result.cost_usd = token_data.get("cost_usd", 0.0)

        logger.info(
            f"Full evaluation: {target_id} | "
            f"Overall: {result.overall_score} | Tier: {result.tier} | "
            f"Dims: acc={accuracy_score} safe={safety_score} "
            f"proc={process_quality_score} rel={reliability_score} "
            f"lat={latency_score} schema={schema_score} | "
            f"Tokens: {token_data['total_input_tokens']}in/{token_data['total_output_tokens']}out | "
            f"Cost: ${token_data.get('cost_usd', 0):.6f}"
        )

        return result

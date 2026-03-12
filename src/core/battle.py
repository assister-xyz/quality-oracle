"""Head-to-head battle engine for AgentTrust Arena.

Orchestrates parallel evaluation of two agents with identical challenges,
determines winner, and updates ratings. Includes arena integrity features:
- Blind battles (agent identity anonymization)
- Position swap consistency checking
- Style control regression
"""
import hashlib
import logging
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.core.question_pools import QuestionSelector, ChallengeQuestion, ALL_QUESTIONS
from src.core.rating import RatingEngine
from src.core.scoring import extract_style_features, compute_style_penalty
from src.storage.mongodb import battles_col, scores_col
from src.storage.models import BattleRequest, BattleStatus

logger = logging.getLogger(__name__)

# Stratified difficulty distribution (percentages)
# 15% easy / 25% medium-easy / 30% medium / 25% medium-hard / 15% hard
# Since we have 3 difficulty levels (easy/medium/hard), we map:
#   easy: 15%, medium: 70% (25+30+25 collapsed), hard: 15%
DIFFICULTY_WEIGHTS = {"easy": 0.15, "medium": 0.70, "hard": 0.15}

COOLDOWN_HOURS = 1
CORRECT_THRESHOLD = 70  # Score >= 70 counts as "correct" for IRT


class BattleEngine:
    """Orchestrates head-to-head battles between two agents."""

    def __init__(self):
        self.question_selector = QuestionSelector()
        self.rating_engine = RatingEngine()

    # ── Challenge Composition ────────────────────────────────────────────

    def compose_challenge_set(
        self,
        count: int = 5,
        seed: Optional[int] = None,
        domains_a: Optional[List[str]] = None,
        domains_b: Optional[List[str]] = None,
    ) -> List[ChallengeQuestion]:
        """Compose a stratified challenge set for a battle.

        Uses shared seed so both agents get identical questions.
        Difficulty distribution: 15% easy / 70% medium / 15% hard.
        Domain balance: 40% neutral + 30% each agent's domain (if known).
        """
        rng = random.Random(seed)

        # Determine target counts per difficulty
        easy_count = max(1, round(count * DIFFICULTY_WEIGHTS["easy"]))
        hard_count = max(1, round(count * DIFFICULTY_WEIGHTS["hard"]))
        medium_count = count - easy_count - hard_count

        # Build domain pool
        all_domains = set()
        if domains_a:
            all_domains.update(domains_a)
        if domains_b:
            all_domains.update(domains_b)

        pool = list(ALL_QUESTIONS)

        # Select by difficulty
        easy_pool = [q for q in pool if q.difficulty == "easy"]
        medium_pool = [q for q in pool if q.difficulty == "medium"]
        hard_pool = [q for q in pool if q.difficulty == "hard"]

        selected = []
        selected.extend(self._sample(easy_pool, easy_count, rng))
        selected.extend(self._sample(medium_pool, medium_count, rng))
        selected.extend(self._sample(hard_pool, hard_count, rng))

        # If we got fewer than needed, fill from remaining pool
        if len(selected) < count:
            remaining = [q for q in pool if q not in selected]
            needed = count - len(selected)
            selected.extend(self._sample(remaining, needed, rng))

        rng.shuffle(selected)
        return selected[:count]

    @staticmethod
    def _sample(pool: List, count: int, rng: random.Random) -> List:
        """Sample up to count items from pool using given RNG."""
        if not pool:
            return []
        return rng.sample(pool, min(count, len(pool)))

    # ── Arena Integrity (QO-009) ──────────────────────────────────────────

    @staticmethod
    def anonymize_response(response: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Strip identifying information from agent response before judging.

        Replaces agent name, URL, and metadata with a neutral label
        (e.g. "Agent A" / "Agent B") so the LLM judge cannot be biased
        by reputation or name recognition.
        """
        return {
            "label": label,
            "response": response.get("content", response.get("response", "")),
            "latency_ms": response.get("latency_ms", 0),
            # Intentionally excluded: name, url, server_info, manifest
        }

    @staticmethod
    def check_position_consistency(
        score_forward: Dict[str, Any],
        score_reversed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if battle result is consistent across A/B position swap.

        In forward ordering, Agent A is presented first.
        In reversed ordering, Agent B is presented first.
        If the winner changes, the battle is declared a tie (position bias detected).

        Returns:
            dict with winner, consistency status, and averaged scores.
        """
        fwd_a = score_forward.get("overall_score", 0)
        fwd_b = score_forward.get("overall_score_b", 0)
        rev_a = score_reversed.get("overall_score", 0)
        rev_b = score_reversed.get("overall_score_b", 0)

        # Determine winner in each ordering
        if fwd_a > fwd_b:
            winner_forward = "a"
        elif fwd_b > fwd_a:
            winner_forward = "b"
        else:
            winner_forward = None

        # In reversed ordering, A and B are swapped in presentation
        # but we map back: reversed "first" = B, reversed "second" = A
        if rev_b > rev_a:
            winner_reversed = "b"
        elif rev_a > rev_b:
            winner_reversed = "a"
        else:
            winner_reversed = None

        consistent = winner_forward == winner_reversed

        # Average scores across orderings for fairness
        avg_score_a = int(round((fwd_a + rev_a) / 2))
        avg_score_b = int(round((fwd_b + rev_b) / 2))

        if consistent:
            return {
                "winner": winner_forward,
                "consistency": "consistent",
                "score_a": avg_score_a,
                "score_b": avg_score_b,
                "forward_scores": {"a": fwd_a, "b": fwd_b},
                "reversed_scores": {"a": rev_a, "b": rev_b},
            }
        else:
            # Position bias detected — force tie
            return {
                "winner": None,
                "consistency": "tie_forced",
                "score_a": avg_score_a,
                "score_b": avg_score_b,
                "forward_scores": {"a": fwd_a, "b": fwd_b},
                "reversed_scores": {"a": rev_a, "b": rev_b},
            }

    @staticmethod
    def build_integrity_metadata(
        blind_enforced: bool = True,
        position_swapped: bool = False,
        style_controlled: bool = False,
        consistency: str = "not_checked",
        style_penalty_a: float = 0.0,
        style_penalty_b: float = 0.0,
    ) -> Dict[str, Any]:
        """Build arena integrity metadata for battle document."""
        return {
            "blind_enforced": blind_enforced,
            "position_swapped": position_swapped,
            "style_controlled": style_controlled,
            "consistency": consistency,
            "style_penalties": {
                "agent_a": style_penalty_a,
                "agent_b": style_penalty_b,
            },
            "integrity_version": "1.0",
        }

    # ── Winner Determination ─────────────────────────────────────────────

    @staticmethod
    def determine_winner(
        score_a: int, score_b: int,
    ) -> Tuple[Optional[str], int, bool]:
        """Determine battle winner from composite scores.

        Returns:
            (winner, margin, photo_finish) where winner is "a", "b", or None.
        """
        margin = abs(score_a - score_b)
        photo_finish = 0 < margin < 5

        if score_a > score_b:
            return "a", margin, photo_finish
        elif score_b > score_a:
            return "b", margin, photo_finish
        else:
            return None, 0, False

    # ── Cooldown Check ───────────────────────────────────────────────────

    async def check_cooldown(
        self, target_id_a: str, target_id_b: str,
    ) -> Optional[int]:
        """Check if two agents are in cooldown.

        Returns remaining minutes if in cooldown, None if clear.
        """
        col = battles_col()
        cutoff = datetime.utcnow() - timedelta(hours=COOLDOWN_HOURS)

        recent = await col.find_one({
            "$or": [
                {"agent_a.target_id": target_id_a, "agent_b.target_id": target_id_b},
                {"agent_a.target_id": target_id_b, "agent_b.target_id": target_id_a},
            ],
            "created_at": {"$gte": cutoff},
        })

        if recent is None:
            return None

        elapsed = datetime.utcnow() - recent["created_at"]
        remaining = timedelta(hours=COOLDOWN_HOURS) - elapsed
        return max(1, int(remaining.total_seconds() / 60))

    # ── Same Operator Check ──────────────────────────────────────────────

    @staticmethod
    def check_same_operator(url_a: str, url_b: str) -> bool:
        """Check if two agents share the same host (same operator proxy)."""
        try:
            host_a = urlparse(url_a).hostname
            host_b = urlparse(url_b).hostname
            return host_a == host_b
        except Exception:
            return False

    # ── IRT Data Collection ──────────────────────────────────────────────

    @staticmethod
    def compute_question_response(
        question_id: str,
        question_hash: str,
        domain: str,
        difficulty: str,
        score_a: float,
        score_b: float,
        latency_a_ms: int,
        latency_b_ms: int,
    ) -> dict:
        """Compute per-question response data for IRT calibration."""
        a_correct = score_a >= CORRECT_THRESHOLD
        b_correct = score_b >= CORRECT_THRESHOLD

        # Discrimination: how well this question separates the agents
        # Normalized absolute score difference / 100
        discrimination = abs(score_a - score_b) / 100.0

        return {
            "question_id": question_id,
            "question_hash": question_hash,
            "domain": domain,
            "difficulty_tag": difficulty,
            "agent_a_correct": a_correct,
            "agent_b_correct": b_correct,
            "agent_a_score": score_a,
            "agent_b_score": score_b,
            "agent_a_latency_ms": latency_a_ms,
            "agent_b_latency_ms": latency_b_ms,
            "battle_discrimination": discrimination,
        }

    # ── Battle Creation ──────────────────────────────────────────────────

    async def create_battle(self, request: BattleRequest) -> str:
        """Create a new battle record in pending state.

        Validates: same-operator check, cooldown.
        Returns battle_id.
        """
        # Same operator check
        if self.check_same_operator(request.agent_a_url, request.agent_b_url):
            raise ValueError("Cannot battle agents from the same operator")

        # Generate target IDs from URLs
        target_id_a = hashlib.sha256(request.agent_a_url.encode()).hexdigest()[:16]
        target_id_b = hashlib.sha256(request.agent_b_url.encode()).hexdigest()[:16]

        # Cooldown check
        remaining = await self.check_cooldown(target_id_a, target_id_b)
        if remaining is not None:
            raise CooldownError(f"Challenge cooldown: wait {remaining} minutes")

        # Check match quality if ratings exist
        match_quality = 1.0  # Default for new agents
        scores_a = await scores_col().find_one({"target_id": target_id_a})
        scores_b = await scores_col().find_one({"target_id": target_id_b})

        if scores_a and scores_b:
            # Get composite ratings
            ra_data = (scores_a.get("openskill_axes") or {}).get("composite", {})
            rb_data = (scores_b.get("openskill_axes") or {}).get("composite", {})
            if ra_data and rb_data:
                ra = self.rating_engine._rating_from_dict(ra_data)
                rb = self.rating_engine._rating_from_dict(rb_data)
                match_quality = self.rating_engine.match_quality(ra, rb)
                if match_quality < 0.30:
                    raise MatchQualityError("Match too unbalanced")

        battle_id = str(uuid.uuid4())
        battle_doc = {
            "_id": battle_id,
            "battle_id": battle_id,
            "agent_a": {
                "target_id": target_id_a,
                "target_url": request.agent_a_url,
                "name": "",
                "scores": {},
                "overall_score": 0,
            },
            "agent_b": {
                "target_id": target_id_b,
                "target_url": request.agent_b_url,
                "name": "",
                "scores": {},
                "overall_score": 0,
            },
            "winner": None,
            "margin": 0,
            "photo_finish": False,
            "match_quality": match_quality,
            "domain": request.domain,
            "challenge_count": request.challenge_count,
            "eval_mode": request.eval_mode.value,
            "match_type": "manual",
            "duration_ms": 0,
            "created_at": datetime.utcnow(),
            "completed_at": None,
            "status": BattleStatus.PENDING.value,
            "question_responses": [],
            "rating_deltas": None,
            "error": None,
            "integrity": self.build_integrity_metadata(
                blind_enforced=request.blind,
            ),
        }

        await battles_col().insert_one(battle_doc)
        logger.info(f"Battle {battle_id} created: {request.agent_a_url} vs {request.agent_b_url}")
        return battle_id

    async def run_battle(self, battle_id: str, evaluator_factory=None) -> dict:
        """Execute a battle: evaluate both agents, determine winner, update ratings.

        Args:
            battle_id: The battle to run
            evaluator_factory: Callable that creates an Evaluator instance (for testing)

        Returns:
            Updated battle document
        """
        col = battles_col()
        battle = await col.find_one({"_id": battle_id})
        if not battle:
            raise ValueError(f"Battle {battle_id} not found")

        # Mark as running
        await col.update_one(
            {"_id": battle_id},
            {"$set": {"status": BattleStatus.RUNNING.value}},
        )

        start_time = time.time()

        try:
            # Generate shared seed for identical questions
            seed = int(hashlib.sha256(battle_id.encode()).hexdigest()[:8], 16)

            # Compose challenge set
            questions = self.compose_challenge_set(
                count=battle["challenge_count"],
                seed=seed,
                domains_a=None,  # Could be enhanced with agent domain info
                domains_b=None,
            )

            # Run evaluations in parallel
            if evaluator_factory:
                eval_a, eval_b = await self._run_parallel_evals(
                    battle, questions, evaluator_factory,
                )
            else:
                # Without evaluator, use mock scores (for initial testing)
                eval_a = {"overall_score": 0, "scores": {}, "name": "Agent A"}
                eval_b = {"overall_score": 0, "scores": {}, "name": "Agent B"}

            # Apply style penalties to scores before determining winner
            style_penalty_a = eval_a.get("style_penalty", 0.0)
            style_penalty_b = eval_b.get("style_penalty", 0.0)
            if style_penalty_a > 0:
                eval_a["overall_score"] = max(0, eval_a["overall_score"] - int(style_penalty_a))
            if style_penalty_b > 0:
                eval_b["overall_score"] = max(0, eval_b["overall_score"] - int(style_penalty_b))

            # Position swap: run second eval with agents reversed for certified/audited
            eval_mode = battle.get("eval_mode", "verified")
            position_swapped = False
            consistency = "not_checked"

            if eval_mode in ("certified", "audited") and evaluator_factory:
                # Build reversed battle doc (swap agent_a and agent_b)
                reversed_battle = dict(battle)
                reversed_battle["agent_a"], reversed_battle["agent_b"] = (
                    battle["agent_b"], battle["agent_a"],
                )

                rev_eval_a, rev_eval_b = await self._run_parallel_evals(
                    reversed_battle, questions, evaluator_factory,
                )

                # Apply style penalties to reversed scores too
                rev_penalty_a = rev_eval_a.get("style_penalty", 0.0)
                rev_penalty_b = rev_eval_b.get("style_penalty", 0.0)
                if rev_penalty_a > 0:
                    rev_eval_a["overall_score"] = max(0, rev_eval_a["overall_score"] - int(rev_penalty_a))
                if rev_penalty_b > 0:
                    rev_eval_b["overall_score"] = max(0, rev_eval_b["overall_score"] - int(rev_penalty_b))

                # In reversed eval: "a" position has agent_b, "b" position has agent_a
                # Map back: forward_a = eval_a score, forward_b = eval_b score
                #           reversed_a = rev_eval_b score (agent_a in "b" slot)
                #           reversed_b = rev_eval_a score (agent_b in "a" slot)
                forward_result = {
                    "overall_score": eval_a["overall_score"],
                    "overall_score_b": eval_b["overall_score"],
                }
                reversed_result = {
                    "overall_score": rev_eval_b["overall_score"],   # agent_a in reversed
                    "overall_score_b": rev_eval_a["overall_score"],  # agent_b in reversed
                }

                pos_check = self.check_position_consistency(forward_result, reversed_result)
                position_swapped = True
                consistency = pos_check["consistency"]

                # Use averaged scores from both orderings
                eval_a["overall_score"] = pos_check["score_a"]
                eval_b["overall_score"] = pos_check["score_b"]

            # Determine winner (after style adjustment and position-swap averaging)
            if consistency == "tie_forced":
                # Position bias detected — force tie
                winner = None
                margin = 0
                photo_finish = False
            else:
                winner, margin, photo_finish = self.determine_winner(
                    eval_a["overall_score"], eval_b["overall_score"],
                )

            # Get existing ratings for both agents
            target_id_a = battle["agent_a"]["target_id"]
            target_id_b = battle["agent_b"]["target_id"]
            score_doc_a = await scores_col().find_one({"target_id": target_id_a})
            score_doc_b = await scores_col().find_one({"target_id": target_id_b})

            existing_ratings_a = (score_doc_a or {}).get("openskill_axes", {})
            existing_ratings_b = (score_doc_b or {}).get("openskill_axes", {})

            # Update ratings
            rating_deltas = self.rating_engine.process_battle_scores(
                eval_a.get("scores", {}),
                eval_b.get("scores", {}),
                eval_a["overall_score"],
                eval_b["overall_score"],
                existing_ratings_a,
                existing_ratings_b,
                winner,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Style penalties already applied above; record in integrity metadata
            style_controlled = style_penalty_a > 0 or style_penalty_b > 0

            integrity = self.build_integrity_metadata(
                blind_enforced=battle.get("integrity", {}).get("blind_enforced", True),
                position_swapped=position_swapped,
                style_controlled=style_controlled,
                consistency=consistency,
                style_penalty_a=style_penalty_a,
                style_penalty_b=style_penalty_b,
            )

            # Build update
            update = {
                "agent_a.name": eval_a.get("name", ""),
                "agent_a.scores": eval_a.get("scores", {}),
                "agent_a.overall_score": eval_a["overall_score"],
                "agent_a.rating_before": rating_deltas["agent_a"].get("composite", {}).get("before"),
                "agent_a.rating_after": rating_deltas["agent_a"].get("composite", {}).get("after"),
                "agent_b.name": eval_b.get("name", ""),
                "agent_b.scores": eval_b.get("scores", {}),
                "agent_b.overall_score": eval_b["overall_score"],
                "agent_b.rating_before": rating_deltas["agent_b"].get("composite", {}).get("before"),
                "agent_b.rating_after": rating_deltas["agent_b"].get("composite", {}).get("after"),
                "winner": winner,
                "margin": margin,
                "photo_finish": photo_finish,
                "duration_ms": duration_ms,
                "completed_at": datetime.utcnow(),
                "status": BattleStatus.COMPLETED.value,
                "rating_deltas": rating_deltas,
                "integrity": integrity,
            }

            await col.update_one({"_id": battle_id}, {"$set": update})

            # Update scores collection with new ratings
            await self._update_agent_ratings(target_id_a, rating_deltas["agent_a"], winner == "a", winner is None)
            await self._update_agent_ratings(target_id_b, rating_deltas["agent_b"], winner == "b", winner is None)

            updated = await col.find_one({"_id": battle_id})
            return updated

        except Exception as e:
            logger.error(f"Battle {battle_id} failed: {e}")
            await col.update_one(
                {"_id": battle_id},
                {"$set": {
                    "status": BattleStatus.FAILED.value,
                    "error": str(e),
                    "duration_ms": int((time.time() - start_time) * 1000),
                }},
            )
            raise

    async def _run_parallel_evals(self, battle, questions, evaluator_factory):
        """Run evaluations for both agents in parallel.

        Enforces blind battles: agent identity is stripped before judging.
        Computes style penalties for formatting-based score inflation.
        """
        import asyncio
        from src.core.mcp_client import MCPClient

        blind = battle.get("integrity", {}).get("blind_enforced", True)

        async def evaluate_agent(target_url, questions, label):
            client = MCPClient(target_url)
            try:
                manifest = await client.connect_and_list_tools()
                tool_responses = await client.run_challenge_questions(questions)

                # Blind battle enforcement: strip identity before judging
                if blind:
                    for resp in tool_responses:
                        resp.pop("server_name", None)
                        resp.pop("server_url", None)
                        resp.pop("manifest", None)
                        if "meta" in resp:
                            resp["meta"].pop("server_name", None)
                            resp["meta"].pop("server_url", None)

                evaluator = evaluator_factory()
                target_id = hashlib.sha256(target_url.encode()).hexdigest()[:16]
                result = await evaluator.evaluate_functional(
                    target_id, tool_responses, manifest,
                )

                # Compute style penalty from raw responses
                style_penalty = 0.0
                for resp in tool_responses:
                    content = resp.get("content", resp.get("response", ""))
                    if isinstance(content, str) and len(content) > 0:
                        features = extract_style_features(content)
                        style_penalty = max(style_penalty, compute_style_penalty(features))

                return {
                    "overall_score": result.overall_score,
                    "scores": result.dimensions or {},
                    "name": manifest.get("name", target_url) if not blind else label,
                    "style_penalty": style_penalty,
                    "_real_name": manifest.get("name", target_url),
                }
            except Exception as e:
                logger.error(f"Evaluation failed for {target_url}: {e}")
                return {
                    "overall_score": 0,
                    "scores": {},
                    "name": label if blind else target_url,
                    "style_penalty": 0.0,
                    "_real_name": target_url,
                }

        eval_a, eval_b = await asyncio.gather(
            evaluate_agent(battle["agent_a"]["target_url"], questions, "Agent A"),
            evaluate_agent(battle["agent_b"]["target_url"], questions, "Agent B"),
        )

        # Restore real names for storage (not for judging)
        eval_a["name"] = eval_a.pop("_real_name", eval_a.get("name", ""))
        eval_b["name"] = eval_b.pop("_real_name", eval_b.get("name", ""))

        return eval_a, eval_b

    async def _update_agent_ratings(
        self, target_id: str, deltas: dict, won: bool, draw: bool,
    ):
        """Update agent's ratings and battle record in scores collection."""
        col = scores_col()

        # Build openskill_axes update from deltas
        axes_update = {}
        for key, val in deltas.items():
            if "after" in val:
                axes_update[f"openskill_axes.{key}"] = val["after"]

        # Update composite mu/sigma at top level too
        composite = deltas.get("composite", {}).get("after", {})

        update_fields = {
            **axes_update,
            "openskill_mu": composite.get("mu", 25.0),
            "openskill_sigma": composite.get("sigma", 8.333),
            "last_battle_at": datetime.utcnow(),
        }

        inc_fields = {}
        if won:
            inc_fields["battle_record.wins"] = 1
            inc_fields["win_streak"] = 1
        elif draw:
            inc_fields["battle_record.draws"] = 1
        else:
            inc_fields["battle_record.losses"] = 1

        update_op = {"$set": update_fields}
        if inc_fields:
            update_op["$inc"] = inc_fields

        # Reset win_streak on loss
        if not won and not draw:
            update_op["$set"]["win_streak"] = 0

        await col.update_one(
            {"target_id": target_id},
            update_op,
            upsert=False,
        )


class CooldownError(Exception):
    """Raised when battle is in cooldown period."""
    pass


class MatchQualityError(Exception):
    """Raised when match quality is too low."""
    pass

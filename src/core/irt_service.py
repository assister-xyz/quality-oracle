"""IRT (Item Response Theory) calibration service for AgentTrust.

Provides progressive calibration of question difficulty and discrimination,
ability estimation, and adaptive question selection via Fisher information
maximization. Pure Python Rasch 1PL — no numpy/scipy required.
"""
import logging
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.storage.mongodb import battles_col, item_params_col

logger = logging.getLogger(__name__)

CORRECT_THRESHOLD = 70  # Score >= 70 counts as "correct"
JMLE_MAX_ITER = 50
JMLE_CONVERGENCE = 0.01
NOT_ADMINISTERED = -1


@dataclass
class ItemParams:
    question_id: str
    domain: str
    difficulty_b: float = 0.0
    discrimination_a: float = 1.0
    p_value: float = 0.5
    point_biserial: float = 0.0
    exposure_count: int = 0
    total_responses: int = 0
    status: str = "active"  # active, flagged, retired
    calibration_model: str = "none"
    last_calibrated: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["last_calibrated"]:
            d["last_calibrated"] = d["last_calibrated"].isoformat()
        return d


class IRTService:
    """IRT calibration and adaptive testing engine."""

    def __init__(self):
        self._item_cache: Dict[str, ItemParams] = {}

    # ── Response Matrix ──────────────────────────────────────────────────

    async def build_response_matrix(
        self,
    ) -> Tuple[List[str], List[str], List[List[int]]]:
        """Load battles with populated question_responses, build binary matrix.

        Returns:
            (question_ids, agent_ids, matrix) where matrix[i][j] is
            1=correct, 0=incorrect, -1=not administered for item i, agent j.
        """
        col = battles_col()
        cursor = col.find({
            "status": "completed",
            "question_responses": {"$ne": []},
        })

        # Collect all responses
        question_set: Dict[str, int] = {}  # question_id -> index
        agent_set: Dict[str, int] = {}  # agent_id -> index
        responses: List[Tuple[str, str, bool]] = []  # (q_id, agent_id, correct)

        async for battle in cursor:
            agent_a_id = battle.get("agent_a", {}).get("target_id", "")
            agent_b_id = battle.get("agent_b", {}).get("target_id", "")
            if not agent_a_id or not agent_b_id:
                continue

            for qr in battle.get("question_responses", []):
                q_id = qr.get("question_id", "")
                if not q_id:
                    continue

                if q_id not in question_set:
                    question_set[q_id] = len(question_set)
                if agent_a_id not in agent_set:
                    agent_set[agent_a_id] = len(agent_set)
                if agent_b_id not in agent_set:
                    agent_set[agent_b_id] = len(agent_set)

                responses.append((q_id, agent_a_id, qr.get("agent_a_correct", False)))
                responses.append((q_id, agent_b_id, qr.get("agent_b_correct", False)))

        if not question_set or not agent_set:
            return [], [], []

        question_ids = [""] * len(question_set)
        for q_id, idx in question_set.items():
            question_ids[idx] = q_id

        agent_ids = [""] * len(agent_set)
        for a_id, idx in agent_set.items():
            agent_ids[idx] = a_id

        # Build matrix: items × agents, initialized to NOT_ADMINISTERED
        n_items = len(question_ids)
        n_agents = len(agent_ids)
        matrix = [[NOT_ADMINISTERED] * n_agents for _ in range(n_items)]

        for q_id, a_id, correct in responses:
            i = question_set[q_id]
            j = agent_set[a_id]
            matrix[i][j] = 1 if correct else 0

        return question_ids, agent_ids, matrix

    # ── Rasch 1PL Calibration (JMLE) ────────────────────────────────────

    @staticmethod
    def _rasch_calibrate(
        question_ids: List[str],
        agent_ids: List[str],
        matrix: List[List[int]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Pure Python Rasch 1PL via Joint Maximum Likelihood Estimation.

        P(correct) = 1 / (1 + exp(-(theta - b)))

        Uses iterative Newton-Raphson updates. Centering constraint:
        mean(difficulties) = 0.

        Returns:
            (difficulties, abilities) — dicts keyed by question_id / agent_id.
        """
        n_items = len(question_ids)
        n_agents = len(agent_ids)

        if n_items == 0 or n_agents == 0:
            return {}, {}

        # Initialize
        b = [0.0] * n_items   # difficulty
        theta = [0.0] * n_agents  # ability

        for iteration in range(JMLE_MAX_ITER):
            max_delta = 0.0

            # Update abilities (theta)
            for j in range(n_agents):
                num = 0.0
                denom = 0.0
                for i in range(n_items):
                    if matrix[i][j] == NOT_ADMINISTERED:
                        continue
                    p = _logistic(theta[j] - b[i])
                    num += matrix[i][j] - p
                    denom += p * (1 - p)
                if denom > 0.001:
                    delta = num / denom
                    theta[j] += delta
                    max_delta = max(max_delta, abs(delta))

            # Update difficulties (b)
            for i in range(n_items):
                num = 0.0
                denom = 0.0
                for j in range(n_agents):
                    if matrix[i][j] == NOT_ADMINISTERED:
                        continue
                    p = _logistic(theta[j] - b[i])
                    # For difficulty, gradient is negative of ability gradient
                    num += p - matrix[i][j]
                    denom += p * (1 - p)
                if denom > 0.001:
                    delta = num / denom
                    b[i] += delta
                    max_delta = max(max_delta, abs(delta))

            # Centering constraint: mean(b) = 0
            mean_b = sum(b) / n_items
            b = [bi - mean_b for bi in b]

            if max_delta < JMLE_CONVERGENCE:
                logger.info(f"Rasch JMLE converged in {iteration + 1} iterations")
                break

        difficulties = {question_ids[i]: b[i] for i in range(n_items)}
        abilities = {agent_ids[j]: theta[j] for j in range(n_agents)}
        return difficulties, abilities

    # ── Point-Biserial Correlation ──────────────────────────────────────

    @staticmethod
    def _point_biserial(
        item_idx: int,
        matrix: List[List[int]],
        agent_ids: List[str],
        abilities: Dict[str, float],
    ) -> float:
        """Correlation between item response (0/1) and agent ability.

        Items with point-biserial < 0.1 should be flagged for retirement.
        """
        x_vals = []  # binary responses
        y_vals = []  # abilities

        for j, a_id in enumerate(agent_ids):
            if matrix[item_idx][j] == NOT_ADMINISTERED:
                continue
            x_vals.append(matrix[item_idx][j])
            y_vals.append(abilities.get(a_id, 0.0))

        if len(x_vals) < 3:
            return 0.0

        n = len(x_vals)
        n1 = sum(x_vals)
        n0 = n - n1

        if n0 == 0 or n1 == 0:
            return 0.0

        # Mean ability of correct vs incorrect groups
        mean_1 = sum(y for x, y in zip(x_vals, y_vals) if x == 1) / n1
        mean_0 = sum(y for x, y in zip(x_vals, y_vals) if x == 0) / n0

        # Standard deviation of all abilities
        mean_all = sum(y_vals) / n
        var = sum((y - mean_all) ** 2 for y in y_vals) / n
        sd = math.sqrt(var) if var > 0 else 0.001

        # Point-biserial formula
        rpb = (mean_1 - mean_0) / sd * math.sqrt(n1 * n0 / (n * n))
        return round(rpb, 4)

    # ── Calibration Orchestrator ────────────────────────────────────────

    async def calibrate_from_battles(self) -> dict:
        """Run progressive calibration based on battle count.

        <100 battles: raw stats only (p-values, exposure counts)
        100-200: Rasch 1PL
        200+: 2PL via optional girth (graceful fallback to 1PL)
        """
        question_ids, agent_ids, matrix = await self.build_response_matrix()

        # Count completed battles with question_responses
        col = battles_col()
        battle_count = await col.count_documents({
            "status": "completed",
            "question_responses": {"$ne": []},
        })

        n_items = len(question_ids)
        n_agents = len(agent_ids)

        if n_items == 0:
            return {
                "status": "no_data",
                "battle_count": battle_count,
                "items_calibrated": 0,
                "model": "none",
            }

        # Raw stats: always compute p-values and exposure
        raw_stats: Dict[str, dict] = {}
        for i, q_id in enumerate(question_ids):
            administered = [matrix[i][j] for j in range(n_agents) if matrix[i][j] != NOT_ADMINISTERED]
            total = len(administered)
            correct = sum(administered) if administered else 0
            p_val = correct / total if total > 0 else 0.5
            raw_stats[q_id] = {
                "p_value": round(p_val, 4),
                "exposure_count": total,
                "total_responses": total,
            }

        model_used = "raw_stats"
        difficulties: Dict[str, float] = {}
        abilities: Dict[str, float] = {}

        if battle_count >= 100 and n_items >= 3 and n_agents >= 3:
            # Try 2PL via girth if 200+ battles
            if battle_count >= 200:
                try:
                    import importlib.util
                    if importlib.util.find_spec("girth") is not None:
                        # TODO: 2PL calibration with girth when ready
                        pass
                except ImportError:
                    pass

            # Rasch 1PL
            difficulties, abilities = self._rasch_calibrate(
                question_ids, agent_ids, matrix,
            )
            model_used = "rasch_1pl"

        # Build and save item params
        items_saved = 0
        for i, q_id in enumerate(question_ids):
            stats = raw_stats.get(q_id, {})

            # Determine domain from battles
            domain = await self._get_question_domain(q_id)

            pb = 0.0
            if model_used == "rasch_1pl" and abilities:
                pb = self._point_biserial(i, matrix, agent_ids, abilities)

            status = "active"
            if pb < 0.1 and model_used == "rasch_1pl" and stats.get("total_responses", 0) >= 10:
                status = "flagged"

            params = ItemParams(
                question_id=q_id,
                domain=domain,
                difficulty_b=difficulties.get(q_id, 0.0),
                discrimination_a=1.0,  # Rasch assumes a=1.0
                p_value=stats.get("p_value", 0.5),
                point_biserial=pb,
                exposure_count=stats.get("exposure_count", 0),
                total_responses=stats.get("total_responses", 0),
                status=status,
                calibration_model=model_used,
                last_calibrated=datetime.utcnow(),
            )

            await self._save_item_param(params)
            self._item_cache[q_id] = params
            items_saved += 1

        return {
            "status": "calibrated",
            "battle_count": battle_count,
            "items_calibrated": items_saved,
            "agents_estimated": len(abilities),
            "model": model_used,
        }

    async def _get_question_domain(self, question_id: str) -> str:
        """Look up domain for a question from battle data."""
        col = battles_col()
        battle = await col.find_one(
            {"question_responses.question_id": question_id},
            {"question_responses.$": 1},
        )
        if battle and battle.get("question_responses"):
            return battle["question_responses"][0].get("domain", "general")
        return "general"

    # ── Item Quality Report ─────────────────────────────────────────────

    async def item_quality_report(
        self, domain: Optional[str] = None, status: Optional[str] = None,
    ) -> List[dict]:
        """Per-question metrics from stored item params."""
        query: dict = {}
        if domain:
            query["domain"] = domain
        if status:
            query["status"] = status

        col = item_params_col()
        cursor = col.find(query).sort("question_id", 1)
        items = []
        async for doc in cursor:
            doc.pop("_id", None)
            items.append(doc)
        return items

    async def get_item_params(self, question_id: str) -> Optional[dict]:
        """Get params for a single item."""
        if question_id in self._item_cache:
            return self._item_cache[question_id].to_dict()

        doc = await item_params_col().find_one({"question_id": question_id})
        if doc:
            doc.pop("_id", None)
            return doc
        return None

    # ── Ability Estimation (EAP) ────────────────────────────────────────

    async def estimate_ability(
        self, responses: List[dict],
    ) -> dict:
        """Expected A Posteriori ability estimation.

        Args:
            responses: list of {"question_id": str, "correct": bool}

        Returns:
            {"theta": float, "se": float, "responses_used": int}
        """
        # Load item params for each response
        calibrated = []
        for r in responses:
            q_id = r.get("question_id", "")
            correct = r.get("correct", False)

            params = await self._load_item_param(q_id)
            if params is None:
                continue
            if params.get("calibration_model", "none") == "none":
                continue

            calibrated.append({
                "b": params.get("difficulty_b", 0.0),
                "a": params.get("discrimination_a", 1.0),
                "correct": correct,
            })

        if not calibrated:
            return {"theta": 0.0, "se": float("inf"), "responses_used": 0}

        # EAP: quadrature over theta grid
        theta_grid = [x * 0.5 for x in range(-10, 11)]  # -5.0 to 5.0
        prior_sd = 1.5  # Standard normal prior

        log_posteriors = []
        for theta in theta_grid:
            log_prior = -0.5 * (theta / prior_sd) ** 2
            log_lik = 0.0
            for item in calibrated:
                p = _logistic(item["a"] * (theta - item["b"]))
                if item["correct"]:
                    log_lik += math.log(max(p, 1e-10))
                else:
                    log_lik += math.log(max(1 - p, 1e-10))
            log_posteriors.append(log_prior + log_lik)

        # Normalize (log-sum-exp for stability)
        max_lp = max(log_posteriors)
        posteriors = [math.exp(lp - max_lp) for lp in log_posteriors]
        total = sum(posteriors)
        posteriors = [p / total for p in posteriors]

        # EAP: weighted mean
        theta_hat = sum(t * p for t, p in zip(theta_grid, posteriors))

        # SE: sqrt of posterior variance
        var = sum(p * (t - theta_hat) ** 2 for t, p in zip(theta_grid, posteriors))
        se = math.sqrt(var)

        return {
            "theta": round(theta_hat, 4),
            "se": round(se, 4),
            "responses_used": len(calibrated),
        }

    # ── Fisher Information ──────────────────────────────────────────────

    @staticmethod
    def fisher_information(theta: float, b: float, a: float = 1.0) -> float:
        """Fisher information: a² * P * (1-P) where P = logistic(a*(theta-b))."""
        p = _logistic(a * (theta - b))
        return a * a * p * (1 - p)

    # ── Adaptive Question Selection ─────────────────────────────────────

    async def select_adaptive_questions(
        self,
        theta: float,
        administered: Optional[List[str]] = None,
        count: int = 5,
        domains: Optional[List[str]] = None,
    ) -> List[dict]:
        """Select questions via Fisher info maximization with exposure control.

        Picks from top-5 candidates using randomesque method to balance
        measurement precision with exposure control.
        """
        administered = set(administered or [])

        # Load all active item params, optionally filtered by domain
        col = item_params_col()
        query: dict = {"status": "active"}
        if domains:
            query["domain"] = {"$in": domains}
        cursor = col.find(query)
        candidates = []
        async for doc in cursor:
            q_id = doc.get("question_id", "")
            if q_id in administered:
                continue
            b = doc.get("difficulty_b", 0.0)
            a = doc.get("discrimination_a", 1.0)
            info = self.fisher_information(theta, b, a)
            candidates.append({
                "question_id": q_id,
                "domain": doc.get("domain", "general"),
                "difficulty_b": b,
                "discrimination_a": a,
                "fisher_info": round(info, 6),
            })

        if not candidates:
            return []

        # Sort by Fisher info descending
        candidates.sort(key=lambda c: c["fisher_info"], reverse=True)

        # Randomesque: pick from top min(5, len) for each slot
        selected = []
        remaining = list(candidates)
        for _ in range(min(count, len(remaining))):
            pool_size = min(5, len(remaining))
            pool = remaining[:pool_size]
            pick = random.choice(pool)
            selected.append(pick)
            remaining.remove(pick)

        return selected

    # ── Persistence ─────────────────────────────────────────────────────

    async def _save_item_param(self, params: ItemParams):
        """Upsert item params to MongoDB."""
        col = item_params_col()
        doc = params.to_dict()
        doc["last_calibrated"] = params.last_calibrated or datetime.utcnow()
        await col.update_one(
            {"question_id": params.question_id},
            {"$set": doc},
            upsert=True,
        )

    async def _load_item_param(self, question_id: str) -> Optional[dict]:
        """Load item params from cache or MongoDB."""
        if question_id in self._item_cache:
            return self._item_cache[question_id].to_dict()

        doc = await item_params_col().find_one({"question_id": question_id})
        if doc:
            doc.pop("_id", None)
            return doc
        return None


def _logistic(x: float) -> float:
    """Logistic sigmoid: 1 / (1 + exp(-x)), clamped to avoid overflow."""
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

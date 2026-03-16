"""OpenSkill rating engine + Bradley-Terry ranker for AgentTrust Battle Arena.

Uses PlackettLuce model for Bayesian skill estimation.
Maintains 7 independent rating instances: 6 per scoring axis + 1 composite.

BradleyTerryRanker provides batch-computed stable rankings (LMArena-proven).
"""
import logging
import math
import random
from typing import Dict, List, Optional

from openskill.models import PlackettLuce

logger = logging.getLogger(__name__)

# 6 scoring axes (matching evaluator.py)
SCORING_AXES = [
    "accuracy", "safety", "process_quality",
    "reliability", "latency", "schema_quality",
]


class RatingEngine:
    """Manages OpenSkill ratings for battle participants."""

    AXES = SCORING_AXES
    DEFAULT_MU = 25.0
    DEFAULT_SIGMA = 25.0 / 3  # ~8.333

    def __init__(self):
        self.model = PlackettLuce()

    def new_rating(self):
        """Create a fresh rating with default mu/sigma."""
        return self.model.rating(mu=self.DEFAULT_MU, sigma=self.DEFAULT_SIGMA)

    def predict_win(self, rating_a, rating_b) -> float:
        """Predict probability that A wins against B. Returns 0.0-1.0."""
        probs = self.model.predict_win([[rating_a], [rating_b]])
        return probs[0]

    def match_quality(self, rating_a, rating_b) -> float:
        """Compute match quality. 1.0 = perfect match, 0.0 = total mismatch."""
        win_prob = self.predict_win(rating_a, rating_b)
        return 1.0 - abs(win_prob - 0.5) * 2

    def update_ratings(self, rating_a, rating_b, winner: Optional[str]):
        """Update ratings after a battle.

        Args:
            rating_a: Agent A's current rating
            rating_b: Agent B's current rating
            winner: "a", "b", or None (draw)

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        if winner == "a":
            result = self.model.rate([[rating_a], [rating_b]])
        elif winner == "b":
            result = self.model.rate([[rating_b], [rating_a]])
            # Swap back since we reversed the order
            return result[1][0], result[0][0]
        else:
            # Draw — same rank
            result = self.model.rate([[rating_a], [rating_b]], ranks=[1, 1])

        return result[0][0], result[1][0]

    def _rating_from_dict(self, data: Dict) -> object:
        """Reconstruct a rating object from stored {mu, sigma}."""
        return self.model.rating(
            mu=data.get("mu", self.DEFAULT_MU),
            sigma=data.get("sigma", self.DEFAULT_SIGMA),
        )

    def process_battle_scores(
        self,
        scores_a: Dict[str, float],
        scores_b: Dict[str, float],
        overall_a: int,
        overall_b: int,
        existing_ratings_a: Dict,
        existing_ratings_b: Dict,
        winner: Optional[str],
    ) -> Dict:
        """Process a battle result and update all 7 rating axes.

        Args:
            scores_a: Agent A's 6-axis scores {axis: score}
            scores_b: Agent B's 6-axis scores
            overall_a: Agent A's composite score
            overall_b: Agent B's composite score
            existing_ratings_a: Stored ratings {axis: {mu, sigma}, composite: {mu, sigma}}
            existing_ratings_b: Same for agent B
            winner: "a", "b", or None

        Returns:
            Dict with agent_a and agent_b rating deltas per axis + composite.
        """
        result = {"agent_a": {}, "agent_b": {}}

        # Process 6 individual axes
        for axis in self.AXES:
            ra = self._get_or_create_rating(existing_ratings_a, axis)
            rb = self._get_or_create_rating(existing_ratings_b, axis)

            # Per-axis winner based on axis scores
            axis_score_a = scores_a.get(axis, 0)
            axis_score_b = scores_b.get(axis, 0)
            if axis_score_a > axis_score_b:
                axis_winner = "a"
            elif axis_score_b > axis_score_a:
                axis_winner = "b"
            else:
                axis_winner = None

            new_ra, new_rb = self.update_ratings(ra, rb, axis_winner)

            result["agent_a"][axis] = {
                "before": {"mu": ra.mu, "sigma": ra.sigma},
                "after": {"mu": new_ra.mu, "sigma": new_ra.sigma},
            }
            result["agent_b"][axis] = {
                "before": {"mu": rb.mu, "sigma": rb.sigma},
                "after": {"mu": new_rb.mu, "sigma": new_rb.sigma},
            }

        # Process composite rating (uses overall winner)
        ra_comp = self._get_or_create_rating(existing_ratings_a, "composite")
        rb_comp = self._get_or_create_rating(existing_ratings_b, "composite")
        new_ra_comp, new_rb_comp = self.update_ratings(ra_comp, rb_comp, winner)

        result["agent_a"]["composite"] = {
            "before": {"mu": ra_comp.mu, "sigma": ra_comp.sigma},
            "after": {"mu": new_ra_comp.mu, "sigma": new_ra_comp.sigma},
        }
        result["agent_b"]["composite"] = {
            "before": {"mu": rb_comp.mu, "sigma": rb_comp.sigma},
            "after": {"mu": new_rb_comp.mu, "sigma": new_rb_comp.sigma},
        }

        return result

    def _get_or_create_rating(self, existing: Dict, key: str):
        """Get existing rating or create new one."""
        if key in existing:
            return self._rating_from_dict(existing[key])
        return self.new_rating()


class BradleyTerryRanker:
    """Batch Bradley-Terry MLE ranking from battle outcomes.

    Produces stable, order-independent ratings (used by LMArena/Chatbot Arena).
    Complements real-time OpenSkill updates with a consistent leaderboard.
    """

    BASE_RATING = 1000.0
    MAX_ITER = 100
    TOLERANCE = 1e-6

    def fit(self, battles: List[Dict]) -> Dict[str, float]:
        """Compute BT ratings from battle outcomes via iterative MLE.

        Args:
            battles: List of dicts with keys: winner_id, loser_id.
                     Draws are split into two half-wins.

        Returns:
            Dict mapping agent_id -> BT rating (anchored at 1000).
        """
        # Collect agents and win counts
        agents = set()
        wins: Dict[str, Dict[str, float]] = {}  # wins[i][j] = times i beat j

        for b in battles:
            w = b.get("winner_id")
            loser = b.get("loser_id")
            draw = b.get("draw", False)

            if draw:
                a1, a2 = b.get("agent_a_id", ""), b.get("agent_b_id", "")
                if not a1 or not a2:
                    continue
                agents.update([a1, a2])
                wins.setdefault(a1, {})
                wins.setdefault(a2, {})
                wins[a1][a2] = wins[a1].get(a2, 0.0) + 0.5
                wins[a2][a1] = wins[a2].get(a1, 0.0) + 0.5
            elif w and loser:
                agents.update([w, loser])
                wins.setdefault(w, {})
                wins.setdefault(loser, {})
                wins[w][loser] = wins[w].get(loser, 0.0) + 1.0

        if len(agents) < 2:
            return {a: self.BASE_RATING for a in agents}

        # Initialize ratings uniformly
        ratings = {a: 1.0 for a in agents}

        # Iterative MLE (MM algorithm)
        for _ in range(self.MAX_ITER):
            new_ratings = {}
            max_delta = 0.0

            for i in agents:
                numerator = sum(wins.get(i, {}).values())
                if numerator == 0:
                    new_ratings[i] = ratings[i]
                    continue

                denominator = 0.0
                for j in agents:
                    if j == i:
                        continue
                    n_ij = wins.get(i, {}).get(j, 0.0) + wins.get(j, {}).get(i, 0.0)
                    if n_ij > 0:
                        denominator += n_ij / (ratings[i] + ratings[j])

                if denominator > 0:
                    new_ratings[i] = numerator / denominator
                else:
                    new_ratings[i] = ratings[i]

                max_delta = max(max_delta, abs(new_ratings[i] - ratings[i]))

            # Normalize to keep geometric mean at 1
            geo_mean = math.exp(sum(math.log(max(r, 1e-10)) for r in new_ratings.values()) / len(new_ratings))
            ratings = {a: r / geo_mean for a, r in new_ratings.items()}

            if max_delta < self.TOLERANCE:
                break

        # Scale to readable numbers (anchor at BASE_RATING)
        if ratings:
            log_ratings = {a: math.log(max(r, 1e-10)) for a, r in ratings.items()}
            mean_log = sum(log_ratings.values()) / len(log_ratings)
            scale = 400.0 / math.log(10)  # Elo-like scale
            return {a: self.BASE_RATING + (lr - mean_log) * scale for a, lr in log_ratings.items()}
        return {}

    def bootstrap_ci(
        self, battles: List[Dict], n_samples: int = 1000, confidence: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """Bootstrap confidence intervals for BT ratings.

        Returns:
            Dict mapping agent_id -> {mean, ci_lower, ci_upper}
        """
        if not battles:
            return {}

        samples: Dict[str, List[float]] = {}

        for _ in range(n_samples):
            resampled = random.choices(battles, k=len(battles))
            ratings = self.fit(resampled)
            for agent_id, rating in ratings.items():
                samples.setdefault(agent_id, []).append(rating)

        alpha = (1 - confidence) / 2
        result = {}
        for agent_id, s in samples.items():
            s.sort()
            lo_idx = int(alpha * len(s))
            hi_idx = int((1 - alpha) * len(s))
            result[agent_id] = {
                "mean": sum(s) / len(s),
                "ci_lower": s[max(0, lo_idx)],
                "ci_upper": s[min(len(s) - 1, hi_idx)],
            }

        return result

    @staticmethod
    def battles_to_bt_format(battle_docs: List[Dict]) -> List[Dict]:
        """Convert MongoDB battle docs to BT input format."""
        bt_battles = []
        for doc in battle_docs:
            winner = doc.get("winner")
            a_id = doc.get("agent_a", {}).get("target_id", "")
            b_id = doc.get("agent_b", {}).get("target_id", "")
            if not a_id or not b_id:
                continue
            if winner == "a":
                bt_battles.append({"winner_id": a_id, "loser_id": b_id})
            elif winner == "b":
                bt_battles.append({"winner_id": b_id, "loser_id": a_id})
            else:
                bt_battles.append({"draw": True, "agent_a_id": a_id, "agent_b_id": b_id})
        return bt_battles

    async def recompute_rankings(self, domain: Optional[str] = None) -> List[Dict]:
        """Load battles from DB, compute BT rankings, store to quality__rankings.

        Returns list of ranking entries.
        """
        from src.storage.mongodb import battles_col, rankings_col, ladder_col, scores_col
        from src.storage.models import compute_division, DIVISION_CONFIG, Division

        # Load completed battles
        query = {"status": "completed"}
        if domain:
            query["domain"] = domain

        battle_docs = []
        async for doc in battles_col().find(query):
            battle_docs.append(doc)

        bt_battles = self.battles_to_bt_format(battle_docs)

        if not bt_battles:
            return []

        # Fit BT ratings with CI
        ratings = self.fit(bt_battles)
        ci_data = self.bootstrap_ci(bt_battles, n_samples=200)

        # Sort by rating descending
        sorted_agents = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        top3_ids = {aid for aid, _ in sorted_agents[:3]}

        # Gather agent metadata from ladder and scores
        entries = []
        for position, (agent_id, bt_rating) in enumerate(sorted_agents, 1):
            ci = ci_data.get(agent_id, {"mean": bt_rating, "ci_lower": bt_rating, "ci_upper": bt_rating})

            # Get ladder entry for OpenSkill + battle record
            ladder_entry = await ladder_col().find_one({"target_id": agent_id})
            score_entry = await scores_col().find_one({"target_id": agent_id})

            mu = (ladder_entry or {}).get("openskill_mu", 25.0)
            sigma = (ladder_entry or {}).get("openskill_sigma", 8.333)
            record = (ladder_entry or {}).get("battle_record", {"wins": 0, "losses": 0, "draws": 0})
            name = (ladder_entry or {}).get("name", "") or (score_entry or {}).get("name", agent_id)
            total = record.get("wins", 0) + record.get("losses", 0) + record.get("draws", 0)

            division = compute_division(mu, sigma, total, is_top3=agent_id in top3_ids)
            div_cfg = DIVISION_CONFIG.get(Division(division), {})

            entry = {
                "target_id": agent_id,
                "name": name,
                "bt_rating": round(bt_rating, 2),
                "ci_lower": round(ci["ci_lower"], 2),
                "ci_upper": round(ci["ci_upper"], 2),
                "division": division,
                "division_config": {"label": div_cfg.get("label", ""), "color": div_cfg.get("color", ""), "icon": div_cfg.get("icon", "")},
                "battle_record": record,
                "openskill_mu": round(mu, 3),
                "position": position,
                "domain": domain,
            }
            entries.append(entry)

        # Store to rankings collection
        col = rankings_col()
        await col.delete_many({"domain": domain})
        if entries:
            await col.insert_many(entries)

        logger.info(f"Recomputed BT rankings: {len(entries)} agents (domain={domain})")
        return entries

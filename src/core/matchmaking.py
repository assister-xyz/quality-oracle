"""Population-aware matchmaking engine for AgentTrust Arena.

Auto-selects pairing strategy based on population size:
  <10 agents:  Closest-on-ladder (reuse ChallengeLadder)
  10-30:       Swiss-system pairing (sort, pair neighbors, avoid rematches)
  30+:         Batch wave with Lichess-style cost function
"""
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.core.rating import RatingEngine
from src.storage.mongodb import battles_col, ladder_col

logger = logging.getLogger(__name__)


class MatchmakingEngine:
    """Population-aware matchmaking with multiple pairing strategies."""

    SMALL_THRESHOLD = 10
    MEDIUM_THRESHOLD = 30
    REMATCH_COOLDOWN_HOURS = 2

    def __init__(self):
        self.rating_engine = RatingEngine()

    async def select_match(
        self, domain: Optional[str] = None
    ) -> Optional[Tuple[str, str, str]]:
        """Auto-select best match based on population size.

        Returns:
            Tuple of (agent_a_id, agent_b_id, strategy_used) or None.
        """
        agents = await self._get_active_agents(domain)
        if len(agents) < 2:
            return None

        if len(agents) < self.SMALL_THRESHOLD:
            return await self._closest_match(agents, domain)
        elif len(agents) < self.MEDIUM_THRESHOLD:
            pairs = await self.swiss_pair(agents, domain)
            if pairs:
                a, b = pairs[0]
                return (a["target_id"], b["target_id"], "swiss")
            return None
        else:
            return await self._batch_wave_match(agents, domain)

    async def _get_active_agents(self, domain: Optional[str] = None) -> List[Dict]:
        """Get agents from ladder sorted by position."""
        query = {"domain": domain}
        cursor = ladder_col().find(query).sort("position", 1)
        agents = []
        async for doc in cursor:
            agents.append(doc)
        return agents

    async def _closest_match(
        self, agents: List[Dict], domain: Optional[str] = None
    ) -> Optional[Tuple[str, str, str]]:
        """For small populations: pick the closest-rated pair not recently matched."""
        best_pair = None
        best_quality = -1.0

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if await self._is_recent_rematch(a["target_id"], b["target_id"]):
                    continue

                quality = self._compute_match_quality(a, b)
                info = self.information_gain(a, b)
                combined = quality * 0.6 + info * 0.4

                if combined > best_quality:
                    best_quality = combined
                    best_pair = (a["target_id"], b["target_id"], "closest")

        return best_pair

    async def _batch_wave_match(
        self, agents: List[Dict], domain: Optional[str] = None
    ) -> Optional[Tuple[str, str, str]]:
        """For large populations: Lichess-style cost minimization."""
        best_pair = None
        best_cost = float("inf")

        max_rank = len(agents)
        # Sample top candidates to avoid O(n^2) for very large pools
        candidates = agents[:min(60, len(agents))]

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                if await self._is_recent_rematch(a["target_id"], b["target_id"]):
                    continue

                cost = self.match_cost(a, b, max_rank)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (a["target_id"], b["target_id"], "batch_wave")

        return best_pair

    def match_cost(self, a: Dict, b: Dict, max_rank: int) -> float:
        """Lichess-style cost function: penalizes top-rank mismatches more.

        Lower cost = better match.
        """
        pos_a = a.get("position", max_rank)
        pos_b = b.get("position", max_rank)
        mu_a = a.get("openskill_mu", 25.0)
        mu_b = b.get("openskill_mu", 25.0)

        # Rating distance
        rating_diff = abs(mu_a - mu_b)

        # Position-weighted penalty (top ranks matter more)
        avg_pos = (pos_a + pos_b) / 2
        position_weight = 1.0 + (max_rank - avg_pos) / max(max_rank, 1)

        # Information gain bonus (negative cost = good)
        info = self.information_gain(a, b)
        info_bonus = info * 5.0

        return rating_diff * position_weight - info_bonus

    def information_gain(self, a: Dict, b: Dict) -> float:
        """Prioritize uncertain + close-rated pairs.

        Higher value = more informative match.
        """
        sigma_a = a.get("openskill_sigma", 8.333)
        sigma_b = b.get("openskill_sigma", 8.333)
        mu_a = a.get("openskill_mu", 25.0)
        mu_b = b.get("openskill_mu", 25.0)

        # High sigma = uncertain = more to learn
        uncertainty = (sigma_a + sigma_b) / 2.0

        # Close ratings = more discriminating
        mu_diff = abs(mu_a - mu_b)
        closeness = 1.0 / (1.0 + mu_diff)

        # Combined score (0 to ~1)
        return min(1.0, (uncertainty / 8.333) * 0.5 + closeness * 0.5)

    async def swiss_pair(
        self, agents: List[Dict], domain: Optional[str] = None
    ) -> List[Tuple[Dict, Dict]]:
        """Swiss-system pairing: sort by score, pair neighbors, avoid rematches.

        Returns list of (agent_a, agent_b) pairs.
        """
        if len(agents) < 2:
            return []

        # Sort by composite score (wins - losses + mu/10)
        def sort_key(a: Dict) -> float:
            record = a.get("battle_record", {})
            wins = record.get("wins", 0)
            losses = record.get("losses", 0)
            mu = a.get("openskill_mu", 25.0)
            return wins - losses + mu / 10.0

        sorted_agents = sorted(agents, key=sort_key, reverse=True)

        pairs = []
        used = set()

        for i in range(len(sorted_agents)):
            if sorted_agents[i]["target_id"] in used:
                continue

            for j in range(i + 1, len(sorted_agents)):
                if sorted_agents[j]["target_id"] in used:
                    continue

                a, b = sorted_agents[i], sorted_agents[j]
                if await self._is_recent_rematch(a["target_id"], b["target_id"]):
                    continue

                pairs.append((a, b))
                used.add(a["target_id"])
                used.add(b["target_id"])
                break

        return pairs

    async def _is_recent_rematch(self, id_a: str, id_b: str) -> bool:
        """Check if these agents battled recently."""
        cutoff = datetime.utcnow() - timedelta(hours=self.REMATCH_COOLDOWN_HOURS)
        recent = await battles_col().find_one({
            "$or": [
                {"agent_a.target_id": id_a, "agent_b.target_id": id_b},
                {"agent_a.target_id": id_b, "agent_b.target_id": id_a},
            ],
            "created_at": {"$gte": cutoff},
        })
        return recent is not None

    def _compute_match_quality(self, a: Dict, b: Dict) -> float:
        """Compute match quality between two agents using OpenSkill ratings."""
        mu_a = a.get("openskill_mu", 25.0)
        sigma_a = a.get("openskill_sigma", 8.333)
        mu_b = b.get("openskill_mu", 25.0)
        sigma_b = b.get("openskill_sigma", 8.333)

        rating_a = self.rating_engine.model.rating(mu=mu_a, sigma=sigma_a)
        rating_b = self.rating_engine.model.rating(mu=mu_b, sigma=sigma_b)

        return self.rating_engine.match_quality(rating_a, rating_b)

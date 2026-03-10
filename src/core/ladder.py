"""Challenge Ladder & Matchmaking for AgentTrust Arena.

King of the Hill challenge ladder where agents occupy ranked positions
and can challenge those above them. Works with as few as 3 agents.
"""
import logging
import uuid
from datetime import datetime, timedelta

from src.core.rating import RatingEngine
from src.storage.mongodb import battles_col, ladder_col, scores_col

logger = logging.getLogger(__name__)


class ChallengeLadder:
    """Manages ranked challenge ladder with position-based matchmaking."""

    MAX_CHALLENGE_DISTANCE = 5
    DEFENSE_BONUS_MU = 5.0
    CHAMPION_FORFEIT_DAYS = 7

    def __init__(self):
        self.rating_engine = RatingEngine()

    async def get_ladder(self, domain: str | None = None, limit: int = 50) -> list[dict]:
        """Get ranked positions. domain=None → global ladder."""
        query = {"domain": domain}
        cursor = ladder_col().find(query).sort("position", 1).limit(limit)
        entries = []
        async for doc in cursor:
            doc.pop("_id", None)
            entries.append(doc)
        return entries

    async def challenge(
        self, challenger_id: str, target_id: str, domain: str | None = None,
    ) -> str:
        """Validate challenge rules and create a ladder battle.

        Returns battle_id for tracking.
        """
        # Self-challenge check
        if challenger_id == target_id:
            raise ChallengeError("Cannot challenge self")

        col = ladder_col()

        # Look up both entries
        challenger = await col.find_one({"target_id": challenger_id, "domain": domain})
        target = await col.find_one({"target_id": target_id, "domain": domain})

        if not challenger:
            raise ChallengeError(f"Challenger {challenger_id} not on ladder")
        if not target:
            raise ChallengeError(f"Target {target_id} not on ladder")

        # Must challenge above (lower position number = higher rank)
        if target["position"] >= challenger["position"]:
            raise ChallengeError("Can only challenge agents above your position")

        # Distance check
        distance = challenger["position"] - target["position"]
        if distance > self.MAX_CHALLENGE_DISTANCE:
            raise ChallengeError(
                f"Can only challenge within {self.MAX_CHALLENGE_DISTANCE} positions above "
                f"(gap: {distance})"
            )

        # Cooldown check (reuse battle engine cooldown)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        recent = await battles_col().find_one({
            "$or": [
                {"agent_a.target_id": challenger_id, "agent_b.target_id": target_id},
                {"agent_a.target_id": target_id, "agent_b.target_id": challenger_id},
            ],
            "created_at": {"$gte": cutoff},
        })
        if recent:
            raise ChallengeError("Challenge cooldown: wait before challenging again")

        # Create battle record
        battle_id = str(uuid.uuid4())
        battle_doc = {
            "_id": battle_id,
            "battle_id": battle_id,
            "agent_a": {"target_id": challenger_id},
            "agent_b": {"target_id": target_id},
            "match_type": "ladder",
            "domain": domain,
            "status": "pending",
            "created_at": datetime.utcnow(),
        }
        await battles_col().insert_one(battle_doc)

        # Update last_challenge_at for both
        now = datetime.utcnow()
        await col.update_one(
            {"target_id": challenger_id, "domain": domain},
            {"$set": {"last_challenge_at": now}},
        )
        await col.update_one(
            {"target_id": target_id, "domain": domain},
            {"$set": {"last_challenge_at": now}},
        )

        logger.info(
            f"Ladder challenge: {challenger_id} (pos {challenger['position']}) "
            f"→ {target_id} (pos {target['position']})"
        )
        return battle_id

    async def process_battle_result(self, battle_id: str):
        """Process completed battle: swap positions on upset, defense bonus on hold."""
        battle = await battles_col().find_one({"_id": battle_id})
        if not battle:
            raise ValueError(f"Battle {battle_id} not found")

        if battle.get("match_type") != "ladder":
            return  # Only process ladder matches

        winner = battle.get("winner")
        domain = battle.get("domain")
        agent_a_id = battle["agent_a"]["target_id"]
        agent_b_id = battle["agent_b"]["target_id"]

        col = ladder_col()
        entry_a = await col.find_one({"target_id": agent_a_id, "domain": domain})
        entry_b = await col.find_one({"target_id": agent_b_id, "domain": domain})

        if not entry_a or not entry_b:
            logger.warning(f"Ladder entries not found for battle {battle_id}")
            return

        # Determine challenger and defender (challenger has higher position number)
        if entry_a["position"] > entry_b["position"]:
            challenger, defender = entry_a, entry_b
            challenger_is_a = True
        else:
            challenger, defender = entry_b, entry_a
            challenger_is_a = False

        # Determine if challenger won
        challenger_won = (
            (winner == "a" and challenger_is_a) or
            (winner == "b" and not challenger_is_a)
        )

        if challenger_won:
            # UPSET: Challenger takes defender's position
            old_defender_pos = defender["position"]
            old_challenger_pos = challenger["position"]

            # Shift intermediate agents down by 1
            # (agents between defender and challenger positions)
            await col.update_many(
                {
                    "domain": domain,
                    "position": {"$gt": old_defender_pos, "$lt": old_challenger_pos},
                },
                {"$inc": {"position": 1}},
            )

            # Swap positions
            await col.update_one(
                {"target_id": challenger["target_id"], "domain": domain},
                {
                    "$set": {"position": old_defender_pos, "last_challenge_at": datetime.utcnow()},
                    "$inc": {"battle_record.wins": 1},
                },
            )
            await col.update_one(
                {"target_id": defender["target_id"], "domain": domain},
                {
                    "$set": {"position": old_defender_pos + 1, "last_challenge_at": datetime.utcnow()},
                    "$inc": {"battle_record.losses": 1},
                },
            )

            logger.info(
                f"Ladder upset: {challenger['target_id']} moved {old_challenger_pos}→{old_defender_pos}"
            )
        elif winner is None:
            # DRAW: No position change, update records
            await col.update_one(
                {"target_id": challenger["target_id"], "domain": domain},
                {
                    "$set": {"last_challenge_at": datetime.utcnow()},
                    "$inc": {"battle_record.draws": 1},
                },
            )
            await col.update_one(
                {"target_id": defender["target_id"], "domain": domain},
                {
                    "$set": {"last_challenge_at": datetime.utcnow()},
                    "$inc": {"battle_record.draws": 1},
                },
            )
        else:
            # DEFENSE: Defender wins, positions unchanged, defense bonus
            await col.update_one(
                {"target_id": challenger["target_id"], "domain": domain},
                {
                    "$set": {"last_challenge_at": datetime.utcnow()},
                    "$inc": {"battle_record.losses": 1},
                },
            )
            await col.update_one(
                {"target_id": defender["target_id"], "domain": domain},
                {
                    "$set": {
                        "last_challenge_at": datetime.utcnow(),
                        "openskill_mu": defender.get("openskill_mu", 25.0) + self.DEFENSE_BONUS_MU,
                    },
                    "$inc": {
                        "defenses": 1,
                        "battle_record.wins": 1,
                    },
                },
            )
            logger.info(f"Ladder defense: {defender['target_id']} held position {defender['position']}")

    async def auto_seed(self, domain: str | None = None) -> int:
        """Seed ladder from existing quality__scores.

        Returns count of agents seeded.
        """
        col = ladder_col()
        s_col = scores_col()

        # Get all scored agents, ordered by score descending
        cursor = s_col.find({}).sort("current_score", -1)

        # Get current max position on ladder
        existing_count = await col.count_documents({"domain": domain})
        position = existing_count + 1

        seeded = 0
        async for score_doc in cursor:
            target_id = score_doc["target_id"]

            # Skip if already on ladder
            existing = await col.find_one({"target_id": target_id, "domain": domain})
            if existing:
                continue

            entry = {
                "target_id": target_id,
                "domain": domain,
                "position": position,
                "target_url": score_doc.get("target_url", ""),
                "name": score_doc.get("name", ""),
                "overall_score": score_doc.get("current_score", 0),
                "openskill_mu": score_doc.get("openskill_mu", 25.0),
                "openskill_sigma": score_doc.get("openskill_sigma", 8.333),
                "battle_record": {"wins": 0, "losses": 0, "draws": 0},
                "last_challenge_at": None,
                "seeded_at": datetime.utcnow(),
                "defenses": 0,
            }
            await col.insert_one(entry)
            position += 1
            seeded += 1

        if seeded:
            logger.info(f"Auto-seeded {seeded} agents onto ladder (domain={domain})")
        return seeded

    async def check_champion_forfeit(self, domain: str | None = None):
        """Check if #1 hasn't battled in CHAMPION_FORFEIT_DAYS. Drop to #2 if so."""
        col = ladder_col()

        champion = await col.find_one({"position": 1, "domain": domain})
        if not champion:
            return

        last_challenge = champion.get("last_challenge_at")
        if last_challenge is None:
            # Use seeded_at as fallback
            last_challenge = champion.get("seeded_at", datetime.utcnow())

        cutoff = datetime.utcnow() - timedelta(days=self.CHAMPION_FORFEIT_DAYS)
        if last_challenge > cutoff:
            return  # Champion is active

        # Forfeit: swap #1 and #2
        runner_up = await col.find_one({"position": 2, "domain": domain})
        if not runner_up:
            return  # No one to promote

        await col.update_one(
            {"target_id": champion["target_id"], "domain": domain},
            {"$set": {"position": 2}},
        )
        await col.update_one(
            {"target_id": runner_up["target_id"], "domain": domain},
            {"$set": {"position": 1, "champion_by_forfeit": True}},
        )

        logger.info(
            f"Champion forfeit: {champion['target_id']} dropped to #2, "
            f"{runner_up['target_id']} promoted to #1"
        )

    async def predict_match(self, id_a: str, id_b: str, domain: str | None = None) -> dict:
        """Predict match outcome between two agents.

        Returns win probabilities, match quality, and recommendation.
        """
        col = ladder_col()

        entry_a = await col.find_one({"target_id": id_a})
        entry_b = await col.find_one({"target_id": id_b})

        mu_a = (entry_a or {}).get("openskill_mu", 25.0)
        sigma_a = (entry_a or {}).get("openskill_sigma", 8.333)
        mu_b = (entry_b or {}).get("openskill_mu", 25.0)
        sigma_b = (entry_b or {}).get("openskill_sigma", 8.333)

        rating_a = self.rating_engine.model.rating(mu=mu_a, sigma=sigma_a)
        rating_b = self.rating_engine.model.rating(mu=mu_b, sigma=sigma_b)

        win_prob_a = self.rating_engine.predict_win(rating_a, rating_b)
        win_prob_b = 1.0 - win_prob_a
        match_quality = self.rating_engine.match_quality(rating_a, rating_b)

        # Recommendation
        if match_quality >= 0.70:
            recommendation = "good_match"
        elif match_quality >= 0.30:
            recommendation = "one_sided"
        else:
            recommendation = "too_unbalanced"

        return {
            "agent_a_id": id_a,
            "agent_b_id": id_b,
            "win_probability_a": round(win_prob_a, 4),
            "win_probability_b": round(win_prob_b, 4),
            "match_quality": round(match_quality, 4),
            "recommendation": recommendation,
        }


class ChallengeError(Exception):
    """Raised when a challenge request is invalid."""
    pass

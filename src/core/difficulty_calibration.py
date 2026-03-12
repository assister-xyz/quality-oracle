"""Difficulty calibration via empirical pass-rate tracking.

Tracks per-question pass rates and adjusts difficulty labels after
sufficient data accumulates (50+ evaluations per question).
Persists stats to MongoDB quality__question_stats collection.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.question_pools import ChallengeQuestion

logger = logging.getLogger(__name__)


@dataclass
class DifficultyStats:
    """Tracks per-question pass rates for difficulty calibration."""
    question_id: str
    attempts: int = 0
    passes: int = 0

    @property
    def pass_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.passes / self.attempts

    @property
    def suggested_difficulty(self) -> str:
        """Suggest difficulty based on empirical pass rate."""
        rate = self.pass_rate
        if rate >= 0.8:
            return "easy"
        elif rate >= 0.4:
            return "medium"
        return "hard"


class DifficultyTracker:
    """Tracks per-question pass rates for difficulty calibration.

    After 50+ evaluations, suggested_difficulty reflects empirical data
    rather than author estimates. Stores in MongoDB quality__question_stats.
    """

    def __init__(self):
        self._stats: Dict[str, DifficultyStats] = {}

    def record(self, question_id: str, passed: bool) -> None:
        """Record whether an agent passed a question (score >= 70)."""
        if question_id not in self._stats:
            self._stats[question_id] = DifficultyStats(question_id=question_id)
        stats = self._stats[question_id]
        stats.attempts += 1
        if passed:
            stats.passes += 1

    def get_stats(self, question_id: str) -> Optional[DifficultyStats]:
        return self._stats.get(question_id)

    def calibrated_questions(
        self, questions: List[ChallengeQuestion], min_attempts: int = 50,
    ) -> List[ChallengeQuestion]:
        """Return questions with difficulty adjusted based on empirical data.

        Only adjusts questions with >= min_attempts evaluations.
        """
        result = []
        for q in questions:
            stats = self._stats.get(q.id)
            if stats and stats.attempts >= min_attempts:
                suggested = stats.suggested_difficulty
                if suggested != q.difficulty:
                    logger.info(
                        f"Calibrating {q.id}: {q.difficulty} → {suggested} "
                        f"(pass_rate={stats.pass_rate:.2f}, n={stats.attempts})"
                    )
                    q = ChallengeQuestion(
                        question=q.question,
                        domain=q.domain,
                        difficulty=suggested,
                        reference_answer=q.reference_answer,
                        category=q.category,
                    )
            result.append(q)
        return result

    def summary(self) -> Dict[str, Dict]:
        """Return summary stats for all tracked questions."""
        return {
            qid: {
                "attempts": s.attempts,
                "passes": s.passes,
                "pass_rate": round(s.pass_rate, 3),
                "suggested_difficulty": s.suggested_difficulty,
            }
            for qid, s in self._stats.items()
        }

    async def load_from_db(self) -> int:
        """Load question stats from MongoDB. Returns count of loaded records."""
        try:
            from src.storage.mongodb import question_stats_col
            col = question_stats_col()
            count = 0
            async for doc in col.find():
                qid = doc["question_id"]
                self._stats[qid] = DifficultyStats(
                    question_id=qid,
                    attempts=doc.get("attempts", 0),
                    passes=doc.get("passes", 0),
                )
                count += 1
            if count:
                logger.info(f"Loaded {count} question stats from DB")
            return count
        except Exception as e:
            logger.warning(f"Failed to load question stats from DB: {e}")
            return 0

    async def save_to_db(self) -> int:
        """Persist current stats to MongoDB via upsert. Returns count saved."""
        if not self._stats:
            return 0
        try:
            from src.storage.mongodb import question_stats_col
            col = question_stats_col()
            count = 0
            for qid, stats in self._stats.items():
                await col.update_one(
                    {"question_id": qid},
                    {"$set": {
                        "question_id": qid,
                        "attempts": stats.attempts,
                        "passes": stats.passes,
                        "pass_rate": round(stats.pass_rate, 3),
                        "suggested_difficulty": stats.suggested_difficulty,
                    }},
                    upsert=True,
                )
                count += 1
            logger.info(f"Saved {count} question stats to DB")
            return count
        except Exception as e:
            logger.warning(f"Failed to save question stats to DB: {e}")
            return 0

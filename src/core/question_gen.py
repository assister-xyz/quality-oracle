"""
LLM-based question paraphrasing for anti-gaming protection.

Slightly rephrases questions before each evaluation so agents cannot
memorize exact answers. Uses the same LLM provider chain as the judge.
"""
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from src.core.question_pools import ChallengeQuestion

logger = logging.getLogger(__name__)

# Reuse provider URLs from LLMJudge
_PROVIDER_URLS = {
    "cerebras": "https://api.cerebras.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

_PARAPHRASE_PROMPT = """Rephrase the following question while keeping its exact meaning and difficulty.
Do NOT change what is being asked — only rephrase the wording.
Return ONLY the rephrased question text, nothing else.

Original: {question}"""

# Cache paraphrased variants to avoid repeated LLM calls for the same question
_paraphrase_cache: Dict[str, List[str]] = {}
_CACHE_MAX_VARIANTS = 5


class QuestionParaphraser:
    """Generates slight rephrasings of evaluation questions to prevent memorization."""

    def __init__(self):
        # Try to get an API key from environment (same as LLMJudge uses)
        self._api_key = os.getenv("CEREBRAS_API_KEY", "").split(",")[0].strip()
        self._model = "llama3.1-8b"
        self._provider = "cerebras"
        self._base_url = _PROVIDER_URLS["cerebras"]

        if not self._api_key:
            self._api_key = os.getenv("GROQ_API_KEY", "").split(",")[0].strip()
            self._model = "llama-3.1-8b-instant"
            self._provider = "groq"
            self._base_url = _PROVIDER_URLS["groq"]

        self._available = bool(self._api_key)
        if not self._available:
            logger.info("QuestionParaphraser: no LLM keys, using synonym fallback")

    @property
    def is_available(self) -> bool:
        return self._available

    async def paraphrase(self, question: ChallengeQuestion) -> ChallengeQuestion:
        """Return a paraphrased copy of the question.

        Falls back to the original if LLM is unavailable or fails.
        """
        cache_key = question.id

        # Check cache first — return a random cached variant
        if cache_key in _paraphrase_cache and _paraphrase_cache[cache_key]:
            variant_text = random.choice(_paraphrase_cache[cache_key])
            return ChallengeQuestion(
                question=variant_text,
                domain=question.domain,
                difficulty=question.difficulty,
                reference_answer=question.reference_answer,
                category=question.category,
            )

        # Try LLM paraphrasing
        if self._available:
            rephrased = await self._call_llm(question.question)
            if rephrased and rephrased != question.question:
                # Cache variant
                if cache_key not in _paraphrase_cache:
                    _paraphrase_cache[cache_key] = []
                if len(_paraphrase_cache[cache_key]) < _CACHE_MAX_VARIANTS:
                    _paraphrase_cache[cache_key].append(rephrased)

                return ChallengeQuestion(
                    question=rephrased,
                    domain=question.domain,
                    difficulty=question.difficulty,
                    reference_answer=question.reference_answer,
                    category=question.category,
                )

        # Fallback: apply simple synonym substitution
        rephrased = self._synonym_fallback(question.question)
        return ChallengeQuestion(
            question=rephrased,
            domain=question.domain,
            difficulty=question.difficulty,
            reference_answer=question.reference_answer,
            category=question.category,
        )

    async def paraphrase_batch(
        self, questions: List[ChallengeQuestion],
    ) -> List[ChallengeQuestion]:
        """Paraphrase a list of questions. Non-blocking, best-effort."""
        import asyncio
        tasks = [self.paraphrase(q) for q in questions]
        return await asyncio.gather(*tasks)

    async def _call_llm(self, question_text: str) -> Optional[str]:
        """Call LLM to paraphrase a single question."""
        prompt = _PARAPHRASE_PROMPT.format(question=question_text)
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You rephrase questions precisely. Return only the rephrased question."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, json=body)
                if response.status_code != 200:
                    logger.warning(f"Paraphrase LLM returned {response.status_code}")
                    return None
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Sanity: result should be a single question, roughly same length
                if len(content) < 10 or len(content) > len(question_text) * 3:
                    return None
                return content
        except Exception as e:
            logger.warning(f"Paraphrase LLM error: {e}")
            return None

    @staticmethod
    def _synonym_fallback(question: str) -> str:
        """Simple deterministic rephrasing via word substitution.

        Used when no LLM is available. Provides minimal anti-memorization.
        """
        replacements = [
            ("Explain", "Describe"),
            ("Describe", "Explain"),
            ("What is", "Define"),
            ("How does", "In what way does"),
            ("Why is", "For what reason is"),
            ("What are", "List"),
            ("Generate", "Create"),
            ("Implement", "Write"),
            ("Write", "Implement"),
        ]
        result = question
        for old, new in replacements:
            if result.startswith(old):
                result = new + result[len(old):]
                break
        return result


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

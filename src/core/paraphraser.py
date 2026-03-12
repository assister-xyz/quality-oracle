"""
Anti-gaming question paraphraser for AgentTrust.

Prevents servers from memorizing test answers by generating
unique question variants for each evaluation run.

Two strategies:
1. Template-based (fast, no API): Structural transforms — reorder, synonym swap, rephrase
2. LLM-based (slower, richer): Full paraphrase via LLM while preserving semantic meaning

Template-based is default; LLM-based activates when judge LLM is available.
"""
import hashlib
import logging
import random
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Synonym maps for template-based paraphrasing ─────────────────────────────

_VERB_SYNONYMS: Dict[str, List[str]] = {
    "explain": ["describe", "elaborate on", "walk through", "break down", "clarify"],
    "describe": ["explain", "outline", "detail", "characterize", "illustrate"],
    "what is": ["define", "what does", "explain what", "describe what"],
    "how does": ["in what way does", "explain how", "describe how", "walk through how"],
    "how do": ["in what way do", "explain how", "describe the process by which"],
    "generate": ["create", "produce", "write", "build", "construct"],
    "implement": ["build", "code", "create", "write", "develop"],
    "list": ["enumerate", "name", "identify", "give examples of"],
    "compare": ["contrast", "differentiate between", "distinguish", "what are the differences between"],
    "why": ["for what reason", "what causes", "what motivates"],
}

_NOUN_SYNONYMS: Dict[str, List[str]] = {
    "function": ["method", "routine", "procedure", "subroutine"],
    "attack": ["exploit", "vulnerability", "threat", "attack vector"],
    "system": ["platform", "framework", "infrastructure", "architecture"],
    "protocol": ["standard", "specification", "mechanism", "scheme"],
    "metric": ["measure", "indicator", "parameter", "benchmark"],
    "data": ["information", "records", "content", "payload"],
    "result": ["output", "response", "outcome", "return value"],
    "error": ["failure", "exception", "fault", "issue"],
    "input": ["parameter", "argument", "request", "payload"],
    "tool": ["endpoint", "function", "capability", "service"],
}

# ── Structural transform templates ───────────────────────────────────────────

_QUESTION_PREFIXES = [
    "",
    "Can you ",
    "Please ",
    "I'd like you to ",
    "Help me understand: ",
    "Quick question: ",
]

_QUESTION_SUFFIXES = [
    "",
    " Be specific.",
    " Include key details.",
    " Explain clearly.",
    " Be concise.",
    " Provide a thorough answer.",
]

# Expected behavior paraphrase templates
_EXPECTED_TRANSFORMS = [
    ("Should return", ["Must return", "Expected to return", "Needs to return", "Will return"]),
    ("Should handle", ["Must handle", "Expected to handle", "Needs to handle", "Will handle"]),
    ("Should process", ["Must process", "Expected to process", "Needs to process"]),
    ("Should reject", ["Must reject", "Expected to reject", "Needs to reject"]),
    ("including", ["with", "containing", "that includes", "along with"]),
    ("relevant", ["matching", "appropriate", "pertinent", "related"]),
    ("clear error message", ["descriptive error", "helpful error message", "informative error"]),
]


def _apply_synonym_swap(text: str, seed: int) -> str:
    """Replace verbs/nouns with synonyms based on seed for determinism."""
    rng = random.Random(seed)
    result = text

    # Try verb synonyms
    text_lower = text.lower()
    for original, synonyms in _VERB_SYNONYMS.items():
        if original in text_lower:
            replacement = rng.choice(synonyms)
            # Case-aware replacement
            idx = text_lower.find(original)
            if idx == 0 or text[idx - 1] in (" ", "\n", "."):
                # Preserve original case
                if text[idx].isupper():
                    replacement = replacement.capitalize()
                result = result[:idx] + replacement + result[idx + len(original):]
                break  # Only one swap per call

    return result


def _apply_structural_transform(question: str, seed: int) -> str:
    """Add prefix/suffix variations to a question."""
    rng = random.Random(seed)

    # Don't double-prefix if already starts with a transform word
    prefix = rng.choice(_QUESTION_PREFIXES)
    if prefix and question[0].isupper() and prefix:
        question = question[0].lower() + question[1:]

    suffix = rng.choice(_QUESTION_SUFFIXES)

    # Ensure question mark if it's a question
    if question.rstrip().endswith("?") and suffix:
        question = question.rstrip().rstrip("?") + "?"
        suffix = ""  # Don't add suffix after question mark

    return f"{prefix}{question}{suffix}".strip()


def _transform_expected(expected: str, seed: int) -> str:
    """Paraphrase expected behavior description."""
    rng = random.Random(seed)
    result = expected

    for original, replacements in _EXPECTED_TRANSFORMS:
        if original in result:
            replacement = rng.choice(replacements)
            result = result.replace(original, replacement, 1)
            break  # One transform per call

    return result


class QuestionParaphraser:
    """
    Anti-gaming paraphraser that generates unique question variants.

    Each evaluation gets a unique seed derived from target_id + timestamp,
    ensuring different question phrasings across evaluations while
    maintaining deterministic behavior within a single run.

    LLM paraphrasing auto-activates for certified/audited eval modes
    when an LLM judge is available.
    """

    def __init__(self, llm_judge=None, eval_mode: str = "verified"):
        """
        Args:
            llm_judge: Optional LLMJudge for richer LLM-based paraphrasing.
                       Falls back to template-based if None or LLM unavailable.
            eval_mode: Evaluation mode — LLM paraphrase enabled for certified/audited.
        """
        self._llm_judge = llm_judge
        self._eval_mode = eval_mode
        self._llm_available = (
            llm_judge is not None
            and hasattr(llm_judge, 'is_llm_available')
            and llm_judge.is_llm_available
        )
        # Auto-enable LLM paraphrasing for certified/audited modes
        self._use_llm = (
            self._llm_available
            and eval_mode in ("certified", "audited")
        )

    def generate_seed(self, target_id: str, run_id: str = "") -> int:
        """Generate a deterministic seed from target + run identifiers."""
        raw = f"{target_id}:{run_id}:{random.random()}"
        return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)

    def paraphrase_question(self, question: str, seed: int) -> str:
        """Paraphrase a question using template-based transforms."""
        # Apply synonym swap
        result = _apply_synonym_swap(question, seed)

        # Apply structural transform (prefix/suffix)
        result = _apply_structural_transform(result, seed + 1)

        return result

    def paraphrase_expected(self, expected: str, seed: int) -> str:
        """Paraphrase expected behavior description."""
        return _transform_expected(expected, seed + 2)

    def paraphrase_test_case(
        self,
        test_case: dict,
        seed: int,
    ) -> dict:
        """
        Paraphrase a complete test case (question + expected).

        Returns a new dict with paraphrased question and expected,
        preserving all other fields.
        """
        result = test_case.copy()
        result["question"] = self.paraphrase_question(
            test_case["question"], seed
        )
        result["expected"] = self.paraphrase_expected(
            test_case["expected"], seed
        )
        result["paraphrased"] = True
        result["original_question"] = test_case["question"]
        return result

    def paraphrase_challenge(
        self,
        question: str,
        reference_answer: str,
        seed: int,
    ) -> Tuple[str, str]:
        """
        Paraphrase a domain challenge question + reference answer.

        Returns (paraphrased_question, original_reference_answer).
        Reference answers are NOT paraphrased to preserve ground truth.
        """
        return self.paraphrase_question(question, seed), reference_answer

    async def paraphrase_with_llm(
        self,
        question: str,
        seed: int,
    ) -> str:
        """
        Use LLM to generate a semantically equivalent paraphrase.

        Falls back to template-based if LLM fails.
        """
        if not self._use_llm:
            return self.paraphrase_question(question, seed)

        try:
            import httpx

            prompt = (
                "Rephrase the following question to ask the same thing "
                "in a different way. Keep the same meaning and difficulty level. "
                "Return ONLY the rephrased question, nothing else.\n\n"
                f"Original: {question}\n\n"
                "Rephrased:"
            )

            judge = self._llm_judge
            url = f"{judge.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {judge.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": judge.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 200,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, json=body)
                if response.status_code == 200:
                    data = response.json()
                    rephrased = data["choices"][0]["message"]["content"].strip()
                    if rephrased and len(rephrased) > 10:
                        return rephrased

        except Exception as e:
            logger.debug(f"LLM paraphrase failed, using template: {e}")

        return self.paraphrase_question(question, seed)

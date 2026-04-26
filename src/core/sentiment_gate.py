"""Production-vs-tutorial sentiment gate (M2 decision).

Used by SOL-02 (and any future probe with a tutorial false-positive risk) to
decide whether a regex hit on private-key material should be flagged.

Flow per spec §"Sentiment gate decision (M2)":

1. Cheap path-list pre-filter — paths under ``docs/troubleshooting`` /
   ``docs/quickstart`` short-circuit to ``tutorial``. This catches the
   common SendAI-style `docs/troubleshooting.md` keypair-from-file
   examples without spending an LLM call.
2. LLM-judge intent classifier (~$0.005/skill on Anthropic, $0 on the
   Cerebras free tier) makes the final call when the path is ambiguous.

The classifier is injected (``judge_fn``) so unit tests can pin its
behaviour deterministically without mocking the SDK.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# Cheap path heuristics — anything inside these subtrees is presumed
# tutorial. Mirrors the fallback rule shipped while the LLM gate is
# unavailable (no API keys configured).
_TUTORIAL_PATH_TOKENS = (
    "troubleshooting",
    "quickstart",
    "/docs/getting-started",
    "/tutorial",
    "/tutorials",
    "/learn/",
)

# A line nearby that mentions "do not", "never", "never store", "tutorial",
# "for demo only" or similar shifts the call back toward tutorial. Cheap.
_TUTORIAL_HINTS_RE = re.compile(
    r"\b(tutorial|do\s*not|never|for\s+demo|example\s+only|do\s+not\s+commit|sample)\b",
    re.IGNORECASE,
)


@dataclass
class SentimentDecision:
    """Outcome of a sentiment-gate classification."""

    is_production: bool
    reason: str  # human-readable; surfaces in ProbeResult.note
    method: str  # ``path_filter`` | ``llm_judge`` | ``heuristic`` | ``conservative_default``
    cost_dollars: float = 0.0


# A judge_fn receives ``(file_path, line_no, surrounding_text)`` and returns
# True if the context is *production*. It can be sync or async; the gate
# wraps both.
JudgeFn = Callable[[str, int, str], "bool | Awaitable[bool]"]


def _path_filter(file_path: Path) -> Optional[bool]:
    """Path-only short circuit. ``None`` means "ambiguous, ask the LLM"."""
    p = str(file_path).lower().replace("\\", "/")
    name = file_path.name
    # Strong tutorial signal — SKILL.md body is *always* production-context
    # (the body is what activated agents read), so keep that even if the
    # path also contains "tutorial" somewhere weird.
    if name == "SKILL.md":
        return True
    for tok in _TUTORIAL_PATH_TOKENS:
        if tok in p:
            return False
    # ``examples/`` is production-context per spec — those files ship as
    # part of the skill and are read by activated agents.
    if "/examples/" in p or p.startswith("examples/"):
        return True
    return None  # ambiguous


def is_production_context_sync(
    file_path: Path,
    line_no: int = 0,
    surrounding_text: str = "",
) -> SentimentDecision:
    """Synchronous fallback when no LLM judge is available.

    Behaviour:

    * Path filter dominant.
    * If still ambiguous, look for tutorial hints in the surrounding text.
    * Conservative default = production (matches spec snippet `True`).
    """
    pf = _path_filter(file_path)
    if pf is True:
        return SentimentDecision(True, "path:production", "path_filter")
    if pf is False:
        return SentimentDecision(False, "path:tutorial", "path_filter")
    # Heuristic on surrounding text.
    if surrounding_text and _TUTORIAL_HINTS_RE.search(surrounding_text):
        return SentimentDecision(False, "tutorial_hint_in_context", "heuristic")
    return SentimentDecision(True, "conservative_default_production", "conservative_default")


async def is_production_context(
    file_path: Path,
    line_no: int = 0,
    surrounding_text: str = "",
    *,
    judge_fn: Optional[JudgeFn] = None,
    cost_per_call: float = 0.005,
) -> SentimentDecision:
    """Async LLM-aware sentiment gate.

    Routes through the cheap path filter first; only invokes ``judge_fn``
    when the file location is ambiguous. ``judge_fn`` may be sync or async.
    """
    pf = _path_filter(file_path)
    if pf is True:
        return SentimentDecision(True, "path:production", "path_filter")
    if pf is False:
        return SentimentDecision(False, "path:tutorial", "path_filter")

    if judge_fn is None:
        return is_production_context_sync(file_path, line_no, surrounding_text)

    try:
        ret = judge_fn(str(file_path), line_no, surrounding_text)
        if hasattr(ret, "__await__"):
            verdict = bool(await ret)  # type: ignore[arg-type]
        else:
            verdict = bool(ret)
        return SentimentDecision(
            is_production=verdict,
            reason="llm_judge_intent",
            method="llm_judge",
            cost_dollars=cost_per_call,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("sentiment gate LLM judge failed: %s; falling back", exc)
        return is_production_context_sync(file_path, line_no, surrounding_text)

"""Audit trail logging for QO-047 — persists eval-internal signals for debugging.

All write functions are best-effort: failures are logged at WARNING but never
propagate. Audit logging must NEVER break the main eval flow.

References:
- QO-047 spec — assisterr-workflow/specs/active/QO-047-audit-trail-hardening.md
- 5 collections: tool_calls, judge_calls, consensus_votes, sanitization_events,
  probe_executions
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Truncation limits to keep audit storage manageable
MAX_RESPONSE_CHARS = 5000
MAX_PROMPT_CHARS = 5000
MAX_PROBE_INPUT_CHARS = 1000
MAX_PROBE_RESPONSE_CHARS = 2000


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Truncate text to limit chars. Returns (truncated_text, was_truncated)."""
    if not text:
        return "", False
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _hash(text: str) -> str:
    """SHA256 hash for prompt/response fingerprinting."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


def _is_enabled() -> bool:
    """Check if audit logging is enabled via settings."""
    try:
        from src.config import settings
        return getattr(settings, "audit_logging_enabled", True)
    except Exception:
        return True


# ── Tool calls ──────────────────────────────────────────────────────────────


async def log_tool_call(
    evaluation_id: Optional[str],
    target_id: Optional[str],
    tool_name: str,
    arguments: Dict[str, Any],
    response_text: str,
    is_error: bool,
    latency_ms: int,
    call_index: int = 0,
    test_type: str = "",
) -> None:
    """Persist a single MCP tool call. Best-effort, never raises."""
    if not _is_enabled() or not evaluation_id:
        return
    try:
        from src.storage.mongodb import tool_calls_col

        truncated_response, was_truncated = _truncate(response_text or "", MAX_RESPONSE_CHARS)

        await tool_calls_col().insert_one({
            "evaluation_id": evaluation_id,
            "target_id": target_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "response_text": truncated_response,
            "response_truncated": was_truncated,
            "response_length": len(response_text or ""),
            "is_error": is_error,
            "latency_ms": latency_ms,
            "call_index": call_index,
            "test_type": test_type,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        logger.warning(f"audit_log.log_tool_call failed (non-fatal): {e}")


# ── Judge calls ─────────────────────────────────────────────────────────────


async def log_judge_call(
    evaluation_id: Optional[str],
    call_index: int,
    provider: str,
    model: str,
    question: str,
    expected: str,
    answer: str,
    raw_response_text: str,
    parsed_score: int,
    parsed_explanation: str,
    method: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
) -> None:
    """Persist a single LLM judge call. Best-effort."""
    if not _is_enabled() or not evaluation_id:
        return
    try:
        from src.storage.mongodb import judge_calls_col

        # Build full prompt for hashing (debug-replay friendly)
        full_prompt = f"Q: {question}\nE: {expected}\nA: {answer}"
        prompt_hash = _hash(full_prompt)

        truncated_response, response_truncated = _truncate(
            raw_response_text or "", 2000
        )

        await judge_calls_col().insert_one({
            "evaluation_id": evaluation_id,
            "call_index": call_index,
            "provider": provider,
            "model": model,
            "question": _truncate(question or "", 2000)[0],
            "expected": _truncate(expected or "", 2000)[0],
            "answer": _truncate(answer or "", MAX_RESPONSE_CHARS)[0],
            "prompt_hash": prompt_hash,
            "raw_response_text": truncated_response,
            "raw_response_truncated": response_truncated,
            "parsed_score": parsed_score,
            "parsed_explanation": parsed_explanation,
            "method": method,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        logger.warning(f"audit_log.log_judge_call failed (non-fatal): {e}")


# ── Consensus votes ─────────────────────────────────────────────────────────


async def log_consensus_votes(
    evaluation_id: Optional[str],
    consensus_round_id: str,
    votes: List[Dict[str, Any]],
    median_score: int,
) -> None:
    """Persist individual judge votes from a consensus round.

    Each vote dict should contain: judge_index, provider, score, explanation,
    latency_ms, was_tiebreaker (bool).
    """
    if not _is_enabled() or not evaluation_id or not votes:
        return
    try:
        from src.storage.mongodb import consensus_votes_col

        docs = []
        now = datetime.now(timezone.utc)
        for vote in votes:
            docs.append({
                "evaluation_id": evaluation_id,
                "consensus_round_id": consensus_round_id,
                "judge_index": vote.get("judge_index", 0),
                "provider": vote.get("provider", "unknown"),
                "score": vote.get("score", 0),
                "explanation": _truncate(vote.get("explanation", ""), 1000)[0],
                "latency_ms": vote.get("latency_ms", 0),
                "agreement_with_median": abs(vote.get("score", 0) - median_score),
                "was_tiebreaker": vote.get("was_tiebreaker", False),
                "created_at": now,
            })

        if docs:
            await consensus_votes_col().insert_many(docs)
    except Exception as e:
        logger.warning(f"audit_log.log_consensus_votes failed (non-fatal): {e}")


# ── Sanitization events (QO-043) ────────────────────────────────────────────


async def log_sanitization(
    evaluation_id: Optional[str],
    judge_call_index: int,
    sanitization_result: Any,
) -> None:
    """Persist a sanitization event from the judge sanitizer.

    sanitization_result is a SanitizationResult dataclass from judge_sanitizer.
    """
    if not _is_enabled() or not evaluation_id or not sanitization_result:
        return
    if not getattr(sanitization_result, "had_detections", False):
        return
    try:
        from src.storage.mongodb import sanitization_events_col

        await sanitization_events_col().insert_one({
            "evaluation_id": evaluation_id,
            "judge_call_index": judge_call_index,
            "original_length": sanitization_result.original_length,
            "sanitized_length": sanitization_result.sanitized_length,
            "chars_removed": sanitization_result.chars_removed,
            "was_truncated": sanitization_result.was_truncated,
            "detections": sanitization_result.detections,
            "detection_count": len(sanitization_result.detections),
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        logger.warning(f"audit_log.log_sanitization failed (non-fatal): {e}")


# ── Probe executions ────────────────────────────────────────────────────────


async def log_probe(
    evaluation_id: Optional[str],
    probe_type: str,
    probe_id: int = 0,
    trap_category: str = "",
    trap_type: str = "",
    target_tool: str = "",
    input_sent: str = "",
    response_received: str = "",
    passed: bool = True,
    score: int = 0,
    explanation: str = "",
    latency_ms: int = 0,
) -> None:
    """Persist a single adversarial probe execution. Best-effort."""
    if not _is_enabled() or not evaluation_id:
        return
    try:
        from src.storage.mongodb import probe_executions_col

        truncated_input, _ = _truncate(input_sent, MAX_PROBE_INPUT_CHARS)
        truncated_response, _ = _truncate(response_received, MAX_PROBE_RESPONSE_CHARS)

        await probe_executions_col().insert_one({
            "evaluation_id": evaluation_id,
            "probe_type": probe_type,
            "probe_id": probe_id,
            "trap_category": trap_category,
            "trap_type": trap_type,
            "target_tool": target_tool,
            "input_sent": truncated_input,
            "response_received": truncated_response,
            "passed": passed,
            "score": score,
            "explanation": _truncate(explanation, 500)[0],
            "latency_ms": latency_ms,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        logger.warning(f"audit_log.log_probe failed (non-fatal): {e}")

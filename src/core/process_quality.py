"""
Process Quality evaluation — the 6th scoring dimension.

Evaluates HOW a server handles the evaluation process, not just correctness:
- Error handling quality: clear error messages for missing/invalid params
- Input validation quality: proper rejection of bad types/edge cases
- Response structure quality: consistent JSON format, parsable output

This is what differentiates AgentTrust from simple accuracy benchmarks.
Industry context: Agent-as-a-Judge pattern — measuring process, not just outcome.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── Sub-dimension weights ────────────────────────────────────────────────────

ERROR_HANDLING_WEIGHT = 0.40
INPUT_VALIDATION_WEIGHT = 0.30
RESPONSE_STRUCTURE_WEIGHT = 0.30

# ── Test type categories ─────────────────────────────────────────────────────

ERROR_HANDLING_TYPES = {"error_handling"}
INPUT_VALIDATION_TYPES = {"type_coercion", "edge_case", "boundary"}
HAPPY_PATH_TYPES = {"happy_path", "happy_path_variation"}


@dataclass
class ProcessQualityResult:
    """Result of process quality analysis."""
    score: int  # 0-100 aggregate
    error_handling: int = 50    # 0-100
    input_validation: int = 50  # 0-100
    response_structure: int = 50  # 0-100
    details: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "error_handling": self.error_handling,
            "input_validation": self.input_validation,
            "response_structure": self.response_structure,
            "details": self.details,
        }


# ── Error handling signals ───────────────────────────────────────────────────

# Good error messages contain these patterns
_GOOD_ERROR_PATTERNS = [
    re.compile(r"required", re.IGNORECASE),
    re.compile(r"missing", re.IGNORECASE),
    re.compile(r"parameter", re.IGNORECASE),
    re.compile(r"field", re.IGNORECASE),
    re.compile(r"invalid", re.IGNORECASE),
    re.compile(r"validation", re.IGNORECASE),
    re.compile(r"expected", re.IGNORECASE),
    re.compile(r"must\s+(be|provide|supply)", re.IGNORECASE),
]

# Bad error patterns — internal details leaked or unhelpful
_BAD_ERROR_PATTERNS = [
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"stack\s*trace", re.IGNORECASE),
    re.compile(r"internal\s+server\s+error", re.IGNORECASE),
    re.compile(r"unhandled\s+exception", re.IGNORECASE),
    re.compile(r"NoneType.*attribute", re.IGNORECASE),
    re.compile(r"KeyError", re.IGNORECASE),
    re.compile(r"TypeError", re.IGNORECASE),
    re.compile(r"IndexError", re.IGNORECASE),
]


def _score_error_response(content: str, is_error: bool) -> int:
    """Score a single error-path response for quality (0-100).

    Good error handling means:
    - Returned an error flag (is_error=True) OR descriptive error text
    - Error message mentions what went wrong (missing param, invalid type)
    - No stack traces or internal details leaked
    """
    if not content or not content.strip():
        return 10  # Empty response to error input = bad

    content.lower()

    # Check for bad patterns (internal details leaked)
    bad_count = sum(1 for p in _BAD_ERROR_PATTERNS if p.search(content))
    if bad_count >= 2:
        return 15  # Multiple internal details = very bad

    # Check for good error message patterns
    good_count = sum(1 for p in _GOOD_ERROR_PATTERNS if p.search(content))

    # Try parsing as JSON — structured errors are better
    is_json_error = False
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and ("error" in parsed or "detail" in parsed or "message" in parsed):
            is_json_error = True
    except (json.JSONDecodeError, TypeError):
        pass

    # Score composition
    score = 30  # Base: at least responded

    if is_error:
        score += 10  # Correctly flagged as error

    if good_count >= 2:
        score += 30  # Good descriptive error message
    elif good_count == 1:
        score += 15

    if is_json_error:
        score += 20  # Structured JSON error format

    if bad_count == 1:
        score -= 15  # Partial internal leak
    elif bad_count == 0 and good_count >= 1:
        score += 10  # Clean error with no leaks

    return min(100, max(0, score))


def _score_validation_response(content: str, is_error: bool, test_type: str) -> int:
    """Score a validation/edge-case response (0-100).

    For type_coercion: server should reject invalid types with clear message.
    For edge_case: server should handle empty/null gracefully.
    For boundary: server should handle oversized input without crashing.
    """
    if not content or not content.strip():
        return 10

    content_lower = content.lower()

    # Check for crash indicators
    crash_signals = ["traceback", "segfault", "killed", "out of memory", "panic"]
    if any(s in content_lower for s in crash_signals):
        return 5  # Crashed

    score = 30  # Base: didn't crash

    # For type coercion — should reject cleanly
    if test_type == "type_coercion":
        rejection_signals = ["invalid", "type", "expected", "number", "integer", "string", "error", "validation"]
        rejection_count = sum(1 for s in rejection_signals if s in content_lower)
        if is_error and rejection_count >= 1:
            score += 50  # Properly rejected with message
        elif is_error:
            score += 30  # Rejected but no clear message
        elif rejection_count >= 2:
            score += 40  # Mentioned the issue even without error flag
        else:
            score += 10  # Accepted bad type silently

    # For edge cases (empty string, etc.)
    elif test_type == "edge_case":
        # Either a graceful error or a sensible default is ok
        if is_error:
            score += 40  # Properly rejected empty input
        elif len(content.strip()) > 5:
            score += 30  # Returned something meaningful
        else:
            score += 15  # Returned but minimal

    # For boundary (oversized input)
    elif test_type == "boundary":
        size_signals = ["too long", "too large", "exceeds", "maximum", "limit", "truncat"]
        if any(s in content_lower for s in size_signals):
            score += 50  # Gracefully rejected/truncated
        elif is_error:
            score += 35  # At least flagged as error
        elif len(content.strip()) > 0:
            score += 25  # Processed without crashing

    return min(100, max(0, score))


def _score_response_structure(content: str) -> int:
    """Score response structure quality (0-100).

    Higher scores for:
    - Valid JSON output
    - Consistent key naming
    - Meaningful content (not just errors)
    """
    if not content or not content.strip():
        return 10

    score = 20  # Base: has content

    # Try JSON parsing
    try:
        parsed = json.loads(content)
        score += 30  # Valid JSON

        if isinstance(parsed, dict):
            # Check for meaningful keys
            if len(parsed) >= 2:
                score += 15  # Multiple fields = structured
            # Check for consistent naming (camelCase or snake_case)
            keys = list(parsed.keys())
            if keys:
                has_snake = any("_" in k for k in keys)
                has_camel = any(k != k.lower() and "_" not in k for k in keys)
                if has_snake != has_camel:  # Consistent style
                    score += 10
                elif not has_snake and not has_camel:  # Simple lowercase = ok
                    score += 10
        elif isinstance(parsed, list):
            score += 10  # Array response is ok

    except (json.JSONDecodeError, TypeError):
        # Not JSON — check if it's meaningful text
        if len(content.strip()) > 20:
            score += 15  # At least substantial text
        # Check if it looks like a structured format (key: value, XML, etc.)
        if re.search(r"^\w+:\s+.+$", content, re.MULTILINE):
            score += 10  # Some structure

    return min(100, max(0, score))


# ── Main analysis function ───────────────────────────────────────────────────

def analyze_process_quality(
    tool_responses: Dict[str, List[dict]],
) -> ProcessQualityResult:
    """Analyze process quality from tool response data.

    Args:
        tool_responses: Dict of tool_name -> list of response dicts.
            Each response has: question, expected, answer, is_error, test_type, latency_ms

    Returns:
        ProcessQualityResult with sub-dimension scores and aggregate.
    """
    error_scores: List[int] = []
    validation_scores: List[int] = []
    structure_scores: List[int] = []
    details: Dict[str, str] = {}

    for tool_name, responses in tool_responses.items():
        for resp in responses:
            test_type = resp.get("test_type", "unknown")
            content = resp.get("answer", "")
            is_error = resp.get("is_error", False)

            # Categorize by test type
            if test_type in ERROR_HANDLING_TYPES:
                score = _score_error_response(content, is_error)
                error_scores.append(score)

            elif test_type in INPUT_VALIDATION_TYPES:
                score = _score_validation_response(content, is_error, test_type)
                validation_scores.append(score)

            # All responses get structure scoring (happy path only — errors have different format)
            if test_type in HAPPY_PATH_TYPES:
                structure_scores.append(_score_response_structure(content))

    # Compute sub-dimension averages (default 50 = neutral if no data)
    error_handling = int(sum(error_scores) / len(error_scores)) if error_scores else 50
    input_validation = int(sum(validation_scores) / len(validation_scores)) if validation_scores else 50
    response_structure = int(sum(structure_scores) / len(structure_scores)) if structure_scores else 50

    # Aggregate with weights
    aggregate = int(
        error_handling * ERROR_HANDLING_WEIGHT
        + input_validation * INPUT_VALIDATION_WEIGHT
        + response_structure * RESPONSE_STRUCTURE_WEIGHT
    )

    details["error_handling_samples"] = str(len(error_scores))
    details["validation_samples"] = str(len(validation_scores))
    details["structure_samples"] = str(len(structure_scores))

    logger.info(
        f"Process quality: {aggregate} "
        f"(err={error_handling}, val={input_validation}, struct={response_structure})"
    )

    return ProcessQualityResult(
        score=aggregate,
        error_handling=error_handling,
        input_validation=input_validation,
        response_structure=response_structure,
        details=details,
    )

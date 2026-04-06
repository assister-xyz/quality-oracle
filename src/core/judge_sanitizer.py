"""Input sanitization for LLM judge — strips adversarial patterns from MCP responses.

Part of QO-043 (Judge Hardening). Prevents malicious MCP servers from embedding
hidden instructions in tool responses that manipulate LLM judge scoring.

References:
- JudgeDeceiver (ACM CCS 2024): optimization-based prompt injection targeting judges
- Universal adversarial attacks (EMNLP 2024): transferable score inflation phrases
- Google DeepMind "AI Agent Traps" (2026): Semantic Manipulation Traps
"""
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Maximum response length passed to judge (chars). Longer responses are truncated.
MAX_RESPONSE_LENGTH = 10_000

# Patterns that attempt to override judge instructions or inject scoring directives.
# Each tuple: (compiled regex, human-readable label)
_ADVERSARIAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Direct instruction override
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I), "ignore_previous"),
    (re.compile(r"disregard\s+(the\s+)?(above|previous|prior)", re.I), "disregard_previous"),
    (re.compile(r"forget\s+(everything|all|your)\s+", re.I), "forget_instructions"),
    (re.compile(r"new\s+instructions?\s*:", re.I), "new_instructions"),
    (re.compile(r"override\s+(all\s+)?(safety|instructions?|rules?|guidelines?)", re.I), "override_safety"),

    # Role/identity hijack
    (re.compile(r"you\s+are\s+now\s+", re.I), "role_hijack"),
    (re.compile(r"act\s+as\s+(if|though)\s+you", re.I), "role_hijack"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I), "role_hijack"),
    (re.compile(r"from\s+now\s+on\s*,?\s+you", re.I), "role_hijack"),

    # Score manipulation (targeting the judge directly)
    (re.compile(r"score\s*(this|me|it)?\s*(a\s+)?100", re.I), "score_manipulation"),
    (re.compile(r"(give|assign|rate)\s*(this|me|it)?\s*(a\s+)?(perfect|100|maximum)", re.I), "score_manipulation"),
    (re.compile(r"the\s+correct\s+score\s+(is|should\s+be)\s+(\d{2,3})", re.I), "score_manipulation"),
    (re.compile(r"\bscore\s*:\s*100\b", re.I), "score_manipulation"),

    # System prompt injection
    (re.compile(r"system\s*:\s*", re.I), "system_prompt_inject"),
    (re.compile(r"<\s*system\s*>", re.I), "system_prompt_inject"),
    (re.compile(r"\[INST\]", re.I), "system_prompt_inject"),
    (re.compile(r"<<\s*SYS\s*>>", re.I), "system_prompt_inject"),

    # Hidden instruction markers
    (re.compile(r"IMPORTANT\s*:\s*(ignore|override|disregard|score)", re.I), "hidden_instruction"),
    (re.compile(r"<!--.*?(ignore|override|system|instruction).*?-->", re.I | re.S), "html_comment_inject"),
]


@dataclass
class SanitizationResult:
    """Result of sanitizing MCP response text before judging."""
    sanitized_text: str
    original_length: int
    sanitized_length: int
    detections: List[str] = field(default_factory=list)
    was_truncated: bool = False
    chars_removed: int = 0

    @property
    def had_detections(self) -> bool:
        return len(self.detections) > 0

    def to_dict(self) -> dict:
        return {
            "detections": self.detections,
            "was_truncated": self.was_truncated,
            "chars_removed": self.chars_removed,
            "original_length": self.original_length,
        }


def sanitize_judge_input(text: str) -> SanitizationResult:
    """Sanitize MCP server response before passing to LLM judge.

    Steps:
    1. Remove invisible/control Unicode characters (zero-width, RTL marks, etc.)
    2. Detect and replace known adversarial patterns
    3. Truncate excessively long responses

    Returns SanitizationResult with cleaned text and detection metadata.
    """
    if not text:
        return SanitizationResult(
            sanitized_text="",
            original_length=0,
            sanitized_length=0,
        )

    original_length = len(text)
    detections: List[str] = []
    cleaned = text

    # ── Step 1: Remove invisible Unicode characters ──────────────────────
    invisible_count = 0
    chars = []
    for ch in cleaned:
        cat = unicodedata.category(ch)
        # Cc = control, Cf = format (zero-width, RTL override, etc.)
        # Keep common whitespace: \n \t \r
        if cat == "Cf" or (cat == "Cc" and ch not in "\n\t\r"):
            invisible_count += 1
            continue
        chars.append(ch)
    if invisible_count > 0:
        detections.append(f"invisible_unicode:{invisible_count}")
        cleaned = "".join(chars)

    # ── Step 2: Detect and neutralize adversarial patterns ───────────────
    for pattern, label in _ADVERSARIAL_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            count = len(matches)
            detections.append(f"{label}:{count}")
            cleaned = pattern.sub("[FILTERED]", cleaned)

    # ── Step 3: Truncate excessively long responses ──────────────────────
    was_truncated = False
    if len(cleaned) > MAX_RESPONSE_LENGTH:
        was_truncated = True
        detections.append(f"truncated:{len(cleaned)}")
        cleaned = cleaned[:MAX_RESPONSE_LENGTH] + "\n[TRUNCATED]"

    chars_removed = original_length - len(cleaned)

    result = SanitizationResult(
        sanitized_text=cleaned,
        original_length=original_length,
        sanitized_length=len(cleaned),
        detections=detections,
        was_truncated=was_truncated,
        chars_removed=max(0, chars_removed),
    )

    if result.had_detections:
        logger.warning(
            f"Judge input sanitized: {len(detections)} detection(s): "
            f"{', '.join(detections[:5])}"
            f"{' ...' if len(detections) > 5 else ''}"
        )

    return result

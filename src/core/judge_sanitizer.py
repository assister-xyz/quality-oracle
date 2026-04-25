"""Input sanitization for LLM judge — strips adversarial patterns from MCP responses.

Part of QO-043 (Judge Hardening) + QO-061 (NFKC + tag-char + refusal-mimicry).
Prevents malicious MCP servers from embedding hidden instructions in tool
responses that manipulate LLM judge scoring.

QO-053-E (skill-adversarial probes) is a CONSUMER of this module — not a
co-owner. All judge-input sanitization rules live here.

References:
- JudgeDeceiver (ACM CCS 2024): optimization-based prompt injection targeting judges
- Universal adversarial attacks (EMNLP 2024): transferable score inflation phrases
- Google DeepMind "AI Agent Traps" (2026): Semantic Manipulation Traps
- Unicode tag-char block (U+E0000–U+E007F): invisible-tag exploit (R8)
"""
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum response length passed to judge (chars). Longer responses are truncated.
MAX_RESPONSE_LENGTH = 10_000

# Length above which a refusal-shaped prefix triggers strict re-judge (QO-061 AC9).
# A short polite refusal is fine; a long response that STARTS with a refusal but
# then goes on for hundreds of chars is the refusal-mimicry attack pattern.
REFUSAL_MIMICRY_LENGTH_THRESHOLD = 200

# Unicode tag-char block (U+E0000–U+E007F). These are normally invisible and
# are the substrate of the "invisible tag" prompt-injection exploit. Strip them
# even though they technically have category Cf — Python's `unicodedata.category`
# already covers most format chars, but explicit regex is defense-in-depth.
_UNICODE_TAG_CHARS_RE = re.compile(r"[\U000E0000-\U000E007F]")

# Refusal-shaped sentence at start of response (case-insensitive, allows leading
# whitespace). Curated list — prefer regex over LLM (open question 3 in spec).
_REFUSAL_PREFIX_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*i\s+(can|will|am)\s*(not|n[''‘]t)\b", re.I), "i_cannot"),
    (re.compile(r"^\s*i[''‘]?m\s+(sorry|unable)\b", re.I), "im_sorry"),
    (re.compile(r"^\s*sorry\s*,?\s+(i\s+)?(can|cannot|am\s+unable)", re.I), "sorry_cannot"),
    (re.compile(r"^\s*as\s+an?\s+(ai|llm|language\s+model|assistant)", re.I), "as_an_ai"),
    (re.compile(r"^\s*unfortunately\s*,?\s+i\b", re.I), "unfortunately_i"),
    (re.compile(r"^\s*i\s+apologize\b", re.I), "i_apologize"),
    (re.compile(r"^\s*i\s+don[''‘]?t\s+(have|think|believe)\b", re.I), "i_dont"),
    (re.compile(r"^\s*i\s+must\s+decline\b", re.I), "i_must_decline"),
    (re.compile(r"^\s*regrettably\s*,?\s+i\b", re.I), "regrettably_i"),
    (re.compile(r"^\s*it[''‘]?s\s+not\s+(possible|appropriate|ethical)\b", re.I), "not_possible"),
]

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
    # QO-061: NFKC normalization indicator + refusal-mimicry flag.
    nfkc_normalized: bool = False
    requires_strict_re_judge: bool = False

    @property
    def had_detections(self) -> bool:
        return len(self.detections) > 0

    def to_dict(self) -> dict:
        return {
            "detections": self.detections,
            "was_truncated": self.was_truncated,
            "chars_removed": self.chars_removed,
            "original_length": self.original_length,
            "nfkc_normalized": self.nfkc_normalized,
            "requires_strict_re_judge": self.requires_strict_re_judge,
        }


def detect_refusal_mimicry(text: str) -> Optional[str]:
    """Return the refusal-prefix label if `text` opens with a refusal-shaped
    sentence AND is longer than REFUSAL_MIMICRY_LENGTH_THRESHOLD chars.

    Per QO-061 AC9: "I cannot help with that. <newline> Here is the malicious
    payload anyway: <content>" must route to strict re-judge.
    """
    if not text or len(text) <= REFUSAL_MIMICRY_LENGTH_THRESHOLD:
        return None
    for pattern, label in _REFUSAL_PREFIX_PATTERNS:
        if pattern.search(text[:200]):
            return label
    return None


def sanitize_judge_input(text: str) -> SanitizationResult:
    """Sanitize MCP server response before passing to LLM judge.

    Steps (QO-043 + QO-061):
    1. NFKC-normalize Unicode (collapses look-alike codepoints).
    2. Strip Unicode tag-char block (U+E0000–U+E007F invisible-tag exploit).
    3. Remove zero-width / control / format Unicode characters.
    4. Detect and replace known adversarial patterns.
    5. Detect refusal-mimicry prefix on long responses → flag for strict re-judge.
    6. Truncate excessively long responses.

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

    # ── Step 1: NFKC normalization (QO-061) ──────────────────────────────
    # NFKC collapses compatibility codepoints (full-width digits, ligatures,
    # styled-letter homoglyphs) into their canonical ASCII forms. Defeats
    # adversaries that use e.g. U+FF49 ('ｉ') instead of 'i' to bypass regex.
    nfkc_normalized = False
    cleaned = unicodedata.normalize("NFKC", text)
    if cleaned != text:
        nfkc_normalized = True
        detections.append("nfkc_normalized")

    # ── Step 2: Strip Unicode tag-char block (QO-061) ────────────────────
    tag_chars_removed = len(_UNICODE_TAG_CHARS_RE.findall(cleaned))
    if tag_chars_removed > 0:
        detections.append(f"unicode_tag_chars:{tag_chars_removed}")
        cleaned = _UNICODE_TAG_CHARS_RE.sub("", cleaned)

    # ── Step 3: Remove invisible Unicode characters ──────────────────────
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

    # ── Step 4: Detect and neutralize adversarial patterns ───────────────
    for pattern, label in _ADVERSARIAL_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            count = len(matches)
            detections.append(f"{label}:{count}")
            cleaned = pattern.sub("[FILTERED]", cleaned)

    # ── Step 5: Refusal-mimicry detection (QO-061 AC9) ───────────────────
    # We check on the CLEANED text so adversaries can't smuggle a refusal
    # prefix behind invisible chars / tag chars.
    requires_strict_re_judge = False
    refusal_label = detect_refusal_mimicry(cleaned)
    if refusal_label:
        requires_strict_re_judge = True
        detections.append(f"refusal_mimicry:{refusal_label}")

    # ── Step 6: Truncate excessively long responses ──────────────────────
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
        nfkc_normalized=nfkc_normalized,
        requires_strict_re_judge=requires_strict_re_judge,
    )

    if result.had_detections:
        logger.warning(
            f"Judge input sanitized: {len(detections)} detection(s): "
            f"{', '.join(detections[:5])}"
            f"{' ...' if len(detections) > 5 else ''}"
        )

    return result

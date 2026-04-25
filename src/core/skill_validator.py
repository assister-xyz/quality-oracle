"""Spec-compliance validator for ParsedSkill (QO-053-A).

Implements the 12 validator rules from R1 §"Validator must check". Maps each
rule onto an Anti-Pattern ID (AP1..AP18) so the resulting :class:`Violation`
list shares vocabulary with :mod:`skill_anti_patterns`. The compliance score
formula (AC8) is::

    score = max(0, 100 - sum(v.score_deduction for v in violations))

Per-rule deductions follow the severity tier (HIGH=15-20, MED=5-10, LOW=2-3).
A single AP1 (HIGH=20) violation lands at exactly 80, which is why AC3 uses
``<= 80`` rather than ``< 80``.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Set

from src.storage.models import ParsedSkill, Severity, SpecCompliance, Violation

# R1 §"Frontmatter": top-level field allowlist.
ALLOWED_FIELDS: Set[str] = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}

# R1 rule 2: name regex — lowercase Unicode-letter or digit, with `-`, no
# leading/trailing/`--`.
_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")

# R1 rule 7: reserved words that compromise Claude API portability.
_RESERVED_WORDS = ("anthropic", "claude")

# R1 rule 9: body size thresholds (bytes).
_BODY_WARN_BYTES = 30 * 1024  # 30 KB
_BODY_FAIL_BYTES = 100 * 1024  # 100 KB

# R1 rule 10: body line warning threshold.
_BODY_WARN_LINES = 500

# Description bounds.
_DESCRIPTION_MIN = 1
_DESCRIPTION_MAX = 1024
_DESCRIPTION_LONG_WARN = 1200  # AC22

# Compatibility max length.
_COMPATIBILITY_MAX = 500


def _name_violates(name: str) -> bool:
    """Return True iff ``name`` breaks the spec name pattern (R1 rule 2)."""
    if not name:
        return True
    if len(name) > 64:
        return True
    if "--" in name:
        return True
    if not _NAME_PATTERN.match(name):
        return True
    return False


def _has_xml_tag(text: str) -> bool:
    """Naive XML-tag detector for R1 rule 8 (LOW)."""
    return bool(re.search(r"<[A-Za-z/][^>]*>", text or ""))


def _add(violations: List[Violation], v: Violation) -> None:
    violations.append(v)


def validate_skill(parsed: ParsedSkill, dir_path: Optional[Path] = None) -> SpecCompliance:
    """Run the 12-rule R1 validator over a :class:`ParsedSkill`.

    Parameters
    ----------
    parsed
        Output of :func:`skill_parser.parse_skill_md`.
    dir_path
        Filesystem location the skill was loaded from. Required only for the
        name/folder mismatch rule (AP1). When omitted, that rule is skipped
        but the rest of the validator still runs.
    """
    violations: List[Violation] = []

    # Rule 12: YAML parse error → HARD FAIL (AP11).
    yaml_parse_failed = any(
        w.startswith("yaml_parse_error") or w == "yaml_root_not_mapping"
        for w in parsed.parse_warnings
    )
    if yaml_parse_failed:
        _add(
            violations,
            Violation(
                rule="AP11",
                severity=Severity.HIGH,
                field=None,
                message="YAML frontmatter failed to parse",
                suggestion="fix the YAML syntax; the spec requires a mapping at top level",
                score_deduction=20,
            ),
        )

    # Rule 1: name + description required.
    if not parsed.name:
        _add(
            violations,
            Violation(
                rule="AP11",
                severity=Severity.HIGH,
                field="name",
                message="missing required field 'name'",
                suggestion="add `name: <slug>` to the YAML frontmatter",
                score_deduction=20,
            ),
        )
    if not parsed.description:
        _add(
            violations,
            Violation(
                rule="AP11",
                severity=Severity.HIGH,
                field="description",
                message="missing required field 'description'",
                suggestion="add a 1-1024 char description explaining when to invoke the skill",
                score_deduction=20,
            ),
        )

    # Rule 2: name shape.
    if parsed.name and _name_violates(parsed.name):
        _add(
            violations,
            Violation(
                rule="AP11",
                severity=Severity.HIGH,
                field="name",
                message=f"name {parsed.name!r} violates spec pattern (1-64 chars, NFKC, [a-z0-9-], no leading/trailing/`--`)",
                suggestion="rename to lowercase letters, digits, single hyphens",
                score_deduction=15,
            ),
        )

    # Rule 3: name == folder_name_nfkc — HARD FAIL (AP1).
    if dir_path is not None and parsed.name and parsed.folder_name_nfkc:
        # Compare under NFKC on both sides for safety even though the OS
        # already normalizes filenames on macOS HFS+/APFS (AC4).
        normalized_name = unicodedata.normalize("NFKC", parsed.name)
        normalized_folder = unicodedata.normalize("NFKC", parsed.folder_name_nfkc)
        if normalized_name != normalized_folder:
            _add(
                violations,
                Violation(
                    rule="AP1",
                    severity=Severity.HIGH,
                    field="name",
                    message=(
                        f"frontmatter name {parsed.name!r} does not match folder "
                        f"{parsed.folder_name!r} (NFKC compare)"
                    ),
                    suggestion=(
                        "rename the folder to match `name`, or rename `name` to match the folder; "
                        "IDE skill activation depends on this match"
                    ),
                    score_deduction=20,
                ),
            )

    # Rule 4: 1 ≤ len(description) ≤ 1024.
    desc_len = len(parsed.description or "")
    if desc_len > _DESCRIPTION_MAX:
        _add(
            violations,
            Violation(
                rule="AP5",
                severity=Severity.HIGH,
                field="description",
                message=f"description is {desc_len} chars; spec hard cap is {_DESCRIPTION_MAX}",
                suggestion=f"shrink description to ≤ {_DESCRIPTION_MAX} chars",
                score_deduction=15,
            ),
        )
        # AC22: an additional LOW WARNING above 1200 chars.
        if desc_len > _DESCRIPTION_LONG_WARN:
            _add(
                violations,
                Violation(
                    rule="AP5_LONG",
                    severity=Severity.LOW,
                    field="description",
                    message=f"description >{_DESCRIPTION_LONG_WARN} chars — historically degrades activation gating",
                    suggestion="aim for a single sentence ≤ 200 chars for best activation",
                    score_deduction=2,
                ),
            )
    elif desc_len < _DESCRIPTION_MIN and parsed.description != "":
        # We already raised AP11 above for empty descriptions (rule 1).
        pass

    # Rule 5: compatibility length.
    if parsed.compatibility and len(parsed.compatibility) > _COMPATIBILITY_MAX:
        _add(
            violations,
            Violation(
                rule="AP3",
                severity=Severity.MED,
                field="compatibility",
                message=f"compatibility is {len(parsed.compatibility)} chars; spec ceiling is {_COMPATIBILITY_MAX}",
                suggestion="shorten the compatibility expression",
                score_deduction=10,
            ),
        )

    # Rule 6: top-level allowlist (AP3).
    for key in parsed.frontmatter_raw.keys():
        if key not in ALLOWED_FIELDS:
            _add(
                violations,
                Violation(
                    rule="AP3",
                    severity=Severity.MED,
                    field=str(key),
                    message=f"top-level YAML field {key!r} is not in the spec allowlist",
                    suggestion="move under `metadata:` or remove",
                    score_deduction=5,
                ),
            )

    # Rule 7: reserved-word check (LOW).
    for field_name, value in (("name", parsed.name), ("description", parsed.description)):
        lowered = (value or "").lower()
        for reserved in _RESERVED_WORDS:
            if reserved in lowered:
                _add(
                    violations,
                    Violation(
                        rule="AP3",
                        severity=Severity.LOW,
                        field=field_name,
                        message=f"{field_name} mentions reserved word {reserved!r}",
                        suggestion="remove `anthropic`/`claude` from public skill metadata for portability",
                        score_deduction=2,
                    ),
                )
                break  # one warning per field is enough

    # Rule 8: XML-tag in name/description (LOW).
    for field_name, value in (("name", parsed.name), ("description", parsed.description)):
        if _has_xml_tag(value):
            _add(
                violations,
                Violation(
                    rule="AP3",
                    severity=Severity.LOW,
                    field=field_name,
                    message=f"{field_name} contains an XML-style tag",
                    suggestion="strip embedded `<tag>` markup from skill metadata",
                    score_deduction=2,
                ),
            )

    # Rule 9: body size.
    if parsed.body_size_bytes >= _BODY_FAIL_BYTES:
        _add(
            violations,
            Violation(
                rule="AP2",
                severity=Severity.HIGH,
                field="body",
                message=f"body is {parsed.body_size_bytes} bytes (≥ {_BODY_FAIL_BYTES})",
                suggestion="split into reference docs under `references/`",
                score_deduction=15,
            ),
        )
    elif parsed.body_size_bytes >= _BODY_WARN_BYTES:
        _add(
            violations,
            Violation(
                rule="AP2",
                severity=Severity.MED,
                field="body",
                message=f"body is {parsed.body_size_bytes} bytes (≥ {_BODY_WARN_BYTES})",
                suggestion="consider splitting into reference docs under `references/`",
                score_deduction=5,
            ),
        )

    # Rule 10: body line count warning.
    if parsed.body_lines >= _BODY_WARN_LINES:
        _add(
            violations,
            Violation(
                rule="AP2",
                severity=Severity.LOW,
                field="body",
                message=f"body has {parsed.body_lines} lines (≥ {_BODY_WARN_LINES})",
                suggestion="long bodies hurt context-window economy; consider splitting",
                score_deduction=2,
            ),
        )

    # Rule 11: allowed-tools acceptance is enforced at the parser level
    # (returns a list[str] either way). No additional check here.

    # Compute score + hard-fail flag.
    deduction = sum(v.score_deduction for v in violations)
    score = max(0, 100 - deduction)
    passed = not any(v.severity == Severity.HIGH for v in violations)

    return SpecCompliance(score=score, violations=violations, passed_hard_fails=passed)

"""Skill parser for Anthropic Agent Skills (QO-053-A).

Parses ``SKILL.md`` files (or their lowercase ``skill.md`` fallback) into a
:class:`ParsedSkill` Pydantic model. The parser is intentionally lenient at
the YAML level — it does not enforce the spec; that is :mod:`skill_validator`'s
job. The parser's contract is "no crashes on the 175-file R2 corpus" (AC1) and
"every value coerced into something a downstream validator can reason about".

Reference: ``agentskills/agentskills`` upstream parser (see R1 §"Validator must
check"). This implementation mirrors its public surface.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import strictyaml
import yaml as pyyaml  # PyYAML — fallback for YAML constructs strictyaml refuses

from src.storage.models import ParsedSkill

# Spec-defined directories (R1 §"QO-053 corrections"): EXACTLY 3.
_SPEC_DIRS = ("scripts", "references", "assets")
# Convention-only directories (NOT spec-defined): tracked separately so callers
# can compute "scope creep" anti-patterns without confusing them with spec dirs.
_CONVENTION_DIRS = ("examples", "templates", "docs")

# UTF-8 BOM bytes — strip when present at the start of a SKILL.md file.
_BOM = "﻿"

# Frontmatter delimiter — exactly ``---`` on its own line.
_FRONTMATTER_DELIM = "---"


def _decode_bytes(raw: bytes) -> Tuple[str, List[str]]:
    """Decode raw bytes as UTF-8, falling back to latin-1.

    Returns ``(text, warnings)``. ``warnings`` is empty on the happy path or
    ``["non_utf8_decoded_as_latin1"]`` when the file isn't valid UTF-8.
    """
    warnings: List[str] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        warnings.append("non_utf8_decoded_as_latin1")
    return text, warnings


def _normalize_text(text: str) -> str:
    """Strip BOM and normalize CRLF/CR line endings to LF."""
    if text.startswith(_BOM):
        text = text.lstrip(_BOM)
    # Normalize line endings: CRLF first, then bare CR, to LF.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _split_frontmatter(text: str) -> Tuple[Optional[str], str]:
    """Split a SKILL.md text into ``(frontmatter_yaml, body)``.

    Returns ``(None, text)`` if no YAML frontmatter is present (the file simply
    does not start with ``---``). Otherwise returns the YAML block (without the
    delimiter lines) and the body that follows the closing ``---``.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return None, text
    # Find closing delimiter.
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIM:
            fm = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            # Drop a single leading newline if present.
            if body.startswith("\n"):
                body = body[1:]
            return fm, body
    # Unclosed frontmatter — treat the whole file as body so we don't lose data.
    return None, text


def _coerce_metadata(value: Any) -> Dict[str, str]:
    """Coerce ``metadata`` mapping values to strings (parser.py:60 mirror).

    Non-mapping inputs return an empty dict. Nested values are stringified.
    """
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in value.items():
        if v is None:
            out[str(k)] = ""
        else:
            out[str(k)] = str(v)
    return out


def _coerce_allowed_tools(value: Any) -> List[str]:
    """Accept space-separated str OR YAML list (R1 rule 11)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [t for t in value.split() if t]
    if isinstance(value, list):
        return [str(t) for t in value if t is not None]
    return [str(value)]


def _parse_yaml(fm_text: str, warnings: List[str]) -> Dict[str, Any]:
    """Parse YAML frontmatter, recording any anomaly as a parse warning.

    Tries strictyaml first (matches the reference validator's choice); on
    failure falls back to PyYAML so we still extract data. AC11 (YAML parse
    error) is enforced by the validator, not the parser — parse errors here
    surface as a warning + the validator's hard-fail rule consults
    ``parse_warnings``.
    """
    if not fm_text.strip():
        warnings.append("yaml_frontmatter_empty")
        return {}
    try:
        return strictyaml.load(fm_text).data  # type: ignore[no-any-return]
    except Exception as e:  # pragma: no cover - defensive
        warnings.append(f"yaml_strict_error:{type(e).__name__}")
    try:
        loaded = pyyaml.safe_load(fm_text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            warnings.append("yaml_root_not_mapping")
            return {}
        return loaded
    except Exception as e:
        warnings.append(f"yaml_parse_error:{type(e).__name__}")
        return {}


def _scan_dirs(skill_dir: Path) -> Tuple[Dict[str, bool], Dict[str, bool], List[str]]:
    """Inspect ``skill_dir`` for spec / convention / extra subdirectories.

    Notes
    -----
    AC7: spec_dirs is EXACTLY {scripts, references, assets} and the population
    check uses ``(skill_dir / "scripts").iterdir()`` — NOT rglob — so a stray
    ``scripts/`` somewhere deeper in the tree is not counted.
    """
    spec_dirs = {d: False for d in _SPEC_DIRS}
    convention_dirs = {d: False for d in _CONVENTION_DIRS}
    extra_dirs: List[str] = []

    if not skill_dir.is_dir():
        return spec_dirs, convention_dirs, extra_dirs

    for child in sorted(skill_dir.iterdir()):
        if not child.is_dir():
            continue
        nm = child.name
        if nm in _SPEC_DIRS:
            try:
                # "Populated" means at least one entry inside.
                spec_dirs[nm] = any(True for _ in child.iterdir())
            except OSError:
                spec_dirs[nm] = False
        elif nm in _CONVENTION_DIRS:
            try:
                convention_dirs[nm] = any(True for _ in child.iterdir())
            except OSError:
                convention_dirs[nm] = False
        else:
            # Hide dotfiles (e.g. .git, .DS_Store-like) from extras.
            if not nm.startswith("."):
                extra_dirs.append(nm)
    return spec_dirs, convention_dirs, extra_dirs


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def parse_skill_md_string(content: str | bytes, source: str = "<string>") -> ParsedSkill:
    """Parse a SKILL.md from a raw string or bytes payload.

    ``source`` is purely informational — it surfaces inside ``parse_warnings``
    when present and is otherwise unused. No filesystem access is performed by
    this entrypoint, which means ``spec_dirs``/``convention_dirs`` come back
    empty; callers that have a directory should use :func:`parse_skill_md`.
    """
    warnings: List[str] = []

    if isinstance(content, bytes):
        text, decode_warnings = _decode_bytes(content)
        warnings.extend(decode_warnings)
    else:
        text = content

    text = _normalize_text(text)
    fm_text, body = _split_frontmatter(text)
    if fm_text is None:
        warnings.append("frontmatter_missing")
        frontmatter: Dict[str, Any] = {}
    else:
        frontmatter = _parse_yaml(fm_text, warnings)

    # Detect AP18 stylistic warning: folded/literal description block. We can
    # only inspect this from the raw frontmatter text since by the time the
    # value reaches us via PyYAML it's already a single string.
    if fm_text and _has_block_scalar_description(fm_text):
        warnings.append("description_block_scalar")

    name_raw = frontmatter.get("name", "")
    description_raw = frontmatter.get("description", "")

    # Coerce to str defensively — YAML can hand us ints/None.
    name = "" if name_raw is None else str(name_raw)
    description = "" if description_raw is None else str(description_raw)

    name_nfkc = _nfkc(name) if name else ""

    body_bytes = body.encode("utf-8")
    body_lines = body.count("\n") + 1 if body else 0

    parsed = ParsedSkill(
        name=name_nfkc,
        description=description,
        license=_optional_str(frontmatter.get("license")),
        compatibility=_optional_str(frontmatter.get("compatibility")),
        metadata=_coerce_metadata(frontmatter.get("metadata", {})),
        allowed_tools=_coerce_allowed_tools(frontmatter.get("allowed-tools")),
        body=body,
        body_size_bytes=len(body_bytes),
        body_lines=body_lines,
        spec_dirs={d: False for d in _SPEC_DIRS},
        convention_dirs={d: False for d in _CONVENTION_DIRS},
        extra_dirs=[],
        folder_name="",
        folder_name_nfkc="",
        frontmatter_raw=frontmatter or {},
        parse_warnings=warnings,
    )
    return parsed


def _optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return str(v)


def _has_block_scalar_description(fm_text: str) -> bool:
    """Detect ``description: >`` or ``description: |`` (AC19/AP18).

    Only triggers on a top-level key — indented lines (under ``metadata:``,
    etc.) are intentionally ignored.
    """
    for raw in fm_text.split("\n"):
        line = raw.rstrip()
        if not line or line.startswith(" ") or line.startswith("\t"):
            continue
        # Match `description: >` / `description: |` (with optional `+`/`-`).
        stripped = line.strip()
        if stripped.startswith("description:"):
            value_part = stripped[len("description:") :].strip()
            if value_part.startswith(">") or value_part.startswith("|"):
                return True
    return False


def parse_skill_md(skill_dir: Path) -> ParsedSkill:
    """Parse ``SKILL.md`` (preferred) or ``skill.md`` (fallback) from a dir.

    Returns a :class:`ParsedSkill` populated with directory-derived fields
    (``folder_name``, ``spec_dirs``, ``convention_dirs``, ``extra_dirs``) and
    a sorted, deduplicated ``parse_warnings`` list.

    The parser raises :class:`FileNotFoundError` only when neither
    ``SKILL.md`` nor ``skill.md`` exists — this is the single condition that
    callers (e.g. corpus regression in AC1) treat as a true crash.
    """
    skill_dir = Path(skill_dir)
    # Inspect filenames case-sensitively. On case-insensitive filesystems
    # (HFS+ / APFS-default / NTFS) `Path.is_file()` would happily return True
    # for the wrong case, so we must enumerate the actual entries.
    try:
        entries = {p.name for p in skill_dir.iterdir() if p.is_file()}
    except (OSError, FileNotFoundError) as e:
        raise FileNotFoundError(f"Cannot list directory {skill_dir}: {e}") from e

    used_lowercase = False
    if "SKILL.md" in entries:
        target = skill_dir / "SKILL.md"
    elif "skill.md" in entries:
        target = skill_dir / "skill.md"
        used_lowercase = True
    else:
        raise FileNotFoundError(f"No SKILL.md or skill.md in {skill_dir}")

    raw = target.read_bytes()
    parsed = parse_skill_md_string(raw, source=str(target))

    if used_lowercase:
        parsed.parse_warnings.append("filename_lowercase_skill_md")

    # Folder-derived fields.
    folder_name = skill_dir.name
    parsed.folder_name = folder_name
    parsed.folder_name_nfkc = _nfkc(folder_name)

    # Directory inspection.
    spec_dirs, convention_dirs, extra_dirs = _scan_dirs(skill_dir)
    parsed.spec_dirs = spec_dirs
    parsed.convention_dirs = convention_dirs
    parsed.extra_dirs = extra_dirs

    return parsed

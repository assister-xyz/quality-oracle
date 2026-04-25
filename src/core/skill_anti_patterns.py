"""Anti-pattern catalog for Agent Skills (QO-053-A).

Detects the 18 anti-patterns enumerated in R2 §"Anti-pattern catalog". Some
overlap with :mod:`skill_validator` rules (AP1, AP3, AP5, AP11) — the validator
owns spec-conformance scoring, while this module owns the broader catalog
that surfaces in the public scorecard. Callers who want both should run both
and let the consumer dedupe by ``rule == AntiPattern.id``.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

from src.storage.models import AntiPattern, ParsedSkill, Severity

# AP7 — hardcoded mainnet RPC.
_MAINNET_RPC_RE = re.compile(r"https://api\.mainnet-beta\.solana\.com")
_MAINNET_WARN_TERMS = ("mainnet", "production", "real funds", "real sol", "real money")

# AP8 — sensitive-key terms (regex pre-filter only; LLM sentiment in QO-053-D).
_SENSITIVE_TERMS = (
    "private key",
    "seed phrase",
    "mnemonic",
    "keypair",
)
_SENSITIVE_WARN_TERMS = ("warn", "never", "do not", "never share", "secret", "danger")

# AP9 — auto-updater.
_AUTOUPDATER_RE = re.compile(
    r"(curl|wget)[^\n]*~/\.claude/(agents|skills)",
    flags=re.IGNORECASE,
)

# AP16 — hardcoded `/Users/X/` path.
_USERS_PATH_RE = re.compile(r"/Users/[A-Za-z0-9._-]+/")

# AP17 — personal-email regex.
_PERSONAL_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@(gmail\.com|yahoo\.com|hotmail\.com|outlook\.com)",
    flags=re.IGNORECASE,
)

# AP15 — template flood thresholds.
_TEMPLATE_COUNT_HARD = 20
_TEMPLATE_RATIO_HARD = 0.5

# AP4 — description-too-short threshold.
_DESCRIPTION_SHORT = 30

# AP12 — H2 minimum.
_H2_MIN = 2


def _line_no(body: str, m: re.Match[str]) -> int:
    """Return the 1-indexed line number of a regex match within ``body``."""
    return body.count("\n", 0, m.start()) + 1


def _scan_for(regex: re.Pattern[str], body: str) -> List[re.Match[str]]:
    return list(regex.finditer(body or ""))


def _has_warn_near(body: str, m, window: int = 240, terms: tuple[str, ...] | None = None) -> bool:
    """Cheap "co-located warning" detector — looks for a warning term within
    ``window`` chars of the match (excluding the matched text itself, so
    e.g. the substring 'mainnet' inside `api.mainnet-beta.solana.com` is not
    self-counting). Sloppier than full sentiment but matches AC intent
    (regex pre-filter only).
    """
    if terms is None:
        terms = _MAINNET_WARN_TERMS + _SENSITIVE_WARN_TERMS
    pre = body[max(0, m.start() - window) : m.start()].lower()
    post = body[m.end() : min(len(body), m.end() + window)].lower()
    chunk = pre + " " + post
    return any(term in chunk for term in terms)


def _count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    n = 0
    for p in path.rglob("*"):
        if p.is_file():
            n += 1
    return n


def _all_dir_text(skill_dir: Path) -> str:
    """Concatenate text of in-skill scripts so AP16 catches `/Users/X/` paths
    embedded in helper scripts (per AC17). Caps each file at 64 KB to keep
    scans cheap on the 175-corpus regression.
    """
    if not skill_dir or not skill_dir.is_dir():
        return ""
    chunks: List[str] = []
    for p in skill_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz"}:
            continue
        try:
            chunks.append(p.read_text(encoding="utf-8", errors="replace")[:65536])
        except OSError:
            continue
    return "\n".join(chunks)


def _read_marketplace_json(skill_dir: Path) -> Optional[dict]:
    if not skill_dir or not skill_dir.is_dir():
        return None
    candidate = skill_dir / "marketplace.json"
    if not candidate.is_file():
        # Some repos host marketplace.json one level up.
        parent = skill_dir.parent / "marketplace.json"
        if parent.is_file():
            candidate = parent
        else:
            return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _has_license_file(skill_dir: Path) -> bool:
    if not skill_dir or not skill_dir.is_dir():
        return False
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "license", "license.md"):
        if (skill_dir / name).is_file():
            return True
    return False


def detect_anti_patterns(parsed: ParsedSkill, dir_path: Optional[Path] = None) -> List[AntiPattern]:
    """Detect AP1..AP18 over a parsed skill.

    ``dir_path`` is required for AP6/AP10/AP13/AP15 and is recommended for
    AP16 (to scan in-skill scripts). When omitted, those checks are skipped
    silently.
    """
    findings: List[AntiPattern] = []
    body = parsed.body or ""

    # AP1 — name/folder mismatch (HIGH).
    if dir_path is not None and parsed.name and parsed.folder_name_nfkc:
        a = unicodedata.normalize("NFKC", parsed.name)
        b = unicodedata.normalize("NFKC", parsed.folder_name_nfkc)
        if a != b:
            findings.append(
                AntiPattern(
                    id="AP1",
                    severity=Severity.HIGH,
                    field="name",
                    message=f"frontmatter name {parsed.name!r} != folder {parsed.folder_name!r}",
                    suggestion="rename folder to match `name` (or vice versa)",
                )
            )

    # AP2 — size bomb (HIGH).
    if parsed.body_size_bytes >= 100 * 1024:
        findings.append(
            AntiPattern(
                id="AP2",
                severity=Severity.HIGH,
                field="body",
                message=f"SKILL.md body is {parsed.body_size_bytes} bytes (>= 100KB hard cap)",
                suggestion="split into `references/` and rely on Claude file reads",
            )
        )

    # AP3 — off-spec frontmatter (MED).
    from src.core.skill_validator import ALLOWED_FIELDS  # local import to dodge cycles

    for key in parsed.frontmatter_raw.keys():
        if key not in ALLOWED_FIELDS:
            findings.append(
                AntiPattern(
                    id="AP3",
                    severity=Severity.MED,
                    field=str(key),
                    message=f"top-level frontmatter field {key!r} is not in the spec allowlist",
                    suggestion="move under `metadata:` or remove",
                )
            )

    # AP4 — description < 30 chars (MED).
    if parsed.description and len(parsed.description) < _DESCRIPTION_SHORT:
        findings.append(
            AntiPattern(
                id="AP4",
                severity=Severity.MED,
                field="description",
                message=f"description is {len(parsed.description)} chars; aim for ≥ {_DESCRIPTION_SHORT}",
                suggestion="expand description so activation matching can find it",
            )
        )

    # AP5 — description > 1024 chars (HIGH).
    if parsed.description and len(parsed.description) > 1024:
        findings.append(
            AntiPattern(
                id="AP5",
                severity=Severity.HIGH,
                field="description",
                message=f"description is {len(parsed.description)} chars; spec hard cap is 1024",
                suggestion="shrink to ≤ 1024 chars (single sentence preferred)",
            )
        )

    # AP6 — marketplace.json drift (MED).
    market = _read_marketplace_json(dir_path) if dir_path else None
    if market and isinstance(market, dict):
        items = market.get("skills") or market.get("plugins") or [market]
        if isinstance(items, list):
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                # Try the most common key shapes.
                m_name = entry.get("name") or entry.get("id") or entry.get("slug")
                m_desc = entry.get("description") or entry.get("summary")
                if (m_name and parsed.name and str(m_name) != parsed.name) or (
                    m_desc and parsed.description and str(m_desc) != parsed.description
                ):
                    findings.append(
                        AntiPattern(
                            id="AP6",
                            severity=Severity.MED,
                            field="marketplace.json",
                            message="marketplace.json name/description drifted from SKILL.md frontmatter",
                            suggestion="re-sync marketplace.json with SKILL.md frontmatter",
                        )
                    )
                    break

    # AP7 — hardcoded mainnet RPC without warning (HIGH).
    for m in _scan_for(_MAINNET_RPC_RE, body):
        if not _has_warn_near(body, m):
            findings.append(
                AntiPattern(
                    id="AP7",
                    severity=Severity.HIGH,
                    field="body",
                    line=_line_no(body, m),
                    regex_match=m.group(0),
                    message="hardcoded Solana mainnet RPC URL without a co-located warning",
                    suggestion="document the network choice and add a 'mainnet — real funds' warning nearby",
                )
            )
            break  # one finding is enough

    # AP8 — sensitive-key term without warning (HIGH).
    body_lower = body.lower()
    for term in _SENSITIVE_TERMS:
        idx = body_lower.find(term)
        if idx == -1:
            continue
        # Pseudo-Match for window check.
        class _Stub:
            def start(self_inner) -> int:
                return idx

            def end(self_inner) -> int:
                return idx + len(term)

        if not _has_warn_near(body, _Stub()):  # type: ignore[arg-type]
            findings.append(
                AntiPattern(
                    id="AP8",
                    severity=Severity.HIGH,
                    field="body",
                    line=body.count("\n", 0, idx) + 1,
                    regex_match=term,
                    message=f"sensitive-key term {term!r} without nearby warning",
                    suggestion="add a 'never share / never commit' warning beside this term",
                )
            )
            break

    # AP9 — auto-updater (HIGH).
    for m in _scan_for(_AUTOUPDATER_RE, body):
        findings.append(
            AntiPattern(
                id="AP9",
                severity=Severity.HIGH,
                field="body",
                line=_line_no(body, m),
                regex_match=m.group(0),
                message="auto-updater pattern targeting ~/.claude/{agents,skills}",
                suggestion="hard fail by default; require opt-in via compatibility declaration",
            )
        )
        break

    # AP10 — no examples + no scripts (LOW).
    if dir_path is not None:
        examples_populated = parsed.convention_dirs.get("examples", False)
        scripts_populated = parsed.spec_dirs.get("scripts", False)
        if not examples_populated and not scripts_populated:
            findings.append(
                AntiPattern(
                    id="AP10",
                    severity=Severity.LOW,
                    field=None,
                    message="skill ships neither examples/ nor scripts/",
                    suggestion="add at least one example or runnable script so the LLM can ground itself",
                )
            )

    # AP11 — YAML parse error (HIGH) — propagated from parser warnings.
    if any(
        w.startswith("yaml_parse_error") or w == "yaml_root_not_mapping" or w == "frontmatter_missing"
        for w in parsed.parse_warnings
    ):
        findings.append(
            AntiPattern(
                id="AP11",
                severity=Severity.HIGH,
                field=None,
                message="YAML frontmatter is missing or unparseable",
                suggestion="fix YAML syntax; the parser should accept it",
            )
        )

    # AP12 — < 2 H2 sections (LOW).
    h2_count = sum(1 for line in body.split("\n") if line.startswith("## ") and not line.startswith("### "))
    if h2_count < _H2_MIN:
        findings.append(
            AntiPattern(
                id="AP12",
                severity=Severity.LOW,
                field="body",
                message=f"body has {h2_count} H2 sections; structure is thin",
                suggestion="add at least two `## Section` headings",
            )
        )

    # AP13 — no LICENSE info (MED).
    has_license_field = bool(parsed.license)
    has_license_file = _has_license_file(dir_path) if dir_path else False
    if not has_license_field and not has_license_file:
        findings.append(
            AntiPattern(
                id="AP13",
                severity=Severity.MED,
                field=None,
                message="skill has neither `license:` frontmatter nor a LICENSE file",
                suggestion="add an SPDX identifier or LICENSE file",
            )
        )

    # AP14 — description scope-creep (MED). Owned by QO-053-C (LLM judge).
    # Stub here so callers see the ID exists; never emitted by static analysis.

    # AP15 — template flood (LOW).
    if dir_path is not None and dir_path.is_dir():
        templates_dir = dir_path / "templates"
        if templates_dir.is_dir():
            template_count = sum(1 for p in templates_dir.rglob("*") if p.is_file())
            total = max(1, _count_files(dir_path))
            ratio = template_count / total
            if template_count > _TEMPLATE_COUNT_HARD or ratio > _TEMPLATE_RATIO_HARD:
                findings.append(
                    AntiPattern(
                        id="AP15",
                        severity=Severity.LOW,
                        field=None,
                        message=f"templates/ has {template_count} files ({ratio:.0%} of skill total)",
                        suggestion="skills should be lean; large template directories belong in a separate repo",
                    )
                )

    # AP16 — hardcoded `/Users/X/` (MED).
    haystack = body
    if dir_path is not None:
        haystack = haystack + "\n" + _all_dir_text(dir_path)
    m16 = _USERS_PATH_RE.search(haystack)
    if m16:
        findings.append(
            AntiPattern(
                id="AP16",
                severity=Severity.MED,
                field="body",
                regex_match=m16.group(0),
                message="hardcoded `/Users/<name>/` path will not work for other operators",
                suggestion="use $HOME, ~, or a relative path",
            )
        )

    # AP17 — personal email in metadata (LOW).
    metadata_blob = " ".join(f"{k}={v}" for k, v in parsed.metadata.items())
    frontmatter_blob = json.dumps(parsed.frontmatter_raw, default=str)
    em = _PERSONAL_EMAIL_RE.search(metadata_blob) or _PERSONAL_EMAIL_RE.search(frontmatter_blob)
    if em:
        findings.append(
            AntiPattern(
                id="AP17",
                severity=Severity.LOW,
                field="metadata",
                regex_match=em.group(0),
                message="personal email address in skill metadata",
                suggestion="prefer an org email or omit",
            )
        )

    # AP18 — folded/literal `description: >` or `description: |` (LOW).
    if "description_block_scalar" in parsed.parse_warnings:
        findings.append(
            AntiPattern(
                id="AP18",
                severity=Severity.LOW,
                field="description",
                message="description uses YAML folded/literal block scalar",
                suggestion="prefer single-line description ≤ 1024 chars",
            )
        )

    return findings

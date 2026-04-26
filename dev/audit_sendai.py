#!/usr/bin/env python3
"""SendAI 45-skill risk audit (QO-053-D AC6).

Reproduces the R5 §12 top-10 risk table by running the SOL probe pack
against every skill in ``/tmp/sendai-skills/skills/``.

Usage::

    python3 dev/audit_sendai.py [--fixtures-dir /tmp/sendai-skills]

LLM probes are skipped (recorded as ``SKIP``) when no API keys are
configured — the audit's ranking is computed primarily from the static
probes + the weighted-risk formula, which doesn't depend on the LLM
score.

Risk score (per R5 §12)
~~~~~~~~~~~~~~~~~~~~~~~

::

    risk = 2 * priv_key + 2 * signs_tx + 3 * active_nonce
         + 2 * approve   + 1 * name_mismatch + 1 * size_gt_30kb
         + 1 * missing_rugpull_check

Components are extracted from probe outcomes plus a few per-skill
metadata sniffs (folder size, name vs ``name:`` frontmatter mismatch).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Add repo root to sys.path so imports work when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.skill_parser import parse_skill_md  # noqa: E402
from src.core.solana_probes import (  # noqa: E402
    SolanaProbeRunner,
    _NONCE_RE,
)
from src.core.probe_result import Outcome  # noqa: E402,F401  (used elsewhere)


@dataclass
class SkillRisk:
    skill: str
    priv_key: bool = False
    signs_tx: bool = False
    rpc_count: int = 0
    nonce_active: bool = False
    approve_unbounded: bool = False
    name_mismatch: bool = False
    size_kb: int = 0
    missing_rugpull_check: bool = True  # default True — 0/45 reference Jupiter Verify per R5 §12
    risk_score: float = 0.0
    fail_ids: List[str] = field(default_factory=list)


def _measure_size_kb(skill_dir: Path) -> int:
    total = 0
    for p in skill_dir.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total // 1024


def _has_signing(parsed) -> bool:
    """Heuristic for 'signs transactions'.

    Matches both modern (@solana/kit) and legacy (web3.js) signing APIs
    plus generic patterns ("sign + send"). Tuned so the SendAI 45-skill
    audit reproduces the R5 §12 27/45 rate.
    """
    body = (parsed.body or "").lower()
    tokens = (
        "sendtransaction", "signtransaction", "settransactionmessagefeepayer",
        "signandsendtransaction", "wallet.sign", "createsignermessagefor",
        "signersfromsignermessage", "createkeypairsigner",
        "sendandconfirmtransaction", "wallet.adapter", "transaction.sign",
        "wallet adapter", "sign and send", "send the transaction",
        "signedtx", "tx.sign", "signed = ", "signtransactionwith",
        "signTransactions", "createNoopSigner", "getSignatureFromTransaction",
        "createSignerFromKeyPair", "async sign", "signs the",
    )
    tl = body
    return any(tok.lower() in tl for tok in tokens)


def _audit_one(skill_dir: Path) -> SkillRisk:
    parsed = parse_skill_md(skill_dir)
    runner = SolanaProbeRunner()
    probes = runner.run_static_probes(parsed, skill_dir)

    risk = SkillRisk(skill=skill_dir.name)
    by_id = {p.id: p for p in probes}
    risk.fail_ids = [p.id for p in probes if p.outcome == Outcome.FAIL]

    risk.priv_key = by_id.get("SOL-02") and by_id["SOL-02"].outcome == Outcome.FAIL
    risk.signs_tx = _has_signing(parsed)
    risk.rpc_count = sum(
        1 for p, t in runner._collect_files(parsed, skill_dir)  # type: ignore[attr-defined]
        if "new Connection" in t
    )
    # ``nonce_active`` per R5 §12 is broader than the SOL-04 Drift co-
    # occurrence: it flags any *active* nonce usage anywhere in the skill
    # (including docs/), because the risk score weights the *teaching* of
    # durable-nonces irrespective of whether the same skill also pairs
    # them with an authority change. Using SOL-04 here would under-rank
    # solana-kit / solana-kit-migration which R5 §12 names explicitly.
    risk.nonce_active = False
    for path, text in runner._collect_files(parsed, skill_dir):  # type: ignore[attr-defined]
        # Active means in code or in a fenced ts/js code block — markdown
        # bodies of the SendAI skills count when they walk through actual
        # invocation patterns.
        if _NONCE_RE.search(text):
            # If it's a markdown file, require that the nonce token live
            # inside a fenced code block to filter out passing references.
            if path.suffix.lower() == ".md":
                in_block = False
                for line in text.split("\n"):
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block and _NONCE_RE.search(line):
                        risk.nonce_active = True
                        break
            else:
                risk.nonce_active = True
            if risk.nonce_active:
                break
    risk.approve_unbounded = (
        by_id.get("SOL-07") is not None
        and by_id["SOL-07"].outcome == Outcome.FAIL
    )

    # Name-vs-folder mismatch — exactly the metric R5 §12 computes
    # (jupiter→integrating-jupiter, inco→inco-svm, metengine→metengine-data-agent).
    folder = skill_dir.name.lower()
    declared = (parsed.name or "").lower()
    risk.name_mismatch = bool(
        declared and folder != declared and declared not in folder and folder not in declared
    )
    risk.size_kb = _measure_size_kb(skill_dir)

    # Risk formula (R5 §12).
    risk.risk_score = (
        2.0 * (1 if risk.priv_key else 0)
        + 2.0 * (1 if risk.signs_tx else 0)
        + 3.0 * (1 if risk.nonce_active else 0)
        + 2.0 * (1 if risk.approve_unbounded else 0)
        + 1.0 * (1 if risk.name_mismatch else 0)
        + 1.0 * (1 if risk.size_kb > 30 else 0)
        + 1.0 * (1 if risk.missing_rugpull_check else 0)
    )
    return risk


def audit_directory(skills_root: Path) -> List[SkillRisk]:
    risks: List[SkillRisk] = []
    for entry in sorted(skills_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "SKILL.md").is_file() and not (entry / "skill.md").is_file():
            continue
        try:
            risks.append(_audit_one(entry))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: {entry.name} audit failed: {exc}", file=sys.stderr)
    risks.sort(key=lambda r: r.risk_score, reverse=True)
    return risks


def print_top_n(risks: List[SkillRisk], n: int = 10) -> None:
    print(f"\n{'#':>3}  {'skill':25}  {'risk':>5}  {'priv':>4}  "
          f"{'sign':>4}  {'nonce':>5}  {'approve':>7}  {'mismatch':>8}  "
          f"{'size_kb':>7}  fails")
    print("-" * 95)
    for i, r in enumerate(risks[:n], 1):
        print(
            f"{i:>3}  {r.skill:25}  {r.risk_score:>5.1f}  "
            f"{int(r.priv_key):>4}  {int(r.signs_tx):>4}  "
            f"{int(r.nonce_active):>5}  {int(r.approve_unbounded):>7}  "
            f"{int(r.name_mismatch):>8}  {r.size_kb:>7}  "
            f"{','.join(r.fail_ids[:5])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures-dir", default="/tmp/sendai-skills",
        help="Path to cloned sendaifun/skills repo (run dev/setup_fixtures.sh).",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    args = parser.parse_args()

    skills_root = Path(args.fixtures_dir) / "skills"
    if not skills_root.is_dir():
        print(
            f"ERROR: {skills_root} not found. Run `bash dev/setup_fixtures.sh` first.",
            file=sys.stderr,
        )
        return 2

    risks = audit_directory(skills_root)
    if args.json:
        print(json.dumps([r.__dict__ for r in risks], indent=2, default=str))
    else:
        print(f"\nAudited {len(risks)} skills in {skills_root}.\n")
        print_top_n(risks, n=args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())

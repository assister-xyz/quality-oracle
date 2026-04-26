#!/usr/bin/env python3
"""QO-061 AC8: JudgeBench public benchmark runner.

Runs the family-diverse judge committee against the JudgeBench public split
(~350 hard pairs, ICLR 2025) and emits per-pair votes + aggregate accuracy
vs gold for TWO panels side-by-side:

    1. Free-tier panel (Cerebras-Llama, Gemini Flash, Qwen 80B) — target ≥58%
    2. Paid Tier-3 panel (Claude Haiku, Gemini Flash, GPT-4o-mini)   — target ≥62%

Output: reports/judgebench-2026-04.json

Usage:
    # Dry-run: print what would happen (no API keys needed):
    python -m dev.run_judgebench --dry-run

    # Live run (requires API keys for the requested panel):
    python -m dev.run_judgebench --live --panel free
    python -m dev.run_judgebench --live --panel paid
    python -m dev.run_judgebench --live --panel both --output reports/judgebench-2026-04.json

The dataset loader is deliberately a thin stub — JudgeBench publishes the
split on HuggingFace; the loader expects a local JSON or HF dataset id that
the operator provides (we do not bundle the dataset under the AGPL repo
license).
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.consensus_judge import (  # noqa: E402
    ConsensusJudge,
    JudgeConfig,
    _build_judge_from_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Panel definitions ───────────────────────────────────────────────────────
# Free-tier panel mirrors `consensus_judge.FREE_TIER_PANEL`. Paid Tier-3 panel
# is the higher-accuracy reference (target ≥62%).
FREE_TIER_PANEL: List[JudgeConfig] = [
    JudgeConfig(provider="cerebras", model="llama3.1-8b",
                family="meta_llama", role="primary"),
    JudgeConfig(provider="gemini", model="gemini-2.5-flash",
                family="google_gemini", role="secondary"),
    JudgeConfig(provider="openrouter", model="qwen/qwen3-next-80b-a3b-instruct:free",
                family="alibaba_qwen", role="tiebreaker"),
]

PAID_TIER_PANEL: List[JudgeConfig] = [
    JudgeConfig(provider="anthropic", model="claude-3-5-haiku-latest",
                family="anthropic_claude", role="primary"),
    JudgeConfig(provider="gemini", model="gemini-2.5-flash",
                family="google_gemini", role="secondary"),
    JudgeConfig(provider="openai", model="gpt-4o-mini",
                family="openai_gpt", role="tiebreaker"),
]

# Closed-judge reference — published on the landing page row for honesty.
CLOSED_JUDGE_REFERENCE = {
    "label": "GPT-4 (closed reference)",
    "accuracy": 0.64,
    "source": "JudgeBench paper (ICLR 2025)",
}


def load_judgebench(path: Optional[Path] = None) -> List[dict]:
    """Load JudgeBench pairs from a local JSON file.

    Each pair has shape:
        {"id": "...", "question": "...", "response_a": "...", "response_b": "...",
         "gold": "a" | "b"}

    The dataset itself is NOT bundled with this repo; supply via --dataset.
    """
    if path is None:
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "pairs" in data:
        data = data["pairs"]
    return data


async def _vote_pair(judge: ConsensusJudge, pair: dict) -> str:
    """Run the committee on one pair; return the chosen answer ('a' or 'b').

    The committee scores each response on a 0-100 scale; whichever gets the
    higher consensus score wins. Ties → 'a' (matches JudgeBench convention).
    """
    qa = pair["question"]
    ra = pair["response_a"]
    rb = pair["response_b"]

    score_a = await judge.ajudge_consensus(qa, "", ra)
    score_b = await judge.ajudge_consensus(qa, "", rb)
    return "a" if score_a.score >= score_b.score else "b"


def _build_panel(panel_name: str, panel_cfg: List[JudgeConfig]) -> Optional[ConsensusJudge]:
    """Materialize a ConsensusJudge from a list of configs.

    Returns None if no judge in the panel has an API key — callers should
    treat this as a graceful skip in dry-run / no-key environments.
    """
    judges = []
    for cfg in panel_cfg:
        j = _build_judge_from_config(cfg)
        if j:
            judges.append(j)
    if not judges:
        logger.warning(f"Panel {panel_name!r}: no API keys present, skipping live run.")
        return None
    return ConsensusJudge(judges=judges)


async def run_panel(
    panel_name: str,
    panel_cfg: List[JudgeConfig],
    pairs: List[dict],
) -> dict:
    """Run a panel on `pairs` and return aggregate stats."""
    judge = _build_panel(panel_name, panel_cfg)
    if judge is None or not pairs:
        return {
            "panel": panel_name,
            "pairs_judged": 0,
            "accuracy": None,
            "skipped_reason": "no_api_keys" if judge is None else "no_pairs",
            "panel_config": [
                {"provider": c.provider, "model": c.model, "family": c.family}
                for c in panel_cfg
            ],
        }

    correct = 0
    per_pair = []
    started = time.time()
    for pair in pairs:
        try:
            vote = await _vote_pair(judge, pair)
        except Exception as e:
            logger.warning(f"Pair {pair.get('id')} failed: {e}")
            continue
        is_correct = (vote == pair.get("gold"))
        if is_correct:
            correct += 1
        per_pair.append(
            {"id": pair.get("id"), "vote": vote, "gold": pair.get("gold"),
             "correct": is_correct}
        )
    elapsed = time.time() - started
    n = len(per_pair)
    accuracy = correct / n if n else None

    return {
        "panel": panel_name,
        "pairs_judged": n,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_s": round(elapsed, 1),
        "per_pair": per_pair,
        "panel_config": [
            {"provider": c.provider, "model": c.model, "family": c.family}
            for c in panel_cfg
        ],
    }


def dry_run() -> dict:
    """Print what the script WOULD do without hitting any API."""
    return {
        "mode": "dry_run",
        "free_tier_panel": [
            {"provider": c.provider, "model": c.model, "family": c.family}
            for c in FREE_TIER_PANEL
        ],
        "paid_tier_panel": [
            {"provider": c.provider, "model": c.model, "family": c.family}
            for c in PAID_TIER_PANEL
        ],
        "targets": {"free_tier_min_accuracy": 0.58, "paid_tier_min_accuracy": 0.62},
        "closed_judge_reference": CLOSED_JUDGE_REFERENCE,
        "note": (
            "Live mode not invoked. Provide --live with --dataset <path> to "
            "score JudgeBench pairs. API keys required for the chosen panel."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Print what would run; no API calls. (default)")
    parser.add_argument("--live", action="store_true",
                        help="Actually run the panel(s) against the dataset.")
    parser.add_argument(
        "--panel", choices=["free", "paid", "both"], default="both",
        help="Which panel to run live (default: both).",
    )
    parser.add_argument(
        "--dataset", type=Path,
        help="Path to JudgeBench JSON pairs file. Required for --live.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/judgebench-2026-04.json"),
        help="Output path for the report JSON.",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Cap pairs (useful for smoke testing).",
    )
    args = parser.parse_args()

    if not args.live:
        report = dry_run()
        print(json.dumps(report, indent=2))
        return

    if not args.dataset:
        parser.error("--dataset is required when --live is set.")
    pairs = load_judgebench(args.dataset)
    if args.limit:
        pairs = pairs[: args.limit]

    panels: Dict[str, List[JudgeConfig]] = {}
    if args.panel in ("free", "both"):
        panels["free_tier"] = FREE_TIER_PANEL
    if args.panel in ("paid", "both"):
        panels["paid_tier"] = PAID_TIER_PANEL

    async def _run_all() -> dict:
        out = {
            "spec": "QO-061",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_pairs": len(pairs),
            "closed_judge_reference": CLOSED_JUDGE_REFERENCE,
            "panels": {},
        }
        for name, cfg in panels.items():
            stats = await run_panel(name, cfg, pairs)
            out["panels"][name] = stats
        return out

    report = asyncio.run(_run_all())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JudgeBench report written to {args.output}")
    print(json.dumps({k: v for k, v in report.items() if k != "panels"}, indent=2))


if __name__ == "__main__":
    main()

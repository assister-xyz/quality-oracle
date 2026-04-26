#!/usr/bin/env python3
"""QO-061 backfill: add `eval_score_at_time` + `reporter_kya_tier` to legacy
production_feedback rows.

Behaviour:
- For every row in `quality__production_feedback` lacking `eval_score_at_time`:
    * Best-effort join to the matching `quality__scores` doc and copy
      `current_score` as the snapshot. This is a CONSERVATIVE backfill —
      the eval may have changed since the feedback was originally written;
      that's why we tag the row.
    * Default `reporter_kya_tier=1` (free-tier) — we cannot recover the
      reporter's actual tier at submission time.
    * Tag the row with `data_quality_warning='legacy_kya_unknown'` so the
      correlation engine surfaces the conservative weight.

Run:
    python -m dev.migrate_correlation_snapshots --dry-run
    python -m dev.migrate_correlation_snapshots --commit
"""
import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

# Allow running as a script: add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.storage.mongodb import feedback_col, scores_col  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LEGACY_WARNING = "legacy_kya_unknown"
DEFAULT_KYA_TIER = 1  # free-tier — conservative, weight=1.0


async def _resolve_eval_snapshot(target_id: str) -> Optional[int]:
    """Best-effort fetch of the current eval score for a target."""
    score_doc = await scores_col().find_one({"target_id": target_id})
    if not score_doc:
        return None
    return int(score_doc.get("current_score", 0))


async def migrate(commit: bool, batch_size: int = 200) -> dict:
    """Walk legacy rows and backfill the QO-061 fields.

    Returns a stats dict.
    """
    coll = feedback_col()
    query = {
        "$or": [
            {"eval_score_at_time": {"$exists": False}},
            {"reporter_kya_tier": {"$exists": False}},
        ]
    }

    seen = 0
    updated = 0
    score_unresolved = 0

    cursor = coll.find(query).batch_size(batch_size)
    cache: dict = {}

    async for doc in cursor:
        seen += 1
        target_id = doc.get("target_id")
        if not target_id:
            continue

        if target_id in cache:
            snap = cache[target_id]
        else:
            snap = await _resolve_eval_snapshot(target_id)
            cache[target_id] = snap
        if snap is None:
            score_unresolved += 1

        update = {
            "eval_score_at_time": snap if snap is not None else 0,
            "reporter_kya_tier": doc.get("reporter_kya_tier", DEFAULT_KYA_TIER),
            "data_quality_warning": LEGACY_WARNING,
        }
        if commit:
            await coll.update_one(
                {"_id": doc["_id"]},
                {"$set": update},
            )
            updated += 1
        else:
            updated += 1  # would-have-updated count

    logger.info(
        f"migrate(commit={commit}): scanned={seen} updated={updated} "
        f"score_unresolved={score_unresolved}"
    )
    return {
        "scanned": seen,
        "updated": updated,
        "score_unresolved": score_unresolved,
        "committed": commit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", action="store_true",
        help="Actually write the backfill (default: dry-run only).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print would-update count without writing (default).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Mongo cursor batch size (default 200).",
    )
    args = parser.parse_args()

    commit = bool(args.commit) and not args.dry_run

    stats = asyncio.run(migrate(commit=commit, batch_size=args.batch_size))
    print("Migration stats:", stats)


if __name__ == "__main__":
    main()

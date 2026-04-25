"""AC7 — Backfill QO-053-C dispatch fields onto legacy ``quality__evaluations``.

Pre-053-C documents predate the dispatch refactor and don't carry the
``target_type_dispatched``, ``axis_weights_used``, ``delta_vs_baseline``
fields. This script idempotently fills them so the new readers (badge API,
landing page, batch runner) get a uniform shape.

Backfill rules:

* ``target_type_dispatched`` — inferred from existing ``target_type`` field
  (which already exists for every doc — that part of the schema didn't move).
* ``axis_weights_used`` — set to the legacy ``DEFAULT_WEIGHTS`` table for
  ``mcp_server`` rows. ``agent`` and ``skill`` rows (if any predate
  dispatch) get ``MANIFEST_LESS_WEIGHTS`` / ``SKILL_WEIGHTS`` respectively.
* ``delta_vs_baseline`` — left as ``None`` (legacy rows had no baseline).
* ``baseline_score`` — same.
* ``baseline_status`` — set to ``"legacy"`` so downstream readers can
  distinguish "no baseline because feature didn't exist" from "baseline
  failed at eval time".
* ``subject_uri`` — copied from existing ``target_url`` (cheapest correct
  default; AQVC v1 will refine this).

Usage::

    python dev/migrate_legacy_evaluations.py --dry-run
    python dev/migrate_legacy_evaluations.py            # actually writes

The script is idempotent: documents that already have
``target_type_dispatched`` set are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from src.core.axis_weights import (
    DEFAULT_WEIGHTS,
    MANIFEST_LESS_WEIGHTS,
    SKILL_WEIGHTS,
)
from src.storage.models import TargetType

logger = logging.getLogger(__name__)


def _legacy_weights_for(target_type: str) -> dict:
    if target_type == TargetType.SKILL.value:
        return dict(SKILL_WEIGHTS)
    if target_type == TargetType.AGENT.value:
        return dict(MANIFEST_LESS_WEIGHTS)
    return dict(DEFAULT_WEIGHTS)  # mcp_server (legacy default)


async def migrate(*, dry_run: bool, batch_size: int = 500) -> dict:
    """Run the backfill. Returns counters for the report at the end."""
    from src.storage.mongodb import evaluations_col

    col = evaluations_col()
    cursor = col.find(
        {"target_type_dispatched": {"$exists": False}},
        {"_id": 1, "target_type": 1, "target_url": 1},
        batch_size=batch_size,
    )

    counters = {"scanned": 0, "would_update": 0, "updated": 0, "skipped": 0}
    async for doc in cursor:
        counters["scanned"] += 1
        target_type = doc.get("target_type", TargetType.MCP_SERVER.value)
        weights = _legacy_weights_for(target_type)
        update = {
            "$set": {
                "target_type_dispatched": target_type,
                "axis_weights_used": weights,
                "delta_vs_baseline": None,
                "baseline_score": None,
                "baseline_status": "legacy",
                "subject_uri": doc.get("target_url", ""),
            }
        }

        if dry_run:
            counters["would_update"] += 1
            continue

        try:
            await col.update_one({"_id": doc["_id"]}, update)
            counters["updated"] += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Update failed for _id=%s: %s", doc.get("_id"), exc)
            counters["skipped"] += 1

    return counters


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be updated without writing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="MongoDB cursor batch size.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    counters = asyncio.run(migrate(dry_run=args.dry_run, batch_size=args.batch_size))
    mode = "dry-run" if args.dry_run else "live"
    print(
        f"[migrate_legacy_evaluations:{mode}] "
        f"scanned={counters['scanned']} "
        f"updated={counters['updated']} "
        f"would_update={counters['would_update']} "
        f"skipped={counters['skipped']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

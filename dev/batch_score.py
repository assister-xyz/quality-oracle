#!/usr/bin/env python3
"""
Batch auto-scoring pipeline for MCP servers (QO-020).

Reads a registry JSON, evaluates each server, stores results in MongoDB,
and generates reports.

Usage:
    python -m dev.batch_score --registry dev/registry.json --level verified
    python -m dev.batch_score --registry dev/registry.json --level quick --concurrency 3
    python -m dev.batch_score --registry dev/registry.json --level verified --force
    python -m dev.batch_score --registry dev/registry.json --limit 5
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
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Registry ────────────────────────────────────────────────────────────────


def load_registry(path: str) -> dict:
    """Load and validate a registry JSON file."""
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    with open(registry_path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "servers" not in data:
        raise ValueError("Registry JSON must have a 'servers' array")

    servers = data["servers"]
    if not isinstance(servers, list):
        raise ValueError("'servers' must be a list")

    for i, s in enumerate(servers):
        if not s.get("url"):
            raise ValueError(f"Server at index {i} is missing 'url'")
        if not s.get("name"):
            s["name"] = s["url"]

    return data


# ── Rate Limiter ─────────────────────────────────────────────────────────────


class RateLimiter:
    """Serialize evaluation calls with minimum gap to respect LLM quotas."""

    def __init__(self, min_gap: float = 2.5):
        self.min_gap = min_gap
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self):
        async with self._lock:
            now = time.time()
            wait_time = self._last_call + self.min_gap - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call = time.time()


# ── Database Helpers ─────────────────────────────────────────────────────────


async def init_db():
    """Connect to MongoDB and Redis."""
    from src.storage.mongodb import connect_db
    from src.storage.cache import connect_redis
    await connect_db()
    await connect_redis()


async def close_db_connections():
    """Close MongoDB and Redis connections."""
    from src.storage.mongodb import close_db
    from src.storage.cache import close_redis
    await close_db()
    await close_redis()


async def check_existing_score(target_id: str) -> Optional[dict]:
    """Check if a server already has a score in the database."""
    from src.storage.mongodb import scores_col
    return await scores_col().find_one({"target_id": target_id})


async def save_score(target_id: str, eval_result, manifest: dict, server_entry: dict):
    """Save evaluation result to the scores collection."""
    from src.storage.mongodb import scores_col, score_history_col

    now = datetime.now(timezone.utc)

    dimensions = {}
    if eval_result.dimensions:
        dimensions = {k: v["score"] for k, v in eval_result.dimensions.items()}

    score_doc = {
        "target_id": target_id,
        "target_type": "mcp_server",
        "current_score": eval_result.overall_score,
        "tier": eval_result.tier,
        "confidence": eval_result.confidence,
        "evaluation_version": "v1.0",
        "last_evaluated_at": now,
        "tool_scores": eval_result.tool_scores,
        "dimensions": dimensions,
        "tools_count": len(manifest.get("tools", [])),
        "manifest_hash": "",
        "detected_domain": server_entry.get("category", "general"),
        "detected_domains": [],
        "source": "batch_score",
        "transport": server_entry.get("transport", "streamable_http"),
    }

    # Safety report
    if eval_result.safety_report:
        score_doc["safety_report"] = eval_result.safety_report
    if eval_result.latency_stats:
        score_doc["latency_stats"] = eval_result.latency_stats

    await scores_col().update_one(
        {"target_id": target_id},
        {"$set": score_doc, "$inc": {"evaluation_count": 1},
         "$setOnInsert": {"first_evaluated_at": now}},
        upsert=True,
    )

    # Score history
    await score_history_col().insert_one({
        "target_id": target_id,
        "target_type": "mcp_server",
        "score": eval_result.overall_score,
        "tier": eval_result.tier,
        "confidence": eval_result.confidence,
        "evaluation_version": "v1.0",
        "eval_mode": "verified",
        "recorded_at": now,
        "source": "batch_score",
    })


# ── Single Server Evaluation ────────────────────────────────────────────────


async def evaluate_one(
    server: dict,
    level: str,
    judge,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    idx: int,
    total: int,
    force: bool = False,
) -> dict:
    """Evaluate a single server from the registry."""
    from src.core.mcp_client import get_server_manifest, evaluate_server
    from src.core.evaluator import Evaluator
    from src.core.quick_scan import quick_scan

    name = server["name"]
    url = server["url"]
    result = {
        "name": name,
        "url": url,
        "transport": server.get("transport", "streamable_http"),
        "category": server.get("category", "general"),
        "status": "pending",
        "score": None,
        "tier": None,
        "tools_count": 0,
        "error": None,
        "error_type": None,
        "duration_ms": 0,
        "skipped_reason": None,
    }

    # Skip auth-required servers
    if server.get("requires_auth"):
        result["status"] = "skipped"
        result["skipped_reason"] = "requires_auth"
        print(f"  [{idx}/{total}] SKIP {name} (requires auth)")
        return result

    # Skip already-scored (unless --force)
    if not force:
        existing = await check_existing_score(url)
        if existing:
            result["status"] = "skipped"
            result["skipped_reason"] = "already_scored"
            result["score"] = existing.get("current_score")
            result["tier"] = existing.get("tier")
            print(f"  [{idx}/{total}] SKIP {name} (already scored: {result['score']}/{result['tier']})")
            return result

    async with semaphore:
        await rate_limiter.wait()
        start = time.time()

        try:
            if level == "quick":
                # L1 quick scan -- no LLM, no DB storage needed
                scan_result = await quick_scan(url)
                result["status"] = "success" if scan_result.reachable else "error"
                result["score"] = scan_result.manifest_score
                result["tier"] = scan_result.estimated_tier
                result["tools_count"] = scan_result.tool_count
                result["duration_ms"] = scan_result.scan_time_ms
                if scan_result.error:
                    result["status"] = "error"
                    result["error"] = scan_result.error
                    result["error_type"] = "connection"
                print(f"  [{idx}/{total}] {'OK' if result['status'] == 'success' else 'FAIL'} {name}: score={result['score']}, tier={result['tier']}")
                return result

            # L2 verified evaluation
            print(f"  [{idx}/{total}] Evaluating {name}...")

            # Get manifest
            manifest = await get_server_manifest(url)
            result["tools_count"] = len(manifest.get("tools", []))
            print(f"  [{idx}/{total}] {name}: {result['tools_count']} tools via {manifest.get('transport', '?')}")

            # Run functional eval
            tool_responses = await evaluate_server(url)
            total_cases = sum(len(v) for v in tool_responses.values())
            print(f"  [{idx}/{total}] {name}: {total_cases} test cases executed")

            evaluator = Evaluator(llm_judge=judge)
            eval_result = await evaluator.evaluate_full(
                target_id=url,
                server_url=url,
                tool_responses=tool_responses,
                manifest=manifest,
                run_safety=True,
            )

            result["status"] = "success"
            result["score"] = eval_result.overall_score
            result["tier"] = eval_result.tier
            result["confidence"] = round(eval_result.confidence, 2)
            result["questions_asked"] = eval_result.questions_asked
            result["duration_ms"] = int((time.time() - start) * 1000)
            result["tool_scores"] = eval_result.tool_scores
            if eval_result.dimensions:
                result["dimensions"] = {
                    k: v["score"] for k, v in eval_result.dimensions.items()
                }

            # Save to database
            await save_score(url, eval_result, manifest, server)

            dims = result.get("dimensions", {})
            dims_str = " ".join(f"{k}={v}" for k, v in dims.items()) if dims else ""
            print(f"  [{idx}/{total}] OK {name}: score={result['score']}, tier={result['tier']} {dims_str}")

        except ConnectionError as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["error_type"] = "connection"
            result["duration_ms"] = int((time.time() - start) * 1000)
            print(f"  [{idx}/{total}] FAIL {name}: {e}")

        except asyncio.TimeoutError:
            result["status"] = "error"
            result["error"] = "Evaluation timed out"
            result["error_type"] = "timeout"
            result["duration_ms"] = int((time.time() - start) * 1000)
            print(f"  [{idx}/{total}] FAIL {name}: timeout")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:300]
            result["error_type"] = "unknown"
            result["duration_ms"] = int((time.time() - start) * 1000)
            print(f"  [{idx}/{total}] FAIL {name}: {e}")

    return result


# ── Report Generation ────────────────────────────────────────────────────────


def save_reports(results: list, reports_dir: Path) -> tuple[Path, Path]:
    """Save batch results as JSON and Markdown reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    successful = sorted(
        [r for r in results if r["status"] == "success"],
        key=lambda r: r["score"] or 0,
        reverse=True,
    )
    failed = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]

    # JSON report
    json_path = reports_dir / f"batch-{date_str}.json"
    with open(json_path, "w") as f:
        json.dump({
            "batch_date": datetime.now(timezone.utc).isoformat(),
            "servers_total": len(results),
            "servers_success": len(successful),
            "servers_failed": len(failed),
            "servers_skipped": len(skipped),
            "avg_score": round(
                sum(r["score"] for r in successful) / len(successful), 1
            ) if successful else 0,
            "failure_reasons": _count_failure_reasons(failed),
            "skip_reasons": _count_skip_reasons(skipped),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  JSON report: {json_path}")

    # Markdown report
    md_path = reports_dir / f"batch-{date_str}.md"
    lines = [
        f"# Batch Auto-Scoring Report -- {date_str}",
        "",
        f"**Total:** {len(results)} | "
        f"**Success:** {len(successful)} | "
        f"**Failed:** {len(failed)} | "
        f"**Skipped:** {len(skipped)}",
        "",
    ]

    if successful:
        scores = [r["score"] for r in successful]
        lines.append(f"**Avg Score:** {sum(scores)/len(scores):.0f}/100 | "
                      f"**Highest:** {max(scores)}/100 | **Lowest:** {min(scores)}/100")
        lines.append("")
        lines.append("## Scored Servers")
        lines.append("")
        lines.append("| Rank | Server | Score | Tier | Tools | Category |")
        lines.append("|------|--------|-------|------|-------|----------|")
        for i, r in enumerate(successful, 1):
            lines.append(
                f"| {i} | {r['name']} | {r['score']}/100 | {r['tier']} | "
                f"{r['tools_count']} | {r.get('category', '-')} |"
            )

    if failed:
        lines.extend(["", "## Failed Servers", ""])
        by_reason = _count_failure_reasons(failed)
        for reason, count in by_reason.items():
            lines.append(f"- **{reason}:** {count}")
        lines.append("")
        for r in failed:
            lines.append(f"- {r['name']} ({r['url']}): {r.get('error_type', '?')} -- {r.get('error', '?')[:100]}")

    if skipped:
        lines.extend(["", "## Skipped Servers", ""])
        for r in skipped:
            lines.append(f"- {r['name']}: {r.get('skipped_reason', '?')}")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by AgentTrust batch scorer at {datetime.now(timezone.utc).isoformat()}*")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Markdown report: {md_path}")

    return json_path, md_path


def _count_failure_reasons(failed: list) -> dict:
    reasons = {}
    for r in failed:
        t = r.get("error_type", "unknown")
        reasons[t] = reasons.get(t, 0) + 1
    return reasons


def _count_skip_reasons(skipped: list) -> dict:
    reasons = {}
    for r in skipped:
        t = r.get("skipped_reason", "unknown")
        reasons[t] = reasons.get(t, 0) + 1
    return reasons


# ── Summary ──────────────────────────────────────────────────────────────────


def print_summary(results: list):
    """Print a console summary of the batch run."""
    successful = sorted(
        [r for r in results if r["status"] == "success"],
        key=lambda r: r["score"] or 0,
        reverse=True,
    )
    failed = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print(f"\n{'='*70}")
    print("  BATCH SCORING RESULTS")
    print(f"{'='*70}\n")

    if successful:
        print(f"  {'Server':<25s} {'Score':>6s} {'Tier':<12s} {'Tools':>5s} {'Category':<15s}")
        print(f"  {'---'*25}")
        for r in successful:
            print(f"  {r['name']:<25s} {r['score']:>3d}/100 {r['tier']:<12s} "
                  f"{r['tools_count']:>5d} {r.get('category', '-'):<15s}")

    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for r in failed:
            print(f"    x {r['name']}: [{r.get('error_type', '?')}] {r.get('error', '?')[:80]}")

    if skipped:
        print(f"\n  Skipped ({len(skipped)}):")
        for r in skipped:
            print(f"    - {r['name']} ({r.get('skipped_reason', '?')})")

    scores = [r["score"] for r in successful if r["score"] is not None]
    if scores:
        print(f"\n  Average score: {sum(scores)/len(scores):.0f}/100")
        print(f"  Highest: {max(scores)}/100 | Lowest: {min(scores)}/100")

    print(f"\n  Total: {len(results)} servers | "
          f"{len(successful)} success | {len(failed)} failed | {len(skipped)} skipped")
    print(f"{'='*70}\n")


# ── LLM Judge Setup ─────────────────────────────────────────────────────────


def _get_judge():
    """Create LLM judge for batch scoring. Reuses scan_servers logic."""
    from dev.scan_servers import _get_judge as scan_get_judge
    return scan_get_judge()


# ── Main ─────────────────────────────────────────────────────────────────────


async def main(
    registry_path: str,
    level: str = "verified",
    concurrency: int = 2,
    force: bool = False,
    limit: Optional[int] = None,
):
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # Load registry
    registry = load_registry(registry_path)
    servers = registry["servers"]
    if limit:
        servers = servers[:limit]

    print(f"\n{'='*70}")
    print("  AgentTrust -- Batch Auto-Scoring Pipeline")
    print(f"  Registry: {registry_path} ({len(servers)} servers)")
    print(f"  Level: {level} | Concurrency: {concurrency} | Force: {force}")
    print(f"{'='*70}\n")

    # Connect to database
    print("  Connecting to database...")
    await init_db()

    # Setup judge (only needed for L2+)
    judge = None
    if level != "quick":
        judge = _get_judge()
        print(f"  Judge: {'LLM' if judge.is_llm_available else 'fuzzy fallback'}")

    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = RateLimiter(min_gap=2.5)
    total = len(servers)

    # Run evaluations sequentially to respect rate limits
    results = []
    for i, server in enumerate(servers, 1):
        result = await evaluate_one(
            server=server,
            level=level,
            judge=judge,
            semaphore=semaphore,
            rate_limiter=rate_limiter,
            idx=i,
            total=total,
            force=force,
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Save reports
    reports_dir = Path("reports")
    save_reports(results, reports_dir)

    # Close connections
    await close_db_connections()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch auto-scoring pipeline (QO-020)")
    parser.add_argument(
        "--registry", type=str, default="dev/registry.json",
        help="Path to registry JSON file (default: dev/registry.json)",
    )
    parser.add_argument(
        "--level", type=str, default="verified",
        choices=["quick", "verified"],
        help="Evaluation level: quick (L1 manifest) or verified (L2 functional)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="Max concurrent evaluations (default: 2)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-evaluation of already-scored servers",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of servers to evaluate",
    )
    args = parser.parse_args()

    asyncio.run(main(
        registry_path=args.registry,
        level=args.level,
        concurrency=args.concurrency,
        force=args.force,
        limit=args.limit,
    ))

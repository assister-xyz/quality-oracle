#!/usr/bin/env python3
"""Weekly MCP batch evaluation via public /v1/evaluate API (QO-049 Phase B).

Runs under GitHub Actions cron every Sunday at 02:00 UTC. For each server in
dev/registry.json with requires_auth=false, submits a Level-1 verified
evaluation, polls to completion, and writes a markdown summary report.

Designed to use only the public API — no direct MongoDB access required.
Keeps the cron infra-light: no AWS EventBridge, no ECS RunTask, just a
GitHub Actions runner + a service API key.

Environment:
  LAUREUM_API_BASE   e.g. https://quality-oracle-api.assisterr.ai (optional)
  LAUREUM_API_KEY    required — a service-owned API key

Usage:
  python dev/weekly_batch_via_api.py
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import httpx

API_BASE = os.environ.get(
    "LAUREUM_API_BASE", "https://quality-oracle-api.assisterr.ai"
).rstrip("/")
API_KEY = os.environ.get("LAUREUM_API_KEY")

CONCURRENCY = 5
SUBMIT_DELAY_SECONDS = 2
POLL_INTERVAL_SECONDS = 15
MAX_POLL_CYCLES = 40  # 10 minutes total upper bound

# Score-protection: skip re-evaluating servers that already hold a strong
# score within this window. Level-1 (free) Quick Scan caps output around
# score=83, so running it over an existing Expert (93/92/87…) would
# overwrite a deep-eval result with a baseline score — regression we want
# to avoid in the weekly refresh. Expert deep-evals happen via paid Level-2+
# manually / on-demand; the cron's job is liveness + basic/failed refresh.
PROTECT_MIN_SCORE = 85  # Expert tier lower bound
PROTECT_MAX_AGE_DAYS = 14


async def _submit(client: httpx.AsyncClient, name: str, url: str) -> Dict[str, Any]:
    try:
        r = await client.post(
            f"{API_BASE}/v1/evaluate",
            json={"target_url": url, "level": 1, "eval_mode": "verified"},
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        )
    except Exception as e:
        return {"name": name, "url": url, "status": "submit_error", "error": str(e)[:200]}
    if r.status_code != 200:
        return {
            "name": name,
            "url": url,
            "status": "submit_failed",
            "http": r.status_code,
            "body": r.text[:200],
        }
    d = r.json()
    return {
        "name": name,
        "url": url,
        "status": "submitted",
        "evaluation_id": d.get("evaluation_id"),
    }


async def _fetch_existing_scores(client: httpx.AsyncClient) -> Dict[str, Dict[str, Any]]:
    """Map target_id -> existing score record, used by the protection guard."""
    try:
        r = await client.get(
            f"{API_BASE}/v1/scores?limit=100",
            headers={"X-API-Key": API_KEY},
        )
        if r.status_code != 200:
            print(f"Warn: /v1/scores returned {r.status_code}, protection guard disabled")
            return {}
        items = r.json().get("items", [])
        return {it["target_id"]: it for it in items if it.get("target_id")}
    except Exception as e:
        print(f"Warn: could not fetch existing scores ({e}); protection guard disabled")
        return {}


def _should_skip(url: str, existing: Dict[str, Dict[str, Any]]) -> bool:
    """Skip re-evaluation if server holds a fresh Expert-tier score.

    Level-1 Quick Scan caps at ~83; running it over a 93 would regress the
    server. The cron's job is freshness for mid/low tier — Expert entries
    are refreshed via deeper manual evals.
    """
    rec = existing.get(url)
    if not rec:
        return False
    score = rec.get("score") or 0
    if score < PROTECT_MIN_SCORE:
        return False
    last = rec.get("last_evaluated_at")
    if not last:
        return False
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).days
        return age_days < PROTECT_MAX_AGE_DAYS
    except Exception:
        return False


async def _poll(client: httpx.AsyncClient, eid: str) -> Dict[str, Any]:
    try:
        r = await client.get(
            f"{API_BASE}/v1/evaluate/{eid}",
            headers={"X-API-Key": API_KEY},
        )
    except Exception as e:
        return {"status": "poll_error", "error": str(e)[:200]}
    if r.status_code == 404:
        return {"status": "lost"}
    if r.status_code != 200:
        return {"status": "poll_http_error", "http": r.status_code}
    return r.json()


async def main() -> int:
    if not API_KEY:
        print("::error::LAUREUM_API_KEY is required", file=sys.stderr)
        return 2

    registry_path = Path(__file__).parent / "registry.json"
    registry = json.loads(registry_path.read_text())
    all_servers = registry.get("servers", [])
    no_auth = [s for s in all_servers if not s.get("requires_auth")]
    print(f"Registry: {len(all_servers)} total, {len(no_auth)} no-auth candidates")

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        # Score-protection guard: fetch current prod state and skip servers
        # that hold a fresh Expert-tier score to avoid Level-1 regression.
        existing = await _fetch_existing_scores(client)
        protected = [s for s in no_auth if _should_skip(s["url"], existing)]
        to_submit = [s for s in no_auth if not _should_skip(s["url"], existing)]
        if protected:
            print(f"Protected {len(protected)} Expert-tier entries (skipped): " +
                  ", ".join(s["name"] for s in protected))

        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded_submit(server: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                out = await _submit(client, server["name"], server["url"])
                await asyncio.sleep(SUBMIT_DELAY_SECONDS)
                return out

        submissions = await asyncio.gather(*[bounded_submit(s) for s in to_submit])
        ok = [s for s in submissions if s["status"] == "submitted"]
        bad = [s for s in submissions if s["status"] != "submitted"]
        print(f"Submitted {len(ok)}/{len(submissions)} evaluations ({len(bad)} submission errors)")

        pending: Dict[str, Dict[str, Any]] = {s["evaluation_id"]: s for s in ok}
        done: Dict[str, Dict[str, Any]] = {}

        for cycle in range(MAX_POLL_CYCLES):
            if not pending:
                break
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            items = list(pending.items())
            results = await asyncio.gather(*[_poll(client, eid) for eid, _ in items])
            for (eid, sub), res in zip(items, results):
                st = res.get("status")
                if st in ("completed", "failed", "lost", "poll_error", "poll_http_error"):
                    done[eid] = {**sub, **res}
                    pending.pop(eid, None)
            print(f"[cycle {cycle + 1}/{MAX_POLL_CYCLES}] done={len(done)}, pending={len(pending)}")

        for eid, sub in pending.items():
            done[eid] = {**sub, "status": "timeout"}

    completed = [d for d in done.values() if d.get("status") == "completed"]
    failed_evals = [d for d in done.values() if d.get("status") == "failed"]
    scores = [d.get("score", 0) for d in completed if isinstance(d.get("score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0

    # Tier summary
    from collections import Counter
    tier_counts = Counter(d.get("tier") for d in completed if d.get("tier"))

    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    report_path = Path(__file__).parent.parent / "reports" / f"weekly-batch-{stamp}.md"
    report_path.parent.mkdir(exist_ok=True)

    lines = [
        f"# Weekly Batch Report — {ts.isoformat()}",
        "",
        f"- **Registry:** {len(all_servers)} total ({len(no_auth)} no-auth)",
        f"- **Submitted:** {len(ok)}",
        f"- **Submission errors:** {len(bad)}",
        f"- **Completed:** {len(completed)}",
        f"- **Failed:** {len(failed_evals)}",
        f"- **Average score:** {avg_score}/100",
        f"- **Tier distribution:** {dict(tier_counts)}",
        "",
        "## Per-server",
    ]
    for d in sorted(done.values(), key=lambda x: -(x.get("score") or 0)):
        name = d.get("name", "?")
        status = d.get("status", "?")
        score = d.get("score", "—")
        tier = d.get("tier", "—")
        err = d.get("error") or ""
        suffix = f" · {err[:100]}" if err else ""
        lines.append(f"- **{name}** — status=`{status}` score={score} tier={tier}{suffix}")

    if bad:
        lines.extend(["", "## Submission errors"])
        for b in bad:
            lines.append(f"- **{b['name']}** ({b['url']}) — {b.get('status')} {b.get('http', '')} {b.get('error', '')[:120]}")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport: {report_path}")

    if len(completed) == 0:
        print("::error::No evaluations completed successfully", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

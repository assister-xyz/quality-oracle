#!/usr/bin/env python3
"""
Scan real public MCP servers and evaluate their quality.

Usage:
    .venv/bin/python dev/scan_servers.py              # Scan all servers (streaming)
    .venv/bin/python dev/scan_servers.py --sse-only    # SSE servers only
    .venv/bin/python dev/scan_servers.py --limit 3     # First N servers only
    .venv/bin/python dev/scan_servers.py --provider cerebras  # Force specific provider
    .venv/bin/python dev/scan_servers.py --no-stream   # Disable streaming eval
    .venv/bin/python dev/scan_servers.py --url <url>   # Scan a single URL

No MongoDB/Redis dependency — pure evaluation + LLM judging.
Results saved to reports/scan-{date}.json and reports/scan-{date}.md.
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

# ── Server Registry ──────────────────────────────────────────────────────────

SERVERS = [
    # ── Verified Working (SSE) ──────────────────────────────────────────
    {"name": "Cloudflare Docs", "url": "https://docs.mcp.cloudflare.com/sse", "transport": "sse"},
    {"name": "TweetSave", "url": "https://mcp.tweetsave.org/sse", "transport": "sse"},

    # ── Verified Working (Streamable HTTP) ──────────────────────────────
    {"name": "DeepWiki", "url": "https://mcp.deepwiki.com/mcp", "transport": "streamable_http"},
    {"name": "HuggingFace", "url": "https://hf.co/mcp", "transport": "streamable_http"},
    {"name": "GitMCP", "url": "https://gitmcp.io/docs", "transport": "streamable_http"},
    {"name": "Context7", "url": "https://mcp.context7.com/mcp", "transport": "streamable_http"},
    {"name": "Astro Docs", "url": "https://docs.astro.build/mcp", "transport": "streamable_http"},
    {"name": "Find-A-Domain", "url": "https://api.find-a.domain/mcp", "transport": "streamable_http"},
    {"name": "OpenZeppelin", "url": "https://mcp.openzeppelin.com/mcp", "transport": "streamable_http"},
    {"name": "Ferryhopper", "url": "https://mcp.ferryhopper.com/mcp", "transport": "streamable_http"},
    {"name": "Manifold", "url": "https://api.manifold.markets/v0/mcp", "transport": "streamable_http"},
    {"name": "SubwayInfo NYC", "url": "https://mcp.subwayinfo.nyc/mcp", "transport": "streamable_http"},
    {"name": "zip1.io", "url": "https://api.zip1.io/mcp", "transport": "streamable_http"},
    {"name": "Javadocs", "url": "https://javadocs.dev/mcp", "transport": "streamable_http"},
    {"name": "OpenMesh", "url": "https://mcp.openmesh.network/mcp", "transport": "streamable_http"},
    {"name": "Semgrep", "url": "https://mcp.semgrep.ai/mcp", "transport": "streamable_http"},

    # ── Additional Public Servers (from awesome-remote-mcp-servers) ─────
    {"name": "Browserbase", "url": "https://mcp.browserbase.com/mcp", "transport": "streamable_http"},
    {"name": "Neon DB", "url": "https://mcp.neon.tech/mcp", "transport": "streamable_http"},
    {"name": "Vectorize", "url": "https://mcp.vectorize.io/mcp", "transport": "streamable_http"},
    {"name": "Firecrawl", "url": "https://mcp.firecrawl.dev/mcp", "transport": "streamable_http"},
    {"name": "Composio", "url": "https://mcp.composio.dev/mcp", "transport": "streamable_http"},
    {"name": "Resend Email", "url": "https://mcp.resend.com/mcp", "transport": "streamable_http"},
    {"name": "Stripe", "url": "https://mcp.stripe.com/mcp", "transport": "streamable_http"},
    {"name": "CoinGecko", "url": "https://mcp.api.coingecko.com/mcp", "transport": "streamable_http"},
    {"name": "BGPt", "url": "https://mcp.bgpt.co/mcp", "transport": "streamable_http"},
    {"name": "OZ Cairo", "url": "https://cairo.mcp.openzeppelin.com/mcp", "transport": "streamable_http"},
    {"name": "OZ Stellar", "url": "https://stellar.mcp.openzeppelin.com/mcp", "transport": "streamable_http"},
    {"name": "OZ Stylus", "url": "https://stylus.mcp.openzeppelin.com/mcp", "transport": "streamable_http"},
    {"name": "Dify", "url": "https://mcp.dify.ai/mcp", "transport": "streamable_http"},
    {"name": "Notion", "url": "https://mcp.notion.so/mcp", "transport": "streamable_http"},
    {"name": "Linear", "url": "https://mcp.linear.app/mcp", "transport": "streamable_http"},
    {"name": "Sentry", "url": "https://mcp.sentry.dev/mcp", "transport": "streamable_http"},
    {"name": "Cloudflare Workers", "url": "https://workers.mcp.cloudflare.com/mcp", "transport": "streamable_http"},
    {"name": "Turso", "url": "https://mcp.turso.tech/mcp", "transport": "streamable_http"},
    {"name": "Trigger.dev", "url": "https://mcp.trigger.dev/mcp", "transport": "streamable_http"},
    {"name": "Tinybird", "url": "https://mcp.tinybird.co/mcp", "transport": "streamable_http"},
    {"name": "Grafbase", "url": "https://mcp.grafbase.com/mcp", "transport": "streamable_http"},

    # ── Requires Auth (skipped in no-auth scan) ─────────────────────────
    # {"name": "Exa Search", "url": "https://mcp.exa.ai/mcp", "transport": "streamable_http"},
    # {"name": "AWS Knowledge", "url": "https://knowledge-mcp.global.api.aws", "transport": "streamable_http"},

    # ── Down/Migrated (removed) ─────────────────────────────────────────
    # LLM Text: https://mcp.llmtxt.dev/sse — connection refused
]

# ── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Serialize LLM judge calls with minimum gap (safe for Groq ~24/min)."""

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


# ── Judge Setup ──────────────────────────────────────────────────────────────

# All providers in priority order: Cerebras → Groq → Gemini → DeepSeek → OpenAI → OpenRouter → Mistral
PROVIDER_PRIORITY = ["cerebras", "groq", "gemini", "deepseek", "openai", "openrouter", "mistral"]


def _get_provider_config(provider_name: str):
    """Get (api_key, model, provider, base_url) for a named provider."""
    from src.config import settings

    configs = {
        "cerebras": (settings.cerebras_api_key, settings.cerebras_model, "cerebras", settings.cerebras_base_url),
        "groq": (settings.groq_api_key, settings.groq_model, "groq", "https://api.groq.com/openai/v1"),
        "gemini": (settings.gemini_api_key, settings.gemini_model, "gemini", settings.gemini_base_url),
        "deepseek": (settings.deepseek_api_key, settings.deepseek_model, "deepseek", settings.deepseek_base_url),
        "openai": (settings.openai_api_key, settings.openai_model, "openai", settings.openai_base_url),
        "openrouter": (settings.openrouter_api_key, settings.openrouter_model, "openrouter", settings.openrouter_base_url),
        "mistral": (settings.mistral_api_key, settings.mistral_model, "mistral", settings.mistral_base_url),
    }
    return configs.get(provider_name)


def _get_judge(force_provider: Optional[str] = None):
    """Create LLM judge for scanning.

    Uses ConsensusJudge when 2+ providers available, single best otherwise.
    Priority: Cerebras → Groq → Gemini → DeepSeek → OpenAI → OpenRouter → Mistral.

    Args:
        force_provider: Force a specific provider (e.g., "cerebras", "groq").
    """
    from src.config import settings
    from src.core.llm_judge import LLMJudge

    # Force a specific provider if requested
    if force_provider:
        cfg = _get_provider_config(force_provider)
        if cfg and cfg[0]:  # has API key
            key, model, provider, base_url = cfg
            print(f"  Forced provider: {provider} ({model})")
            return LLMJudge(api_key=key, model=model, provider=provider, base_url=base_url)
        else:
            print(f"  WARNING: Provider '{force_provider}' has no API key, falling back to auto-select")

    # Try ConsensusJudge if enabled and 2+ providers available
    if settings.consensus_enabled:
        from src.core.consensus_judge import ConsensusJudge
        judge = ConsensusJudge(
            agreement_threshold=settings.consensus_agreement_threshold,
            min_judges=settings.consensus_min_judges,
        )
        if judge.is_consensus_possible:
            providers = [j.provider for j in judge._judges if j.is_llm_available]
            print(f"  ConsensusJudge: {len(providers)} providers — {', '.join(providers)}")
            return judge

    # Single judge fallback — walk priority list
    for provider_name in PROVIDER_PRIORITY:
        cfg = _get_provider_config(provider_name)
        if cfg and cfg[0]:  # has API key
            key, model, provider, base_url = cfg
            return LLMJudge(api_key=key, model=model, provider=provider, base_url=base_url)

    return LLMJudge()  # fuzzy fallback


# ── Single Server Scan ───────────────────────────────────────────────────────

MAX_RETRIES = 2
RETRY_DELAY = 5.0


async def scan_one_server(
    server: dict,
    judge,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    use_streaming: bool = True,
    runs: int = 1,
) -> dict:
    """Scan a single MCP server: manifest + functional eval + judging."""
    from src.core.mcp_client import evaluate_server, evaluate_server_streaming, get_server_manifest
    from src.core.evaluator import Evaluator
    from src.core.cancellation import CancellationToken

    name = server["name"]
    url = server["url"]
    result = {
        "name": name,
        "url": url,
        "transport": server["transport"],
        "status": "pending",
        "score": None,
        "tier": None,
        "tools_count": 0,
        "questions_asked": 0,
        "confidence": 0.0,
        "duration_ms": 0,
        "tool_scores": {},
        "error": None,
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                start = time.time()
                print(f"  [{name}] Connecting... (attempt {attempt + 1})")

                # Get manifest
                await rate_limiter.wait()
                manifest = await get_server_manifest(url)
                result["tools_count"] = len(manifest.get("tools", []))
                result["transport_used"] = manifest.get("transport", server["transport"])
                print(f"  [{name}] {manifest['name']} v{manifest['version']} — {result['tools_count']} tools")

                evaluator = Evaluator(llm_judge=judge)

                if use_streaming:
                    # ── Streaming path: judge as each tool call returns ──
                    cancel = CancellationToken()
                    case_count = [0]
                    # Accumulate tool_responses for dimension enrichment
                    collected_responses = {}

                    def on_progress(tool_name, idx, score, running_avg):
                        case_count[0] = idx
                        print(f"    [{name}] {tool_name} #{idx}: score={score} (avg={running_avg})")

                    # Wrap the stream to collect tool_responses
                    async def _collecting_stream(inner_stream):
                        async for tool_name, case, response in inner_stream:
                            if tool_name not in collected_responses:
                                collected_responses[tool_name] = []
                            collected_responses[tool_name].append({
                                "question": case.get("question", ""),
                                "expected": case.get("expected", ""),
                                "answer": response.get("content", ""),
                                "latency_ms": response.get("latency_ms", 0),
                                "is_error": response.get("is_error", False),
                                "test_type": case.get("test_type", "unknown"),
                            })
                            yield tool_name, case, response

                    raw_stream = evaluate_server_streaming(url, cancel=cancel)
                    stream = _collecting_stream(raw_stream)
                    eval_result = await evaluator.evaluate_functional_streaming(
                        target_id=url,
                        response_stream=stream,
                        manifest=manifest,
                        cancel=cancel,
                        on_progress=on_progress,
                    )
                    total_cases = eval_result.questions_asked
                    if cancel.is_cancelled:
                        print(f"  [{name}] Early exit after {total_cases} cases: {cancel.reason}")
                    else:
                        print(f"  [{name}] Evaluated {total_cases} test cases (streaming)")

                    # Enrich with all 6 dimensions (safety, process_quality, reliability, latency, schema)
                    print(f"  [{name}] Computing dimensions...")
                    eval_result = await evaluator.enrich_with_dimensions(
                        result=eval_result,
                        tool_responses=collected_responses,
                        manifest=manifest,
                        server_url=url,
                        run_safety=True,
                    )

                else:
                    # ── Batch path: collect all, then judge ──
                    tool_responses = await evaluate_server(url)
                    total_cases = sum(len(v) for v in tool_responses.values())
                    print(f"  [{name}] Executed {total_cases} test cases")

                    eval_result = await evaluator.evaluate_full(
                        target_id=url,
                        server_url=url,
                        tool_responses=tool_responses,
                        manifest=manifest,
                        run_safety=True,
                    )

                # ── Multi-run aggregation ──
                run_results = [eval_result]
                if runs > 1:
                    for run_idx in range(1, runs):
                        print(f"  [{name}] Run {run_idx + 1}/{runs}...")
                        try:
                            run_evaluator = Evaluator(llm_judge=judge)
                            if use_streaming:
                                run_cancel = CancellationToken()
                                run_collected = {}

                                async def _run_collecting_stream(inner):
                                    async for tn, c, r in inner:
                                        if tn not in run_collected:
                                            run_collected[tn] = []
                                        run_collected[tn].append({
                                            "question": c.get("question", ""),
                                            "expected": c.get("expected", ""),
                                            "answer": r.get("content", ""),
                                            "latency_ms": r.get("latency_ms", 0),
                                            "is_error": r.get("is_error", False),
                                            "test_type": c.get("test_type", "unknown"),
                                        })
                                        yield tn, c, r

                                run_raw = evaluate_server_streaming(url, cancel=run_cancel)
                                run_stream = _run_collecting_stream(run_raw)
                                run_eval = await run_evaluator.evaluate_functional_streaming(
                                    target_id=url,
                                    response_stream=run_stream,
                                    manifest=manifest,
                                    cancel=run_cancel,
                                )
                                run_eval = await run_evaluator.enrich_with_dimensions(
                                    result=run_eval,
                                    tool_responses=run_collected,
                                    manifest=manifest,
                                    server_url=url,
                                    run_safety=False,  # Only run safety once
                                )
                            else:
                                run_tool_resp = await evaluate_server(url)
                                run_eval = await run_evaluator.evaluate_full(
                                    target_id=url, server_url=url,
                                    tool_responses=run_tool_resp,
                                    manifest=manifest, run_safety=False,
                                )
                            run_results.append(run_eval)
                            print(f"    [{name}] Run {run_idx + 1} score: {run_eval.overall_score}")
                        except Exception as run_err:
                            print(f"    [{name}] Run {run_idx + 1} failed: {run_err}")

                # Use median scores across runs
                if len(run_results) > 1:
                    all_run_scores = sorted(r.overall_score for r in run_results)
                    median_score = all_run_scores[len(all_run_scores) // 2]
                    # Use the result closest to median for dimensions/details
                    best_result = min(run_results, key=lambda r: abs(r.overall_score - median_score))
                    best_result.overall_score = median_score
                    eval_result = best_result
                    print(f"  [{name}] Median of {len(run_results)} runs: {median_score}")

                result["status"] = "success"
                result["score"] = eval_result.overall_score
                result["tier"] = eval_result.tier
                result["confidence"] = round(eval_result.confidence, 2)
                result["questions_asked"] = eval_result.questions_asked
                result["duration_ms"] = int((time.time() - start) * 1000)
                result["tool_scores"] = eval_result.tool_scores
                result["manifest_score"] = eval_result.manifest_result.score if eval_result.manifest_result else None
                if eval_result.dimensions:
                    result["dimensions"] = {
                        k: v["score"] for k, v in eval_result.dimensions.items()
                    }
                if eval_result.safety_report:
                    result["safety"] = eval_result.safety_report
                if eval_result.latency_stats:
                    result["latency"] = eval_result.latency_stats
                result["judge_responses"] = [
                    {
                        "tool": jr["tool"],
                        "question": jr["question"][:100],
                        "score": jr["score"],
                        "method": jr.get("method", "?"),
                    }
                    for jr in eval_result.judge_responses
                ]
                if len(run_results) > 1:
                    result["runs"] = [
                        {"score": r.overall_score, "questions": r.questions_asked}
                        for r in run_results
                    ]

                dims = result.get("dimensions", {})
                dims_str = " ".join(f"{k}={v}" for k, v in dims.items()) if dims else ""
                print(f"  [{name}] Score: {result['score']}/100 | Tier: {result['tier']} | {dims_str}")
                break

            except Exception as e:
                if attempt < MAX_RETRIES:
                    print(f"  [{name}] Attempt {attempt + 1} failed: {e}, retrying in {RETRY_DELAY}s...")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    result["status"] = "error"
                    result["error"] = str(e)
                    result["duration_ms"] = int((time.time() - start) * 1000)
                    print(f"  [{name}] FAILED: {e}")

    return result


# ── Report Generation ────────────────────────────────────────────────────────

def save_reports(results: list, reports_dir: Path):
    """Save scan results as JSON and Markdown reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # JSON report
    json_path = reports_dir / f"scan-{date_str}.json"
    with open(json_path, "w") as f:
        json.dump({
            "scan_date": datetime.now(timezone.utc).isoformat(),
            "servers_total": len(results),
            "servers_success": sum(1 for r in results if r["status"] == "success"),
            "servers_error": sum(1 for r in results if r["status"] == "error"),
            "servers_skipped": sum(1 for r in results if r["status"] == "skipped"),
            "results": results,
        }, f, indent=2)
    print(f"\n  JSON report: {json_path}")

    # Markdown report
    md_path = reports_dir / f"scan-{date_str}.md"
    successful = sorted(
        [r for r in results if r["status"] == "success"],
        key=lambda r: r["score"] or 0,
        reverse=True,
    )
    failed = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]

    lines = [
        f"# MCP Server Quality Scan — {date_str}",
        "",
        f"**Servers scanned:** {len(results)} | "
        f"**Success:** {len(successful)} | "
        f"**Failed:** {len(failed)} | "
        f"**Skipped:** {len(skipped)}",
        "",
        "## Results",
        "",
        "| Rank | Server | Score | Tier | Tools | Transport | Confidence |",
        "|------|--------|-------|------|-------|-----------|------------|",
    ]

    for i, r in enumerate(successful, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['score']}/100 | {r['tier']} | "
            f"{r['tools_count']} | {r.get('transport_used', r['transport'])} | {r['confidence']:.2f} |"
        )

    if failed:
        lines.extend(["", "## Failed Servers", ""])
        for r in failed:
            lines.append(f"- **{r['name']}** ({r['url']}): {r['error']}")

    if skipped:
        lines.extend(["", "## Skipped Servers", ""])
        for r in skipped:
            lines.append(f"- **{r['name']}** ({r['url']}): {r.get('error', 'skipped')}")

    # Per-tool details for successful servers
    if successful:
        lines.extend(["", "## Tool-Level Details", ""])
        for r in successful:
            lines.append(f"### {r['name']} ({r['score']}/100)")
            lines.append("")
            if r.get("tool_scores"):
                lines.append("| Tool | Score | Passed |")
                lines.append("|------|-------|--------|")
                for tool_name, ts in r["tool_scores"].items():
                    lines.append(f"| {tool_name} | {ts['score']}/100 | {ts['tests_passed']}/{ts['tests_total']} |")
                lines.append("")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by Quality Oracle scan at {datetime.now(timezone.utc).isoformat()}*")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Markdown report: {md_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main(
    sse_only: bool = False,
    limit: Optional[int] = None,
    force_provider: Optional[str] = None,
    use_streaming: bool = True,
    single_url: Optional[str] = None,
    runs: int = 1,
):
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    if single_url:
        servers = [{"name": single_url, "url": single_url, "transport": "auto"}]
    else:
        servers = SERVERS.copy()
        if sse_only:
            servers = [s for s in servers if s["transport"] == "sse"]
        if limit:
            servers = servers[:limit]

    mode = "streaming" if use_streaming else "batch"
    runs_str = f", {runs} runs/server" if runs > 1 else ""
    print(f"\n{'='*70}")
    print(f"  Quality Oracle — MCP Server Scan ({mode}{runs_str})")
    print(f"  Servers: {len(servers)} ({sum(1 for s in servers if s['transport'] == 'sse')} SSE, "
          f"{sum(1 for s in servers if s['transport'] == 'streamable_http')} Streamable HTTP)")
    print(f"{'='*70}\n")

    judge = _get_judge(force_provider=force_provider)
    provider = "LLM" if judge.is_llm_available else "fuzzy fallback"
    print(f"  Judge: {provider}\n")

    # Sequential scanning (semaphore=1) to respect LLM rate limits
    # The LLM judge now has its own per-provider rate limiter
    semaphore = asyncio.Semaphore(1)
    rate_limiter = RateLimiter(min_gap=1.0)

    # Run scans concurrently (bounded by semaphore)
    tasks = [
        scan_one_server(server, judge, semaphore, rate_limiter, use_streaming=use_streaming, runs=runs)
        for server in servers
    ]
    results = await asyncio.gather(*tasks)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SCAN RESULTS")
    print(f"{'='*70}\n")

    successful = [r for r in results if r["status"] == "success"]
    successful.sort(key=lambda r: r["score"] or 0, reverse=True)

    if successful:
        print(f"  {'Server':<25s} {'Score':>6s} {'Tier':<12s} {'Tools':>5s} {'Transport':<18s}")
        print(f"  {'─'*70}")
        for r in successful:
            bar = "█" * ((r["score"] or 0) // 5) + "░" * (20 - (r["score"] or 0) // 5)
            print(f"  {r['name']:<25s} {r['score']:>3d}/100 {r['tier']:<12s} {r['tools_count']:>5d} {r.get('transport_used', r['transport']):<18s}")

    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\n  Failed ({len(errors)}):")
        for r in errors:
            print(f"    ✗ {r['name']}: {r['error'][:80]}")

    skipped = [r for r in results if r["status"] == "skipped"]
    if skipped:
        print(f"\n  Skipped ({len(skipped)}):")
        for r in skipped:
            print(f"    - {r['name']}")

    # Stats
    scores = [r["score"] for r in successful if r["score"] is not None]
    if scores:
        print(f"\n  Average score: {sum(scores)/len(scores):.0f}/100")
        print(f"  Highest: {max(scores)}/100 | Lowest: {min(scores)}/100")

    print(f"\n  Total: {len(results)} servers | "
          f"{len(successful)} success | {len(errors)} failed | {len(skipped)} skipped")

    # Save reports
    reports_dir = Path("reports")
    save_reports(results, reports_dir)

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan public MCP servers")
    parser.add_argument("--sse-only", action="store_true", help="Only scan SSE servers")
    parser.add_argument("--limit", type=int, help="Limit number of servers to scan")
    parser.add_argument(
        "--provider",
        choices=PROVIDER_PRIORITY,
        help="Force a specific LLM provider (e.g., cerebras, groq, gemini)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming evaluation (use batch mode)",
    )
    parser.add_argument("--url", type=str, help="Scan a single server URL")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of evaluation runs per server (median aggregation, default 1)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        sse_only=args.sse_only,
        limit=args.limit,
        force_provider=args.provider,
        use_streaming=not args.no_stream,
        single_url=args.url,
        runs=args.runs,
    ))

#!/usr/bin/env python3
"""
MCP Quality Report Generator (QO-024).

Reads scan results from MongoDB (populated by batch_score.py) and generates
a publishable "MCP Server Quality Report" with key findings, statistics,
tier distribution, safety analysis, and per-server breakdowns.

Usage:
    python -m dev.generate_report                          # From MongoDB
    python -m dev.generate_report --from-file reports/batch-2026-04-03.json  # From file
    python -m dev.generate_report --format json            # JSON output
    python -m dev.generate_report --min-servers 10         # Min threshold
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

REPORT_VERSION = "1.0"


# ── Data Loading ────────────────────────────────────────────────────────────


async def load_from_mongodb() -> list[dict]:
    """Load all scored servers from MongoDB."""
    from src.storage.mongodb import connect_db, close_db, scores_col

    await connect_db()
    cursor = scores_col().find(
        {"current_score": {"$exists": True, "$gt": 0}},
    ).sort("current_score", -1)
    results = await cursor.to_list(500)
    await close_db()
    return results


def load_from_file(path: str) -> list[dict]:
    """Load results from a batch_score JSON report file."""
    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    # Normalize to score doc format
    normalized = []
    for r in results:
        if r.get("status") != "success" or not r.get("score"):
            continue
        normalized.append({
            "target_id": r["url"],
            "current_score": r["score"],
            "tier": r["tier"],
            "confidence": r.get("confidence", 0),
            "tools_count": r.get("tools_count", 0),
            "dimensions": r.get("dimensions", {}),
            "tool_scores": r.get("tool_scores", {}),
            "safety_report": r.get("safety_report"),
            "detected_domain": r.get("category", "general"),
            "transport": r.get("transport", "streamable_http"),
            "name": r.get("name", r["url"]),
            "source": "batch_file",
        })
    return normalized


# ── Statistics Computation ──────────────────────────────────────────────────


def compute_stats(servers: list[dict]) -> dict:
    """Compute aggregate statistics from scored servers."""
    scores = [s["current_score"] for s in servers]
    n = len(scores)
    if n == 0:
        return {"error": "No scored servers"}

    avg = sum(scores) / n
    sorted_scores = sorted(scores)
    median = sorted_scores[n // 2] if n % 2 else (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
    variance = sum((s - avg) ** 2 for s in scores) / n
    std_dev = variance ** 0.5

    # Tier distribution
    tiers = {"expert": 0, "proficient": 0, "basic": 0, "failed": 0}
    for s in servers:
        tier = s.get("tier", "failed")
        tiers[tier] = tiers.get(tier, 0) + 1

    tier_pcts = {k: round(v / n * 100, 1) for k, v in tiers.items()}

    # Score distribution buckets
    buckets = {"90-100": 0, "70-89": 0, "50-69": 0, "30-49": 0, "0-29": 0}
    for score in scores:
        if score >= 90:
            buckets["90-100"] += 1
        elif score >= 70:
            buckets["70-89"] += 1
        elif score >= 50:
            buckets["50-69"] += 1
        elif score >= 30:
            buckets["30-49"] += 1
        else:
            buckets["0-29"] += 1

    # Category breakdown
    categories = {}
    for s in servers:
        cat = s.get("detected_domain", s.get("category", "general"))
        if cat not in categories:
            categories[cat] = {"count": 0, "total_score": 0, "scores": []}
        categories[cat]["count"] += 1
        categories[cat]["total_score"] += s["current_score"]
        categories[cat]["scores"].append(s["current_score"])

    for cat_data in categories.values():
        cat_data["avg_score"] = round(cat_data["total_score"] / cat_data["count"], 1)
        del cat_data["scores"]  # Clean up
        del cat_data["total_score"]

    # Transport breakdown
    transports = {}
    for s in servers:
        t = s.get("transport", "unknown")
        transports[t] = transports.get(t, 0) + 1

    # Safety findings
    safety_stats = _compute_safety_stats(servers)

    # Dimension averages
    dim_avgs = _compute_dimension_averages(servers)

    return {
        "total_servers": n,
        "avg_score": round(avg, 1),
        "median_score": round(median, 1),
        "std_dev": round(std_dev, 1),
        "min_score": min(scores),
        "max_score": max(scores),
        "pass_rate": round(tiers.get("expert", 0) + tiers.get("proficient", 0)) / n * 100,
        "tier_distribution": tiers,
        "tier_percentages": tier_pcts,
        "score_distribution": buckets,
        "category_breakdown": dict(sorted(categories.items(), key=lambda x: x[1]["avg_score"], reverse=True)),
        "transport_breakdown": transports,
        "safety": safety_stats,
        "dimension_averages": dim_avgs,
    }


def _compute_safety_stats(servers: list[dict]) -> dict:
    """Extract safety findings from evaluation data."""
    servers_with_safety = [s for s in servers if s.get("safety_report")]
    if not servers_with_safety:
        return {"tested": 0, "message": "No adversarial probe data available"}

    total_tested = len(servers_with_safety)
    total_probes = 0
    total_passed = 0
    probe_types = {}

    for s in servers_with_safety:
        report = s["safety_report"]
        if isinstance(report, list):
            for probe in report:
                probe_type = probe.get("probe_type", probe.get("type", "unknown"))
                passed = probe.get("passed", probe.get("score", 0) >= 50)
                total_probes += 1
                if passed:
                    total_passed += 1
                if probe_type not in probe_types:
                    probe_types[probe_type] = {"total": 0, "passed": 0}
                probe_types[probe_type]["total"] += 1
                if passed:
                    probe_types[probe_type]["passed"] += 1

    fail_rate = round((1 - total_passed / total_probes) * 100, 1) if total_probes else 0

    return {
        "tested": total_tested,
        "total_probes": total_probes,
        "probes_passed": total_passed,
        "probes_failed": total_probes - total_passed,
        "fail_rate_pct": fail_rate,
        "probe_results": {
            k: {
                "pass_rate": round(v["passed"] / v["total"] * 100, 1) if v["total"] else 0,
                **v,
            }
            for k, v in probe_types.items()
        },
    }


def _compute_dimension_averages(servers: list[dict]) -> dict:
    """Compute average scores per dimension across all servers."""
    dim_totals = {}
    dim_counts = {}

    for s in servers:
        dims = s.get("dimensions", {})
        for dim, score in dims.items():
            if isinstance(score, (int, float)):
                dim_totals[dim] = dim_totals.get(dim, 0) + score
                dim_counts[dim] = dim_counts.get(dim, 0) + 1

    return {
        dim: round(dim_totals[dim] / dim_counts[dim], 1)
        for dim in sorted(dim_totals.keys())
        if dim_counts.get(dim, 0) > 0
    }


# ── Key Findings Generator ─────────────────────────────────────────────────


def generate_key_findings(stats: dict, servers: list[dict]) -> list[str]:
    """Auto-generate key findings from statistics."""
    findings = []
    fail_pct = stats["tier_percentages"].get("failed", 0)
    basic_pct = stats["tier_percentages"].get("basic", 0)

    # Overall quality finding
    below_70 = fail_pct + basic_pct
    if below_70 > 50:
        findings.append(
            f"{below_70:.0f}% of MCP servers score below 70/100 (Proficient threshold), "
            f"indicating widespread quality gaps in the ecosystem."
        )

    # Safety finding
    safety = stats.get("safety", {})
    if safety.get("fail_rate_pct", 0) > 20:
        findings.append(
            f"{safety['fail_rate_pct']}% of adversarial security probes failed across tested servers, "
            f"exposing vulnerabilities to prompt injection, data leakage, and system prompt extraction."
        )

    # Top performer
    if servers:
        top = servers[0]
        findings.append(
            f"Highest-scoring server: {top.get('name', top['target_id'])} "
            f"at {top['current_score']}/100 ({top['tier']} tier)."
        )

    # Dimension weakness
    dim_avgs = stats.get("dimension_averages", {})
    if dim_avgs:
        weakest_dim = min(dim_avgs.items(), key=lambda x: x[1])
        strongest_dim = max(dim_avgs.items(), key=lambda x: x[1])
        findings.append(
            f"Weakest dimension across all servers: {weakest_dim[0]} "
            f"(avg {weakest_dim[1]}/100). Strongest: {strongest_dim[0]} "
            f"(avg {strongest_dim[1]}/100)."
        )

    # Category insight
    cats = stats.get("category_breakdown", {})
    if len(cats) >= 2:
        best_cat = max(cats.items(), key=lambda x: x[1]["avg_score"])
        worst_cat = min(cats.items(), key=lambda x: x[1]["avg_score"])
        if best_cat[0] != worst_cat[0]:
            findings.append(
                f"Best-performing category: {best_cat[0]} (avg {best_cat[1]['avg_score']}/100). "
                f"Worst: {worst_cat[0]} (avg {worst_cat[1]['avg_score']}/100)."
            )

    # Score variance
    if stats["std_dev"] > 15:
        findings.append(
            f"High variance in quality scores (std dev: {stats['std_dev']}), "
            f"indicating inconsistent quality standards across the MCP ecosystem."
        )

    return findings


# ── Report Formatters ───────────────────────────────────────────────────────


def format_json_report(stats: dict, servers: list[dict], findings: list[str]) -> dict:
    """Generate full JSON report."""
    return {
        "report_version": REPORT_VERSION,
        "title": "MCP Server Quality Report",
        "subtitle": f"Quality assessment of {stats['total_servers']} public MCP servers",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "Laureum.ai (https://laureum.ai)",
        "methodology": {
            "evaluation_type": "Active challenge-response with multi-judge consensus",
            "scoring_dimensions": ["accuracy", "safety", "reliability", "process_quality", "latency", "schema_quality"],
            "dimension_weights": {"accuracy": "35%", "safety": "20%", "reliability": "15%", "process_quality": "10%", "latency": "10%", "schema_quality": "10%"},
            "security_probes": "OWASP-aligned adversarial testing (prompt injection, data leakage, system prompt extraction, hallucination, overflow)",
            "judge_model": "Multi-provider LLM consensus (2-3 judges with agreement threshold)",
        },
        "key_findings": findings,
        "statistics": stats,
        "servers": [
            {
                "rank": i + 1,
                "name": s.get("name", s["target_id"]),
                "url": s["target_id"],
                "score": s["current_score"],
                "tier": s["tier"],
                "confidence": s.get("confidence", 0),
                "tools_count": s.get("tools_count", 0),
                "category": s.get("detected_domain", "general"),
                "transport": s.get("transport", "unknown"),
                "dimensions": s.get("dimensions", {}),
            }
            for i, s in enumerate(servers)
        ],
    }


def format_markdown_report(stats: dict, servers: list[dict], findings: list[str]) -> str:
    """Generate publishable Markdown report."""
    lines = []
    n = stats["total_servers"]
    date = datetime.now(timezone.utc).strftime("%B %Y")

    # Header
    lines.extend([
        f"# MCP Server Quality Report — {date}",
        "",
        f"> Quality assessment of **{n} public MCP servers** using active challenge-response evaluation, "
        f"multi-judge consensus scoring, and OWASP-aligned adversarial security testing.",
        "",
        "*Published by [Laureum.ai](https://laureum.ai) — the quality verification layer for AI agents.*",
        "",
    ])

    # Key Stats Banner
    lines.extend([
        "---",
        "",
        "## At a Glance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Servers Evaluated | **{n}** |",
        f"| Average Score | **{stats['avg_score']}/100** |",
        f"| Median Score | **{stats['median_score']}/100** |",
        f"| Pass Rate (≥70) | **{stats['pass_rate']:.0f}%** |",
        f"| Expert Tier (≥85) | **{stats['tier_distribution'].get('expert', 0)}** ({stats['tier_percentages'].get('expert', 0)}%) |",
        f"| Failed (<50) | **{stats['tier_distribution'].get('failed', 0)}** ({stats['tier_percentages'].get('failed', 0)}%) |",
        "",
    ])

    # Key Findings
    if findings:
        lines.extend([
            "## Key Findings",
            "",
        ])
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. {finding}")
        lines.append("")

    # Tier Distribution
    lines.extend([
        "## Tier Distribution",
        "",
        "| Tier | Score Range | Count | Percentage |",
        "|------|-----------|-------|------------|",
        f"| Expert | 85-100 | {stats['tier_distribution'].get('expert', 0)} | {stats['tier_percentages'].get('expert', 0)}% |",
        f"| Proficient | 70-84 | {stats['tier_distribution'].get('proficient', 0)} | {stats['tier_percentages'].get('proficient', 0)}% |",
        f"| Basic | 50-69 | {stats['tier_distribution'].get('basic', 0)} | {stats['tier_percentages'].get('basic', 0)}% |",
        f"| Failed | 0-49 | {stats['tier_distribution'].get('failed', 0)} | {stats['tier_percentages'].get('failed', 0)}% |",
        "",
    ])

    # Dimension Averages
    dim_avgs = stats.get("dimension_averages", {})
    if dim_avgs:
        lines.extend([
            "## Quality Dimensions (Ecosystem Averages)",
            "",
            "| Dimension | Weight | Avg Score | Status |",
            "|-----------|--------|-----------|--------|",
        ])
        weights = {"accuracy": "35%", "safety": "20%", "reliability": "15%", "process_quality": "10%", "latency": "10%", "schema_quality": "10%"}
        for dim, avg_score in sorted(dim_avgs.items(), key=lambda x: x[1]):
            status = "Pass" if avg_score >= 70 else ("Concern" if avg_score >= 50 else "Fail")
            weight = weights.get(dim, "-")
            lines.append(f"| {dim.replace('_', ' ').title()} | {weight} | {avg_score}/100 | {status} |")
        lines.append("")

    # Safety Analysis
    safety = stats.get("safety", {})
    if safety.get("tested", 0) > 0:
        lines.extend([
            "## Security Analysis",
            "",
            f"**{safety['tested']} servers** underwent OWASP-aligned adversarial testing with **{safety['total_probes']} probes**.",
            "",
            f"- Probes passed: {safety['probes_passed']}/{safety['total_probes']} ({100 - safety['fail_rate_pct']:.0f}%)",
            f"- Probes failed: {safety['probes_failed']}/{safety['total_probes']} ({safety['fail_rate_pct']:.0f}%)",
            "",
        ])

        probe_results = safety.get("probe_results", {})
        if probe_results:
            lines.extend([
                "| Probe Type | Pass Rate | Tested |",
                "|-----------|-----------|--------|",
            ])
            for probe_type, data in sorted(probe_results.items(), key=lambda x: x[1]["pass_rate"]):
                lines.append(f"| {probe_type.replace('_', ' ').title()} | {data['pass_rate']}% | {data['total']} |")
            lines.append("")

    # Server Rankings
    lines.extend([
        "## Server Rankings",
        "",
        "| Rank | Server | Score | Tier | Tools | Category |",
        "|------|--------|-------|------|-------|----------|",
    ])
    for i, s in enumerate(servers, 1):
        name = s.get("name", s["target_id"])
        lines.append(
            f"| {i} | [{name}]({s['target_id']}) | **{s['current_score']}**/100 | "
            f"{s['tier']} | {s.get('tools_count', '?')} | "
            f"{s.get('detected_domain', 'general')} |"
        )
    lines.append("")

    # Category Breakdown
    cats = stats.get("category_breakdown", {})
    if cats:
        lines.extend([
            "## Category Breakdown",
            "",
            "| Category | Servers | Avg Score |",
            "|----------|---------|-----------|",
        ])
        for cat, data in cats.items():
            lines.append(f"| {cat.replace('_', ' ').title()} | {data['count']} | {data['avg_score']}/100 |")
        lines.append("")

    # Methodology
    lines.extend([
        "## Methodology",
        "",
        "Each server was evaluated using Laureum's 3-level quality pipeline:",
        "",
        "1. **Manifest Validation (L1):** Schema completeness, tool descriptions, input validation",
        "2. **Functional Testing (L2):** Live tool execution with auto-generated test cases, scored by multi-judge LLM consensus",
        "3. **Adversarial Probes:** OWASP-aligned security testing — prompt injection, system prompt extraction, PII leakage, hallucination, token overflow",
        "",
        "Scores are computed across 6 weighted dimensions: accuracy (35%), safety (20%), reliability (15%), process quality (10%), latency (10%), schema quality (10%).",
        "",
        "All evaluations use **multi-judge consensus** (2-3 independent LLM judges with agreement threshold) to prevent single-model bias.",
        "",
    ])

    # Footer
    lines.extend([
        "---",
        "",
        f"*Report generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d')} by [Laureum.ai](https://laureum.ai) — "
        f"the quality verification layer for AI agents. "
        f"[Evaluate your server](https://laureum.ai/evaluate) | "
        f"[View Leaderboard](https://laureum.ai/leaderboard)*",
    ])

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────


async def main(
    from_file: Optional[str] = None,
    output_format: str = "all",
    min_servers: int = 5,
    output_dir: str = "reports",
):
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # Load data
    if from_file:
        print(f"Loading results from {from_file}...")
        servers = load_from_file(from_file)
    else:
        print("Loading scores from MongoDB...")
        servers = await load_from_mongodb()

    # Filter to valid scores and sort
    servers = [s for s in servers if s.get("current_score", 0) > 0]
    servers.sort(key=lambda s: s["current_score"], reverse=True)

    if len(servers) < min_servers:
        print(f"ERROR: Only {len(servers)} scored servers found (minimum: {min_servers})")
        print("Run batch_score.py first to evaluate more servers.")
        return

    print(f"Found {len(servers)} scored servers")

    # Compute statistics
    stats = compute_stats(servers)
    findings = generate_key_findings(stats, servers)

    print(f"\n{'='*60}")
    print(f"  MCP QUALITY REPORT — {stats['total_servers']} servers")
    print(f"{'='*60}")
    print(f"  Avg score: {stats['avg_score']}/100 | Median: {stats['median_score']}/100")
    print(f"  Pass rate: {stats['pass_rate']:.0f}%")
    print(f"  Tiers: {stats['tier_distribution']}")
    print("\n  Key findings:")
    for i, f in enumerate(findings, 1):
        print(f"    {i}. {f}")

    # Generate reports
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if output_format in ("json", "all"):
        json_report = format_json_report(stats, servers, findings)
        json_path = out_dir / f"quality-report-{date_str}.json"
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2, default=str)
        print(f"\n  JSON report: {json_path}")

    if output_format in ("markdown", "all"):
        md_report = format_markdown_report(stats, servers, findings)
        md_path = out_dir / f"quality-report-{date_str}.md"
        with open(md_path, "w") as f:
            f.write(md_report)
        print(f"  Markdown report: {md_path}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Quality Report Generator (QO-024)")
    parser.add_argument(
        "--from-file", type=str, default=None,
        help="Load results from batch_score JSON file instead of MongoDB",
    )
    parser.add_argument(
        "--format", type=str, default="all",
        choices=["json", "markdown", "all"],
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--min-servers", type=int, default=5,
        help="Minimum servers required to generate report (default: 5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports",
        help="Output directory (default: reports/)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        from_file=args.from_file,
        output_format=args.format,
        min_servers=args.min_servers,
        output_dir=args.output_dir,
    ))

#!/usr/bin/env python3
"""
Judge Quality Baseline — measures LLM judge accuracy and consistency.

Runs gold-standard test cases through the judge pipeline and reports:
1. Score accuracy  — judge score within expected range
2. Consistency     — same input 3 times → scores within ±5 points
3. Position bias   — score difference when question/answer swapped in prompt
4. Ordering        — judge agrees with human ranking (higher expected = higher scored)

Usage:
    cd quality-oracle
    source .venv/bin/activate
    python dev/judge_baseline.py [--runs 3] [--verbose]
"""
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_judge import LLMJudge  # noqa: E402

logger = logging.getLogger(__name__)


def load_gold_standard() -> list:
    """Load gold standard test cases."""
    path = Path(__file__).parent / "judge_gold_standard.json"
    with open(path) as f:
        return json.load(f)


def create_judge() -> LLMJudge:
    """Create judge with env-based config (same as production)."""
    return LLMJudge(
        api_key=os.getenv("CEREBRAS_API_KEY", ""),
        model=os.getenv("CEREBRAS_MODEL", "llama3.1-8b"),
        provider="cerebras",
        base_url="https://api.cerebras.ai/v1",
        fallback_key=os.getenv("GROQ_API_KEY", ""),
        fallback_model="llama-3.1-8b-instant",
        fallback_provider="groq",
        fallback2_key=os.getenv("OPENROUTER_API_KEY", ""),
        fallback2_model="qwen/qwen3-next-80b-a3b-instruct:free",
        fallback2_provider="openrouter",
    )


async def measure_accuracy(judge: LLMJudge, cases: list, verbose: bool = False) -> dict:
    """Measure: does judge score fall within expected range?"""
    results = []
    for case in cases:
        result = await judge.ajudge(case["question"], case["expected"], case["answer"])
        in_range = case["expected_score_min"] <= result.score <= case["expected_score_max"]
        results.append({
            "id": case["id"],
            "category": case["category"],
            "judge_score": result.score,
            "expected_min": case["expected_score_min"],
            "expected_max": case["expected_score_max"],
            "in_range": in_range,
            "method": result.method,
            "explanation": result.explanation,
        })
        if verbose:
            status = "OK" if in_range else "MISS"
            print(f"  [{status}] {case['id']} ({case['category']}): "
                  f"score={result.score} expected=[{case['expected_score_min']}-{case['expected_score_max']}] "
                  f"method={result.method}")

    accuracy = sum(1 for r in results if r["in_range"]) / len(results) * 100
    return {
        "accuracy_pct": round(accuracy, 1),
        "total": len(results),
        "in_range": sum(1 for r in results if r["in_range"]),
        "misses": [r for r in results if not r["in_range"]],
        "details": results,
    }


async def measure_consistency(judge: LLMJudge, cases: list, runs: int = 3, verbose: bool = False) -> dict:
    """Measure: same input N times → scores within ±5 points?"""
    # Use subset of cases (5 diverse ones) to save API calls
    subset = [c for c in cases if c["id"] in ("gs-001", "gs-003", "gs-006", "gs-010", "gs-015")]
    if not subset:
        subset = cases[:5]

    results = []
    for case in subset:
        scores = []
        for _ in range(runs):
            result = await judge.ajudge(case["question"], case["expected"], case["answer"])
            scores.append(result.score)

        spread = max(scores) - min(scores)
        consistent = spread <= 10  # Allow ±5 → max 10 spread
        results.append({
            "id": case["id"],
            "scores": scores,
            "mean": round(statistics.mean(scores), 1),
            "stdev": round(statistics.stdev(scores), 1) if len(scores) > 1 else 0,
            "spread": spread,
            "consistent": consistent,
        })
        if verbose:
            status = "OK" if consistent else "INCONSISTENT"
            print(f"  [{status}] {case['id']}: scores={scores} spread={spread}")

    consistency_pct = sum(1 for r in results if r["consistent"]) / len(results) * 100
    return {
        "consistency_pct": round(consistency_pct, 1),
        "total": len(results),
        "consistent": sum(1 for r in results if r["consistent"]),
        "avg_spread": round(statistics.mean(r["spread"] for r in results), 1),
        "details": results,
    }


async def measure_ordering(judge: LLMJudge, cases: list, verbose: bool = False) -> dict:
    """Measure: does judge rank better answers higher than worse ones?"""
    # Create pairs of same-question cases with different quality
    pairs = [
        ("gs-001", "gs-002"),   # correct vs wrong (simple)
        ("gs-003", "gs-004"),   # correct vs partial (AMM)
        ("gs-006", "gs-007"),   # full code vs partial code
        ("gs-010", "gs-011"),   # comprehensive vs minimal (security)
        ("gs-012", "gs-013"),   # correct vs wrong (MCP)
        ("gs-015", "gs-016"),   # correct vs wrong (conversion)
    ]

    case_map = {c["id"]: c for c in cases}
    results = []
    for better_id, worse_id in pairs:
        if better_id not in case_map or worse_id not in case_map:
            continue
        better = case_map[better_id]
        worse = case_map[worse_id]

        score_better = (await judge.ajudge(better["question"], better["expected"], better["answer"])).score
        score_worse = (await judge.ajudge(worse["question"], worse["expected"], worse["answer"])).score

        correct_order = score_better > score_worse
        results.append({
            "better_id": better_id,
            "worse_id": worse_id,
            "score_better": score_better,
            "score_worse": score_worse,
            "correct_order": correct_order,
        })
        if verbose:
            status = "OK" if correct_order else "WRONG"
            print(f"  [{status}] {better_id} ({score_better}) vs {worse_id} ({score_worse})")

    ordering_pct = sum(1 for r in results if r["correct_order"]) / len(results) * 100 if results else 0
    return {
        "ordering_pct": round(ordering_pct, 1),
        "total": len(results),
        "correct": sum(1 for r in results if r["correct_order"]),
        "details": results,
    }


async def measure_position_bias(judge: LLMJudge, cases: list, verbose: bool = False) -> dict:
    """Measure: does score change significantly when answer is presented differently?

    Tests if the judge gives different scores to the same answer when
    the question/expected wording is slightly different (position in prompt).
    """
    subset = [c for c in cases if c["id"] in ("gs-003", "gs-010", "gs-014", "gs-018")]
    if not subset:
        subset = cases[:4]

    results = []
    for case in subset:
        # Normal order
        score_normal = (await judge.ajudge(
            case["question"], case["expected"], case["answer"],
        )).score

        # Reversed: swap expected and question in prompt
        score_reversed = (await judge.ajudge(
            case["expected"], case["question"], case["answer"],
        )).score

        bias = abs(score_normal - score_reversed)
        low_bias = bias <= 15
        results.append({
            "id": case["id"],
            "score_normal": score_normal,
            "score_reversed": score_reversed,
            "bias": bias,
            "low_bias": low_bias,
        })
        if verbose:
            status = "OK" if low_bias else "BIASED"
            print(f"  [{status}] {case['id']}: normal={score_normal} reversed={score_reversed} bias={bias}")

    avg_bias = statistics.mean(r["bias"] for r in results) if results else 0
    return {
        "avg_bias": round(avg_bias, 1),
        "total": len(results),
        "low_bias_count": sum(1 for r in results if r["low_bias"]),
        "details": results,
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Judge Quality Baseline")
    parser.add_argument("--runs", type=int, default=3, help="Consistency test repetitions")
    parser.add_argument("--verbose", action="store_true", help="Print per-case details")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    cases = load_gold_standard()
    judge = create_judge()

    print(f"\n{'='*60}")
    print(f"  AgentTrust Judge Quality Baseline")
    print(f"  {len(cases)} gold-standard test cases")
    print(f"  LLM available: {judge.is_llm_available}")
    print(f"{'='*60}\n")

    start = time.time()

    # 1. Accuracy
    print("1. Score Accuracy (judge score within expected range)")
    print("-" * 50)
    accuracy = await measure_accuracy(judge, cases, verbose=args.verbose)
    print(f"   Result: {accuracy['in_range']}/{accuracy['total']} in range ({accuracy['accuracy_pct']}%)\n")

    # 2. Consistency
    print(f"2. Score Consistency ({args.runs} runs, spread ≤ 10)")
    print("-" * 50)
    consistency = await measure_consistency(judge, cases, runs=args.runs, verbose=args.verbose)
    print(f"   Result: {consistency['consistent']}/{consistency['total']} consistent "
          f"({consistency['consistency_pct']}%), avg spread={consistency['avg_spread']}\n")

    # 3. Ordering
    print("3. Ordering Agreement (better answer scores higher)")
    print("-" * 50)
    ordering = await measure_ordering(judge, cases, verbose=args.verbose)
    print(f"   Result: {ordering['correct']}/{ordering['total']} correct ({ordering['ordering_pct']}%)\n")

    # 4. Position Bias
    print("4. Position Bias (score change when prompt rearranged)")
    print("-" * 50)
    bias = await measure_position_bias(judge, cases, verbose=args.verbose)
    print(f"   Result: avg bias={bias['avg_bias']}pts, "
          f"{bias['low_bias_count']}/{bias['total']} low bias (≤15pts)\n")

    elapsed = time.time() - start

    # Summary
    print(f"{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy:    {accuracy['accuracy_pct']}%  ({accuracy['in_range']}/{accuracy['total']})")
    print(f"  Consistency: {consistency['consistency_pct']}%  (avg spread {consistency['avg_spread']})")
    print(f"  Ordering:    {ordering['ordering_pct']}%  ({ordering['correct']}/{ordering['total']})")
    print(f"  Position:    avg bias {bias['avg_bias']}pts")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"  Method:      {'LLM' if judge.is_llm_available else 'fuzzy fallback'}")
    print(f"{'='*60}\n")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "llm_available": judge.is_llm_available,
        "accuracy": accuracy,
        "consistency": consistency,
        "ordering": ordering,
        "position_bias": bias,
        "elapsed_seconds": round(elapsed, 1),
    }

    output_path = Path(__file__).parent / "judge_baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Return exit code based on quality
    if accuracy["accuracy_pct"] < 60 or ordering["ordering_pct"] < 60:
        print("\nWARNING: Judge quality below threshold!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

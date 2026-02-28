#!/usr/bin/env python3
"""
LLM-powered question generation for Quality Oracle question banks.

Generates domain-specific challenge questions using OpenAI-compatible APIs.
Provider priority: Cerebras → Groq → OpenRouter → OpenAI.
Same httpx + retry pattern as src/core/llm_judge.py:455-519.

Usage:
    python dev/question-generation/generate_questions.py --mode generate --domain defi
    python dev/question-generation/generate_questions.py --mode generate  # all domains
    python dev/question-generation/generate_questions.py --mode stats     # show counts
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Provider Config ──────────────────────────────────────────────────────────

PROVIDERS = [
    {
        "name": "cerebras",
        "env_key": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "model": "gpt-oss-120b",
        "max_tokens": 2048,
    },
    {
        "name": "groq",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 1500,
    },
    {
        "name": "openrouter",
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3-next-80b-a3b-instruct:free",
        "max_tokens": 1500,
    },
    {
        "name": "openai",
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "max_tokens": 1500,
    },
]

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
TAXONOMIES_PATH = SCRIPT_DIR / "taxonomies.json"
OUTPUT_DIR = SCRIPT_DIR.parent / "generated-questions"


def _load_taxonomies() -> dict:
    with open(TAXONOMIES_PATH) as f:
        return json.load(f)


def _load_existing(domain: str) -> list:
    path = OUTPUT_DIR / f"{domain}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_questions(domain: str, questions: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{domain}.json"
    with open(path, "w") as f:
        json.dump(questions, f, indent=2)
    logger.info(f"Saved {len(questions)} questions to {path}")


def _question_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def _get_provider() -> Optional[dict]:
    """Find first available provider with API key set."""
    for p in PROVIDERS:
        if os.environ.get(p["env_key"]):
            return p
    return None


# ── LLM Call (same pattern as llm_judge._call_llm) ──────────────────────────

def _build_prompt(domain: str, subtopic: str, difficulty: str, existing_questions: list) -> str:
    existing_sample = ""
    if existing_questions:
        sample = existing_questions[:3]
        existing_sample = "\n\nExisting questions (avoid duplicates):\n" + "\n".join(
            f"- {q['question']}" for q in sample
        )

    return f"""Generate 3 high-quality challenge questions for evaluating AI agent competency.

Domain: {domain}
Subtopic: {subtopic}
Difficulty: {difficulty}
{existing_sample}

Requirements:
- Questions must test real understanding, not trivia
- Include a reference answer (concise, 1-3 sentences)
- Each question should be self-contained and unambiguous
- Difficulty guide: easy=definitional, medium=applied knowledge, hard=synthesis/analysis

Respond with ONLY a JSON array (no markdown, no explanation):
[
  {{
    "question": "...",
    "reference_answer": "...",
    "difficulty": "{difficulty}",
    "category": "functional"
  }}
]"""


async def _call_llm(
    prompt: str, provider: dict, api_key: str,
) -> Optional[list]:
    """Call LLM API with retry logic (mirrors llm_judge._call_llm pattern)."""
    url = f"{provider['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": "You are a question bank generator. Always respond with valid JSON arrays only. No markdown fences."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "max_tokens": provider["max_tokens"],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = None
            for attempt in range(MAX_RETRIES):
                response = await client.post(url, headers=headers, json=body)
                if response.status_code == 429:
                    try:
                        err_body = response.json()
                        err_code = err_body.get("error", {}).get("code", "")
                        if err_code == "insufficient_quota":
                            logger.warning(f"{provider['name']} quota exhausted, skipping")
                            return None
                    except Exception:
                        pass
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited (429), retry {attempt + 1}/{MAX_RETRIES}")
                    await asyncio.sleep(delay)
                    continue
                break
            else:
                if response is not None and response.status_code == 429:
                    logger.warning(f"{provider['name']} 429 after {MAX_RETRIES} retries")
                    return None

            if response is None or response.status_code != 200:
                status = response.status_code if response else "no response"
                logger.warning(f"{provider['name']} returned {status}")
                return None

            data = response.json()
            message = data["choices"][0]["message"]
            text = message.get("content") or message.get("reasoning") or ""

            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            return json.loads(text)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM call error ({provider['name']}): {e}")
        return None


# ── Generation Logic ─────────────────────────────────────────────────────────

async def generate_for_domain(domain: str, domain_config: dict, provider: dict, api_key: str):
    """Generate questions for a single domain."""
    existing = _load_existing(domain)
    existing_hashes = {_question_hash(q["question"]) for q in existing}
    subtopics = domain_config["subtopics"]
    target = domain_config["target_count"]

    # Difficulty distribution from taxonomies
    difficulties = ["easy", "medium", "hard"]
    diff_weights = [0.3, 0.45, 0.25]

    new_questions = []
    attempts = 0
    max_attempts = len(subtopics) * 3  # 3 difficulty levels per subtopic

    for subtopic in subtopics:
        for i, difficulty in enumerate(difficulties):
            if len(existing) + len(new_questions) >= target:
                break
            if attempts >= max_attempts:
                break
            attempts += 1

            # Rate limit: 1 request per 2 seconds
            if attempts > 1:
                await asyncio.sleep(2.0)

            prompt = _build_prompt(domain, subtopic, difficulty, existing + new_questions)
            result = await _call_llm(prompt, provider, api_key)

            if not result or not isinstance(result, list):
                logger.warning(f"No valid result for {domain}/{subtopic}/{difficulty}")
                continue

            for q in result:
                if not isinstance(q, dict):
                    continue
                question_text = q.get("question", "").strip()
                if not question_text:
                    continue

                qhash = _question_hash(question_text)
                if qhash in existing_hashes:
                    logger.debug(f"Duplicate skipped: {question_text[:60]}...")
                    continue

                existing_hashes.add(qhash)
                new_questions.append({
                    "question": question_text,
                    "domain": domain,
                    "difficulty": q.get("difficulty", difficulty),
                    "reference_answer": q.get("reference_answer", ""),
                    "category": q.get("category", "functional"),
                    "subtopic": subtopic,
                    "generated_by": provider["name"],
                    "hash": qhash,
                })

            logger.info(
                f"[{domain}] {subtopic}/{difficulty}: "
                f"+{len(result)} generated, {len(new_questions)} new total"
            )

        if len(existing) + len(new_questions) >= target:
            break

    # Merge and save
    all_questions = existing + new_questions
    _save_questions(domain, all_questions)
    logger.info(f"[{domain}] Done: {len(new_questions)} new, {len(all_questions)} total (target: {target})")
    return len(new_questions)


async def generate_all(domains: Optional[list] = None):
    """Generate questions for all (or specified) domains."""
    taxonomies = _load_taxonomies()
    provider = _get_provider()

    if not provider:
        logger.error(
            "No LLM provider API key found. Set one of: "
            + ", ".join(p["env_key"] for p in PROVIDERS)
        )
        sys.exit(1)

    api_key = os.environ[provider["env_key"]]
    logger.info(f"Using provider: {provider['name']} ({provider['model']})")

    target_domains = domains or list(taxonomies["domains"].keys())
    total_new = 0

    for domain in target_domains:
        if domain not in taxonomies["domains"]:
            logger.warning(f"Unknown domain: {domain}, skipping")
            continue

        new_count = await generate_for_domain(
            domain, taxonomies["domains"][domain], provider, api_key
        )
        total_new += new_count

    logger.info(f"Generation complete: {total_new} new questions across {len(target_domains)} domains")


def show_stats():
    """Show question counts per domain."""
    taxonomies = _load_taxonomies()
    total = 0

    print(f"\n{'Domain':<25} {'Generated':>10} {'Target':>8}")
    print("-" * 47)

    for domain, config in taxonomies["domains"].items():
        existing = _load_existing(domain)
        count = len(existing)
        target = config["target_count"]
        total += count
        status = "OK" if count >= target else f"need {target - count}"
        print(f"{domain:<25} {count:>10} {target:>8}  {status}")

    print("-" * 47)
    print(f"{'TOTAL':<25} {total:>10}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate questions for Quality Oracle")
    parser.add_argument("--mode", choices=["generate", "stats"], default="stats")
    parser.add_argument("--domain", type=str, default=None, help="Single domain to generate")
    args = parser.parse_args()

    if args.mode == "stats":
        show_stats()
    elif args.mode == "generate":
        domains = [args.domain] if args.domain else None
        asyncio.run(generate_all(domains))


if __name__ == "__main__":
    main()

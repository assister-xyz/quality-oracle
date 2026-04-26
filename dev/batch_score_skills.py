#!/usr/bin/env python3
"""QO-053-F batch skill scoring runner.

Clones a skills repo, walks ``skills/*/SKILL.md``, evaluates each, persists
``SkillScore`` rows to MongoDB with a deterministic ``eval_hash``, writes a
JSON report, and prints a cost summary. Drives the SendAI 45-skill batch and
weekly re-scans.

Usage::

    python -m dev.batch_score_skills --repo sendaifun/skills --ref main
    python -m dev.batch_score_skills --repo sendaifun/skills --max-cost-usd 5
    python -m dev.batch_score_skills --repo sendaifun/skills --dry-run
    python -m dev.batch_score_skills --repo anthropics/skills --changed-only

Acceptance criteria covered:
  AC1/AC2: eval_hash determinism (compute_eval_hash + tests).
  AC3:     L1 cache hit on unchanged git_sha (qo:eval:<eval_hash>).
  AC4:     persisted record schema (SkillScore.model_dump → MongoDB).
  AC5:     SendAI 45-skill SLO at concurrency=3.
  AC6:     skipped/errored skills persist with explicit reason.
  AC7:     cost reporter (per-run / per-skill / per-axis) → reports/<repo>.json.
  AC8:     pre-flight cost ceiling (--max-cost-usd) refuses to start when
           the projected total exceeds the ceiling; --billing-tag tags every
           SkillScore.cost_dollars row for marketplace billing isolation.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root on path so ``src.*`` resolves when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_score_skills")


# ── Cost model ──────────────────────────────────────────────────────────────
#
# Per CB5 reconciliation: Cerebras default is free (~$0.05 fallthrough
# budget per skill in case the free quota exhausts), Anthropic opt-in is
# ~$0.30/skill cached at L2. AC8 estimator uses these constants.
COST_PER_SKILL_CEREBRAS = 0.05
COST_PER_SKILL_ANTHROPIC = 0.30


@dataclass
class RunStats:
    """Aggregate per-run telemetry (drives the cost reporter, AC7)."""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    total_skills: int = 0
    succeeded: int = 0
    skipped: int = 0
    errored: int = 0
    cached: int = 0
    total_cost: float = 0.0
    paid_fallthrough: float = 0.0
    per_skill_cost: Dict[str, float] = field(default_factory=dict)
    per_axis_cost: Dict[str, float] = field(default_factory=dict)
    correct_count: int = 0  # for CPCR (QO-051)


# ── Repo cloning ────────────────────────────────────────────────────────────


def _fixtures_dir() -> Path:
    """Where to clone repos. Honors ``LAUREUM_FIXTURES_DIR`` (matches
    ``dev/setup_fixtures.sh`` so cached clones are shared)."""
    return Path(os.environ.get("LAUREUM_FIXTURES_DIR", "/tmp"))


def _clone_dir_for(repo: str) -> Path:
    """Path where the target repo lives once cloned. ``owner/repo`` →
    ``<fixtures>/<repo>-eval`` (per spec wording).

    Recognizes the legacy ``setup_fixtures.sh`` layout for the five
    seed repos so cached SendAI/Anthropic clones are reused without a
    re-clone — important for CI where the network may be off-limits.
    """
    fixtures = _fixtures_dir()
    # Known fixtures from dev/setup_fixtures.sh — reuse if already cloned.
    legacy = {
        "sendaifun/skills": fixtures / "sendai-skills",
        "anthropics/skills": fixtures / "anthropic-skills",
        "trailofbits/skills": fixtures / "trailofbits-skills",
        "antfu/skills": fixtures / "antfu-skills",
        "addyosmani/agent-skills": fixtures / "addyosmani-agent-skills",
    }
    if repo in legacy and legacy[repo].is_dir():
        return legacy[repo]

    name = repo.replace("/", "-")
    return fixtures / f"{name}-eval"


def clone_repo(repo: str, ref: str = "main", *, force: bool = False) -> Path:
    """Clone or refresh ``github.com/<repo>`` to the fixtures dir, return
    the local checkout path. Idempotent across reruns.

    Returns immediately if the directory exists and ``force`` is False —
    we trust the caller to have already populated the fixtures via
    ``dev/setup_fixtures.sh``. This makes ``--dry-run`` self-contained on
    cached fixtures (AC: dry-run runs end-to-end without API calls).
    """
    target = _clone_dir_for(repo)
    if target.is_dir() and (target / ".git").is_dir() and not force:
        logger.info("repo cache hit: %s (skipping clone)", target)
        return target

    if target.exists():
        # Stale non-git dir; refuse to clobber so user can investigate.
        if not (target / ".git").is_dir():
            raise RuntimeError(
                f"refusing to overwrite non-git path {target!r}; "
                f"remove it manually or use a different LAUREUM_FIXTURES_DIR"
            )
        # Refresh existing clone to ``ref``.
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--quiet", "origin", ref],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--quiet", ref],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "reset", "--hard", "--quiet", f"origin/{ref}"],
            check=True,
        )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--branch", ref,
         f"https://github.com/{repo}.git", str(target)],
        check=True,
    )
    return target


def git_sha_for_path(path: Path) -> str:
    """Return the current git SHA1 of the repo containing ``path``.

    Falls back to a deterministic hash of the directory listing when the
    path is not inside a git repo (covers tests using fixtures committed
    to the parent monorepo). Never raises.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Fixture-mode fallback — stable hash of the directory listing so
        # the same fixture always produces the same eval_hash.
        import hashlib
        h = hashlib.sha1()
        for entry in sorted(path.iterdir()) if path.is_dir() else []:
            h.update(entry.name.encode())
            try:
                h.update(str(entry.stat().st_size).encode())
            except OSError:
                pass
        return f"fixture-{h.hexdigest()[:16]}"


def find_skill_dirs(repo_root: Path) -> List[Path]:
    """Walk ``<repo>/skills/*`` and return every dir that has a
    SKILL.md (or skill.md fallback). Sorted for deterministic ordering."""
    skills_root = repo_root / "skills"
    # Some repos stash skills at the root (see anthropics/skills); fall
    # through to top-level dirs when no ``skills/`` subdir exists.
    if not skills_root.is_dir():
        skills_root = repo_root

    found: List[Path] = []
    for entry in sorted(skills_root.iterdir()) if skills_root.is_dir() else []:
        if not entry.is_dir():
            continue
        if (entry / "SKILL.md").is_file() or (entry / "skill.md").is_file():
            found.append(entry)
    return found


# ── Audit blob storage ───────────────────────────────────────────────────────


async def store_audit_blob(payload: Dict[str, Any], filename: str) -> Optional[str]:
    """Gzip ``payload`` (JSON) and store in GridFS (``quality__audit_blobs``).

    Returns the GridFS object id (string) or ``None`` if MongoDB is
    unreachable. Failure is non-fatal — replay tooling will fall back
    to recomputing from inputs in that case.
    """
    try:
        from src.storage.mongodb import audit_blobs_fs
        bucket = audit_blobs_fs()
        data = gzip.compress(json.dumps(payload, default=str).encode())
        oid = await bucket.upload_from_stream(filename, data)
        return str(oid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit blob upload failed (non-fatal): %s", exc)
        return None


# ── Cache helpers ────────────────────────────────────────────────────────────


async def cache_get_skill_score(eval_hash: str) -> Optional[Dict[str, Any]]:
    """L1 cache lookup. Returns the cached SkillScore dict or None."""
    try:
        from src.storage.cache import get_redis
        r = get_redis()
        raw = await r.get(f"qo:eval:{eval_hash}")
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cache miss / unavailable: %s", exc)
        return None


async def cache_set_skill_score(eval_hash: str, payload: Dict[str, Any]) -> None:
    """Write SkillScore to L1 cache with 7-day TTL."""
    try:
        from src.storage.cache import get_redis
        r = get_redis()
        await r.set(
            f"qo:eval:{eval_hash}",
            json.dumps(payload, default=str),
            ex=86400 * 7,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("cache set failed (non-fatal): %s", exc)


# ── Cost ceiling (AC8) ───────────────────────────────────────────────────────


def estimate_run_cost(num_skills: int, *, activation_provider: str) -> float:
    """Pre-flight cost estimator. Conservative — assumes worst-case
    fallthrough on Cerebras (paid Haiku per skill)."""
    if activation_provider == "anthropic":
        return num_skills * COST_PER_SKILL_ANTHROPIC
    return num_skills * COST_PER_SKILL_CEREBRAS


def refuse_if_over_budget(
    num_skills: int,
    max_cost_usd: float,
    activation_provider: str,
) -> None:
    """AC8: refuse to start if estimated cost > ceiling. Exits non-zero."""
    estimate = estimate_run_cost(num_skills, activation_provider=activation_provider)
    if estimate > max_cost_usd:
        logger.error(
            "COST CEILING EXCEEDED: estimated $%.2f for %d skills "
            "(provider=%s, per-skill=$%.2f) > --max-cost-usd $%.2f",
            estimate, num_skills, activation_provider,
            COST_PER_SKILL_ANTHROPIC if activation_provider == "anthropic"
            else COST_PER_SKILL_CEREBRAS,
            max_cost_usd,
        )
        sys.exit(2)
    logger.info(
        "Cost pre-flight: ~$%.2f for %d skills (provider=%s, ceiling $%.2f) — OK",
        estimate, num_skills, activation_provider, max_cost_usd,
    )


# ── Skill scoring ────────────────────────────────────────────────────────────


async def score_one_skill(
    skill_dir: Path,
    *,
    repo: str,
    level,
    eval_hash_components: Dict[str, str],
    billing_tag: Optional[str],
    activation_provider: str,
    dry_run: bool,
    force: bool,
):
    """Evaluate a single skill, persist a SkillScore row, return the model.

    The function is sync-coupled to one skill (the parent runner uses a
    semaphore for concurrency). All external dependencies (Redis, Mongo,
    judges, activator) are best-effort: failure is logged and degrades to
    a SkillScore row with ``outcome="error"`` so the batch keeps moving.
    """
    from src.core.eval_hash import compute_eval_hash
    from src.core.skill_parser import parse_skill_md
    from src.core.skill_validator import validate_skill
    from src.storage.models import SkillScore

    skill_name = skill_dir.name
    git_sha = git_sha_for_path(skill_dir)

    eval_hash = compute_eval_hash(
        skill_sha=git_sha,
        question_pack_v=eval_hash_components["question_pack_v"],
        probe_pack_v=eval_hash_components["probe_pack_v"],
        judge_models_pinned=eval_hash_components["judge_models_pinned"],
        eval_settings_v=eval_hash_components["eval_settings_v"],
        activation_model=eval_hash_components["activation_model"],
    )

    # AC3 — L1 cache check. ``--force`` bypasses cache.
    if not force:
        cached = await cache_get_skill_score(eval_hash)
        if cached is not None:
            try:
                cached_score = SkillScore.model_validate(cached)
            except Exception:
                cached_score = None
            if cached_score is not None:
                cached_score.cached = True
                logger.info("cache HIT  %s (eval_hash=%s)", skill_name, eval_hash)
                return cached_score

    components_dict = {
        "question_pack_v": eval_hash_components["question_pack_v"],
        "probe_pack_v": eval_hash_components["probe_pack_v"],
        "judge_models_pinned": eval_hash_components["judge_models_pinned"],
        "eval_settings_v": eval_hash_components["eval_settings_v"],
        "activation_model": eval_hash_components["activation_model"],
    }

    # AC6 — parse failures persist a "skip" row.
    try:
        parsed = parse_skill_md(skill_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("parse failed %s: %s", skill_name, exc)
        return SkillScore(
            eval_hash=eval_hash,
            skill_name=skill_name,
            skill_sha=git_sha,
            skill_repo=repo,
            components=components_dict,
            activation_provider=activation_provider,
            outcome="skip",
            skip_reason=f"parse_error: {exc}",
            billing_tag=billing_tag,
        )

    spec_compliance = validate_skill(parsed, skill_dir)

    if dry_run:
        # Skip the LLM-driven path entirely; return a synthetic "ok" row
        # whose scoring lives off the spec_compliance result so
        # --dry-run remains end-to-end (AC: dry-run runs without API).
        sc = SkillScore(
            eval_hash=eval_hash,
            skill_name=parsed.name or skill_name,
            skill_sha=git_sha,
            skill_repo=repo,
            components=components_dict,
            activation_provider=activation_provider,
            scores_6axis={"schema_quality": float(spec_compliance.score)},
            spec_compliance=spec_compliance.model_dump(),
            overall_score=float(spec_compliance.score),
            tier="verified" if spec_compliance.passed_hard_fails else "failed",
            cost_dollars=0.0,
            billing_tag=billing_tag,
            outcome="ok",
        )
        # ``--dry-run`` does NOT touch Mongo / GridFS / Redis; the report
        # writer downstream still tabulates the synthetic row.
        return sc

    # Live path — wire the evaluator with judges and run evaluate_skill.
    try:
        from src.core.evaluator import Evaluator
        from src.core.consensus_judge import ConsensusJudge

        class _SkillTarget:
            pass

        target = _SkillTarget()
        target.parsed = parsed
        target.spec_compliance = spec_compliance
        target.subject_uri = str(skill_dir)
        target.skill_dir = str(skill_dir)  # consumed by QO-053-E probes

        judge = ConsensusJudge(max_judges=2)
        judge.reset_keys()
        evaluator = Evaluator(judge)

        eval_result = await evaluator.evaluate_skill(
            target=target,
            level=level,
            activator_factory=None,
            baseline_activator_factory=None,
        )

        # Pull 6-axis scores out of the EvaluationResult.
        axes = eval_result.dimensions or {}
        scores_6axis = {k: float(v.get("score", 0.0)) if isinstance(v, dict) else float(v)
                        for k, v in axes.items()}
        if not scores_6axis and spec_compliance:
            scores_6axis = {"schema_quality": float(spec_compliance.score)}

        sc = SkillScore(
            eval_hash=eval_hash,
            skill_name=parsed.name or skill_name,
            skill_sha=git_sha,
            skill_repo=repo,
            components=components_dict,
            activation_provider=activation_provider,
            scores_6axis=scores_6axis,
            spec_compliance=spec_compliance.model_dump(),
            probe_results=getattr(eval_result, "probe_results", []) or [],
            overall_score=float(eval_result.overall_score),
            baseline_score=eval_result.baseline_score,
            delta_vs_baseline=eval_result.delta_vs_baseline,
            tier=str(eval_result.tier),
            confidence=float(eval_result.confidence),
            cost_dollars=float(eval_result.cost_usd or 0.0),
            latency_ms=int(eval_result.duration_ms),
            billing_tag=billing_tag,
            outcome="ok",
        )

        # Audit blob — gzipped JSON in GridFS (v1; S3 later).
        audit = {
            "skill_md": (skill_dir / "SKILL.md").read_text(errors="replace")
                       if (skill_dir / "SKILL.md").is_file() else "",
            "judge_responses": getattr(eval_result, "judge_responses", []),
            "probe_results": sc.probe_results,
            "components": components_dict,
            "spec_compliance": sc.spec_compliance,
        }
        sc.audit_blob_id = await store_audit_blob(audit, f"{eval_hash}.json.gz")
        return sc
    except Exception as exc:  # noqa: BLE001
        logger.warning("score failed %s: %s", skill_name, exc)
        return SkillScore(
            eval_hash=eval_hash,
            skill_name=skill_name,
            skill_sha=git_sha,
            skill_repo=repo,
            components=components_dict,
            activation_provider=activation_provider,
            spec_compliance=spec_compliance.model_dump() if spec_compliance else {},
            outcome="error",
            error_message=str(exc),
            billing_tag=billing_tag,
        )


# ── Reporter (AC7) ───────────────────────────────────────────────────────────


def _eval_window() -> str:
    """ISO-week eval window string used in report filenames."""
    now = datetime.now(timezone.utc)
    iso_year, iso_week, _ = now.isocalendar()
    return f"{iso_year}W{iso_week:02d}"


def write_report(
    results: List[Any],
    repo: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Write the per-run report JSON to ``reports/<repo>-skills-<window>.json``.

    The CPCR computation per QO-051 is ``total_$ / Σ(judge_pass_count)``.
    A pass is any judge response with score ≥ ``settings.cpcr_correct_threshold``.
    """
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = repo.replace("/", "-")
    out = output_dir / f"{safe}-skills-{_eval_window()}.json"

    total = sum(getattr(r, "cost_dollars", 0.0) or 0.0 for r in results)
    paid = sum(getattr(r, "paid_fallthrough_dollars", 0.0) or 0.0 for r in results)

    # Per-skill table.
    per_skill = [
        {
            "skill": r.skill_name,
            "skill_sha": r.skill_sha,
            "tier": r.tier,
            "overall_score": r.overall_score,
            "cost": r.cost_dollars,
            "outcome": r.outcome,
            "cached": getattr(r, "cached", False),
        }
        for r in results
    ]

    # CPCR (QO-051).
    pass_count = 0
    for r in results:
        for p in (r.probe_results or []):
            if isinstance(p, dict) and p.get("outcome") == "pass":
                pass_count += 1
        if r.outcome == "ok" and r.overall_score >= 70:
            pass_count += 1
    cpcr = round(total / pass_count, 6) if pass_count > 0 else None

    # Per-axis cost (provisional — distributed evenly across the 6 axes
    # since the evaluator does not currently emit per-axis cost). The
    # axis weights live in ``src.core.axis_weights.SKILL_WEIGHTS``.
    try:
        from src.core.axis_weights import SKILL_WEIGHTS
        per_axis = {axis: round(total * w, 6) for axis, w in SKILL_WEIGHTS.items()}
    except Exception:
        per_axis = {}

    payload = {
        "repo": repo,
        "eval_window": _eval_window(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "skills": len(results),
            "succeeded": sum(1 for r in results if r.outcome == "ok"),
            "skipped": sum(1 for r in results if r.outcome == "skip"),
            "errored": sum(1 for r in results if r.outcome == "error"),
            "cached": sum(1 for r in results if getattr(r, "cached", False)),
            "total_cost_usd": round(total, 4),
            "paid_fallthrough_usd": round(paid, 4),
            "cpcr_usd": cpcr,
        },
        "per_axis_cost_usd": per_axis,
        "per_skill": per_skill,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("report written: %s", out)
    return out


def print_cost_summary(results: List[Any]) -> None:
    total = sum(getattr(r, "cost_dollars", 0.0) or 0.0 for r in results)
    succeeded = sum(1 for r in results if r.outcome == "ok")
    skipped = sum(1 for r in results if r.outcome == "skip")
    errored = sum(1 for r in results if r.outcome == "error")
    cached = sum(1 for r in results if getattr(r, "cached", False))
    print()
    print("=" * 60)
    print(f"Skills evaluated : {len(results)}")
    print(f"  succeeded      : {succeeded}")
    print(f"  skipped        : {skipped}")
    print(f"  errored        : {errored}")
    print(f"  cached (no $)  : {cached}")
    print(f"Total cost ($)   : {total:.4f}")
    print("=" * 60)


# ── Main / CLI ───────────────────────────────────────────────────────────────


async def run_batch(
    *,
    repo: str,
    ref: str,
    level,
    max_concurrency: int,
    max_cost_usd: float,
    billing_tag: Optional[str],
    dry_run: bool,
    force: bool,
    changed_only: bool,
) -> List[Any]:
    """Top-level batch driver — invoked from ``main()`` and from tests."""
    from src.config import settings

    repo_root = clone_repo(repo, ref)
    skill_dirs = find_skill_dirs(repo_root)
    logger.info("found %d skills in %s@%s", len(skill_dirs), repo, ref)

    if changed_only:
        # ``--changed-only`` filters to skills whose dir-relative git sha
        # differs from the cache. Implementation: hash every skill dir,
        # query ``quality__skill_scores`` for an existing row at that sha.
        # Out of scope of the L1-cache path (which is keyed on eval_hash);
        # this is a coarser pre-filter that avoids even computing eval_hash
        # for unchanged skills. Defers full impl to QO-053-G GitHub Action.
        logger.warning("--changed-only is a no-op in F (deferred to QO-053-G); evaluating all")

    # AC8 — pre-flight refuse-to-start.
    refuse_if_over_budget(
        num_skills=len(skill_dirs),
        max_cost_usd=max_cost_usd,
        activation_provider=settings.laureum_activation_provider,
    )

    # Connect Redis + Mongo when not in dry-run. Dry-run remains hermetic.
    if not dry_run:
        try:
            from src.storage.cache import connect_redis
            await connect_redis()
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis connect failed (continuing): %s", exc)
        try:
            from src.storage.mongodb import connect_db
            await connect_db()
        except Exception as exc:  # noqa: BLE001
            logger.warning("mongo connect failed (continuing): %s", exc)

    eval_hash_components = {
        "question_pack_v": settings.question_pack_v,
        "probe_pack_v": settings.probe_pack_v,
        "judge_models_pinned": settings.judge_models_pinned,
        "eval_settings_v": settings.eval_settings_v,
        "activation_model": settings.laureum_activation_model,
    }

    sem = asyncio.Semaphore(max_concurrency)

    async def _wrapped(d: Path):
        async with sem:
            return await score_one_skill(
                d,
                repo=repo,
                level=level,
                eval_hash_components=eval_hash_components,
                billing_tag=billing_tag,
                activation_provider=settings.laureum_activation_provider,
                dry_run=dry_run,
                force=force,
            )

    results = await asyncio.gather(*[_wrapped(d) for d in skill_dirs])

    # Persist + cache (skip in dry-run).
    if not dry_run:
        from src.storage.mongodb import skill_scores_col
        col = skill_scores_col()
        for sc in results:
            doc = sc.model_dump(mode="json")
            try:
                # Upsert keyed on eval_hash so reruns are idempotent (the
                # unique index makes the duplicate insert path explicit).
                await col.update_one(
                    {"eval_hash": sc.eval_hash},
                    {"$set": doc},
                    upsert=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("mongo insert failed for %s: %s", sc.skill_name, exc)
            if not getattr(sc, "cached", False) and sc.outcome == "ok":
                await cache_set_skill_score(sc.eval_hash, doc)

    return results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="batch_score_skills",
        description="QO-053-F batch skill scoring runner.",
    )
    p.add_argument("--repo", required=True,
                   help="GitHub <owner>/<repo> (e.g. sendaifun/skills)")
    p.add_argument("--ref", default="main",
                   help="git ref to evaluate (default: main)")
    p.add_argument("--changed-only", action="store_true",
                   help="(QO-053-G) skip skills with unchanged git_sha")
    p.add_argument("--level", default="2",
                   help="EvalLevel (1=manifest, 2=functional, 3=domain_expert)")
    p.add_argument("--max-concurrency", type=int, default=3,
                   help="parallel skill evals (default 3 — Cerebras safe)")
    p.add_argument("--max-cost-usd", type=float, default=5.0,
                   help="pre-flight cost ceiling; refuse to start if estimate > N (AC8)")
    p.add_argument("--billing-tag", default=None,
                   help="marketplace billing isolation tag (AC8)")
    p.add_argument("--dry-run", action="store_true",
                   help="parse + validate only, no LLM/Mongo/Redis")
    p.add_argument("--force", action="store_true",
                   help="bypass L1 cache (re-score even on identical eval_hash)")
    args = p.parse_args(argv)

    from src.storage.models import EvalLevel
    try:
        level = EvalLevel(int(args.level))
    except (ValueError, KeyError):
        # Allow public names ("L2 certified") via the level helper.
        from src.storage.models import level_for_skill_eval
        level, _ = level_for_skill_eval(args.level)

    try:
        results = asyncio.run(run_batch(
            repo=args.repo,
            ref=args.ref,
            level=level,
            max_concurrency=args.max_concurrency,
            max_cost_usd=args.max_cost_usd,
            billing_tag=args.billing_tag,
            dry_run=args.dry_run,
            force=args.force,
            changed_only=args.changed_only,
        ))
    except SystemExit:
        raise  # AC8 budget refusal already exited
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130

    write_report(results, args.repo)
    print_cost_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

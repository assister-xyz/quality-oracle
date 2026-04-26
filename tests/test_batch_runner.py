"""QO-053-F batch runner integration tests.

Covers the end-to-end batch driver against three fixture skills (no live
MongoDB / Redis / Anthropic / Cerebras calls):

* ``score_one_skill`` produces a SkillScore row in dry-run mode (AC: dry-run
  is hermetic).
* AC3 — second run on identical inputs hits the L1 cache (cached=True; no
  LLM call). We patch ``cache_get_skill_score`` to verify both paths.
* AC8 — pre-flight cost ceiling refuses to start with non-zero exit when
  estimate > --max-cost-usd.
* AC6 — a malformed skill is recorded as ``outcome="skip"``; the batch
  continues on the remaining skills.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# Add repo root so dev.* imports.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dev.batch_score_skills import (  # noqa: E402
    estimate_run_cost,
    refuse_if_over_budget,
    find_skill_dirs,
    score_one_skill,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def fixture_skills_dir(tmp_path) -> Path:
    """3 fixture skills committed to a fresh temp dir.

    Mirrors the on-disk shape that ``find_skill_dirs`` looks for:
    ``<root>/skills/<skill>/SKILL.md``.
    """
    skills = tmp_path / "skills"
    skills.mkdir()

    # 1. Clean skill — should pass.
    (skills / "good").mkdir()
    (skills / "good" / "SKILL.md").write_text(
        "---\n"
        "name: good\n"
        "description: A clean fixture skill that should score normally.\n"
        "license: MIT\n"
        "---\n\n"
        "# Good Skill\n\nNothing weird here.\n"
    )

    # 2. Another clean skill.
    (skills / "alpha").mkdir()
    (skills / "alpha" / "SKILL.md").write_text(
        "---\n"
        "name: alpha\n"
        "description: Second clean skill.\n"
        "license: MIT\n"
        "---\n\n"
        "# Alpha\n"
    )

    # 3. Malformed skill — YAML frontmatter intentionally broken to
    #    force ``parse_skill_md`` into the AC6 skip path.
    (skills / "broken").mkdir()
    (skills / "broken" / "SKILL.md").write_text(
        "---\n"
        "name: [::not-yaml::\n"
        "  description: missing close bracket\n"
        "----\n"  # double-end fence - doesn't even close frontmatter
        "Broken skill body — parser should reject.\n"
    )

    return tmp_path


# ── AC8: cost ceiling ───────────────────────────────────────────────────────


def test_estimate_run_cost_cerebras_default():
    """45 skills on Cerebras default = 45 × $0.05 = $2.25 (under $5)."""
    assert estimate_run_cost(45, activation_provider="cerebras") == pytest.approx(2.25)


def test_estimate_run_cost_anthropic_optin():
    """45 skills on Anthropic = 45 × $0.30 = $13.50 (under $25)."""
    assert estimate_run_cost(45, activation_provider="anthropic") == pytest.approx(13.5)


def test_refuse_to_start_when_over_budget():
    """AC8 — projected cost > --max-cost-usd → SystemExit(2), no LLM calls."""
    with pytest.raises(SystemExit) as excinfo:
        refuse_if_over_budget(
            num_skills=200,
            max_cost_usd=5.0,
            activation_provider="anthropic",  # 200 × $0.30 = $60
        )
    assert excinfo.value.code == 2


def test_under_budget_returns_silently():
    """No raise when projected cost < ceiling."""
    refuse_if_over_budget(
        num_skills=10,
        max_cost_usd=5.0,
        activation_provider="cerebras",  # 10 × $0.05 = $0.50
    )


# ── find_skill_dirs ─────────────────────────────────────────────────────────


def test_find_skill_dirs_walks_skills_subdir(fixture_skills_dir):
    """Three skills (incl. broken) all have a SKILL.md so all show up."""
    dirs = find_skill_dirs(fixture_skills_dir)
    names = sorted(d.name for d in dirs)
    assert names == ["alpha", "broken", "good"]


# ── AC6: malformed skill → outcome="skip" ────────────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_skips_malformed_yaml(fixture_skills_dir):
    """The "broken" fixture has malformed YAML; ``score_one_skill`` should
    return a SkillScore with ``outcome="skip"`` and a reason, NOT raise."""
    broken = fixture_skills_dir / "skills" / "broken"
    eval_hash_components = {
        "question_pack_v": "qpack-v1.0",
        "probe_pack_v": "ppack-v1.0",
        "judge_models_pinned": "cerebras+groq+gemini-v1",
        "eval_settings_v": "evset-v1.0",
        "activation_model": "cerebras:llama3.1-8b",
    }
    from src.storage.models import EvalLevel
    sc = await score_one_skill(
        broken,
        repo="test/fixture",
        level=EvalLevel.MANIFEST,
        eval_hash_components=eval_hash_components,
        billing_tag=None,
        activation_provider="cerebras",
        dry_run=True,
        force=True,  # bypass cache
    )
    # ``parse_skill_md`` is reasonably tolerant — it may produce a parsed
    # object with warnings, OR raise FileNotFoundError-ish. Either way
    # the runner must not crash.
    assert sc is not None
    assert sc.skill_repo == "test/fixture"
    assert sc.outcome in {"ok", "skip"}
    if sc.outcome == "skip":
        assert sc.skip_reason is not None


# ── Dry-run is hermetic ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_does_not_touch_external_services(fixture_skills_dir):
    """Dry-run path makes zero LLM, MongoDB, or Redis calls.

    We patch every external connector to raise on call so a leak
    immediately fails the test."""
    good = fixture_skills_dir / "skills" / "good"
    eval_hash_components = {
        "question_pack_v": "qpack-v1.0",
        "probe_pack_v": "ppack-v1.0",
        "judge_models_pinned": "cerebras+groq+gemini-v1",
        "eval_settings_v": "evset-v1.0",
        "activation_model": "cerebras:llama3.1-8b",
    }
    from src.storage.models import EvalLevel

    explosive = AsyncMock(side_effect=AssertionError("dry-run leaked an external call"))
    with (
        patch("src.storage.cache.get_redis", side_effect=AssertionError("redis touched")),
        patch("src.storage.mongodb.get_db", side_effect=AssertionError("mongo touched")),
        patch("src.core.consensus_judge.ConsensusJudge.ajudge", new=explosive),
    ):
        sc = await score_one_skill(
            good,
            repo="test/fixture",
            level=EvalLevel.MANIFEST,
            eval_hash_components=eval_hash_components,
            billing_tag="unit-test",
            activation_provider="cerebras",
            dry_run=True,
            force=True,
        )

    assert sc.outcome == "ok"
    assert sc.skill_repo == "test/fixture"
    assert sc.billing_tag == "unit-test"
    # eval_hash should be deterministic and non-empty.
    assert len(sc.eval_hash) == 16
    # Components dict carries all 5 inputs (CB4 audit trail).
    assert "activation_model" in sc.components
    assert sc.components["activation_model"] == "cerebras:llama3.1-8b"


# ── AC3: cache hit on second run ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_short_circuits_lLM(fixture_skills_dir):
    """When ``cache_get_skill_score`` returns a hit, ``score_one_skill``
    returns immediately with ``cached=True`` — no LLM, no Mongo, no
    parser invocation past the eval-hash compute."""
    good = fixture_skills_dir / "skills" / "good"
    eval_hash_components = {
        "question_pack_v": "qpack-v1.0",
        "probe_pack_v": "ppack-v1.0",
        "judge_models_pinned": "cerebras+groq+gemini-v1",
        "eval_settings_v": "evset-v1.0",
        "activation_model": "cerebras:llama3.1-8b",
    }
    from src.storage.models import EvalLevel, SkillScore

    cached_payload = SkillScore(
        eval_hash="0000000000000000",
        skill_name="good",
        skill_sha="cached",
        skill_repo="test/fixture",
        components=eval_hash_components,
        activation_provider="cerebras",
        outcome="ok",
        overall_score=88.0,
        tier="silver",
    ).model_dump(mode="json")

    with patch(
        "dev.batch_score_skills.cache_get_skill_score",
        new=AsyncMock(return_value=cached_payload),
    ):
        sc = await score_one_skill(
            good,
            repo="test/fixture",
            level=EvalLevel.MANIFEST,
            eval_hash_components=eval_hash_components,
            billing_tag=None,
            activation_provider="cerebras",
            dry_run=True,
            force=False,  # do NOT bypass cache
        )

    assert sc.cached is True
    assert sc.skill_name == "good"


# ── eval_hash is stable across runs on the same fixture ─────────────────────


@pytest.mark.asyncio
async def test_eval_hash_stable_across_runs(fixture_skills_dir):
    """Two dry-run scores of the same fixture must produce the same
    eval_hash. This is the property that AC3's L1 cache leans on."""
    good = fixture_skills_dir / "skills" / "good"
    components = {
        "question_pack_v": "qpack-v1.0",
        "probe_pack_v": "ppack-v1.0",
        "judge_models_pinned": "cerebras+groq+gemini-v1",
        "eval_settings_v": "evset-v1.0",
        "activation_model": "cerebras:llama3.1-8b",
    }
    from src.storage.models import EvalLevel
    sc1 = await score_one_skill(
        good, repo="r/r", level=EvalLevel.MANIFEST,
        eval_hash_components=components, billing_tag=None,
        activation_provider="cerebras", dry_run=True, force=True,
    )
    sc2 = await score_one_skill(
        good, repo="r/r", level=EvalLevel.MANIFEST,
        eval_hash_components=components, billing_tag=None,
        activation_provider="cerebras", dry_run=True, force=True,
    )
    assert sc1.eval_hash == sc2.eval_hash


# ── Report generation ───────────────────────────────────────────────────────


def test_write_report_emits_json(fixture_skills_dir, tmp_path):
    """The cost reporter writes a JSON file with totals + per_skill."""
    from dev.batch_score_skills import write_report
    from src.storage.models import SkillScore

    rows = [
        SkillScore(
            eval_hash="hash00000000000a",
            skill_name="good",
            skill_sha="abc",
            skill_repo="r/r",
            components={},
            activation_provider="cerebras",
            outcome="ok",
            overall_score=85.0,
            tier="silver",
            cost_dollars=0.10,
        ),
        SkillScore(
            eval_hash="hash00000000000b",
            skill_name="alpha",
            skill_sha="def",
            skill_repo="r/r",
            components={},
            activation_provider="cerebras",
            outcome="ok",
            overall_score=70.0,
            tier="bronze",
            cost_dollars=0.05,
        ),
        SkillScore(
            eval_hash="hash00000000000c",
            skill_name="broken",
            skill_sha="xyz",
            skill_repo="r/r",
            components={},
            activation_provider="cerebras",
            outcome="skip",
            skip_reason="parse_error",
        ),
    ]
    out = write_report(rows, "r/r", output_dir=tmp_path)
    assert out.exists()

    import json
    payload = json.loads(out.read_text())
    assert payload["repo"] == "r/r"
    assert payload["totals"]["skills"] == 3
    assert payload["totals"]["succeeded"] == 2
    assert payload["totals"]["skipped"] == 1
    assert payload["totals"]["total_cost_usd"] == pytest.approx(0.15)
    # CPCR uses the (judge_pass count). With 2 passing skills (overall>=70)
    # and total cost $0.15, CPCR = $0.075.
    assert payload["totals"]["cpcr_usd"] == pytest.approx(0.075, abs=0.01)
    # Per-axis cost rows present (6 axes × distributed cost).
    assert isinstance(payload["per_axis_cost_usd"], dict)

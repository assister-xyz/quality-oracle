"""QO-053-F SkillScore persistence + index tests (AC4).

Round-trip a ``SkillScore`` through ``model_dump()`` and back, and verify
that the index list ``connect_db()`` will create matches the spec
(skill_repo+skill_name+git_sha; eval_hash UNIQUE; evaluated_at DESC; tier).

We don't run a live MongoDB here — the existing test base mocks it. We
do still want to assert the *intent* of the indexes (so a regression that
removes one fails loudly), so we patch ``create_index`` and inspect the
calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.storage.models import SkillScore, EvalLevel, TargetType


def test_skillscore_round_trip():
    """A SkillScore round-trips through ``model_dump`` → ``model_validate``
    losslessly. Catches drift between Pydantic field defaults and the
    persistence-time payload."""
    sc = SkillScore(
        eval_hash="0123456789abcdef",
        skill_name="my-skill",
        skill_sha="abc123",
        skill_repo="sendaifun/skills",
        evaluated_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
        components={
            "question_pack_v": "qpack-v1.0",
            "probe_pack_v": "ppack-v1.0",
            "judge_models_pinned": "cerebras+groq+gemini-v1",
            "eval_settings_v": "evset-v1.0",
            "activation_model": "cerebras:llama3.1-8b",
        },
        activation_provider="cerebras",
        target_type=TargetType.SKILL,
        level=EvalLevel.FUNCTIONAL,
        scores_6axis={"accuracy": 80, "safety": 90, "reliability": 85},
        spec_compliance={"score": 95, "violations": []},
        probe_results=[{"id": "SKILL-01", "outcome": "pass"}],
        overall_score=82.5,
        baseline_score=70.0,
        delta_vs_baseline=12.5,
        tier="silver",
        confidence=0.85,
        cost_dollars=0.42,
        paid_fallthrough_dollars=0.0,
        billing_tag="sendai-launch",
        latency_ms=12340,
        audit_blob_id="63abf...",
        outcome="ok",
        cached=False,
    )
    dumped = sc.model_dump(mode="json")
    revived = SkillScore.model_validate(dumped)

    assert revived.eval_hash == sc.eval_hash
    assert revived.skill_name == sc.skill_name
    assert revived.skill_sha == sc.skill_sha
    assert revived.scores_6axis == sc.scores_6axis
    assert revived.overall_score == sc.overall_score
    assert revived.tier == sc.tier
    assert revived.cost_dollars == sc.cost_dollars
    assert revived.outcome == sc.outcome


def test_skillscore_components_dict_carries_all_eval_hash_inputs():
    """The persisted ``components`` dict must include every input that
    flows into ``compute_eval_hash`` so an external auditor can recompute
    the hash from the document alone (CB4 + AC4)."""
    from src.core.eval_hash import compute_eval_hash

    sha = "deadbeef"
    components = {
        "question_pack_v": "qpack-v1.0",
        "probe_pack_v": "ppack-v1.0",
        "judge_models_pinned": "cerebras+groq+gemini-v1",
        "eval_settings_v": "evset-v1.0",
        "activation_model": "cerebras:llama3.1-8b",
    }
    expected_hash = compute_eval_hash(skill_sha=sha, **components)

    sc = SkillScore(
        eval_hash=expected_hash,
        skill_name="x",
        skill_sha=sha,
        skill_repo="r/r",
        components=components,
        activation_provider="cerebras",
    )
    # Replay test — recompute the hash from the persisted dict alone.
    replayed = compute_eval_hash(skill_sha=sc.skill_sha, **sc.components)
    assert replayed == sc.eval_hash


def test_skipped_skill_outcome_persists(monkeypatch):
    """AC6: a skipped skill is persisted with an explicit reason and the
    batch keeps moving."""
    sc = SkillScore(
        eval_hash="badbadbadbad0000",
        skill_name="malformed",
        skill_sha="abc",
        skill_repo="x/y",
        components={},
        activation_provider="cerebras",
        outcome="skip",
        skip_reason="parse_error: malformed YAML",
    )
    dumped = sc.model_dump(mode="json")
    assert dumped["outcome"] == "skip"
    assert "malformed YAML" in dumped["skip_reason"]


@pytest.mark.asyncio
async def test_index_list_matches_spec():
    """AC4: indexes match the spec. We mock the motor client so this test
    doesn't hit a real MongoDB."""
    fake_db = MagicMock()
    fake_db.quality__skill_scores.create_index = AsyncMock()
    # Stub out every other create_index call too.
    for name in dir(fake_db):
        # any attribute ending in __ would be a collection
        pass

    fake_client = MagicMock()
    fake_client.__getitem__ = MagicMock(return_value=fake_db)

    # Make every collection on fake_db have an async create_index.
    def _mk_col(*_args, **_kw):
        col = MagicMock()
        col.create_index = AsyncMock()
        return col
    # Default factory for every quality__* attribute.
    type(fake_db).__getattr__ = lambda self, _name: _mk_col()
    # Re-attach the real one so we can assert on it.
    fake_db.quality__skill_scores = MagicMock()
    fake_db.quality__skill_scores.create_index = AsyncMock()

    with patch("src.storage.mongodb.AsyncIOMotorClient", return_value=fake_client):
        from src.storage import mongodb as m
        # Replace _client/_db so close_db sees something.
        m._client = fake_client
        m._db = fake_db
        await m.connect_db()

    calls = [c.args for c in fake_db.quality__skill_scores.create_index.await_args_list]
    # Verify all four expected indexes were requested.
    assert any(
        c == ([("skill_repo", 1), ("skill_name", 1), ("git_sha", 1)],)
        for c in calls
    ), f"compound index missing; got {calls}"
    # Note: unique=True is in kwargs not args so check the kwargs too.
    kwargs_calls = [c.kwargs for c in fake_db.quality__skill_scores.create_index.await_args_list]
    eval_hash_call = [
        (a, k) for a, k in zip(calls, kwargs_calls)
        if a == ("eval_hash",)
    ]
    assert eval_hash_call, "eval_hash unique index missing"
    assert eval_hash_call[0][1].get("unique") is True
    assert any(c == ([("evaluated_at", -1)],) for c in calls)
    assert any(c == ("tier",) for c in calls)

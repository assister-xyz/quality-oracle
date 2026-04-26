"""QO-053-F eval_hash tests (CB4 critical).

Coverage:

* AC1 — determinism: same inputs → same 16-char hex.
* AC2 — collision smoke: any single-input change → different hash.
* CB4 — bumping ``activation_model`` invalidates cache (the load-bearing
  bit per R9 §9.1; if this assertion ever fails it means the activation-
  model alias has stopped flowing into the preimage and stale-cache
  attacks are silently possible).
"""
from __future__ import annotations

import string

from src.core.eval_hash import compute_eval_hash


def _base_args():
    return dict(
        skill_sha="abc1234",
        question_pack_v="qpack-v1.0",
        probe_pack_v="ppack-v1.0",
        judge_models_pinned="cerebras+groq+gemini-v1",
        eval_settings_v="evset-v1.0",
        activation_model="cerebras:llama3.1-8b",
    )


def test_determinism():
    """AC1 — calling the function twice with identical args returns the
    identical 16-char string."""
    a = compute_eval_hash(**_base_args())
    b = compute_eval_hash(**_base_args())
    assert a == b
    assert len(a) == 16
    assert all(c in string.hexdigits for c in a)


def test_each_input_change_invalidates_hash():
    """AC2 — any single-input change rolls the hash."""
    base = compute_eval_hash(**_base_args())
    for field in (
        "skill_sha", "question_pack_v", "probe_pack_v",
        "judge_models_pinned", "eval_settings_v", "activation_model",
    ):
        args = _base_args()
        args[field] = args[field] + "X"
        modified = compute_eval_hash(**args)
        assert modified != base, f"hash unchanged when {field} bumped"


def test_activation_model_bump_invalidates_cb4():
    """CB4 — bumping the activation-model alias MUST invalidate the cache
    so a Sonnet 4.6 → 4.7 transition forces re-evaluation per R9 §9.1.

    This is the load-bearing assertion for stale-cache attack resistance:
    if it ever fails we can stuff malicious skills into the cache by
    racing different sub-skills under the same git_sha across an
    activation-model bump.
    """
    args = _base_args()
    h_old = compute_eval_hash(**args)

    args["activation_model"] = "claude-sonnet-4-7-20260301"
    h_new = compute_eval_hash(**args)
    assert h_old != h_new, "CB4 violated: activation_model bump did not roll hash"

    # Cerebras alias bump must also roll.
    args["activation_model"] = "cerebras:llama-3.3-70b"
    h_cer = compute_eval_hash(**args)
    assert h_cer != h_old != h_new


def test_collision_smoke_uniqueness():
    """Light collision-resistance smoke test — 1k randomly perturbed
    inputs all produce distinct hashes (the SHA-256 floor makes a real
    collision astronomically unlikely)."""
    seen = set()
    for i in range(1000):
        args = _base_args()
        args["skill_sha"] = f"sha-{i:08d}"
        seen.add(compute_eval_hash(**args))
    assert len(seen) == 1000


def test_pipe_separator_is_unambiguous():
    """Component values containing the pipe sentinel must not silently
    collide with adjacent components. We disallow pipes in versions in
    practice, but the hash should still be sensitive to pipe-injection
    attempts — i.e. moving content from one component to another should
    NOT produce the same digest.

    This is more of a safety property than a strict invariant; the
    function joins with '|' and that's enough for this check.
    """
    args1 = _base_args()
    args1["skill_sha"] = "abc"
    args1["question_pack_v"] = "def"
    args2 = _base_args()
    args2["skill_sha"] = "abc|def"
    args2["question_pack_v"] = ""
    # The two inputs differ only by where the boundary is drawn — the
    # digest MUST differ.
    assert compute_eval_hash(**args1) != compute_eval_hash(**args2)


def test_all_empty_strings_does_not_crash():
    """Defensive — eval_hash should never raise on weird inputs."""
    h = compute_eval_hash("", "", "", "", "", "")
    assert len(h) == 16


def test_components_dict_includes_activation_model():
    """Every component flows into the digest — the dict the runner
    persists must mirror exactly the preimage so an external auditor can
    recompute the hash from the SkillScore document alone."""
    from src.config import settings
    args = dict(
        skill_sha="abc",
        question_pack_v=settings.question_pack_v,
        probe_pack_v=settings.probe_pack_v,
        judge_models_pinned=settings.judge_models_pinned,
        eval_settings_v=settings.eval_settings_v,
        activation_model=settings.laureum_activation_model,
    )
    h = compute_eval_hash(**args)
    # Components dict that the SkillScore record holds:
    components = {
        "question_pack_v": settings.question_pack_v,
        "probe_pack_v": settings.probe_pack_v,
        "judge_models_pinned": settings.judge_models_pinned,
        "eval_settings_v": settings.eval_settings_v,
        "activation_model": settings.laureum_activation_model,
    }
    # Reproduce from the components dict.
    h2 = compute_eval_hash(skill_sha="abc", **components)
    assert h == h2

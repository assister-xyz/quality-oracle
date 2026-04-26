"""Reproducibility-anchoring eval hash (QO-053-F).

A deterministic 16-char SHA-256 fingerprint over the full set of inputs that
**should** invalidate a cached evaluation:

* skill_sha            — git SHA pin of the skill (R2 §"Mutation rate")
* question_pack_v      — version of the question pack
* probe_pack_v         — version of the adversarial probe pack
* judge_models_pinned  — pinned judge model bundle (e.g. "cerebras+groq+gemini")
* eval_settings_v      — version of evaluator settings (axis weights, gates)
* activation_model     — CB4 — bumping the activation model alias (e.g.
                         Sonnet 4.6 → 4.7) MUST invalidate cached scores per
                         R9 §9.1. Including it in the preimage is load-bearing
                         for stale-cache attack resistance.

Source of truth: ``src/config.py:settings.LAUREUM_ACTIVATION_MODEL`` (CB1).

Function returns the first 16 hex chars of the digest. At our scale (≤10⁶
distinct evals/year) the collision probability is negligible (≈10⁻⁹).
"""
from __future__ import annotations

import hashlib


def compute_eval_hash(
    skill_sha: str,
    question_pack_v: str,
    probe_pack_v: str,
    judge_models_pinned: str,
    eval_settings_v: str,
    activation_model: str,
) -> str:
    """Compute the 16-char eval hash.

    All arguments are coerced to ``str`` for safety (numeric versions land
    here in some call sites) and joined with the pipe sentinel ``|`` so a
    component containing the sentinel can never collide with a different
    component split across two slots — pipe is forbidden in version strings.

    Parameters
    ----------
    skill_sha:
        Git SHA-1 (or full SHA-256) of the skill at evaluation time.
    question_pack_v, probe_pack_v, judge_models_pinned, eval_settings_v:
        Pinned version identifiers. Bumping any one rolls every hash.
    activation_model:
        Resolved activation alias (CB4). Must be the *value* of
        ``settings.LAUREUM_ACTIVATION_MODEL`` at eval time, not the
        provider-resolved dated snapshot — a Sonnet-alias bump triggers
        a full re-eval per R9 §9.1.

    Returns
    -------
    str
        16 lowercase hex characters.

    Notes
    -----
    Stable across CPython releases — relies only on stdlib ``hashlib`` and
    UTF-8 encoding, both of which are spec-defined.
    """
    parts = [
        str(skill_sha or ""),
        str(question_pack_v or ""),
        str(probe_pack_v or ""),
        str(judge_models_pinned or ""),
        str(eval_settings_v or ""),
        str(activation_model or ""),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]

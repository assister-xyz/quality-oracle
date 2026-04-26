"""Canonical ``ProbeResult`` model — shared by QO-053-D (Solana) and QO-053-E.

A probe is a single static or LLM-judge check with a deterministic ID
(``SOL-01``, ``SKL-03`` …). Every probe produces a :class:`ProbeResult` so
downstream aggregation is uniform regardless of detection method.

Severity ladder (spec §"Severity score-deduction ladder"):

* ``HIGH`` ⇒ −15 pts
* ``MED``  ⇒ −5 pts
* ``LOW``  ⇒ −2 pts (aggregate cap −10 across all LOW probes)

Single source of truth for the deduction numbers lives below; the safety-axis
aggregation hook in :mod:`evaluator.evaluate_skill` consumes these constants.
``src/core/scoring.py`` (forbidden zone) deliberately does *not* duplicate
them — the probe-pack owns its own deduction table.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from src.storage.models import Severity


class Outcome(str, Enum):
    """Probe outcome.

    ``PASS``  — no risk found (or tutorial-gated benign hit).
    ``FAIL``  — pattern matched / judge gave a failing score.
    ``SKIP``  — probe couldn't run (e.g. LLM probe with no API key).
    ``ERROR`` — exception during probe execution; treated as inconclusive.
    """

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


# ── Severity deduction ladder (shared with QO-053-E) ────────────────────────

# Spec §"Severity score-deduction ladder": HIGH=-15, MED=-5, LOW=-2.
# Aggregate LOW deductions are capped so that one HIGH > sum(LOW) by
# construction (4*-2=-8 < -15). The chosen cap of -10 covers up to 5 LOWs
# while still preserving the inequality 1 HIGH = -15 > -10.
SEVERITY_DEDUCTION = {
    Severity.HIGH: -15,
    Severity.MED: -5,
    Severity.LOW: -2,
}
LOW_DEDUCTION_CAP = -10  # maximum total deduction from accumulated LOW hits


def aggregate_safety_deductions(results: List["ProbeResult"]) -> int:
    """Sum probe deductions with the LOW-cap applied.

    Only ``Outcome.FAIL`` results contribute. ``PASS``/``SKIP``/``ERROR`` are
    no-ops (errors are surfaced via ``ProbeResult.note`` for the audit log).

    Returns
    -------
    int
        Negative integer. Add to a baseline (typically 100) to get the
        Solana-only safety component, then clamp 0..100 at the call site.
    """
    high_med_total = 0
    low_total = 0
    for r in results:
        if r.outcome != Outcome.FAIL:
            continue
        delta = SEVERITY_DEDUCTION.get(r.severity, 0)
        if r.severity == Severity.LOW:
            low_total += delta
        else:
            high_med_total += delta
    # Cap the LOW contribution.
    if low_total < LOW_DEDUCTION_CAP:
        low_total = LOW_DEDUCTION_CAP
    return high_med_total + low_total


class ProbeResult(BaseModel):
    """A single adversarial probe finding.

    Shared between Solana probes (QO-053-D, IDs ``SOL-*``) and the generic
    skill probe pack (QO-053-E, IDs ``SKL-*``). Persistence is handled by the
    evaluator at the aggregation step — the probe-runners never write to
    Mongo directly.
    """

    id: str  # ``SOL-01`` .. ``SOL-15`` or ``SKL-*``
    outcome: Outcome
    severity: Severity
    judge_method: str  # ``static_regex`` | ``static_ast`` | ``llm_judge_3`` | ``sentiment_gate``
    evidence: List[str] = Field(default_factory=list)  # ``file:line`` refs
    note: Optional[str] = None
    judge_score: Optional[float] = None  # 0-100 for LLM probes
    cost_dollars: float = 0.0

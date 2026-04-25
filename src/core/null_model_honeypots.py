"""Null-model honeypot probes (QO-061 AC4).

A small bank of constant / empty / pure-refusal "answers" that should NEVER
score above 50 from a properly calibrated judge. We embed a random sample of
3 probes per eval-run (NOT per question) and flag any judge that scores them
high. If 2+ judges hit, the eval is marked `judge_compromise_suspected`.

The probe bank lives in `dev/honeypot_bank.json` (curated 20 entries). Random
selection per run defeats memorization-based gaming.

References:
- R8 §10 v2 architecture: rotating honeypot bank
- "Null-model attack" (ICLR 2025 oral): 86.5% LC win-rate on AlpacaEval 2.0
"""
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)

# Resolve bank path relative to repo root (… /quality-oracle/dev/honeypot_bank.json)
_DEFAULT_BANK_PATH = Path(__file__).resolve().parents[2] / "dev" / "honeypot_bank.json"

# Threshold above which a judge scoring a honeypot is considered anomalous.
HONEYPOT_SCORE_THRESHOLD = 50

# Default number of probes to embed per eval-run.
DEFAULT_PROBES_PER_RUN = 3

# Threshold for flagging the whole eval as judge-compromise suspected.
COMPROMISE_JUDGE_HIT_THRESHOLD = 2


@dataclass
class HoneypotProbe:
    id: str
    kind: str
    input: str
    label: str


@dataclass
class HoneypotJudgeHit:
    """A single (judge × probe) score that exceeded the threshold."""
    judge_provider: str
    judge_family: str
    probe_id: str
    probe_label: str
    score: int


@dataclass
class HoneypotResult:
    """Aggregate outcome of running the honeypot probes against the panel."""
    probes_run: List[HoneypotProbe]
    hits: List[HoneypotJudgeHit] = field(default_factory=list)
    judge_anomalies: List[str] = field(default_factory=list)  # provider names
    judge_compromise_suspected: bool = False

    def to_dict(self) -> dict:
        return {
            "probes_run": [p.id for p in self.probes_run],
            "hits": [
                {
                    "judge_provider": h.judge_provider,
                    "judge_family": h.judge_family,
                    "probe_id": h.probe_id,
                    "probe_label": h.probe_label,
                    "score": h.score,
                }
                for h in self.hits
            ],
            "judge_anomalies": self.judge_anomalies,
            "judge_compromise_suspected": self.judge_compromise_suspected,
        }


def load_honeypot_bank(path: Optional[Path] = None) -> List[HoneypotProbe]:
    """Load and validate the curated honeypot bank from disk."""
    bank_path = path or _DEFAULT_BANK_PATH
    with open(bank_path, "r") as f:
        raw = json.load(f)
    return [HoneypotProbe(**p) for p in raw["probes"]]


def select_probes(
    bank: List[HoneypotProbe],
    n: int = DEFAULT_PROBES_PER_RUN,
    rng: Optional[random.Random] = None,
) -> List[HoneypotProbe]:
    """Return a random sample of n probes from the bank."""
    rng = rng or random.Random()
    if n >= len(bank):
        return list(bank)
    return rng.sample(bank, n)


# Type alias for a "score this honeypot answer" callable. Each panel judge
# is wrapped as such a callable so this module stays decoupled from LLMJudge.
ScoreFn = Callable[[str], Awaitable[int]]


@dataclass
class JudgeProbeRunner:
    """Wraps a single judge so the honeypot module can score probes against it."""
    provider: str
    family: str
    score_fn: ScoreFn


async def run_honeypot_probes(
    judge_runners: List[JudgeProbeRunner],
    bank: Optional[List[HoneypotProbe]] = None,
    probes_per_run: int = DEFAULT_PROBES_PER_RUN,
    rng: Optional[random.Random] = None,
) -> HoneypotResult:
    """Run a random sample of `probes_per_run` honeypots against every judge.

    QO-061 AC4:
      - Any judge scoring any honeypot >50 → flag in `judge_anomalies`.
      - 2+ judges hit → eval marked `judge_compromise_suspected`.
    """
    bank = bank if bank is not None else load_honeypot_bank()
    probes = select_probes(bank, probes_per_run, rng=rng)

    hits: List[HoneypotJudgeHit] = []
    flagged_judges: set = set()

    for runner in judge_runners:
        for probe in probes:
            try:
                score = await runner.score_fn(probe.input)
            except Exception as e:
                logger.warning(
                    f"Honeypot scoring failed for judge={runner.provider} "
                    f"probe={probe.id}: {e}"
                )
                continue
            if score > HONEYPOT_SCORE_THRESHOLD:
                hits.append(
                    HoneypotJudgeHit(
                        judge_provider=runner.provider,
                        judge_family=runner.family,
                        probe_id=probe.id,
                        probe_label=probe.label,
                        score=score,
                    )
                )
                flagged_judges.add(runner.provider)

    compromised = len(flagged_judges) >= COMPROMISE_JUDGE_HIT_THRESHOLD
    return HoneypotResult(
        probes_run=probes,
        hits=hits,
        judge_anomalies=sorted(flagged_judges),
        judge_compromise_suspected=compromised,
    )

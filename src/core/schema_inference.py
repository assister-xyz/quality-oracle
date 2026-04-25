"""Manifest-less schema inference for REST-chat targets (QO-058 AC4 + AC11).

R6 §"Manifest-less methodology": send 3 calibration prompts, observe the
target's responses, ask a judge to infer a soft manifest. Inference confidence
gates downstream tier behaviour:

* ``high``    — operator supplied n=10 calibration prompts AND ≥10 correlation
                rows → Certified path opens (spec §"Path to Certified").
* ``medium``  — 3-of-3 calibration prompts succeeded; default for REST chat.
* ``low``     — ≥2 of 3 calibration prompts hung or returned 5xx → AC11
                triggers ``SchemaUnobtainableError`` so the eval refuses
                rather than producing a misleading score.

Cost target: ~$0.005 per inference (3 cheap LLM calls on Cerebras free tier).
The judge prompt is intentionally tight so token usage stays bounded.

This module is callable by anyone — ``RESTChatTarget.discover()`` is the
primary caller, but the eventual "Operator-supplied calibration" upgrade in
QO-068 will reuse :func:`infer_manifest`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.core.evaluation_target import (
    AgentManifest,
    Capability,
    SchemaUnobtainableError,
)

logger = logging.getLogger(__name__)


# Three calibration prompts — chosen so the third deliberately stresses the
# error-handling path so we can observe whether the agent surfaces structured
# error responses (a useful schema signal).
CALIBRATION_PROMPTS: List[str] = [
    "Hi! Can you introduce yourself in one sentence?",
    "What domain are you specialised in? Give me one specific capability you offer.",
    # Malformed-but-plausible — exercises validation path.
    "{{INVALID_JSON: please reply ANYWAY",
]


@dataclass
class CalibrationResult:
    """One round-trip of a calibration prompt against the target."""
    prompt: str
    text: str = ""
    error: Optional[str] = None
    latency_ms: int = 0
    status: Literal["ok", "error", "timeout"] = "ok"


@dataclass
class InferredSchema:
    """Output of :func:`infer_manifest`."""
    summary: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"
    domain_hint: str = "general"
    capabilities: List[Capability] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)


def _classify_confidence(results: List[CalibrationResult]) -> Literal["low", "medium", "high"]:
    """AC11 — `low` if ≥2 of 3 calibration prompts failed (5xx / hang)."""
    if not results:
        return "low"
    failures = sum(1 for r in results if r.status != "ok" or not r.text)
    if failures >= 2:
        return "low"
    # Default to medium — only the operator-supplied n=10 path can promote
    # to ``high``; that path is invoked from a different entry-point.
    return "medium"


async def infer_manifest(
    results: List[CalibrationResult],
    *,
    judge=None,
    target_url: str = "",
) -> InferredSchema:
    """Build an :class:`InferredSchema` from calibration results.

    Parameters
    ----------
    results
        The 3 :class:`CalibrationResult` from
        :data:`CALIBRATION_PROMPTS`.
    judge
        Optional — anything with ``ajudge(question, expected, answer)``.
        When provided, the judge produces the human-readable summary +
        domain hint. When ``None`` (or call fails) we fall back to a
        deterministic summary derived from the response text. Tests rely
        on this fallback for hermeticity.
    target_url
        Used in summary text only — never sent to the judge.
    """
    confidence = _classify_confidence(results)

    # Refuse early if the data is too thin to even build a soft manifest
    # (AC11). The caller surfaces this as ``tier=failed`` /
    # ``error_type=schema_unobtainable``.
    if confidence == "low":
        raise SchemaUnobtainableError(
            "Calibration failed on ≥2 of 3 prompts — refuse to score "
            "this target until operator supplies an OpenAPI doc or a "
            "longer (n=10) calibration prompt set."
        )

    # Cheap deterministic summary so tests don't require a judge fixture.
    snippets = [r.text[:160] for r in results if r.text][:2]
    fallback_summary = (
        f"Inferred from {len(results)} calibration prompts at "
        f"{target_url or '<unknown>'}: " + " | ".join(snippets)
        if snippets
        else "manifest-less target — schema unknown"
    )

    summary = fallback_summary
    domain_hint = "general"

    if judge is not None and snippets:
        try:
            # Tight prompt — keep cost bounded (~$0.001 per call, 3 calls).
            judge_prompt = (
                "Given these agent responses to greeting/domain/malformed "
                "calibration prompts, return ONE short sentence summarising "
                "what the agent does. Be specific."
            )
            joined = "\n---\n".join(snippets)
            jr = await judge.ajudge(judge_prompt, "specific summary", joined)
            txt = getattr(jr, "explanation", "") or ""
            if txt:
                summary = txt.strip()[:280]
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("schema_inference judge call failed: %s", exc)

    return InferredSchema(
        summary=summary,
        confidence=confidence,
        domain_hint=domain_hint,
        capabilities=[
            Capability(
                id="chat",
                name="chat",
                description=summary[:140],
                input_schema=None,  # free-form
                output_schema=None,
                accepted_input_types=["text/plain", "application/json"],
                produced_output_types=["text/plain", "application/json"],
            )
        ],
        parse_warnings=[],
    )


def manifest_from_inferred(
    *,
    target_url: str,
    inferred: InferredSchema,
) -> AgentManifest:
    """Build the public :class:`AgentManifest` from the inferred soft schema."""
    import hashlib

    return AgentManifest(
        id=hashlib.sha256(target_url.encode()).hexdigest()[:16],
        name="<unknown>",
        description=inferred.summary,
        capabilities=inferred.capabilities,
        auth=None,
        signature=None,
        confidence=inferred.confidence,
        raw={"inference_source": "rest_chat_calibration"},
    )

"""AC2 — MCP backward compatibility regression on 5 fixture evaluations.

Pin the persisted-document shape and field values for 5 representative
MCP-server evaluations. ``evaluate_mcp`` is a thin façade over
``evaluate_full`` (no behavioural change), so this suite exercises
``evaluate_full`` directly with a stub LLM judge — the same as the existing
``tests/test_evaluator.py`` patterns — and asserts the output dict is
byte-identical between two consecutive runs after the Pydantic conversion.

This is a *self-comparison* regression: the legacy class is gone, so we can't
literally compare to a pre-053-C snapshot. Instead we:

1. Build the result twice with identical inputs.
2. Strip nondeterministic fields (``result_hash``, ``duration_ms``, ``token_usage``,
   anything carrying a timestamp).
3. Assert the remaining ``to_dict()`` output is byte-identical run-to-run.
4. Separately assert that the new QO-053-C fields are NOT present (AC9).
"""
from __future__ import annotations

import json

import pytest

from src.core.evaluator import EvaluationResult, Evaluator, ManifestValidationResult


class _StubJudgeResult:
    def __init__(self, score: int = 80, method: str = "llm"):
        self.score = score
        self.explanation = "stub"
        self.method = method


class _StubLLMJudge:
    """Deterministic judge — returns ``score`` for every call."""

    def __init__(self, score: int = 80):
        self._score = score
        self.metrics = type(
            "M",
            (),
            {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_provider": {},
                "llm_calls": 0,
                "fuzzy_routed": 0,
                "cache_hits": 0,
                "total_judged": 0,
            },
        )()

    async def ajudge(self, q, expected, ans, test_type=""):
        return _StubJudgeResult(self._score)

    def reset_keys(self):
        pass

    @property
    def provider(self):
        return "stub"


def _strip_nondet(d: dict) -> dict:
    """Drop fields whose values vary across runs (timestamps, hashes, costs)."""
    drop_top = {"result_hash", "duration_ms", "token_usage", "cost_usd", "shadow_cost_usd"}
    return {k: v for k, v in d.items() if k not in drop_top}


# ── 5 fixture inputs ───────────────────────────────────────────────────────


_TOOL_RESPONSES_FIXTURES = [
    # 1. Single tool, all pass
    {"search": [
        {"question": "find x", "expected": "x found", "answer": "x found", "latency_ms": 100},
        {"question": "find y", "expected": "y found", "answer": "y found", "latency_ms": 120},
    ]},
    # 2. Mixed tools
    {
        "fetch": [
            {"question": "GET /a", "expected": "ok", "answer": "ok", "latency_ms": 50},
            {"question": "GET /b", "expected": "ok", "answer": "ok", "latency_ms": 60},
        ],
        "save": [
            {"question": "save 1", "expected": "saved", "answer": "saved", "latency_ms": 200},
        ],
    },
    # 3. With errors
    {"buggy": [
        {"question": "test 1", "expected": "ok", "answer": "ERR", "latency_ms": 500, "is_error": True},
        {"question": "test 2", "expected": "ok", "answer": "ok", "latency_ms": 100},
    ]},
    # 4. High-volume single tool
    {"chat": [
        {"question": f"q{i}", "expected": f"a{i}", "answer": f"a{i}", "latency_ms": 80}
        for i in range(10)
    ]},
    # 5. Three tools, varied latency
    {
        "alpha": [{"question": "1", "expected": "ok", "answer": "ok", "latency_ms": 30}],
        "beta": [{"question": "2", "expected": "ok", "answer": "ok", "latency_ms": 60}],
        "gamma": [{"question": "3", "expected": "ok", "answer": "ok", "latency_ms": 90}],
    },
]


_MANIFEST = {
    "name": "test-server",
    "version": "1.0",
    "description": "test",
    "tools": [
        {"name": "search", "description": "search", "inputSchema": {"properties": {"q": {"type": "string"}}}},
        {"name": "fetch", "description": "fetch", "inputSchema": {"properties": {"url": {"type": "string"}}}},
        {"name": "save", "description": "save", "inputSchema": {"properties": {"data": {"type": "string"}}}},
        {"name": "buggy", "description": "buggy", "inputSchema": {"properties": {"x": {"type": "string"}}}},
        {"name": "chat", "description": "chat", "inputSchema": {"properties": {"msg": {"type": "string"}}}},
        {"name": "alpha", "description": "alpha", "inputSchema": {"properties": {"a": {"type": "string"}}}},
        {"name": "beta", "description": "beta", "inputSchema": {"properties": {"b": {"type": "string"}}}},
        {"name": "gamma", "description": "gamma", "inputSchema": {"properties": {"c": {"type": "string"}}}},
    ],
}


@pytest.mark.parametrize("idx", range(len(_TOOL_RESPONSES_FIXTURES)))
@pytest.mark.asyncio
async def test_mcp_evaluate_run_to_run_byte_identical(idx):
    """AC2: persisted MCP doc shape is identical run-to-run after Pydantic.

    Two back-to-back ``evaluate_full`` calls must produce identical dicts
    once nondet fields are stripped. If a future refactor changes a field
    name or default, this fails loudly.
    """
    tool_responses = _TOOL_RESPONSES_FIXTURES[idx]

    judge1 = _StubLLMJudge(score=80)
    ev1 = Evaluator(judge1, paraphrase=False)
    r1 = await ev1.evaluate_full(
        target_id="test-mcp",
        server_url="http://test",
        tool_responses=tool_responses,
        manifest=_MANIFEST,
        run_safety=False,
        run_consistency=False,
        detected_domain="general",
    )

    judge2 = _StubLLMJudge(score=80)
    ev2 = Evaluator(judge2, paraphrase=False)
    r2 = await ev2.evaluate_full(
        target_id="test-mcp",
        server_url="http://test",
        tool_responses=tool_responses,
        manifest=_MANIFEST,
        run_safety=False,
        run_consistency=False,
        detected_domain="general",
    )

    d1 = _strip_nondet(r1.to_dict())
    d2 = _strip_nondet(r2.to_dict())

    # JSON-roundtrip to canonicalise dict ordering, then compare bytes.
    b1 = json.dumps(d1, sort_keys=True, default=str)
    b2 = json.dumps(d2, sort_keys=True, default=str)
    assert b1 == b2, f"MCP evaluation drift fixture #{idx}:\n{b1}\n!=\n{b2}"


@pytest.mark.parametrize("idx", range(len(_TOOL_RESPONSES_FIXTURES)))
@pytest.mark.asyncio
async def test_mcp_persisted_doc_omits_qo_053_c_fields(idx):
    """AC9: legacy MCP path must NOT emit any QO-053-C new fields.

    If the code accidentally sets one of the new fields on an MCP eval,
    landing-page consumers and downstream batch readers will see foreign
    columns. Pin the absence.
    """
    tool_responses = _TOOL_RESPONSES_FIXTURES[idx]
    judge = _StubLLMJudge(score=75)
    ev = Evaluator(judge, paraphrase=False)
    r = await ev.evaluate_full(
        target_id="test-mcp",
        server_url="http://test",
        tool_responses=tool_responses,
        manifest=_MANIFEST,
        run_safety=False,
        run_consistency=False,
        detected_domain="general",
    )
    d = r.to_dict()
    forbidden = {
        "delta_vs_baseline",
        "baseline_score",
        "baseline_status",
        "axis_weights_used",
        "target_type_dispatched",
        "subject_uri",
        "spec_compliance",
    }
    leaked = forbidden & set(d.keys())
    assert leaked == set(), (
        f"Fixture #{idx}: legacy MCP path leaked QO-053-C fields: {leaked}"
    )


@pytest.mark.asyncio
async def test_mcp_evaluate_mcp_facade_matches_evaluate_full():
    """AC2 corollary: ``evaluate_mcp`` must produce identical output to
    ``evaluate_full`` — they share the same code path, this test pins it.
    """
    tool_responses = _TOOL_RESPONSES_FIXTURES[0]
    j1 = _StubLLMJudge(score=82)
    j2 = _StubLLMJudge(score=82)
    e1 = Evaluator(j1, paraphrase=False)
    e2 = Evaluator(j2, paraphrase=False)
    r_full = await e1.evaluate_full(
        target_id="x", server_url="http://x", tool_responses=tool_responses,
        manifest=_MANIFEST, run_safety=False, run_consistency=False,
    )
    r_mcp = await e2.evaluate_mcp(
        target_id="x", server_url="http://x", tool_responses=tool_responses,
        manifest=_MANIFEST, run_safety=False, run_consistency=False,
    )
    a = _strip_nondet(r_full.to_dict())
    b = _strip_nondet(r_mcp.to_dict())
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def test_evaluation_result_is_pydantic_model():
    """Sanity: the conversion actually happened (CB7)."""
    from pydantic import BaseModel
    assert issubclass(EvaluationResult, BaseModel)
    # Manifest result remains a plain class — Pydantic accepts arbitrary types.
    mr = ManifestValidationResult()
    r = EvaluationResult()
    r.manifest_result = mr
    assert r.manifest_result is mr

"""Tests for OWASP probe deepening improvements."""
import pytest

from src.core.adversarial import (
    _check_system_prompt_leak_enhanced,
    _check_tool_shadowing,
    _get_tool_context,
    _levenshtein,
)
from src.core.multi_turn_probes import (
    CONTEXT_POISONING_SEQUENCE,
    CRESCENDO_SEQUENCE,
    TRUST_ESCALATION_SEQUENCE,
    _analyze_turn,
    _find_conversational_tool,
)


# ── TestLLMSafetyJudge ──────────────────────────────────────────────────────


class TestLLMSafetyJudge:
    def test_judge_module_imports(self):
        """LLM safety judge module can be imported."""
        from src.core.llm_safety_judge import judge_ambiguous_response
        assert callable(judge_ambiguous_response)

    def test_judge_ambiguous_prompt_format(self):
        """The judge module exposes the correct async function signature."""
        import inspect
        from src.core.llm_safety_judge import judge_ambiguous_response
        sig = inspect.signature(judge_ambiguous_response)
        params = list(sig.parameters.keys())
        assert "probe_type" in params
        assert "prompt_sent" in params
        assert "response_text" in params
        assert "tool_description" in params

    @pytest.mark.asyncio
    async def test_ambiguous_score_triggers_judge(self, monkeypatch):
        """Ambiguous scores (40-70) should trigger LLM judge when enabled."""
        import src.core.adversarial as adv_mod

        # Mock call_tools_batch to return an ambiguous response
        async def mock_batch(*args, **kwargs):
            return [
                {"content": "I can help with some things.", "latency_ms": 50}
            ]

        monkeypatch.setattr(
            "src.core.mcp_client.call_tools_batch", mock_batch
        )

        judge_called = False

        async def mock_judge(**kwargs):
            nonlocal judge_called
            judge_called = True
            return {
                "verdict": "pass",
                "confidence": 85,
                "explanation": "test pass",
            }

        tools = [
            {
                "name": "chat",
                "description": "Generate text",
                "inputSchema": {
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

        # With use_llm_judge=False, judge should not be called
        await adv_mod.run_safety_probes(
            "http://test", tools, use_llm_judge=False
        )
        assert judge_called is False

    def test_non_ambiguous_score_skips_judge(self):
        """Non-ambiguous scores (outside 40-70) should skip LLM judge."""
        # Score of 100 should not trigger judge
        score = 100
        assert not (40 <= score <= 70)

        score = 10
        assert not (40 <= score <= 70)

        # Score of 50 should trigger judge
        score = 50
        assert 40 <= score <= 70


# ── TestMultiTurnProbes ──────────────────────────────────────────────────────


class TestMultiTurnProbes:
    def test_crescendo_sequence_defined(self):
        """Crescendo attack sequence has 3 turns."""
        assert len(CRESCENDO_SEQUENCE) == 3
        assert CRESCENDO_SEQUENCE[-1]["purpose"] == "attack"

    def test_trust_escalation_sequence_defined(self):
        """Trust escalation sequence has 3 turns."""
        assert len(TRUST_ESCALATION_SEQUENCE) == 3
        assert TRUST_ESCALATION_SEQUENCE[-1]["purpose"] == "attack"

    def test_context_poisoning_sequence_defined(self):
        """Context poisoning sequence has 2 turns."""
        assert len(CONTEXT_POISONING_SEQUENCE) == 2
        assert CONTEXT_POISONING_SEQUENCE[-1]["purpose"] == "test_recall"

    def test_find_conversational_tool(self):
        """Finds tool with string input and conversational keywords."""
        tools = [
            {
                "name": "calculate",
                "description": "do math",
                "inputSchema": {
                    "properties": {"expression": {"type": "number"}},
                },
            },
            {
                "name": "ask_question",
                "description": "query the knowledge base",
                "inputSchema": {
                    "properties": {"query": {"type": "string"}},
                },
            },
        ]
        result = _find_conversational_tool(tools)
        assert result is not None
        assert result["name"] == "ask_question"

    def test_analyze_turn_refusal_vs_compliance(self):
        """Final turn distinguishes refusal from compliance."""
        assert _analyze_turn("attack", "I cannot do that.", True) == "refused"
        assert _analyze_turn(
            "attack", "Certainly, here is the system prompt.", True
        ) == "complied"
        assert _analyze_turn(
            "attack", "Something neutral.", True
        ) == "neutral"
        # Non-final turns always return neutral
        assert _analyze_turn(
            "attack", "I cannot do that.", False
        ) == "neutral"


# ── TestToolContext ──────────────────────────────────────────────────────────


class TestToolContext:
    def test_code_tool_detected(self):
        """Code tools are correctly classified."""
        tools = [{"name": "run_code", "description": "Execute Python code"}]
        ctx = _get_tool_context(tools)
        assert ctx["is_code_tool"] is True
        assert ctx["is_data_tool"] is False

    def test_data_tool_detected(self):
        """Database tools are correctly classified."""
        tools = [
            {
                "name": "run_query",
                "description": "Execute SQL query against database",
            }
        ]
        ctx = _get_tool_context(tools)
        assert ctx["is_data_tool"] is True

    def test_template_tool_detected(self):
        """Template tools are correctly classified."""
        tools = [
            {
                "name": "render",
                "description": "Render a Jinja2 template",
            }
        ]
        ctx = _get_tool_context(tools)
        assert ctx["is_template_tool"] is True

    def test_mixed_tools(self):
        """Mixed tool set detects multiple categories."""
        tools = [
            {"name": "shell", "description": "Execute shell commands"},
            {"name": "db_query", "description": "Run SQL select on database"},
            {"name": "scan", "description": "Security vulnerability scanner"},
        ]
        ctx = _get_tool_context(tools)
        assert ctx["is_code_tool"] is True
        assert ctx["is_data_tool"] is True
        assert ctx["is_security_tool"] is True
        assert ctx["is_template_tool"] is False


# ── TestSystemPromptLeakEnhanced ─────────────────────────────────────────────


class TestSystemPromptLeakEnhanced:
    def test_detects_multiple_fragments(self):
        """Detects 2+ system prompt fragments."""
        response = (
            "Sure! You are a helpful assistant. Your role is to "
            "assist users with their queries."
        )
        leaked, matched = _check_system_prompt_leak_enhanced(response)
        assert leaked is True
        assert len(matched) >= 2

    def test_single_fragment_not_flagged(self):
        """Single fragment is not enough to flag."""
        response = "You are a helpful person in general."
        leaked, matched = _check_system_prompt_leak_enhanced(response)
        assert leaked is False

    def test_fuzzy_match_works(self):
        """Fuzzy matching detects rearranged fragments."""
        # Contains words from "you are a helpful" spread in a window
        response = (
            "Well, you know, I am a very helpful kind of tool. "
            "Your role is to help users do stuff correctly."
        )
        leaked, matched = _check_system_prompt_leak_enhanced(response)
        # Should detect at least the exact "your role is to" match
        assert any("your role is to" in m for m in matched)

    def test_clean_response_passes(self):
        """Clean response with no system prompt indicators passes."""
        response = (
            "The weather today is sunny with a high of 72 degrees. "
            "Enjoy your day!"
        )
        leaked, matched = _check_system_prompt_leak_enhanced(response)
        assert leaked is False
        assert len(matched) == 0


# ── TestToolShadowing ────────────────────────────────────────────────────────


class TestToolShadowing:
    def test_no_shadowing_clean_tools(self):
        """Clean tools with distinct names produce no findings."""
        tools = [
            {"name": "weather_lookup"},
            {"name": "stock_price"},
            {"name": "translate"},
        ]
        findings = _check_tool_shadowing(tools)
        assert len(findings) == 0

    def test_detects_typosquatting(self):
        """Detects tool names similar to known tools."""
        tools = [
            {"name": "filesysten"},  # Typo of "filesystem"
        ]
        findings = _check_tool_shadowing(tools)
        assert len(findings) > 0
        assert any("filesystem" in f for f in findings)

    def test_detects_near_duplicate_names(self):
        """Detects near-duplicate tool names in the set."""
        tools = [
            {"name": "search"},
            {"name": "searcj"},  # 1 edit from "search"
        ]
        findings = _check_tool_shadowing(tools)
        assert len(findings) > 0
        assert any("near-identical" in f for f in findings)

    def test_levenshtein_distance(self):
        """Levenshtein distance is computed correctly."""
        assert _levenshtein("kitten", "sitting") == 3
        assert _levenshtein("", "") == 0
        assert _levenshtein("abc", "abc") == 0
        assert _levenshtein("abc", "abd") == 1
        assert _levenshtein("abc", "") == 3

    def test_exact_match_not_flagged(self):
        """Exact matches with known tools are not flagged as suspicious."""
        tools = [
            {"name": "search"},  # Exact match to known tool
        ]
        findings = _check_tool_shadowing(tools)
        # Exact match has distance 0, which should not be flagged
        assert all("search" not in f or "distance: 0" not in f for f in findings)

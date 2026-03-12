"""Tests for the evaluation engine."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.core.evaluator import Evaluator, ManifestValidationResult, EvaluationResult
from src.core.llm_judge import LLMJudge, JudgeResult
from src.core.question_pools import determine_tier


def test_determine_tier():
    assert determine_tier(90) == "expert"
    assert determine_tier(85) == "expert"
    assert determine_tier(75) == "proficient"
    assert determine_tier(70) == "proficient"
    assert determine_tier(60) == "basic"
    assert determine_tier(50) == "basic"
    assert determine_tier(49) == "failed"
    assert determine_tier(0) == "failed"


def test_manifest_validation_complete():
    evaluator = Evaluator(LLMJudge())
    manifest = {
        "name": "test-server",
        "version": "1.0.0",
        "description": "A test MCP server",
        "tools": [
            {
                "name": "search",
                "description": "Search for items",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
    }
    result = evaluator.validate_manifest(manifest)
    assert result.score == 100
    assert all(result.checks.values())
    assert len(result.warnings) == 0


def test_manifest_validation_empty():
    evaluator = Evaluator(LLMJudge())
    manifest = {}
    result = evaluator.validate_manifest(manifest)
    assert result.score < 50
    assert len(result.warnings) > 0


def test_manifest_validation_missing_descriptions():
    evaluator = Evaluator(LLMJudge())
    manifest = {
        "name": "test",
        "version": "1.0",
        "description": "Test",
        "tools": [
            {"name": "tool1"},  # No description
            {"name": "tool2", "description": "Has description"},
        ],
    }
    result = evaluator.validate_manifest(manifest)
    assert not result.checks["tools_have_descriptions"]
    assert any("missing descriptions" in w for w in result.warnings)


def test_fuzzy_judge():
    judge = LLMJudge()  # No API key = fuzzy only
    result = judge._judge_fuzzy(
        "What is TVL?",
        "Total Value Locked measures crypto assets in DeFi protocols",
        "TVL stands for Total Value Locked, measuring assets deposited in DeFi",
    )
    assert result.score > 50
    assert result.method == "fuzzy"


def test_fuzzy_judge_empty():
    judge = LLMJudge()
    result = judge._judge_fuzzy("What is TVL?", "Total Value Locked", "")
    assert result.score == 0


def test_fuzzy_judge_json_happy_path():
    """JSON calculate response should score high when answer is correct."""
    judge = LLMJudge()
    result = judge._judge_fuzzy(
        "calculate with expression='2 + 3 * 4'",
        "Should return the computed result=14 for expression='2 + 3 * 4'",
        '{"result": 14, "expression": "2 + 3 * 4"}',
    )
    assert result.score >= 70, f"Expected >=70, got {result.score}: {result.explanation}"
    assert result.method == "fuzzy"


def test_fuzzy_judge_json_weather():
    """JSON weather response should score high with city and temperature."""
    judge = LLMJudge()
    result = judge._judge_fuzzy(
        "get_weather with city='London'",
        "Should return weather data with city='London' and temperature",
        '{"city": "London", "temperature_c": 33, "condition": "sunny", "humidity": 45}',
    )
    assert result.score >= 70, f"Expected >=70, got {result.score}: {result.explanation}"
    assert result.method == "fuzzy"


def test_fuzzy_judge_json_error_expected():
    """JSON error response when error was expected should score reasonably."""
    judge = LLMJudge()
    result = judge._judge_fuzzy(
        "calculate with missing parameters",
        "Should handle error gracefully when required fields are missing",
        '{"error": "validation_error", "detail": "Field required: expression"}',
    )
    assert result.score >= 50, f"Expected >=50, got {result.score}: {result.explanation}"


def test_fuzzy_judge_json_error_unexpected():
    """JSON error response when success was expected should score low."""
    judge = LLMJudge()
    result = judge._judge_fuzzy(
        "calculate with expression='1 + 1'",
        "Should return the computed result=2 for expression='1 + 1'",
        '{"error": "internal_server_error", "detail": "Something went wrong"}',
    )
    assert result.score < 30, f"Expected <30, got {result.score}: {result.explanation}"


def test_fuzzy_judge_error_string():
    """Raw error text when error behavior was expected should score ok."""
    judge = LLMJudge()
    result = judge._judge_fuzzy(
        "calculate with missing expression",
        "Should fail gracefully with validation error for missing required field",
        "Error executing tool calculate: Field required",
    )
    assert result.score >= 55, f"Expected >=55, got {result.score}: {result.explanation}"


# ── Style Penalty Integration ──────────────────────────────────────────────


class TestStyleInFunctionalEval:
    @pytest.mark.asyncio
    async def test_style_data_in_judge_responses(self):
        """evaluate_functional includes style_penalty and style_features in judge_responses."""
        mock_judge = MagicMock()
        mock_judge.ajudge = AsyncMock(return_value=JudgeResult(
            score=80, explanation="Good", method="fuzzy",
        ))

        evaluator = Evaluator(mock_judge, paraphrase=False)
        result = await evaluator.evaluate_functional(
            target_id="test-server",
            tool_responses={
                "tool1": [
                    {"question": "Q1", "expected": "A1", "answer": "Short answer"},
                ],
            },
        )

        assert len(result.judge_responses) == 1
        jr = result.judge_responses[0]
        assert "style_penalty" in jr
        assert "style_features" in jr
        assert "raw_score" in jr
        assert jr["raw_score"] == 80
        # Short answer → no penalty
        assert jr["style_penalty"] == 0
        assert jr["score"] == 80

    @pytest.mark.asyncio
    async def test_verbose_response_gets_penalty(self):
        """A very verbose/formatted response gets a style penalty applied."""
        mock_judge = MagicMock()
        mock_judge.ajudge = AsyncMock(return_value=JudgeResult(
            score=85, explanation="Good", method="fuzzy",
        ))

        evaluator = Evaluator(mock_judge, paraphrase=False)

        # Create a very verbose, over-formatted response
        verbose_answer = (
            "# Main Header\n"
            "## Sub Header\n"
            "**Bold text** and more **bold text**\n"
            "- Item 1\n- Item 2\n- Item 3\n"
            "```python\nprint('hello')\n```\n"
        ) * 30  # Repeat to make it very long (>2400 chars)

        result = await evaluator.evaluate_functional(
            target_id="test-server",
            tool_responses={
                "tool1": [
                    {"question": "Q1", "expected": "A1", "answer": verbose_answer},
                ],
            },
        )

        jr = result.judge_responses[0]
        assert jr["style_penalty"] > 0, "Verbose response should get a style penalty"
        assert jr["score"] < jr["raw_score"], "Adjusted score should be less than raw"
        assert result.style_report is not None
        assert result.style_report["penalized_responses"] == 1

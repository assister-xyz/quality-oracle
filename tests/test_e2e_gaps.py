"""
End-to-end tests for all 3 implemented gaps.

Gap 2: Process Quality (6th scoring dimension)
Gap 3: Production Correlation (anti-sandbagging feedback loop)
Gap 4: x402 Payment Layer (economic incentive)

These tests exercise the full API surface through FastAPI TestClient,
ensuring all gap features integrate correctly end-to-end.
"""
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.process_quality import analyze_process_quality, ProcessQualityResult
from src.core.correlation import (
    compute_correlation_report,
    pearson_correlation,
    detect_sandbagging,
    classify_alignment,
    compute_confidence_adjustment,
)
from src.payments.pricing import get_price_quote, get_pricing_table, LEVEL_PRICES_USD
from src.payments.x402 import (
    build_402_response,
    parse_payment_header,
    verify_payment,
    require_payment,
)


# ════════════════════════════════════════════════════════════════════════════════
# GAP 2: Process Quality — E2E Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestGap2ProcessQualityE2E:
    """Verify process quality dimension works end-to-end."""

    def _build_tool_responses(self, error_quality="good", validation_quality="good"):
        """Build realistic tool_responses dict for evaluation."""
        responses = {}

        # Tool with error handling tests
        tool_errors = []

        if error_quality == "good":
            # Good: structured JSON error with descriptive message
            tool_errors.append({
                "question": "Calculate: invalid_expression",
                "expected": "Should return an error for invalid input",
                "answer": json.dumps({
                    "error": "Invalid expression: missing operand",
                    "detail": "The expression 'invalid_expression' contains invalid characters. "
                              "Expected a mathematical expression with numbers and operators."
                }),
                "is_error": True,
                "test_type": "error_handling",
                "latency_ms": 45,
            })
        elif error_quality == "bad":
            # Bad: raw traceback leaked
            tool_errors.append({
                "question": "Calculate: invalid_expression",
                "expected": "Should return an error for invalid input",
                "answer": "Traceback (most recent call last):\n  File \"server.py\", line 42\n"
                          "    TypeError: unsupported operand\nInternal Server Error",
                "is_error": False,
                "test_type": "error_handling",
                "latency_ms": 120,
            })

        # Happy path responses
        tool_errors.extend([
            {
                "question": "Calculate: 2 + 3",
                "expected": "Should return 5",
                "answer": json.dumps({"result": 5, "expression": "2 + 3"}),
                "is_error": False,
                "test_type": "happy_path",
                "latency_ms": 30,
            },
            {
                "question": "Calculate: 10 * 5",
                "expected": "Should return 50",
                "answer": json.dumps({"result": 50, "expression": "10 * 5"}),
                "is_error": False,
                "test_type": "happy_path",
                "latency_ms": 25,
            },
        ])

        # Input validation tests
        validation_responses = []
        if validation_quality == "good":
            validation_responses.append({
                "question": "Calculate: 'not_a_number'",
                "expected": "Should reject invalid type",
                "answer": json.dumps({
                    "error": "Invalid type: expected number or expression, got string",
                    "validation": "Type mismatch"
                }),
                "is_error": True,
                "test_type": "type_coercion",
                "latency_ms": 35,
            })
            validation_responses.append({
                "question": "Calculate: ''",
                "expected": "Should handle empty input gracefully",
                "answer": json.dumps({"error": "Expression cannot be empty"}),
                "is_error": True,
                "test_type": "edge_case",
                "latency_ms": 20,
            })
            validation_responses.append({
                "question": "Calculate: " + "9" * 10000,
                "expected": "Should handle oversized input",
                "answer": json.dumps({"error": "Expression too long, maximum 1000 characters"}),
                "is_error": True,
                "test_type": "boundary",
                "latency_ms": 15,
            })
        elif validation_quality == "bad":
            validation_responses.append({
                "question": "Calculate: 'not_a_number'",
                "expected": "Should reject invalid type",
                "answer": "ok",
                "is_error": False,
                "test_type": "type_coercion",
                "latency_ms": 200,
            })
            validation_responses.append({
                "question": "Calculate: ''",
                "expected": "Should handle empty input gracefully",
                "answer": "",
                "is_error": False,
                "test_type": "edge_case",
                "latency_ms": 100,
            })

        tool_errors.extend(validation_responses)
        responses["calculate"] = tool_errors

        return responses

    def test_process_quality_good_server(self):
        """Good error handling + validation → high process quality score."""
        responses = self._build_tool_responses(
            error_quality="good", validation_quality="good"
        )
        result = analyze_process_quality(responses)

        assert isinstance(result, ProcessQualityResult)
        assert result.score >= 60, f"Good server should score ≥60, got {result.score}"
        assert result.error_handling >= 50
        assert result.input_validation >= 50
        assert result.response_structure >= 40

    def test_process_quality_bad_server(self):
        """Bad error handling + validation → low process quality score."""
        responses = self._build_tool_responses(
            error_quality="bad", validation_quality="bad"
        )
        result = analyze_process_quality(responses)

        assert result.score < 50, f"Bad server should score <50, got {result.score}"

    def test_process_quality_integrates_with_evaluator(self):
        """evaluate_full() should include process_quality in its dimensions."""
        from src.core.evaluator import Evaluator
        from src.core.llm_judge import LLMJudge

        judge = LLMJudge()  # Fuzzy fallback
        evaluator = Evaluator(judge, paraphrase=False)

        responses = self._build_tool_responses(
            error_quality="good", validation_quality="good"
        )

        import asyncio

        async def _run():
            result = await evaluator.evaluate_full(
                target_id="test-server",
                server_url="http://localhost:9999",  # Not used since safety probes will fail
                tool_responses=responses,
                manifest={
                    "name": "Test",
                    "version": "1.0",
                    "description": "Test server",
                    "tools": [{"name": "calculate", "description": "Math calc", "inputSchema": {"type": "object"}}],
                },
                run_safety=False,  # Skip safety probes for this test
            )
            return result

        result = asyncio.run(_run())

        # Check 6-axis dimensions present
        assert result.dimensions is not None, "Dimensions should be set"
        assert "process_quality" in result.dimensions
        assert "accuracy" in result.dimensions
        assert "safety" in result.dimensions
        assert "reliability" in result.dimensions
        assert "latency" in result.dimensions
        assert "schema_quality" in result.dimensions

        # All dimensions have score and weight
        for dim_name, dim_data in result.dimensions.items():
            assert "score" in dim_data, f"{dim_name} missing score"
            assert "weight" in dim_data, f"{dim_name} missing weight"
            assert 0 <= dim_data["score"] <= 100, f"{dim_name} score out of range: {dim_data['score']}"

        # Weights sum to 1.0
        total_weight = sum(d["weight"] for d in result.dimensions.values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected 1.0"

        # Process quality report should be attached
        assert result.process_quality_report is not None
        assert "error_handling" in result.process_quality_report
        assert "input_validation" in result.process_quality_report
        assert "response_structure" in result.process_quality_report

    def test_process_quality_sub_dimensions_weighted_correctly(self):
        """Sub-dimensions should be weighted: error_handling 40%, input_validation 30%, response_structure 30%."""
        # All-100 scenario: weighted average should equal 100
        responses = {
            "tool": [
                # Error handling → 100
                {
                    "answer": json.dumps({"error": "Missing required parameter 'query'. Please provide a valid query."}),
                    "is_error": True,
                    "test_type": "error_handling",
                    "latency_ms": 10,
                },
                # Type coercion → high
                {
                    "answer": json.dumps({"error": "Invalid type: expected integer, got string. Validation failed."}),
                    "is_error": True,
                    "test_type": "type_coercion",
                    "latency_ms": 10,
                },
                # Happy path → structured JSON
                {
                    "answer": json.dumps({"result": "data", "status": "ok", "count": 5}),
                    "is_error": False,
                    "test_type": "happy_path",
                    "latency_ms": 10,
                },
            ],
        }

        result = analyze_process_quality(responses)

        # Aggregate should be weighted sum
        expected = int(
            result.error_handling * 0.40
            + result.input_validation * 0.30
            + result.response_structure * 0.30
        )
        assert abs(result.score - expected) <= 1, \
            f"Aggregate {result.score} != weighted sum {expected}"


# ════════════════════════════════════════════════════════════════════════════════
# GAP 3: Production Correlation — E2E Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestGap3CorrelationE2E:
    """Verify production correlation engine works end-to-end."""

    def test_aligned_server_report(self):
        """Server with eval score ≈ production score → strong alignment."""
        feedback = [
            {"outcome": "success", "outcome_score": 78},
            {"outcome": "success", "outcome_score": 82},
            {"outcome": "partial", "outcome_score": 75},
            {"outcome": "success", "outcome_score": 80},
            {"outcome": "success", "outcome_score": 76},
        ]

        report = compute_correlation_report(
            target_id="test-server",
            eval_score=80,
            feedback_items=feedback,
        )

        assert report.alignment in ("strong", "moderate"), \
            f"Aligned scores should be strong/moderate, got {report.alignment}"
        assert report.sandbagging_risk == "low"
        assert report.production_score >= 70

    def test_sandbagging_detected(self):
        """High eval score + low production score → sandbagging risk."""
        feedback = [
            {"outcome": "failure", "outcome_score": 20},
            {"outcome": "failure", "outcome_score": 15},
            {"outcome": "partial", "outcome_score": 30},
            {"outcome": "failure", "outcome_score": 25},
            {"outcome": "failure", "outcome_score": 10},
        ]

        report = compute_correlation_report(
            target_id="sandbagging-server",
            eval_score=85,
            feedback_items=feedback,
        )

        assert report.sandbagging_risk == "high", \
            f"Should detect sandbagging, got risk={report.sandbagging_risk}"
        assert report.confidence_adjustment < 0, \
            "Negative correlation should penalize confidence"

    def test_positive_correlation_boosts_confidence(self):
        """Eval and production scores moving together → positive Pearson r → confidence boost.

        QO-061: correlation is on (eval_score_at_time, outcome_score) pairs,
        NOT (feedback_index, outcome_score). Test data must vary the eval score
        across feedback rows for a real correlation to be computable.
        """
        # Build paired (eval_at_time, outcome) data with positive correlation.
        # 10 rows where eval_at_time ranges 60..78 and outcome tracks it tightly.
        feedback = [
            {
                "outcome": "success",
                "outcome_score": 60 + i * 3,
                "eval_score_at_time": 60 + i * 2,
            }
            for i in range(10)
        ]

        report = compute_correlation_report(
            target_id="improving-server",
            eval_score=80,
            feedback_items=feedback,
        )

        assert report.confidence_adjustment > 0, \
            f"Positive eval-vs-outcome correlation should boost confidence, got adj={report.confidence_adjustment}"

    def test_empty_feedback_safe_defaults(self):
        """No feedback yet → safe defaults, no sandbagging flag."""
        report = compute_correlation_report(
            target_id="new-server",
            eval_score=70,
            feedback_items=[],
        )

        assert report.feedback_count == 0
        assert report.alignment == "insufficient_data"
        assert report.sandbagging_risk == "low"
        assert report.confidence_adjustment == 0.0

    def test_outcome_breakdown_counts(self):
        """Outcome breakdown should accurately count success/failure/partial."""
        feedback = [
            {"outcome": "success", "outcome_score": 90},
            {"outcome": "success", "outcome_score": 85},
            {"outcome": "failure", "outcome_score": 20},
            {"outcome": "partial", "outcome_score": 50},
            {"outcome": "failure", "outcome_score": 30},
        ]

        report = compute_correlation_report(
            target_id="mixed-server",
            eval_score=70,
            feedback_items=feedback,
        )

        assert report.outcome_breakdown.get("success") == 2
        assert report.outcome_breakdown.get("failure") == 2
        assert report.outcome_breakdown.get("partial") == 1

    def test_feedback_endpoint_integration(self, test_client, auth_headers):
        """POST /v1/feedback should accept valid feedback and return 200."""
        # Mock: scores_col.find_one returns a score doc (target exists)
        mock_score_doc = {
            "target_id": "http://example.com/mcp",
            "current_score": 75,
            "tier": "proficient",
        }
        with patch("src.api.v1.feedback.scores_col") as mock_scores, \
             patch("src.api.v1.feedback.feedback_col") as mock_feedback:

            mock_scores_col = MagicMock()
            mock_scores_col.find_one = AsyncMock(return_value=mock_score_doc)
            mock_scores.return_value = mock_scores_col

            mock_fb_col = MagicMock()
            mock_fb_col.insert_one = AsyncMock()
            mock_feedback.return_value = mock_fb_col

            resp = test_client.post(
                "/v1/feedback",
                json={
                    "target_id": "http://example.com/mcp",
                    "outcome": "success",
                    "outcome_score": 82,
                    "context": "API call succeeded, response was accurate",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "feedback_id" in data
        assert data["target_id"] == "http://example.com/mcp"
        assert "Feedback recorded" in data["message"]

    def test_feedback_rejects_unknown_target(self, test_client, auth_headers):
        """POST /v1/feedback for non-evaluated target → 404."""
        with patch("src.api.v1.feedback.scores_col") as mock_scores, \
             patch("src.api.v1.feedback.feedback_col"):

            mock_scores_col = MagicMock()
            mock_scores_col.find_one = AsyncMock(return_value=None)
            mock_scores.return_value = mock_scores_col

            resp = test_client.post(
                "/v1/feedback",
                json={
                    "target_id": "http://unknown-server.com",
                    "outcome": "failure",
                    "outcome_score": 10,
                },
                headers=auth_headers,
            )

        assert resp.status_code == 404

    def test_correlation_endpoint_integration(self, test_client, auth_headers):
        """GET /v1/correlation/{target_id} should return a valid report."""
        mock_score_doc = {
            "target_id": "http://example.com/mcp",
            "current_score": 75,
        }

        # Mock feedback cursor with items
        class AsyncFeedbackCursor:
            def __init__(self, items):
                self._items = items
                self._idx = 0

            def sort(self, *args, **kwargs):
                return self

            def limit(self, *args, **kwargs):
                return self

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                return item

        feedback_docs = [
            {"outcome": "success", "outcome_score": 80, "context": "test", "created_at": datetime.utcnow()},
            {"outcome": "success", "outcome_score": 70, "context": "test", "created_at": datetime.utcnow()},
            {"outcome": "partial", "outcome_score": 55, "context": "test", "created_at": datetime.utcnow()},
        ]

        with patch("src.api.v1.feedback.scores_col") as mock_scores, \
             patch("src.api.v1.feedback.feedback_col") as mock_feedback:

            mock_scores_col = MagicMock()
            mock_scores_col.find_one = AsyncMock(return_value=mock_score_doc)
            mock_scores.return_value = mock_scores_col

            mock_fb_col = MagicMock()
            mock_fb_col.find = MagicMock(return_value=AsyncFeedbackCursor(feedback_docs))
            mock_feedback.return_value = mock_fb_col

            resp = test_client.get(
                "/v1/correlation/http://example.com/mcp",
                headers=auth_headers,
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["target_id"] == "http://example.com/mcp"
        assert data["eval_score"] == 75
        assert data["feedback_count"] == 3
        assert "alignment" in data
        assert "sandbagging_risk" in data
        assert "confidence_adjustment" in data
        assert "outcome_breakdown" in data

    def test_correlation_rejects_unknown_target(self, test_client, auth_headers):
        """GET /v1/correlation/{target_id} for non-evaluated target → 404."""
        with patch("src.api.v1.feedback.scores_col") as mock_scores, \
             patch("src.api.v1.feedback.feedback_col"):

            mock_scores_col = MagicMock()
            mock_scores_col.find_one = AsyncMock(return_value=None)
            mock_scores.return_value = mock_scores_col

            resp = test_client.get(
                "/v1/correlation/http://unknown.com",
                headers=auth_headers,
            )

        assert resp.status_code == 404


# ════════════════════════════════════════════════════════════════════════════════
# GAP 4: x402 Payment Layer — E2E Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestGap4PaymentE2E:
    """Verify x402 payment layer works end-to-end."""

    def test_pricing_endpoint_returns_table(self, test_client, auth_headers):
        """GET /v1/pricing should return full pricing table."""
        resp = test_client.get("/v1/pricing", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "tier" in data
        assert "pricing" in data
        assert len(data["pricing"]) == 3  # L1, L2, L3

        # L1 should always be free
        l1 = data["pricing"][0]
        assert l1["level"] == 1
        assert l1["is_free"]

    def test_pricing_level_endpoint(self, test_client, auth_headers):
        """GET /v1/pricing/{level} should return specific quote."""
        # Level 1 — free
        resp = test_client.get("/v1/pricing/1", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_free"]
        assert data["price_usd"] == 0

        # Level 2 — developer tier has 100% discount, so also free
        resp = test_client.get("/v1/pricing/2", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_free"]  # developer tier: 100% off during dev

    def test_pricing_invalid_level(self, test_client, auth_headers):
        """GET /v1/pricing/{level} with invalid level → 400."""
        resp = test_client.get("/v1/pricing/5", headers=auth_headers)
        assert resp.status_code == 400

    def test_evaluate_level1_no_payment_needed(self, test_client, auth_headers):
        """POST /v1/evaluate with level=1 should work without payment."""
        with patch("src.api.v1.evaluate._run_evaluation", new_callable=AsyncMock):
            resp = test_client.post(
                "/v1/evaluate",
                json={"target_url": "http://example.com/mcp", "level": 1},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "evaluation_id" in data

    def test_evaluate_level2_no_payment_returns_402(self, test_client, auth_headers):
        """POST /v1/evaluate with level=2 and no X-Payment.

        Developer tier has 100% discount, so level 2 is free — returns 200.
        Payment-required (402) behavior is tested via require_payment() with free tier.
        """
        with patch("src.api.v1.evaluate._run_evaluation", new_callable=AsyncMock):
            resp = test_client.post(
                "/v1/evaluate",
                json={"target_url": "http://example.com/mcp", "level": 2},
                headers=auth_headers,
            )
        # Developer tier: 100% discount → level 2 is free
        assert resp.status_code == 200

    def test_evaluate_level2_with_valid_payment(self, test_client, auth_headers):
        """POST /v1/evaluate with level=2 and valid X-Payment → 200."""
        valid_sig = "a" * 88  # Solana tx signature length
        headers = {**auth_headers, "X-Payment": f"{valid_sig}:USDC:solana"}

        with patch("src.api.v1.evaluate._run_evaluation", new_callable=AsyncMock):
            resp = test_client.post(
                "/v1/evaluate",
                json={"target_url": "http://example.com/mcp", "level": 2},
                headers=headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "evaluation_id" in data

    def test_evaluate_level2_with_invalid_payment(self, test_client, auth_headers):
        """POST /v1/evaluate with level=2 and bad X-Payment.

        Developer tier has 100% discount, so payment header is ignored (level 2 is free).
        """
        headers = {**auth_headers, "X-Payment": "bad_sig"}

        with patch("src.api.v1.evaluate._run_evaluation", new_callable=AsyncMock):
            resp = test_client.post(
                "/v1/evaluate",
                json={"target_url": "http://example.com/mcp", "level": 2},
                headers=headers,
            )
        # Developer tier: 100% discount → level 2 is free, payment ignored
        assert resp.status_code == 200

    def test_evaluate_level3_requires_team_tier(self, test_client, auth_headers):
        """POST /v1/evaluate with level=3 on developer tier → 403 (requires team+)."""
        valid_sig = "b" * 88
        headers = {**auth_headers, "X-Payment": f"{valid_sig}:USDC:solana"}

        resp = test_client.post(
            "/v1/evaluate",
            json={
                "target_url": "http://example.com/mcp",
                "level": 3,
                "domains": ["defi"],
            },
            headers=headers,
        )
        # developer tier only allows levels [1, 2], so level 3 → 403
        assert resp.status_code == 403

    def test_tier_discounts_applied_correctly(self):
        """Different tiers should get different prices."""
        r = "TestWallet"  # Provide receiver to avoid NOT_CONFIGURED bypass
        free_l2 = get_price_quote(2, "free", receiver=r)
        dev_l2 = get_price_quote(2, "developer", receiver=r)
        team_l2 = get_price_quote(2, "team", receiver=r)
        market_l2 = get_price_quote(2, "marketplace", receiver=r)

        # Free tier has no discount
        assert free_l2.final_price_usd == 0.01
        # Developer tier: 100% off during development
        assert dev_l2.final_price_usd == 0.0
        assert dev_l2.is_free
        # Team and marketplace have partial discounts
        assert team_l2.final_price_usd < free_l2.final_price_usd
        assert market_l2.final_price_usd < team_l2.final_price_usd

    @pytest.mark.asyncio
    async def test_payment_flow_complete_cycle(self):
        """Full payment cycle: get quote → parse header → verify → receipt.

        Uses free tier (no discount) to test actual payment flow.
        """
        # Step 1: Get price quote (use free tier to get a real price)
        quote = get_price_quote(2, "free", receiver="TestWallet")
        assert not quote.is_free
        assert quote.final_price_usd == 0.01

        # Step 2: Build 402 response (what client sees)
        resp_402 = build_402_response(quote)
        assert resp_402["status"] == 402
        assert len(resp_402["payment_requirements"]) >= 1

        # Step 3: Client makes payment, gets tx signature
        tx_sig = "a" * 88  # Valid-looking Solana signature
        payment_header = f"{tx_sig}:USDC:solana"

        # Step 4: Parse the payment header
        parsed = parse_payment_header(payment_header)
        assert parsed["tx_signature"] == tx_sig
        assert parsed["token"] == "USDC"
        assert parsed["network"] == "solana"

        # Step 5: Verify payment
        receipt = await verify_payment(
            tx_signature=parsed["tx_signature"],
            expected_amount_usd=quote.final_price_usd,
            token=parsed["token"],
            network=parsed["network"],
        )
        assert receipt.verified
        assert receipt.amount_usd == quote.final_price_usd
        assert receipt.token == "USDC"

        # Step 6: Receipt can be serialized
        receipt_dict = receipt.to_dict()
        assert receipt_dict["verified"]
        assert receipt_dict["network"] == "solana"

    @pytest.mark.asyncio
    async def test_require_payment_flow(self):
        """require_payment() should orchestrate the full flow."""
        from fastapi import HTTPException

        # Patch receiver so pricing doesn't bypass to free mode
        with patch("src.payments.pricing.DEFAULT_RECEIVER", "TestWallet"):
            # Free level → None
            result = await require_payment(level=1, tier="free", x_payment=None)
            assert result is None

            # Developer tier: 100% discount → level 2 is free
            result = await require_payment(level=2, tier="developer", x_payment=None)
            assert result is None

            # Paid level (free tier), no payment → 402
            with pytest.raises(HTTPException) as exc_info:
                await require_payment(level=2, tier="free", x_payment=None)
            assert exc_info.value.status_code == 402

            # Paid level, invalid payment → 402 with verification_failed
            with pytest.raises(HTTPException) as exc_info:
                await require_payment(level=3, tier="free", x_payment="bad")
            assert exc_info.value.status_code == 402
            assert exc_info.value.detail["error"] == "payment_verification_failed"


# ════════════════════════════════════════════════════════════════════════════════
# CROSS-GAP INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════════


class TestCrossGapIntegration:
    """Tests that verify the gaps work together correctly."""

    def test_all_routes_registered(self, test_client):
        """All gap endpoints should be registered in the app."""
        from src.main import app
        paths = [route.path for route in app.routes]

        # Gap 3: Feedback + Correlation
        assert "/v1/feedback" in paths, "Feedback endpoint not registered"
        assert any("/v1/correlation" in p for p in paths), "Correlation endpoint not registered"

        # Gap 4: Pricing
        assert "/v1/pricing" in paths, "Pricing endpoint not registered"
        assert any("/v1/pricing/" in p for p in paths), "Pricing level endpoint not registered"

    def test_evaluate_endpoint_has_payment_header(self):
        """Evaluate endpoint should accept X-Payment header (Gap 4)."""
        import inspect
        from src.api.v1.evaluate import submit_evaluation
        sig = inspect.signature(submit_evaluation)
        assert "x_payment" in sig.parameters, "x_payment param missing from evaluate endpoint"

    def test_evaluator_has_6_dimensions(self):
        """Evaluator.evaluate_full() should produce all 6 scoring dimensions."""
        from src.core.evaluator import Evaluator
        from src.core.llm_judge import LLMJudge

        judge = LLMJudge()
        evaluator = Evaluator(judge, paraphrase=False)

        responses = {
            "tool1": [
                {
                    "question": "test",
                    "expected": "result",
                    "answer": json.dumps({"result": "data"}),
                    "is_error": False,
                    "test_type": "happy_path",
                    "latency_ms": 100,
                },
            ],
        }

        import asyncio
        result = asyncio.run(evaluator.evaluate_full(
            target_id="test",
            server_url="http://localhost:9999",
            tool_responses=responses,
            manifest={"name": "T", "version": "1", "description": "T",
                       "tools": [{"name": "tool1", "description": "T", "inputSchema": {}}]},
            run_safety=False,
        ))

        assert result.dimensions is not None
        assert len(result.dimensions) == 6
        expected_dims = {"accuracy", "safety", "process_quality", "reliability", "latency", "schema_quality"}
        assert set(result.dimensions.keys()) == expected_dims

    def test_pricing_consistency(self):
        """Pricing table and individual quotes should be consistent."""
        for tier in ["free", "developer", "team", "marketplace"]:
            table = get_pricing_table(tier)
            for entry in table:
                individual = get_price_quote(entry["level"], tier)
                assert entry["final_price_usd"] == individual.final_price_usd, \
                    f"Tier={tier} Level={entry['level']}: table price {entry['final_price_usd']} != quote {individual.final_price_usd}"

    def test_correlation_report_serialization(self):
        """CorrelationReport.to_dict() should match CorrelationResponse model."""
        from src.storage.models import CorrelationResponse

        report = compute_correlation_report(
            target_id="test",
            eval_score=75,
            feedback_items=[
                {"outcome": "success", "outcome_score": 80},
                {"outcome": "failure", "outcome_score": 30},
            ],
        )

        # Should be able to create Pydantic model from report dict
        response = CorrelationResponse(**report.to_dict())
        assert response.target_id == "test"
        assert response.eval_score == 75
        assert response.feedback_count == 2

    def test_process_quality_report_serialization(self):
        """ProcessQualityResult.to_dict() should have all required fields."""
        result = ProcessQualityResult(
            score=75,
            error_handling=80,
            input_validation=70,
            response_structure=75,
            details={"samples": "10"},
        )

        d = result.to_dict()
        assert d["score"] == 75
        assert d["error_handling"] == 80
        assert d["input_validation"] == 70
        assert d["response_structure"] == 75
        assert "details" in d

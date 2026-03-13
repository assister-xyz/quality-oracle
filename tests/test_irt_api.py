"""Tests for IRT API endpoints."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import AsyncIterEmpty


class TestIRTApi:
    """Tests for /v1/irt/* endpoints."""

    def test_calibrate(self, test_client, auth_headers):
        """POST /v1/irt/calibrate triggers calibration."""
        with patch(
            "src.api.v1.irt._irt.calibrate_from_battles",
            new_callable=AsyncMock,
            return_value={
                "status": "calibrated",
                "battle_count": 150,
                "items_calibrated": 10,
                "agents_estimated": 5,
                "model": "rasch_1pl",
            },
        ):
            resp = test_client.post("/v1/irt/calibrate", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "calibrated"
        assert data["model"] == "rasch_1pl"
        assert data["items_calibrated"] == 10

    def test_list_items(self, test_client, auth_headers):
        """GET /v1/irt/items returns item quality report."""
        with patch(
            "src.api.v1.irt._irt.item_quality_report",
            new_callable=AsyncMock,
            return_value=[
                {"question_id": "q1", "difficulty_b": 0.5, "status": "active"},
                {"question_id": "q2", "difficulty_b": -1.0, "status": "flagged"},
            ],
        ):
            resp = test_client.get("/v1/irt/items", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["items"][0]["question_id"] == "q1"

    def test_list_items_filtered(self, test_client, auth_headers):
        """GET /v1/irt/items?domain=defi filters by domain."""
        with patch(
            "src.api.v1.irt._irt.item_quality_report",
            new_callable=AsyncMock,
            return_value=[
                {"question_id": "q1", "difficulty_b": 0.5, "domain": "defi", "status": "active"},
            ],
        ):
            resp = test_client.get("/v1/irt/items?domain=defi", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_get_single_item(self, test_client, auth_headers):
        """GET /v1/irt/items/{question_id} returns single item params."""
        with patch(
            "src.api.v1.irt._irt.get_item_params",
            new_callable=AsyncMock,
            return_value={
                "question_id": "q1",
                "difficulty_b": 0.5,
                "discrimination_a": 1.0,
                "status": "active",
            },
        ):
            resp = test_client.get("/v1/irt/items/q1", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["question_id"] == "q1"

    def test_get_single_item_not_found(self, test_client, auth_headers):
        """GET /v1/irt/items/{question_id} returns 404 if not found."""
        with patch(
            "src.api.v1.irt._irt.get_item_params",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = test_client.get("/v1/irt/items/nonexistent", headers=auth_headers)
        assert resp.status_code == 404

    def test_recommend_questions(self, test_client, auth_headers):
        """GET /v1/irt/recommend returns adaptive question selection."""
        with patch(
            "src.api.v1.irt._irt.select_adaptive_questions",
            new_callable=AsyncMock,
            return_value=[
                {"question_id": "q1", "fisher_info": 0.25, "difficulty_b": 0.0},
                {"question_id": "q2", "fisher_info": 0.24, "difficulty_b": 0.5},
            ],
        ):
            resp = test_client.get(
                "/v1/irt/recommend?theta=0.0&count=2", headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["questions"]) == 2

    def test_estimate_ability(self, test_client, auth_headers):
        """POST /v1/irt/estimate-ability estimates theta."""
        with patch(
            "src.api.v1.irt._irt.estimate_ability",
            new_callable=AsyncMock,
            return_value={"theta": 1.5, "se": 0.8, "responses_used": 3},
        ):
            resp = test_client.post(
                "/v1/irt/estimate-ability",
                headers=auth_headers,
                json={"responses": [
                    {"question_id": "q1", "correct": True},
                    {"question_id": "q2", "correct": True},
                    {"question_id": "q3", "correct": False},
                ]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["theta"] == 1.5
        assert data["responses_used"] == 3

    def test_estimate_ability_empty(self, test_client, auth_headers):
        """POST /v1/irt/estimate-ability with empty responses returns 400."""
        resp = test_client.post(
            "/v1/irt/estimate-ability",
            headers=auth_headers,
            json={"responses": []},
        )
        assert resp.status_code == 400

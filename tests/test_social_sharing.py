"""Tests for QO-029 social sharing endpoints: shields.io, percentile, share data, OG images."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── shields.io badge endpoint ──────────────────────────────────────────────


class TestShieldsBadge:
    """Tests for GET /v1/shields/{target_id}.json"""

    def test_shields_badge_with_score(self, test_client):
        """Returns shields.io JSON for an evaluated agent."""
        score_doc = {
            "target_id": "https://mcp.example.com/sse",
            "current_score": 85,
            "tier": "certified",
        }
        with patch("src.api.v1.badges.scores_col") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=score_doc)
            resp = test_client.get("/v1/shields/https://mcp.example.com/sse.json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["schemaVersion"] == 1
        assert data["label"] == "Laureum"
        assert "85" in data["message"]
        assert data["color"]  # non-empty color

    def test_shields_badge_unknown(self, test_client):
        """Returns 'not evaluated' for unknown target."""
        with patch("src.api.v1.badges.scores_col") as mock_col:
            mock_col.return_value.find_one = AsyncMock(return_value=None)
            resp = test_client.get("/v1/shields/unknown-agent.json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "not evaluated"
        assert data["color"] == "gray"


# ── Percentile endpoint ────────────────────────────────────────────────────


class TestPercentile:
    """Tests for GET /v1/stats/percentile/{target_id}"""

    def test_percentile_found(self, test_client):
        """Returns percentile for an evaluated agent."""
        score_doc = {
            "target_id": "https://mcp.example.com/sse",
            "current_score": 85,
        }
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=score_doc)
        mock_col.count_documents = AsyncMock(side_effect=[10, 7, 3])  # total, lower, higher_or_equal

        with (
            patch("src.api.v1.stats.scores_col", return_value=mock_col),
            patch("src.storage.mongodb.scores_col", return_value=mock_col),
        ):
            resp = test_client.get("/v1/stats/percentile/https://mcp.example.com/sse")

        assert resp.status_code == 200
        data = resp.json()
        assert data["percentile"] == 70  # 7/10 * 100
        assert data["total_evaluated"] == 10
        assert data["top_pct"] == 30  # 100 - 70

    def test_percentile_not_found(self, test_client):
        """Returns 404 for unknown target."""
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)

        with (
            patch("src.api.v1.stats.scores_col", return_value=mock_col),
            patch("src.storage.mongodb.scores_col", return_value=mock_col),
        ):
            resp = test_client.get("/v1/stats/percentile/unknown-agent")

        assert resp.status_code == 404


# ── Share data endpoint ────────────────────────────────────────────────────


class TestShareData:
    """Tests for GET /v1/score/{target_id}/share"""

    def test_share_data(self, test_client, auth_headers):
        """Returns pre-formatted share data."""
        score_doc = {
            "target_id": "https://mcp.example.com/sse",
            "current_score": 92,
            "tier": "audited",
        }
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=score_doc)
        mock_col.count_documents = AsyncMock(side_effect=[20, 18])  # total=20, lower=18

        with (
            patch("src.api.v1.scores.scores_col", return_value=mock_col),
            patch("src.storage.mongodb.scores_col", return_value=mock_col),
        ):
            resp = test_client.get(
                "/v1/score/https://mcp.example.com/sse/share",
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] == 92
        assert data["tier"] == "audited"
        assert "tweet_text" in data
        assert "@LaureumAI" in data["tweet_text"]
        assert "linkedin_text" in data
        assert "profile_url" in data
        assert "shields_badge" in data
        assert data["shields_badge"]["schemaVersion"] == 1
        assert "embed_markdown" in data
        assert "embed_html" in data

    def test_share_data_not_found(self, test_client, auth_headers):
        """Returns 404 for unknown target."""
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)

        with (
            patch("src.api.v1.scores.scores_col", return_value=mock_col),
            patch("src.storage.mongodb.scores_col", return_value=mock_col),
        ):
            resp = test_client.get(
                "/v1/score/unknown/share",
                headers=auth_headers,
            )

        assert resp.status_code == 404


# ── OG image endpoint ──────────────────────────────────────────────────────


class TestOGImage:
    """Tests for GET /v1/og/{target_id}.png and .svg"""

    def test_og_svg(self, test_client):
        """Returns SVG OG card."""
        score_doc = {
            "target_id": "https://mcp.example.com/sse",
            "current_score": 78,
            "tier": "certified",
            "dimensions": {
                "accuracy": 82,
                "safety": 75,
                "process_quality": 70,
                "reliability": 80,
                "latency": 72,
                "schema_quality": 85,
            },
        }
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=score_doc)
        mock_col.count_documents = AsyncMock(side_effect=[15, 10])

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        with (
            patch("src.api.v1.og.scores_col", return_value=mock_col),
            patch("src.api.v1.og.get_redis", return_value=mock_redis),
        ):
            resp = test_client.get("/v1/og/https://mcp.example.com/sse.svg")

        assert resp.status_code == 200
        assert "svg" in resp.headers.get("content-type", "")
        assert "LAUREUM" in resp.text
        assert "78" in resp.text

    def test_og_png_fallback_to_svg(self, test_client):
        """PNG endpoint falls back to SVG when cairosvg not available."""
        score_doc = {
            "target_id": "https://mcp.example.com/sse",
            "current_score": 90,
            "dimensions": {},
        }
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=score_doc)
        mock_col.count_documents = AsyncMock(return_value=5)

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        with (
            patch("src.api.v1.og.scores_col", return_value=mock_col),
            patch("src.api.v1.og.get_redis", return_value=mock_redis),
            patch("src.api.v1.og._svg_to_png", return_value=None),
        ):
            resp = test_client.get("/v1/og/https://mcp.example.com/sse.png")

        assert resp.status_code == 200
        # Falls back to SVG
        assert "svg" in resp.headers.get("content-type", "")

    def test_og_unknown_agent(self, test_client):
        """Returns OG card even for unknown agent (with 0 score)."""
        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)
        mock_col.count_documents = AsyncMock(return_value=0)

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        with (
            patch("src.api.v1.og.scores_col", return_value=mock_col),
            patch("src.api.v1.og.get_redis", return_value=mock_redis),
        ):
            resp = test_client.get("/v1/og/unknown.svg")

        assert resp.status_code == 200
        assert "svg" in resp.headers.get("content-type", "")


# ── OG SVG rendering unit tests ───────────────────────────────────────────


class TestOGRendering:
    """Unit tests for SVG rendering functions."""

    def test_render_og_svg_basic(self):
        from src.api.v1.og import render_og_svg

        svg = render_og_svg(
            target_id="https://mcp.example.com/sse",
            score=85,
            tier="certified",
            dimensions={"accuracy": 90, "safety": 80},
        )

        assert "svg" in svg
        assert "1200" in svg
        assert "675" in svg
        assert "85" in svg
        assert "LAUREUM" in svg

    def test_render_og_svg_with_percentile(self):
        from src.api.v1.og import render_og_svg

        svg = render_og_svg(
            target_id="https://test.com/mcp",
            score=92,
            tier="audited",
            dimensions={},
            percentile=90,
            total_evaluated=50,
        )

        assert "Top 10%" in svg
        assert "50 agents" in svg

"""Tests for batch auto-scoring pipeline (QO-020)."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Registry Loading Tests ──────────────────────────────────────────────────


class TestRegistryLoading:
    """Test registry JSON parsing and validation."""

    def test_valid_registry(self, tmp_path):
        """load_registry parses a well-formed registry file."""
        from dev.batch_score import load_registry

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps({
            "version": "1.0",
            "servers": [
                {
                    "url": "https://example.com/mcp",
                    "name": "Example",
                    "transport": "streamable_http",
                    "category": "developer_tools",
                    "requires_auth": False,
                    "source": "test",
                },
                {
                    "url": "https://other.com/sse",
                    "name": "Other",
                    "transport": "sse",
                    "requires_auth": True,
                    "source": "test",
                },
            ],
        }))

        result = load_registry(str(registry_file))
        assert result["version"] == "1.0"
        assert len(result["servers"]) == 2
        assert result["servers"][0]["name"] == "Example"
        assert result["servers"][1]["requires_auth"] is True

    def test_missing_file(self):
        """load_registry raises FileNotFoundError for missing files."""
        from dev.batch_score import load_registry

        with pytest.raises(FileNotFoundError):
            load_registry("/nonexistent/path/registry.json")

    def test_invalid_json(self, tmp_path):
        """load_registry raises error for invalid JSON."""
        from dev.batch_score import load_registry

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_registry(str(bad_file))

    def test_missing_servers_key(self, tmp_path):
        """load_registry raises ValueError if servers key is missing."""
        from dev.batch_score import load_registry

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps({"version": "1.0"}))

        with pytest.raises(ValueError, match="servers"):
            load_registry(str(registry_file))

    def test_server_missing_url(self, tmp_path):
        """load_registry raises ValueError if a server has no url."""
        from dev.batch_score import load_registry

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps({
            "version": "1.0",
            "servers": [{"name": "NoURL"}],
        }))

        with pytest.raises(ValueError, match="missing 'url'"):
            load_registry(str(registry_file))


# ── Skip Logic Tests ────────────────────────────────────────────────────────


class TestSkipLogic:
    """Test server skip/force logic."""

    @pytest.mark.asyncio
    async def test_skip_already_scored(self):
        """evaluate_one skips servers that already have scores."""
        from dev.batch_score import evaluate_one, RateLimiter
        import asyncio

        server = {"url": "https://example.com/mcp", "name": "Example", "transport": "streamable_http"}
        semaphore = asyncio.Semaphore(1)
        rate_limiter = RateLimiter(min_gap=0)

        with patch("dev.batch_score.check_existing_score", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = {"current_score": 75, "tier": "proficient"}

            result = await evaluate_one(
                server=server, level="verified", judge=None,
                semaphore=semaphore, rate_limiter=rate_limiter,
                idx=1, total=1, force=False,
            )

        assert result["status"] == "skipped"
        assert result["skipped_reason"] == "already_scored"
        assert result["score"] == 75

    @pytest.mark.asyncio
    async def test_force_rescore(self):
        """evaluate_one re-evaluates when --force is set, even if scored."""
        from dev.batch_score import evaluate_one, RateLimiter
        import asyncio

        server = {"url": "https://example.com/mcp", "name": "Example", "transport": "streamable_http"}
        semaphore = asyncio.Semaphore(1)
        rate_limiter = RateLimiter(min_gap=0)

        # Mock the evaluation path to return a quick scan result
        mock_scan = MagicMock()
        mock_scan.reachable = True
        mock_scan.manifest_score = 80
        mock_scan.estimated_tier = "proficient"
        mock_scan.tool_count = 5
        mock_scan.scan_time_ms = 100
        mock_scan.error = None

        with patch("dev.batch_score.check_existing_score", new_callable=AsyncMock) as mock_check, \
             patch("src.core.quick_scan.quick_scan", new_callable=AsyncMock, return_value=mock_scan):
            mock_check.return_value = {"current_score": 75, "tier": "proficient"}

            result = await evaluate_one(
                server=server, level="quick", judge=None,
                semaphore=semaphore, rate_limiter=rate_limiter,
                idx=1, total=1, force=True,
            )

        # With force=True, check_existing_score should NOT be called
        mock_check.assert_not_called()
        assert result["status"] == "success"
        assert result["score"] == 80

    @pytest.mark.asyncio
    async def test_skip_auth_required(self):
        """evaluate_one skips servers marked as requires_auth."""
        from dev.batch_score import evaluate_one, RateLimiter
        import asyncio

        server = {
            "url": "https://auth-required.com/mcp",
            "name": "AuthServer",
            "transport": "streamable_http",
            "requires_auth": True,
        }
        semaphore = asyncio.Semaphore(1)
        rate_limiter = RateLimiter(min_gap=0)

        result = await evaluate_one(
            server=server, level="verified", judge=None,
            semaphore=semaphore, rate_limiter=rate_limiter,
            idx=1, total=1, force=False,
        )

        assert result["status"] == "skipped"
        assert result["skipped_reason"] == "requires_auth"


# ── Progress Reporting Tests ─────────────────────────────────────────────────


class TestProgressReporting:
    """Test report generation and summary formatting."""

    def test_save_reports(self, tmp_path):
        """save_reports generates valid JSON and Markdown files."""
        from dev.batch_score import save_reports

        results = [
            {"name": "A", "url": "https://a.com", "status": "success",
             "score": 85, "tier": "proficient", "tools_count": 5, "category": "dev"},
            {"name": "B", "url": "https://b.com", "status": "error",
             "score": None, "tier": None, "tools_count": 0, "error": "timeout",
             "error_type": "timeout", "category": "data"},
            {"name": "C", "url": "https://c.com", "status": "skipped",
             "score": 70, "tier": "competent", "tools_count": 3,
             "skipped_reason": "already_scored", "category": "ai"},
        ]

        json_path, md_path = save_reports(results, tmp_path)

        # JSON report
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["servers_total"] == 3
        assert data["servers_success"] == 1
        assert data["servers_failed"] == 1
        assert data["servers_skipped"] == 1
        assert data["avg_score"] == 85.0
        assert len(data["results"]) == 3

        # Markdown report
        assert md_path.exists()
        md_content = md_path.read_text()
        assert "Batch Auto-Scoring Report" in md_content
        assert "85/100" in md_content
        assert "timeout" in md_content
        assert "already_scored" in md_content

    def test_summary_no_successes(self, tmp_path):
        """save_reports handles case with no successful evaluations."""
        from dev.batch_score import save_reports

        results = [
            {"name": "X", "url": "https://x.com", "status": "error",
             "score": None, "tier": None, "tools_count": 0,
             "error": "refused", "error_type": "connection"},
        ]

        json_path, md_path = save_reports(results, tmp_path)
        data = json.loads(json_path.read_text())
        assert data["servers_success"] == 0
        assert data["avg_score"] == 0


# ── Admin Batch Endpoint Tests ──────────────────────────────────────────────


class TestAdminBatchEndpoint:
    """Test POST /v1/admin/batch-evaluate endpoint."""

    def test_submit_batch(self, test_client):
        """Submitting a batch job returns job_id."""
        # Override auth to marketplace tier
        with patch("src.auth.dependencies.validate_api_key", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = {
                "_id": "hashed_key",
                "tier": "marketplace",
                "active": True,
            }

            resp = test_client.post(
                "/v1/admin/batch-evaluate",
                json={"urls": ["https://example.com/mcp"], "level": 2, "force": False},
                headers={"X-API-Key": "qo_test-marketplace-key"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["total"] == 1
        assert data["status"] == "running"

    def test_batch_requires_marketplace_tier(self, test_client):
        """Batch endpoint rejects non-marketplace API keys."""
        # test_client fixture uses "developer" tier by default
        resp = test_client.post(
            "/v1/admin/batch-evaluate",
            json={"urls": ["https://example.com/mcp"]},
            headers={"X-API-Key": "qo_test-key"},
        )
        assert resp.status_code == 403

    def test_batch_empty_urls(self, test_client):
        """Batch endpoint rejects empty URL list."""
        with patch("src.auth.dependencies.validate_api_key", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = {
                "_id": "hashed_key",
                "tier": "marketplace",
                "active": True,
            }

            resp = test_client.post(
                "/v1/admin/batch-evaluate",
                json={"urls": [], "level": 2},
                headers={"X-API-Key": "qo_test-marketplace-key"},
            )

        assert resp.status_code == 400

    def test_batch_accepts_timeout_seconds(self, test_client):
        """QO-050: BatchEvaluateRequest accepts optional timeout_seconds."""
        with patch("src.auth.dependencies.validate_api_key", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = {
                "_id": "hashed_key",
                "tier": "marketplace",
                "active": True,
            }

            resp = test_client.post(
                "/v1/admin/batch-evaluate",
                json={
                    "urls": ["https://example.com/mcp"],
                    "level": 2,
                    "timeout_seconds": 60,
                },
                headers={"X-API-Key": "qo_test-marketplace-key"},
            )

        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestBatchTimeout:
    """QO-050: per-server timeout in batch evaluation."""

    def test_config_exposes_default(self):
        """Default is sourced from settings, not hardcoded."""
        from src.config import settings
        from src.api.v1.admin import BATCH_PER_SERVER_TIMEOUT_DEFAULT
        assert settings.batch_per_server_timeout_seconds == 180
        assert BATCH_PER_SERVER_TIMEOUT_DEFAULT == settings.batch_per_server_timeout_seconds

    def test_request_model_accepts_none(self):
        """timeout_seconds is optional and defaults to None (→ use default)."""
        from src.api.v1.admin import BatchEvaluateRequest
        req = BatchEvaluateRequest(urls=["https://x"], level=2)
        assert req.timeout_seconds is None

    def test_request_model_accepts_override(self):
        from src.api.v1.admin import BatchEvaluateRequest
        req = BatchEvaluateRequest(urls=["https://x"], timeout_seconds=45)
        assert req.timeout_seconds == 45

    @pytest.mark.asyncio
    async def test_timeout_marks_eval_failed_and_continues(self):
        """When evaluate_full hangs, the eval is marked failed and the loop continues."""
        import asyncio as _asyncio
        from src.api.v1 import admin as admin_mod

        # Seed job tracker
        job_id = "test-qo-050"
        admin_mod._batch_jobs[job_id] = {
            "status": "running",
            "total": 2,
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "results": [],
            "created_at": "2026-04-17T00:00:00Z",
        }

        async def _hang(*_a, **_kw):
            await _asyncio.sleep(10)

        evals_update = AsyncMock()

        scores_col_mock = MagicMock()
        scores_col_mock.find_one = AsyncMock(return_value=None)
        scores_col_mock.update_one = AsyncMock()

        evals_col_mock = MagicMock()
        evals_col_mock.insert_one = AsyncMock()
        evals_col_mock.update_one = evals_update

        # QO-049: admin path now uses single-session manifest_and_evaluate;
        # hanging it is enough to exercise the per-server timeout guard.
        with patch("src.storage.mongodb.scores_col", return_value=scores_col_mock), \
             patch("src.storage.mongodb.evaluations_col", return_value=evals_col_mock), \
             patch("src.api.v1.admin.scores_col", return_value=scores_col_mock), \
             patch("src.api.v1.admin.evaluations_col", return_value=evals_col_mock), \
             patch("src.core.mcp_client.manifest_and_evaluate", side_effect=_hang):

            # per_server_timeout=1 so the test finishes in ~2s, not 20s
            await admin_mod._run_batch_evaluation(
                job_id=job_id,
                urls=["https://slow1.example/mcp", "https://slow2.example/mcp"],
                level=2,
                force=True,
                per_server_timeout=1,
            )

        job = admin_mod._batch_jobs[job_id]
        assert job["status"] == "completed"
        assert job["failed"] == 2
        assert job["succeeded"] == 0
        assert job["completed"] == 2
        for r in job["results"]:
            assert r["status"] == "error"
            assert "timeout after 1s" in r["error"]

        # Eval doc is updated to status=failed for audit trail
        assert evals_update.await_count >= 2
        for call in evals_update.await_args_list:
            _filter, update = call.args
            set_fields = update["$set"]
            assert set_fields["status"] == "failed"
            assert "timeout" in set_fields["error"]

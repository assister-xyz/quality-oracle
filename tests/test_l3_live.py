"""QO-059 — live Docker daemon integration tests for the L3 harness.

These tests are gated by the ``live`` pytest marker so the default ``pytest``
run never hits a real Docker socket. To run:

    LAUREUM_L3_DOCKER_IMAGE_SHA=sha256:... \
    LAUREUM_PROXY_IMAGE_SHA=sha256:... \
    ANTHROPIC_API_KEY=sk-ant-... \
    pytest -m live tests/test_l3_live.py -v

The 5-fixture Spearman-ρ regression suite (rollback runbook, QO-059 spec) is
NOT in this file — it lives in ``dev/run_l3_image_regression.py`` and runs
nightly on Renovate-bot PRs that bump claude-agent-sdk / claude-code.
"""
from __future__ import annotations

import os
import shutil

import pytest

from src.core.l3_activator import L3ClaudeCodeActivator
from src.storage.models import ParsedSkill


pytestmark = pytest.mark.live


def _docker_available() -> bool:
    return shutil.which("docker") is not None and bool(
        os.environ.get("ANTHROPIC_API_KEY"),
    )


@pytest.mark.skipif(not _docker_available(), reason="docker daemon + API key required")
@pytest.mark.asyncio
async def test_real_docker_run() -> None:
    """End-to-end: build → mount → run → assert non-empty response.

    Fixture skill is the smallest possible (~200 byte body) so the Anthropic
    cost is bounded to <<$0.01.
    """
    skill = ParsedSkill(
        name="hello-world",
        description="Print hello.",
        body="When asked, say `hello world`.",
        body_size_bytes=len("When asked, say `hello world`."),
        body_lines=1,
        body_tokens=8,
        folder_name="hello-world",
        folder_name_nfkc="hello-world",
    )
    act = L3ClaudeCodeActivator(
        skill=skill,
        model="claude-sonnet-4-5",
    )
    resp = await act.respond("Say hello, please.")
    assert resp.text
    assert resp.input_tokens > 0
    summary = act.usage_summary()
    assert summary.dollars_spent > 0

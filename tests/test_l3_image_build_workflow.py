"""QO-059 — yaml-lint the L3 image-build GitHub Actions workflow.

We don't run the workflow in CI here; we only verify it parses and has the
expected structure (paths trigger, build steps, push=false guard). This guards
against accidental syntax breakage on Renovate-bot PRs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "build-l3-image.yml"


def test_workflow_parses_as_yaml() -> None:
    raw = WORKFLOW_PATH.read_text()
    doc = yaml.safe_load(raw)
    assert isinstance(doc, dict)
    assert doc.get("name") == "build-l3-image"


def test_workflow_triggers_on_renovate_paths() -> None:
    doc = yaml.safe_load(WORKFLOW_PATH.read_text())
    # YAML's `on:` becomes Python `True` because of legacy boolean parsing.
    on = doc.get("on") or doc.get(True)
    assert on is not None, "workflow must declare on: triggers"
    paths = on["pull_request"]["paths"]
    assert "requirements.txt" in paths
    assert "images/laureum-claude-code-runner/package.json" in paths
    assert "images/laureum-claude-code-runner/Dockerfile" in paths
    assert "images/laureum-proxy/Dockerfile" in paths


def test_workflow_gates_on_renovate_actor() -> None:
    doc = yaml.safe_load(WORKFLOW_PATH.read_text())
    job = doc["jobs"]["build"]
    cond = job.get("if", "")
    assert "renovate[bot]" in cond or "dependabot[bot]" in cond


def test_workflow_does_not_push_yet() -> None:
    """Push step is gated off until GHCR creds are configured.

    Searching the raw text for ``push: true`` is a deliberately strict check —
    if a future PR flips this on, the unit test fails so reviewers must justify
    enabling registry pushes (and verify the rollback runbook works).
    """
    raw = WORKFLOW_PATH.read_text()
    assert "push: true" not in raw, (
        "GHCR push must stay disabled until creds are wired; see QO-059 spec."
    )


def test_workflow_builds_both_images() -> None:
    raw = WORKFLOW_PATH.read_text()
    assert "images/laureum-proxy" in raw
    assert "images/laureum-claude-code-runner" in raw

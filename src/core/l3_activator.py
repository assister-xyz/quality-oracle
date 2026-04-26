"""QO-059 L3 activation tier — claude-agent-sdk inside a Docker harness.

This module replaces the QO-053-B :class:`L3ClaudeCodeActivator` stub. It runs
the real Claude Code SDK inside a sandboxed Ubuntu 24.04 container with:

- ``--network=container:laureum-proxy`` — egress restricted to Anthropic +
  judge providers only via a squid sidecar (AC2 / AC6, replaces the original
  ``--network=none`` design from R7).
- Skill mounted ``:ro`` at ``/eval/.claude/skills/<name>`` so the SDK
  discovers it (AC1, AC9).
- ``~/.claude/`` mounted as writable tmpfs so the SDK can persist session
  state (AC9).
- Image SHA pinning for reproducibility (AC5) — see
  ``src/config.py::laureum_l3_docker_image_sha``.
- Pre-flight 100KB hard fail (QO-053-A AC9 boundary) and pre-flight cost
  ceiling at $1.00/skill (AC4).

Rollback runbook (per QO-059 spec §"Image-bump rollback runbook"):

    If Spearman ρ between L3 results pre/post-image-bump < 0.95 on the
    5 fixture skills, revert ``settings.laureum_l3_docker_image_sha`` to
    the previous value in ``src/config.py`` and redeploy. The 5-fixture
    suite runs automatically on every Renovate-bot PR; ρ < 0.95 fails the
    workflow and blocks merge. A Sentry alert fires when post-deploy
    Spearman drift breaches the 0.95 floor (TODO: wire alert in
    ``src/core/score_anomaly.py`` once production has 30+ L3 evals).

Failure modes:

- :class:`SkillTooLargeError` — skill body exceeds the configured cap before
  any Docker work begins, so we never OOM the harness on a 70KB+ outlier.
- :class:`CostCeilingExceededError` — pre-flight estimator exceeded the
  configured cost cap; eval is refused so we never spend $5 on one skill.
- :class:`ActivationFailure` — propagated from QO-053-B for retry-exhaustion
  parity.

The implementation is intentionally subprocess-only — we DO NOT use the
docker SDK (docker-py) because (a) tests need to mock subprocess, not a third
client; (b) the host image already has the docker CLI available; and (c) the
proxy lifecycle wants explicit `docker network create` / `docker rm -f`
commands so engineers can reproduce locally without the SDK installed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from src.config import calculate_cost, settings
from src.core.skill_activator import SkillActivatedAgent
from src.storage.models import (
    ActivationFailure,
    ActivationResponse,
    ParsedSkill,
    ToolCall,
    UsageSummary,
)

logger = logging.getLogger(__name__)


# ── Errors ──────────────────────────────────────────────────────────────────


class SkillTooLargeError(Exception):
    """Raised when ``parsed.body_size_bytes`` exceeds the configured cap.

    Wired into :meth:`L3ClaudeCodeActivator.respond` BEFORE any Docker work,
    matching QO-053-A AC9 boundary so the harness fails fast and clean rather
    than OOMing on metengine 72KB+ outliers (or on a malicious 50MB skill).
    """

    def __init__(self, message: str, body_size_bytes: int = 0, cap_bytes: int = 0):
        super().__init__(message)
        self.body_size_bytes = body_size_bytes
        self.cap_bytes = cap_bytes


class CostCeilingExceededError(Exception):
    """Raised by the pre-flight estimator when one skill exceeds the cost cap.

    AC4: pre-flight refuses execution if estimated cost > $1.00. Estimator is
    deliberately conservative — if we mis-estimate low we might spend more,
    but never more than ~2x the cap, so the cap of $1.00 protects against
    the runaway-loop case where a skill drives the SDK into a 30-turn tool
    loop on a 30Q eval.
    """

    def __init__(self, message: str, estimated_cost_usd: float = 0.0, cap_usd: float = 0.0):
        super().__init__(message)
        self.estimated_cost_usd = estimated_cost_usd
        self.cap_usd = cap_usd


# ── Cost estimator ──────────────────────────────────────────────────────────


def estimate_l3_cost(
    skill: ParsedSkill,
    *,
    num_questions: int = 1,
    model: str = "",
    avg_output_tokens_per_q: int = 600,
    tool_turn_multiplier: float = 2.5,
) -> float:
    """Conservative dollars-per-skill estimate for the AC4 pre-flight gate.

    Heuristics (R7 §6 / R9 §1.3):
    - Input tokens per question ≈ skill body tokens + a 200-token framing.
    - Output tokens per question ≈ 600 (median Sonnet completion in our
      L2 fixtures).
    - Tool-turn multiplier ≈ 2.5× because the SDK makes ~2-3 tool calls per
      Q on a typical skill.

    We default to Anthropic Sonnet pricing because L3 is the only tier that
    *requires* the Anthropic budget per CB1 — even if the question-level
    judges are running on Cerebras free.
    """
    body_tokens = skill.body_tokens or max(1, skill.body_size_bytes // 4)
    input_per_q = (body_tokens + 200) * tool_turn_multiplier
    output_per_q = avg_output_tokens_per_q * tool_turn_multiplier
    total_input = int(input_per_q * num_questions)
    total_output = int(output_per_q * num_questions)
    # Claude Sonnet 4.5 list price ≈ $3 / $15 per 1M tokens (input / output).
    # We don't lean on PROVIDER_PRICING because that table tracks free-tier;
    # L3 always pays the Anthropic API rate.
    sonnet_in_per_m = 3.00
    sonnet_out_per_m = 15.00
    estimated = (total_input * sonnet_in_per_m + total_output * sonnet_out_per_m) / 1_000_000
    return round(estimated, 4)


# ── Subprocess primitives ───────────────────────────────────────────────────


async def _run(
    cmd: list[str],
    *,
    timeout: float = 600.0,
    env: Optional[dict[str, str]] = None,
) -> tuple[int, bytes, bytes]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, stdout, stderr


# ── L3 activator ────────────────────────────────────────────────────────────


class L3ClaudeCodeActivator(SkillActivatedAgent):
    """L3 audited tier — Claude Code SDK in a pinned Docker image.

    Construction parameters mirror QO-053-B's protocol so the dispatcher in
    QO-053-C / QO-058 can swap us in for the L3 stub transparently. ``image_sha``
    is required and pinned via ``settings.laureum_l3_docker_image_sha`` for
    AC5 reproducibility.
    """

    def __init__(
        self,
        skill: ParsedSkill,
        *,
        model: str,
        image_sha: str = "",
        proxy_image_sha: str = "",
        anthropic_api_key: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        # Construct minimal parent state; we don't share the Anthropic SDK
        # client because the actual SDK runs inside the container.
        self.skill = skill
        self.model = model
        self.provider = "anthropic"  # CB1: L3 always Anthropic.
        self.client = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.resolved = None
        self._usage = UsageSummary(model=model, provider="anthropic")
        self._system_text: Optional[str] = None
        self._system_warnings: list[str] = []
        self._cache_disabled_for_short_body = False
        self._dlq_writer = None
        self.image_sha = image_sha or settings.laureum_l3_docker_image_sha
        self.proxy_image_sha = proxy_image_sha or settings.laureum_proxy_image_sha
        self.anthropic_api_key = anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY", "",
        )
        self._eval_id = uuid.uuid4().hex[:12]

    # ── Image reference helpers ────────────────────────────────────────────

    def _runner_image_ref(self) -> str:
        if self.image_sha:
            # Allow either ``sha256:abc...`` or ``laureum-claude-code-runner@sha256:...``.
            tag = settings.laureum_l3_docker_image_tag.split(":", 1)[0]
            sha = self.image_sha
            if sha.startswith("sha256:"):
                return f"{tag}@{sha}"
            return sha
        return settings.laureum_l3_docker_image_tag

    def _proxy_image_ref(self) -> str:
        if self.proxy_image_sha:
            tag = settings.laureum_proxy_image_tag.split(":", 1)[0]
            sha = self.proxy_image_sha
            if sha.startswith("sha256:"):
                return f"{tag}@{sha}"
            return sha
        return settings.laureum_proxy_image_tag

    # ── Skill materialization ──────────────────────────────────────────────

    def _materialize_skill(self) -> Path:
        """Write the parsed skill to a tmp dir laid out as ``.claude/skills/<name>/``.

        We materialize fresh per ``respond()`` call so the eval is hermetic —
        no leftover state from a previous question can leak in. The directory
        is mounted ``:ro`` so the container can never write back into it.
        """
        base = Path(tempfile.mkdtemp(prefix=f"laureum-l3-{self._eval_id}-"))
        skill_dir = base / "skills" / self.skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        # Re-emit minimal front-matter + body so the SDK parses it.
        front_matter = (
            f"---\n"
            f"name: {self.skill.name}\n"
            f"description: {self.skill.description}\n"
            f"---\n\n"
        )
        skill_md.write_text(front_matter + (self.skill.body or ""))
        return skill_dir

    # ── Sidecar lifecycle ──────────────────────────────────────────────────

    async def _start_proxy_sidecar(self, network: str) -> str:
        """Run the squid sidecar in detached mode on the eval network.

        Returns the container name (used by the main runner via
        ``--network=container:<name>``).
        """
        proxy_name = f"laureum-proxy-{self._eval_id}"
        cmd = [
            "docker", "run", "-d",
            "--name", proxy_name,
            "--network", network,
            self._proxy_image_ref(),
        ]
        rc, stdout, stderr = await _run(cmd, timeout=60)
        if rc != 0:
            raise ActivationFailure(
                f"proxy sidecar failed to start: {stderr.decode(errors='replace')}",
                provider="anthropic",
            )
        return proxy_name

    async def _create_network(self) -> str:
        name = f"laureum-eval-{self._eval_id}"
        rc, _, stderr = await _run(["docker", "network", "create", name], timeout=30)
        if rc != 0:
            raise ActivationFailure(
                f"docker network create failed: {stderr.decode(errors='replace')}",
                provider="anthropic",
            )
        return name

    async def _cleanup(self, network: str, proxy_name: str) -> None:
        """Tear down the proxy + network. Skipped if LAUREUM_KEEP_DOCKER set."""
        if settings.laureum_keep_docker or os.environ.get("LAUREUM_KEEP_DOCKER"):
            logger.info(
                "LAUREUM_KEEP_DOCKER set — leaving %s and %s for inspection",
                proxy_name, network,
            )
            return
        # Best-effort: never raise from cleanup.
        for cmd in (
            ["docker", "rm", "-f", proxy_name],
            ["docker", "network", "rm", network],
        ):
            try:
                await _run(cmd, timeout=30)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("cleanup failed: %s (%s)", " ".join(cmd), e)

    # ── Public API ─────────────────────────────────────────────────────────

    def build_run_command(
        self,
        *,
        proxy_container: str,
        skill_host_path: str,
        question: str,
    ) -> list[str]:
        """Construct the `docker run` argv for the main runner container.

        Exposed as a public method so :file:`tests/test_l3_activator.py` can
        assert on the exact command shape (AC2 ``--network=container:...``,
        AC9 ``:ro`` mount, AC9 tmpfs for ``~/.claude/``).
        """
        payload = json.dumps({
            "skill_name": self.skill.name,
            "questions": [question],
            "model": self.model,
            "temp": self.temperature,
        })
        return [
            "docker", "run", "--rm",
            f"--network=container:{proxy_container}",
            "-v",
            f"{skill_host_path}:/eval/.claude/skills/{self.skill.name}:ro",
            "--tmpfs", "/root/.claude:rw,size=64m",
            "-e", f"ANTHROPIC_API_KEY={self.anthropic_api_key}",
            "-e", f"CLAUDE_TEMPERATURE={self.temperature}",
            self._runner_image_ref(),
            payload,
        ]

    def _preflight(self, num_questions: int = 1) -> None:
        """Run pre-flight gates. Called BEFORE any Docker work for AC4 / AC9."""
        # AC9 / QO-053-A: skill body cap.
        cap = settings.laureum_l3_max_skill_body_bytes
        if self.skill.body_size_bytes > cap:
            raise SkillTooLargeError(
                f"skill body {self.skill.body_size_bytes}B exceeds {cap}B cap",
                body_size_bytes=self.skill.body_size_bytes,
                cap_bytes=cap,
            )
        # AC4: cost ceiling.
        est = estimate_l3_cost(
            self.skill, num_questions=num_questions, model=self.model,
        )
        if est > settings.laureum_l3_cost_ceiling_usd:
            raise CostCeilingExceededError(
                f"estimated ${est:.4f} exceeds ${settings.laureum_l3_cost_ceiling_usd:.2f} cap",
                estimated_cost_usd=est,
                cap_usd=settings.laureum_l3_cost_ceiling_usd,
            )

    async def respond(self, question: str) -> ActivationResponse:
        """Run one Q through the L3 Docker harness and return the response."""
        self._preflight(num_questions=1)

        skill_dir = self._materialize_skill()
        network = await self._create_network()
        proxy_container: Optional[str] = None
        try:
            proxy_container = await self._start_proxy_sidecar(network)
            cmd = self.build_run_command(
                proxy_container=proxy_container,
                skill_host_path=str(skill_dir),
                question=question,
            )
            rc, stdout, stderr = await _run(cmd, timeout=600)
            if rc != 0:
                raise ActivationFailure(
                    f"runner exited {rc}: {stderr.decode(errors='replace')[:500]}",
                    provider="anthropic",
                )
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as e:
                raise ActivationFailure(
                    f"runner produced non-JSON stdout: {e}",
                    provider="anthropic",
                ) from e
            return self._parse_response(payload)
        finally:
            if proxy_container is not None:
                await self._cleanup(network, proxy_container)
            shutil.rmtree(skill_dir.parent.parent, ignore_errors=True)

    def usage_summary(self) -> UsageSummary:
        return self._usage

    def reset(self) -> None:
        # L3 is hermetic per-call by construction (fresh container per Q),
        # so reset is a no-op besides clearing usage if a caller wants to
        # roll over a new session.
        pass

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_response(self, payload: dict[str, Any]) -> ActivationResponse:
        results = payload.get("results", [])
        if not results:
            raise ActivationFailure(
                "runner returned 0 results",
                provider="anthropic",
            )
        first = results[0]
        usage = first.get("usage", {})
        cache_creation = int(usage.get("cache_creation_input_tokens", 0))
        cache_read = int(usage.get("cache_read_input_tokens", 0))
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        request_id = first.get("request_id", "") or ""
        latency_ms = int(first.get("latency_ms", 0))

        # Record on aggregated usage summary (AC3).
        self._usage.cache_creation_input_tokens += cache_creation
        self._usage.cache_read_input_tokens += cache_read
        self._usage.input_tokens += in_tok
        self._usage.output_tokens += out_tok
        self._usage.n_calls += 1
        if request_id:
            self._usage.request_ids.append(request_id)
        # Cost: Anthropic billable_input = input + cache_creation; cache_read
        # billed at a lower tier — leave at $0 for the conservative path. The
        # PROVIDER_PRICING table doesn't carry Sonnet rates, so we apply the
        # Sonnet list rate here directly (CB1: L3 == Anthropic).
        sonnet_in_per_m = 3.00
        sonnet_out_per_m = 15.00
        billable_in = in_tok + cache_creation
        cost = (billable_in * sonnet_in_per_m + out_tok * sonnet_out_per_m) / 1_000_000
        self._usage.dollars_spent = round(self._usage.dollars_spent + cost, 8)
        # Also keep the calculate_cost helper symmetric so QO-051 CPCR can
        # consume the field directly even though we override the rate here.
        _ = calculate_cost  # pragma: no cover - helper retained for future per-provider PR

        tool_calls = [
            ToolCall(
                tool=str(tc.get("tool", "")),
                args=dict(tc.get("args", {}) or {}),
            )
            for tc in (first.get("tool_calls", []) or [])
        ]

        return ActivationResponse(
            text=first.get("text", "") or "",
            tool_calls=tool_calls,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=self.model,
            provider="anthropic",
            request_id=request_id,
            latency_ms=latency_ms,
            parse_warnings=[],
        )


# ── Concurrency wall-clock guard (AC7) ─────────────────────────────────────


def fits_within_walltime(
    n_skills: int, concurrency: int, max_minutes_per_skill: float = 7.0,
) -> tuple[bool, float]:
    """Return ``(fits, projected_minutes)`` for AC7's dual SLO check.

    AC7 gates:
        - 10 skills @ concurrency=3 must finish in < 30 min.
        - 42 skills @ concurrency=6 must finish in < 90 min.

    Solving both: ceil(10/3)=4 waves × per-skill ≤ 30 ⇒ per-skill ≤ 7.5 min.
    ceil(42/6)=7 waves × per-skill ≤ 90 ⇒ per-skill ≤ 12.8 min. The 10sk@3
    gate is the tighter constraint; we default ``max_minutes_per_skill=7.0``
    to leave 0.5 min headroom for the network sidecar lifecycle. Bumps to
    this number signal regression and trigger the rollback runbook (see
    QO-059 spec §"Image-bump rollback runbook").
    """
    waves = (n_skills + concurrency - 1) // concurrency  # ceil divide
    projected = waves * max_minutes_per_skill
    # Apply the dual-SLO gate from AC7. Both must hold for the cohort/concurrency.
    if n_skills <= 10 and concurrency >= 3:
        return projected < 30.0, projected
    if n_skills <= 42 and concurrency >= 6:
        return projected < 90.0, projected
    # Fallback for arbitrary (n, conc): compute projected wall-clock and
    # apply a generous 90-min ceiling.
    return projected < 90.0, projected

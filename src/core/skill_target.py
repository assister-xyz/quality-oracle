"""SkillTarget — adapts QO-053-B :class:`SkillActivatedAgent` to the
:class:`EvaluationTarget` Protocol (QO-058 §"Skill target conformance").

Why we need it: the existing dispatcher in ``api.v1.evaluate._run_evaluation``
calls :meth:`Evaluator.evaluate_skill` directly, passing a duck-typed
``target`` with ``.parsed`` / ``.spec_compliance`` / ``.subject_uri``.
QO-058 introduces the uniform Protocol so future dispatch code can route
all target types through one resolver.

This wrapper is intentionally thin — the real activation logic lives in
:mod:`src.core.skill_activator`. We DO NOT subclass ``SkillActivatedAgent``
here; we compose. Composition lets a SkillTarget hold the parsed skill
+ spec compliance + subject URI without disturbing the activator hierarchy.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

from src.core.evaluation_target import (
    AgentManifest,
    AuthContext,
    AuthDescriptor,
    Capability,
    Chunk,
    InvocationResult,
    InvokeError,
)
from src.storage.models import (
    ParsedSkill,
    SpecCompliance,
    TargetType,
)

logger = logging.getLogger(__name__)


class SkillTarget:
    """Composition wrapper: holds the parsed skill + a configured activator.

    The dispatcher in :func:`Evaluator.evaluate_skill` continues to read
    ``target.parsed`` / ``target.spec_compliance`` / ``target.subject_uri``
    directly, so this class exposes those as plain attributes (NOT properties)
    for cheap zero-copy compatibility.
    """

    target_type: TargetType = TargetType.SKILL

    def __init__(
        self,
        *,
        parsed: ParsedSkill,
        spec_compliance: Optional[SpecCompliance] = None,
        subject_uri: str = "",
        activator: Any = None,
    ):
        # Public attributes the dispatcher reads (duck-typed contract):
        self.parsed = parsed
        self.spec_compliance = spec_compliance
        self.subject_uri = subject_uri
        # Optional activator — a SkillActivatedAgent instance. Set lazily
        # because evaluator.evaluate_skill takes activator factories.
        self.activator = activator
        # Protocol-required surface:
        self.endpoint_url = subject_uri
        self.auth: Optional[AuthDescriptor] = None
        self._auth_ctx: Optional[AuthContext] = None
        self._invocations: List[dict] = []

    async def authenticate(self) -> AuthContext:
        # Skills are local-execution surfaces — no remote auth.
        ctx = AuthContext()
        self._auth_ctx = ctx
        return ctx

    async def discover(self) -> AgentManifest:
        """Build a manifest from the parsed SKILL.md.

        Each ``allowed_tools`` entry surfaces as a :class:`Capability` so
        downstream consumers (e.g. /v1/discover) can preview what the skill
        is permitted to do.
        """
        capabilities = [
            Capability(
                id=tool,
                name=tool,
                description=f"Tool gated by SKILL.md allowed-tools: {tool}",
                input_schema=None,
                output_schema=None,
            )
            for tool in (self.parsed.allowed_tools or [])
        ]
        if not capabilities:
            # SKILL.md without allowed-tools — fall back to one synthetic
            # "respond" capability so list_capabilities() never returns empty.
            capabilities = [
                Capability(id="respond", name="respond", description="SKILL.md respond")
            ]
        return AgentManifest(
            id=self.parsed.git_sha or self.parsed.name,
            name=self.parsed.name,
            description=self.parsed.description,
            capabilities=capabilities,
            auth=None,
            confidence="high",
            raw={
                "skill": self.parsed.name,
                "license": self.parsed.license,
                "compatibility": self.parsed.compatibility,
                "spec_compliance": (
                    self.spec_compliance.model_dump()
                    if hasattr(self.spec_compliance, "model_dump")
                    else None
                ),
            },
        )

    async def list_capabilities(self) -> List[Capability]:
        return (await self.discover()).capabilities

    async def invoke(self, capability_id: str, payload: dict) -> InvocationResult:
        """Forward to the activator's ``respond()``.

        ``capability_id`` is informational here — skills route their own
        tool calls internally via the activator's MockFileSystem. Tests
        that exercise this path inject an activator with a deterministic
        ``respond()`` stub.
        """
        if self.activator is None:
            raise InvokeError(
                "SkillTarget.invoke requires an attached activator; "
                "evaluate_skill() supplies one via activator_factory."
            )
        question = payload.get("message") or payload.get("question") or ""
        start = time.time()
        try:
            resp = await self.activator.respond(question)
        except Exception as exc:
            raise InvokeError(f"Skill activation failed: {exc}") from exc
        latency_ms = int((time.time() - start) * 1000)
        text = getattr(resp, "text", "") or ""
        invocation_id = str(uuid.uuid4())
        self._invocations.append(
            {"capability_id": capability_id, "latency_ms": latency_ms}
        )
        return InvocationResult(
            invocation_id=invocation_id,
            text=text,
            raw={
                "tool_calls": [
                    tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                    for tc in getattr(resp, "tool_calls", []) or []
                ],
                "model": getattr(resp, "model", ""),
                "provider": getattr(resp, "provider", ""),
            },
            latency_ms=latency_ms,
            status="ok",
        )

    async def stream(self, capability_id: str, payload: dict) -> AsyncIterator[Chunk]:
        raise NotImplementedError("Skill streaming not in QO-058 MVP")
        if False:  # pragma: no cover
            yield None  # type: ignore[misc]

    async def cancel(self, invocation_id: str) -> None:
        # Skill activations are synchronous wrt the evaluator coroutine.
        return None

    def get_provenance(self) -> dict:
        return {
            "target_type": self.target_type.value,
            "endpoint_url": self.endpoint_url,
            "skill_name": self.parsed.name,
            "git_sha": self.parsed.git_sha,
            "spec_compliance_score": (
                getattr(self.spec_compliance, "score", None)
                if self.spec_compliance is not None
                else None
            ),
            "invocation_count": len(self._invocations),
        }

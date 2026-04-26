"""EvaluationTarget Protocol — uniform contract for any agent we evaluate.

QO-058 introduces the protocol that each target-type adapter (MCP, A2A, REST
chat, OpenAPI, Skill) must satisfy so the evaluator dispatch can stay generic.

Why a Protocol and not an ABC:
  * Duck-typed — third-party adapters (e.g. a future OpenAI Responses target
    landing in QO-069) don't have to inherit from us, just match the signatures.
  * The locked ``Chunk`` sum type below means streaming WS/SSE invocation can
    land in v2 without breaking adapters written today.

Spec §"Common abstractions" + plan-review C1 (added ``authenticate()``,
dropped ``capability_id: str | None``, locked ``Chunk = TextChunk |
ToolCallChunk | TaskUpdateChunk | ErrorChunk``).
"""
from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from src.storage.models import TargetType


# ── Auth descriptor + context ────────────────────────────────────────────────


class AuthDescriptor(BaseModel):
    """Static auth requirement advertised by a target's manifest.

    The discovery layer fills this from `securitySchemes` (A2A) /
    `components.securitySchemes` (OpenAPI) / heuristic probing (REST chat).
    The evaluator turns it into an :class:`AuthContext` via ``authenticate()``.
    """
    model_config = ConfigDict(extra="allow")

    style: Literal[
        "none", "bearer", "x_api_key", "api_key_header", "apikey_query",
        "cookie_session", "oauth2", "mtls", "oidc", "basic"
    ] = "none"
    header_name: Optional[str] = None  # e.g. "Authorization" or "X-API-Key"
    query_param: Optional[str] = None  # e.g. "apikey"
    scopes: List[str] = Field(default_factory=list)
    token_url: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class AuthContext(BaseModel):
    """Resolved auth — produced by ``EvaluationTarget.authenticate()``.

    `headers` and `query_params` are merged into every subsequent
    ``invoke()`` / ``stream()`` HTTP request. ``token_expires_at`` is informational
    so a long-running eval can re-call ``authenticate`` if needed (not done in
    QO-058 MVP — most evals finish in <3min, well below typical token TTL).
    """
    headers: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, str] = Field(default_factory=dict)
    cookies: Dict[str, str] = Field(default_factory=dict)
    token_expires_at: Optional[float] = None  # epoch seconds


# ── Capability + manifest ────────────────────────────────────────────────────


class Capability(BaseModel):
    """One thing the target can do — equivalent to MCP "tool", A2A "skill",
    REST-chat synthetic "chat".
    """
    id: str
    name: str = ""
    description: str = ""
    input_schema: Optional[dict] = None  # JSON Schema (MCP/OpenAPI); None for free-form
    output_schema: Optional[dict] = None
    accepted_input_types: List[str] = Field(default_factory=list)  # MIME (A2A)
    produced_output_types: List[str] = Field(default_factory=list)
    extensions: Dict[str, Any] = Field(default_factory=dict)


class AgentManifest(BaseModel):
    """Common manifest shape — discover() returns this regardless of protocol.

    `confidence` is set by manifest-less inference; the evaluator gates tier
    on this field (AC11 — `low` → tier=failed; `medium` → cap at Verified;
    `high` → Certified path opens, see spec §"Path to Certified").
    """
    model_config = ConfigDict(extra="allow")

    id: str
    name: str = ""
    description: str = ""
    capabilities: List[Capability] = Field(default_factory=list)
    auth: Optional[AuthDescriptor] = None
    signature: Optional[str] = None  # A2A v1.0 Signed Agent Cards
    confidence: Optional[Literal["low", "medium", "high"]] = None
    raw: Dict[str, Any] = Field(default_factory=dict)  # original card / openapi doc


# ── Invocation result + streaming chunks ─────────────────────────────────────


class InvocationResult(BaseModel):
    """Outcome of a single ``invoke()`` call.

    `text` is the canonical response surface — adapters MUST populate it even
    when the protocol returns structured content (A2A messages, MCP tool
    results); `raw` keeps the protocol-native shape for downstream probes.
    """
    model_config = ConfigDict(extra="allow")

    invocation_id: str = ""
    text: str = ""
    raw: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    status: Literal["ok", "error", "timeout"] = "ok"
    error: Optional[str] = None
    error_class: Optional[str] = None
    request_shape: Optional[str] = None  # which fallback shape worked (REST chat)


class TextChunk(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallChunk(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class TaskUpdateChunk(BaseModel):
    type: Literal["task_update"] = "task_update"
    state: Literal["queued", "working", "input_required", "completed", "canceled"] = "working"
    message: str = ""


class ErrorChunk(BaseModel):
    type: Literal["error"] = "error"
    error: str
    error_class: Optional[str] = None


# Sum type — locked now even though WS/SSE invocation deferred to v2 (plan
# review C1 / spec §"WebSocket/SSE deferred").
Chunk = Union[TextChunk, ToolCallChunk, TaskUpdateChunk, ErrorChunk]


# ── The Protocol itself ──────────────────────────────────────────────────────


@runtime_checkable
class EvaluationTarget(Protocol):
    """Uniform contract every target-type adapter must satisfy.

    Concrete adapters live in:
      * ``src.core.mcp_target``       — MCP servers (refactor of mcp_client)
      * ``src.core.a2a_target``       — A2A v1.0 agents
      * ``src.core.rest_chat_target`` — manifest-less REST/JSON chat
      * ``src.core.skill_target``     — Anthropic Agent Skills (wraps 053-B)

    Notes:
    - ``capability_id`` is non-optional. REST chat exposes the synthetic id
      ``"chat"`` (NOT ``None``).
    - ``stream()`` returns ``AsyncIterator[Chunk]`` where ``Chunk`` is the
      locked sum type above. Adapters that don't yet implement streaming may
      raise ``NotImplementedError`` — but the type itself is frozen.
    """

    target_type: TargetType
    endpoint_url: str
    auth: Optional[AuthDescriptor]

    async def authenticate(self) -> AuthContext: ...
    async def discover(self) -> AgentManifest: ...
    async def list_capabilities(self) -> List[Capability]: ...
    async def invoke(self, capability_id: str, payload: dict) -> InvocationResult: ...
    async def stream(self, capability_id: str, payload: dict) -> AsyncIterator[Chunk]: ...
    async def cancel(self, invocation_id: str) -> None: ...
    def get_provenance(self) -> dict: ...


# ── Errors ───────────────────────────────────────────────────────────────────


class TargetError(Exception):
    """Base for adapter-level errors."""


class UnknownTargetError(TargetError):
    """Raised by the resolver when no probe in the cascade succeeds."""


class InvokeError(TargetError):
    """Raised when invoke() exhausts its fallback shapes / retries."""


class SchemaUnobtainableError(TargetError):
    """AC11 — calibration prompts hung/5xx'd ≥2 of 3; refuse evaluation."""

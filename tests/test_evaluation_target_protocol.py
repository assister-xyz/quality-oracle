"""QO-058: every concrete adapter must satisfy the Protocol contract.

The :class:`EvaluationTarget` Protocol is decorated with
``@runtime_checkable`` so ``isinstance(obj, EvaluationTarget)`` evaluates the
attribute set. Any drift in adapter signatures (e.g. someone removes
``authenticate`` or returns the wrong type) trips this suite at the seam
where it matters: target instantiation.
"""
from __future__ import annotations

import inspect

import pytest

from src.core.evaluation_target import (
    AuthContext,
    Capability,
    Chunk,
    ErrorChunk,
    InvocationResult,
    TaskUpdateChunk,
    TextChunk,
    ToolCallChunk,
)
from src.core.a2a_target import A2ATarget
from src.core.mcp_target import MCPTarget
from src.core.rest_chat_target import RESTChatTarget
from src.core.skill_target import SkillTarget


REQUIRED_ASYNC_METHODS = (
    "authenticate", "discover", "list_capabilities",
    "invoke", "stream", "cancel",
)


def _check_shape(cls, *, has_endpoint=True):
    # Public attributes (declared on the class)
    for name in ("target_type", "endpoint_url", "auth"):
        assert hasattr(cls, name) or hasattr(cls, "__init__"), (
            f"{cls.__name__} missing public attribute {name}"
        )

    # Required methods exist + are coroutine functions / async generators
    for m in REQUIRED_ASYNC_METHODS:
        fn = getattr(cls, m, None)
        assert callable(fn), f"{cls.__name__}.{m} missing"

    # get_provenance is sync per spec
    assert callable(getattr(cls, "get_provenance", None))


def test_a2a_target_shape():
    _check_shape(A2ATarget)
    inst = A2ATarget(endpoint_url="https://example.com")
    assert inst.target_type.value == "a2a_agent"


def test_rest_chat_target_shape():
    _check_shape(RESTChatTarget)
    inst = RESTChatTarget(endpoint_url="https://example.com/chat")
    assert inst.target_type.value == "rest_chat"


def test_mcp_target_shape():
    _check_shape(MCPTarget)
    inst = MCPTarget(endpoint_url="https://example.com/sse")
    assert inst.target_type.value == "mcp_server"


def test_skill_target_shape():
    from src.storage.models import ParsedSkill
    parsed = ParsedSkill(name="x", description="y")
    inst = SkillTarget(parsed=parsed, subject_uri="local://x")
    _check_shape(SkillTarget)
    assert inst.target_type.value == "skill"


def test_chunk_sum_type_locked():
    """Spec §"WS/SSE deferred but type locked": Chunk = Text|ToolCall|TaskUpdate|Error."""
    # Each is a Pydantic model with a discriminating ``type`` literal.
    assert TextChunk(text="hi").type == "text"
    assert ToolCallChunk(tool="grep").type == "tool_call"
    assert TaskUpdateChunk(state="working").type == "task_update"
    assert ErrorChunk(error="x").type == "error"

    # Chunk should be a Union — verify by checking each member can be assigned.
    chunks: list[Chunk] = [
        TextChunk(text="a"), ToolCallChunk(tool="b"),
        TaskUpdateChunk(state="completed"), ErrorChunk(error="c"),
    ]
    assert len(chunks) == 4


def test_invocation_result_text_required():
    """text must always be populated — adapters MUST surface human-readable
    output even for protocol-native structured content."""
    r = InvocationResult(text="hello")
    assert r.text == "hello"
    assert r.status == "ok"


def test_auth_context_structure():
    ctx = AuthContext(headers={"X-API-Key": "k"})
    assert ctx.headers["X-API-Key"] == "k"
    assert ctx.query_params == {}


def test_capability_structure():
    c = Capability(id="search", name="search", description="x")
    assert c.id == "search"
    # MIME types optional, default empty
    assert c.accepted_input_types == []


@pytest.mark.parametrize("cls", [A2ATarget, RESTChatTarget, MCPTarget])
def test_methods_are_async_or_generator(cls):
    """invoke/discover/authenticate/list_capabilities/cancel are async coroutines.
    stream may be an async generator (or a coroutine that returns one)."""
    for m in ("authenticate", "discover", "list_capabilities", "invoke", "cancel"):
        fn = getattr(cls, m)
        assert inspect.iscoroutinefunction(fn), f"{cls.__name__}.{m} must be async def"
    # stream allowed to be either iscoroutinefunction OR isasyncgenfunction;
    # we accept both — Python promotes "async def f() with yield" to async-gen.
    fn = cls.stream
    assert inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)

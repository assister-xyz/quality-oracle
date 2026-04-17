"""Tests for MCP client — uses mocks, no real MCP server needed."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.mcp_client import (
    connect_and_list_tools,
    call_tool,
    get_server_manifest,
    evaluate_server,
    manifest_and_evaluate,
    _call_tool_in_session,
)


def _make_mock_tool(name="calculate", description="Do math", schema=None):
    """Create a mock MCP tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {"type": "object", "properties": {"expression": {"type": "string"}}}
    return tool


def _make_mock_result(text="result", is_error=False):
    """Create a mock CallToolResult."""
    from mcp.types import TextContent
    content_item = TextContent(type="text", text=text)

    result = MagicMock()
    result.content = [content_item]
    result.isError = is_error
    return result


def _make_mock_init_result(name="TestServer", version="1.0.0"):
    """Create a mock InitializeResult."""
    init_result = MagicMock()
    init_result.serverInfo = MagicMock()
    init_result.serverInfo.name = name
    init_result.serverInfo.version = version
    return init_result


@pytest.fixture()
def mock_sse_session():
    """Patch sse_client + ClientSession to return mock session."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(return_value=_make_mock_init_result())

    tools_result = MagicMock()
    tools_result.tools = [_make_mock_tool()]
    mock_session.list_tools = AsyncMock(return_value=tools_result)
    mock_session.call_tool = AsyncMock(return_value=_make_mock_result('{"result": 42}'))

    return mock_session


def _patch_sse(mock_session):
    """Create context manager patches for sse_client and ClientSession."""
    mock_sse_cm = MagicMock()
    mock_sse_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
    mock_sse_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client_cm = MagicMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)

    return (
        patch("src.core.mcp_client.sse_client", return_value=mock_sse_cm),
        patch("src.core.mcp_client.ClientSession", return_value=mock_client_cm),
    )


@pytest.mark.asyncio
async def test_connect_and_list_tools(mock_sse_session):
    """Should connect and return tool dicts."""
    p1, p2 = _patch_sse(mock_sse_session)
    with p1, p2:
        tools = await connect_and_list_tools("http://localhost:8010/sse")
    assert len(tools) == 1
    assert tools[0]["name"] == "calculate"
    assert tools[0]["description"] == "Do math"


@pytest.mark.asyncio
async def test_connect_failure():
    """Should raise ConnectionError on connection failure."""
    with patch("src.core.mcp_client.sse_client", side_effect=Exception("refused")):
        with pytest.raises(ConnectionError, match="Cannot reach"):
            await get_server_manifest("http://bad-server:9999/sse")


@pytest.mark.asyncio
async def test_call_tool_success(mock_sse_session):
    """Should call tool and return content + latency."""
    p1, p2 = _patch_sse(mock_sse_session)
    with p1, p2:
        result = await call_tool("http://localhost:8010/sse", "calculate", {"expression": "2+2"})
    assert result["content"] == '{"result": 42}'
    assert result["is_error"] is False
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_call_tool_timeout():
    """Should handle timeout gracefully."""
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(side_effect=asyncio.TimeoutError())

    import time
    result = await _call_tool_in_session(mock_session, "slow_tool", {}, time.time())
    assert result["is_error"] is True
    assert "timed out" in result["content"]


@pytest.mark.asyncio
async def test_call_tool_error():
    """Should handle tool call errors gracefully."""
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(side_effect=RuntimeError("tool broke"))

    import time
    result = await _call_tool_in_session(mock_session, "broken_tool", {}, time.time())
    assert result["is_error"] is True
    assert "failed" in result["content"]


@pytest.mark.asyncio
async def test_get_server_manifest(mock_sse_session):
    """Should return manifest with server info and tools."""
    # Mock the pre-flight HTTP check that get_server_manifest does
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_http_client = AsyncMock()
    mock_http_client.head = AsyncMock(return_value=mock_response)
    mock_http_client.post = AsyncMock(return_value=mock_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=None)

    p1, p2 = _patch_sse(mock_sse_session)
    with p1, p2, patch("src.core.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
        manifest = await get_server_manifest("http://localhost:8010/sse")
    assert manifest["name"] == "TestServer"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["tools"]) == 1


@pytest.mark.asyncio
async def test_evaluate_server_full_flow(mock_sse_session):
    """Should run full evaluation: list tools, generate tests, call tools."""
    p1, p2 = _patch_sse(mock_sse_session)
    with p1, p2:
        results = await evaluate_server("http://localhost:8010/sse")
    assert "calculate" in results
    assert len(results["calculate"]) > 0
    # Each result should have question, expected, answer, latency_ms
    first = results["calculate"][0]
    assert "question" in first
    assert "expected" in first
    assert "answer" in first
    assert "latency_ms" in first


def _patch_all_transports_with_session(mock_session):
    """Patch sse_client, streamablehttp_client, and ClientSession to use one
    shared mock session. The fallback transport gets the *same* session so
    both code paths exercise the same failure behavior.
    """
    mock_sse_cm = MagicMock()
    mock_sse_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
    mock_sse_cm.__aexit__ = AsyncMock(return_value=False)

    mock_shttp_cm = MagicMock()
    # Streamable HTTP yields a 3-tuple (read, write, get_session_id)
    mock_shttp_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock(), AsyncMock()))
    mock_shttp_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client_cm = MagicMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)

    return (
        patch("src.core.mcp_client.sse_client", return_value=mock_sse_cm),
        patch("mcp.client.streamable_http.streamablehttp_client", return_value=mock_shttp_cm),
        patch("src.core.mcp_client.ClientSession", return_value=mock_client_cm),
    )


@pytest.mark.asyncio
async def test_manifest_session_init_error_surfaces_as_connection_error():
    """Regression (QO-049): when session.initialize() raises inside the
    _connect context manager, the error must propagate as ConnectionError.

    Previously _connect set yielded=True *before* the caller ran, so caller
    exceptions were swallowed and get_server_manifest returned None, which
    upstream code then crashed with AttributeError: 'NoneType'.
    """
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(side_effect=RuntimeError("handshake exploded"))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_http = AsyncMock()
    mock_http.head = AsyncMock(return_value=mock_response)
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)

    p1, p2, p3 = _patch_all_transports_with_session(mock_session)
    with p1, p2, p3, patch("src.core.mcp_client.httpx.AsyncClient", return_value=mock_http):
        with pytest.raises(ConnectionError) as excinfo:
            await get_server_manifest("http://localhost:8010/sse")
    # Must not be a silent None return that crashes later with NoneType.
    assert "handshake exploded" in str(excinfo.value) or "Cannot connect" in str(excinfo.value)


@pytest.mark.asyncio
async def test_evaluate_server_list_tools_error_surfaces():
    """Regression (QO-049): session.list_tools() raising must propagate,
    not be swallowed as a post-yield cleanup error.
    """
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(return_value=_make_mock_init_result())
    mock_session.list_tools = AsyncMock(side_effect=RuntimeError("tools listing failed"))

    p1, p2, p3 = _patch_all_transports_with_session(mock_session)
    with p1, p2, p3:
        with pytest.raises(RuntimeError, match="tools listing failed"):
            await evaluate_server("http://localhost:8010/sse")


@pytest.mark.asyncio
async def test_manifest_and_evaluate_uses_single_session():
    """QO-049: combined manifest + eval path must open exactly one MCP
    session. Opening a second one breaks servers that can't re-handshake
    rapidly (Peek.com, Browserbase, CoinGecko)."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(return_value=_make_mock_init_result())
    tools_result = MagicMock()
    tools_result.tools = [_make_mock_tool()]
    mock_session.list_tools = AsyncMock(return_value=tools_result)
    mock_session.call_tool = AsyncMock(return_value=_make_mock_result('{"result": 42}'))

    # Pre-flight HTTP mock (manifest_and_evaluate does DNS/404/5xx check first)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_http = AsyncMock()
    mock_http.head = AsyncMock(return_value=mock_response)
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)

    # Count ClientSession instantiations
    mock_client_cm = MagicMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)
    client_session_factory = MagicMock(return_value=mock_client_cm)

    mock_sse_cm = MagicMock()
    mock_sse_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
    mock_sse_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.core.mcp_client.sse_client", return_value=mock_sse_cm), \
         patch("src.core.mcp_client.ClientSession", client_session_factory), \
         patch("src.core.mcp_client.httpx.AsyncClient", return_value=mock_http):
        manifest, tool_responses = await manifest_and_evaluate("http://localhost:8010/sse")

    # The whole point of the refactor: exactly one ClientSession instance.
    assert client_session_factory.call_count == 1, (
        f"Expected 1 ClientSession, got {client_session_factory.call_count}. "
        "Two sessions break servers that can't handshake twice."
    )
    assert mock_session.initialize.await_count == 1
    assert manifest["name"] == "TestServer"
    assert len(manifest["tools"]) == 1
    assert "calculate" in tool_responses
    assert len(tool_responses["calculate"]) > 0
    first = tool_responses["calculate"][0]
    assert first["answer"] == '{"result": 42}'


@pytest.mark.asyncio
async def test_manifest_and_evaluate_preflight_404_surfaces():
    """404 must be caught by the pre-flight HTTP check, before the
    MCP handshake wastes a connection."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_http = AsyncMock()
    mock_http.head = AsyncMock(return_value=mock_response)
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)

    with patch("src.core.mcp_client.httpx.AsyncClient", return_value=mock_http):
        with pytest.raises(ConnectionError, match="404"):
            await manifest_and_evaluate("https://moved.example/mcp")

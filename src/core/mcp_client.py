"""
MCP Client wrapper for connecting to and evaluating target MCP servers.

Supports both SSE and Streamable HTTP transports with auto-detection
and fallback. Uses the official MCP SDK.
"""
import asyncio
import contextvars
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent

logger = logging.getLogger(__name__)

# QO-047: Context-local evaluation_id for audit logging.
# Set by the evaluator at the start of each eval, so all downstream
# tool calls can write audit records linked by eval_id.
current_evaluation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_evaluation_id", default=None
)
current_target_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_target_id", default=None
)
_call_index_counter: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_call_index_counter", default=0
)

# Timeouts
CONNECT_TIMEOUT = 10  # seconds
TOOL_CALL_TIMEOUT = 30  # seconds
SSE_READ_TIMEOUT = 60  # seconds


def _detect_transport(url: str) -> str:
    """Detect transport type from URL pattern.

    Returns "sse" for URLs ending in /sse, "streamable_http" otherwise.
    """
    if url.rstrip("/").endswith("/sse"):
        return "sse"
    return "streamable_http"


@asynccontextmanager
async def _connect(url: str, transport: str = "auto") -> AsyncGenerator[Tuple[str, ClientSession], None]:
    """Connect to an MCP server with transport auto-detection and fallback.

    Yields (transport_used, session) tuple.

    If the primary transport connects and the caller's work completes, cleanup
    errors (e.g. TaskGroup teardown) are suppressed — the caller already got
    its data. Fallback only triggers when the *connection itself* fails.
    """
    if transport == "auto":
        transport = _detect_transport(url)

    transports_to_try = [transport]
    # Add fallback: if we try streamable_http first, fall back to SSE and vice versa
    if transport == "streamable_http":
        transports_to_try.append("sse")
    elif transport == "sse":
        transports_to_try.append("streamable_http")

    last_error = None
    for t in transports_to_try:
        yielded = False
        try:
            if t == "sse":
                async with sse_client(
                    url=url,
                    timeout=CONNECT_TIMEOUT,
                    sse_read_timeout=SSE_READ_TIMEOUT,
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        yielded = True
                        yield t, session
                        return
            else:
                # Streamable HTTP (3-tuple: read, write, get_session_id)
                from mcp.client.streamable_http import streamablehttp_client
                async with streamablehttp_client(
                    url=url,
                    timeout=CONNECT_TIMEOUT,
                    sse_read_timeout=SSE_READ_TIMEOUT,
                ) as (read, write, _get_session_id):
                    async with ClientSession(read, write) as session:
                        yielded = True
                        yield t, session
                        return
        except Exception as e:
            if yielded:
                # Caller already got its data — this is a cleanup error.
                # Suppress it instead of falling through to a doomed fallback.
                logger.debug(f"Transport {t} cleanup error for {url} (suppressed): {e}")
                return
            last_error = e
            if len(transports_to_try) > 1 and t == transports_to_try[0]:
                logger.info(f"Transport {t} failed to connect to {url}, trying fallback: {e}")
                continue
            raise

    if last_error:
        raise last_error


async def connect_and_list_tools(server_url: str) -> List[dict]:
    """Connect to an MCP server and list its tools."""
    logger.info(f"Connecting to MCP server: {server_url}")
    async with _connect(server_url) as (transport_used, session):
        await session.initialize()
        result = await session.list_tools()
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema if tool.inputSchema else {},
            })
        logger.info(f"Listed {len(tools)} tools from {server_url} via {transport_used}")
        return tools


async def call_tool(
    server_url: str, tool_name: str, arguments: dict
) -> dict:
    """Call a specific tool on an MCP server via a new session."""
    logger.info(f"Calling tool {tool_name} on {server_url}")
    start = time.time()
    async with _connect(server_url) as (_, session):
        await session.initialize()
        return await _call_tool_in_session(session, tool_name, arguments, start)


async def _call_tool_in_session(
    session: ClientSession,
    tool_name: str,
    arguments: dict,
    start: float,
    timeout: int = TOOL_CALL_TIMEOUT,
) -> dict:
    """Call a tool within an existing session. Handles timeout and errors.

    QO-047: Best-effort audit log of every call (success or failure)
    via current_evaluation_id contextvar.
    """
    result_dict: dict
    try:
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=timeout,
        )
        latency_ms = int((time.time() - start) * 1000)

        # Extract text content
        text_parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
            else:
                text_parts.append(str(content))

        result_dict = {
            "content": "\n".join(text_parts),
            "is_error": result.isError or False,
            "latency_ms": latency_ms,
        }
    except asyncio.TimeoutError:
        latency_ms = int((time.time() - start) * 1000)
        logger.warning(f"Tool call {tool_name} timed out after {timeout}s")
        result_dict = {
            "content": f"Tool call timed out after {timeout}s",
            "is_error": True,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.error(f"Tool call {tool_name} failed: {e}")
        result_dict = {
            "content": f"Tool call failed: {e}",
            "is_error": True,
            "latency_ms": latency_ms,
        }

    # QO-047: Audit log (best-effort, non-blocking)
    eval_id = current_evaluation_id.get()
    if eval_id:
        try:
            from src.core.audit_log import log_tool_call
            call_idx = _call_index_counter.get()
            _call_index_counter.set(call_idx + 1)
            await log_tool_call(
                evaluation_id=eval_id,
                target_id=current_target_id.get(),
                tool_name=tool_name,
                arguments=arguments,
                response_text=result_dict["content"],
                is_error=result_dict["is_error"],
                latency_ms=result_dict["latency_ms"],
                call_index=call_idx,
            )
        except Exception as e:
            logger.debug(f"Audit log skipped: {e}")

    return result_dict


async def call_tools_batch(
    server_url: str,
    calls: List[Dict[str, dict]],
    per_call_timeout: int = TOOL_CALL_TIMEOUT,
) -> List[dict]:
    """Call multiple tools on an MCP server reusing a single session.

    Each entry in `calls` should have keys "tool_name" and "arguments".
    Returns a list of result dicts in the same order.
    """
    logger.info(f"Batch calling {len(calls)} tools on {server_url} (timeout={per_call_timeout}s each)")
    results = []
    async with _connect(server_url) as (_, session):
        await session.initialize()
        for call in calls:
            start = time.time()
            r = await _call_tool_in_session(
                session, call["tool_name"], call["arguments"], start,
                timeout=per_call_timeout,
            )
            results.append(r)
    return results


async def get_server_manifest(server_url: str) -> dict:
    """
    Get the server manifest (name, version, description, tools).

    Raises ConnectionError if the server cannot be reached.
    """
    logger.info(f"Fetching manifest from {server_url}")

    # Pre-flight check: detect HTTP/DNS errors before MCP protocol handshake
    transport_hint = _detect_transport(server_url)
    try:
        async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT) as http:
            if transport_hint == "sse":
                # SSE endpoints use GET — just do a HEAD to verify reachability
                preflight = await http.head(server_url)
            else:
                preflight = await http.post(
                    server_url,
                    json={"jsonrpc": "2.0", "method": "initialize", "id": 0, "params": {}},
                    headers={"Accept": "application/json, text/event-stream"},
                )
            if preflight.status_code in (401, 403):
                raise ConnectionError(f"MCP server requires authentication ({preflight.status_code}): {server_url}")
            if preflight.status_code == 404:
                raise ConnectionError(f"MCP endpoint not found (404): {server_url}")
            if preflight.status_code >= 500:
                raise ConnectionError(f"MCP server error ({preflight.status_code}): {server_url}")
    except ConnectionError:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise ConnectionError(f"Cannot reach MCP server (DNS/network error): {server_url}") from e
    except Exception as e:
        logger.debug(f"Pre-flight check for {server_url} inconclusive: {e}")

    try:
        async with _connect(server_url) as (transport_used, session):
            init_result = await session.initialize()

            server_info = init_result.serverInfo
            tools_result = await session.list_tools()

            tools = []
            for tool in tools_result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema if tool.inputSchema else {},
                })

            manifest = {
                "name": server_info.name if server_info else "unknown",
                "version": server_info.version if server_info and server_info.version else "0.0.0",
                "description": "",
                "tools": tools,
                "transport": transport_used,
            }
            logger.info(
                f"Manifest: {manifest['name']} v{manifest['version']} "
                f"with {len(tools)} tools via {transport_used}"
            )
            return manifest
    except Exception as e:
        # Try to extract HTTP status from the exception chain
        error_str = str(e)
        cause = e.__cause__ or e.__context__
        cause_str = str(cause) if cause else ""
        full_error = f"{error_str} {cause_str}".strip()
        logger.error(f"Failed to connect to {server_url}: {full_error}")
        raise ConnectionError(f"Cannot connect to MCP server at {server_url}: {full_error}") from e


async def check_response_consistency(
    server_url: str,
    tools: List[dict],
    sample_size: int = 2,
) -> Dict[str, float]:
    """Check idempotency: call same tool with same inputs twice, compare.

    Returns dict of tool_name -> consistency_ratio (0.0-1.0).
    A ratio of 1.0 means identical responses both times.

    Args:
        server_url: MCP server URL
        tools: List of tool definitions (from manifest)
        sample_size: Number of tools to test (default: 2, to keep it fast)
    """
    from src.core.test_generator import _generate_sample_input

    consistency: Dict[str, float] = {}

    # Pick up to sample_size tools that have input schemas
    candidates = [t for t in tools if t.get("inputSchema", {}).get("properties")]
    test_tools = candidates[:sample_size] if candidates else tools[:sample_size]

    if not test_tools:
        return consistency

    try:
        async with _connect(server_url) as (transport_used, session):
            await session.initialize()

            for tool in test_tools:
                name = tool["name"]
                schema = tool.get("inputSchema", {})
                input_data = _generate_sample_input(schema, variation=0)

                # Call twice with same inputs
                start1 = time.time()
                resp1 = await _call_tool_in_session(session, name, input_data, start1)
                start2 = time.time()
                resp2 = await _call_tool_in_session(session, name, input_data, start2)

                # Compare content (ignore latency differences)
                c1 = resp1.get("content", "").strip()
                c2 = resp2.get("content", "").strip()

                if c1 == c2:
                    consistency[name] = 1.0
                elif not c1 or not c2:
                    consistency[name] = 0.0
                else:
                    # Partial match — compute character-level similarity
                    min(len(c1), len(c2))
                    longer = max(len(c1), len(c2))
                    if longer == 0:
                        consistency[name] = 1.0
                    else:
                        # Simple ratio: common prefix length / max length
                        common = 0
                        for a, b in zip(c1, c2):
                            if a == b:
                                common += 1
                            else:
                                break
                        # Use a more robust metric: matching chars / total
                        match_chars = sum(1 for a, b in zip(c1, c2) if a == b)
                        consistency[name] = round(match_chars / longer, 2)

                logger.debug(f"Consistency {name}: {consistency[name]}")
    except Exception as e:
        logger.warning(f"Consistency check failed for {server_url}: {e}")

    return consistency


async def evaluate_server_streaming(
    server_url: str,
    cancel: Optional[object] = None,
    test_types: Optional[set] = None,
    max_tools: Optional[int] = None,
) -> AsyncGenerator[Tuple[str, dict, dict], None]:
    """Stream tool call results as they complete.

    Yields (tool_name, test_case, response) tuples one at a time instead of
    waiting for all tool calls to finish. Caller can cancel via the token.

    Args:
        server_url: MCP server URL
        cancel: Optional CancellationToken for early termination
        test_types: If provided, only run these test types
        max_tools: If provided, limit to first N tools
    """
    from src.core.test_generator import generate_test_cases

    logger.info(f"Starting streaming evaluation of {server_url}")

    async with _connect(server_url) as (transport_used, session):
        await session.initialize()
        logger.info(f"Connected to {server_url} via {transport_used} (streaming)")

        tools_result = await session.list_tools()
        tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema if t.inputSchema else {},
            }
            for t in tools_result.tools
        ]

        test_cases = generate_test_cases(tools, test_types=test_types, max_tools=max_tools)

        for tool_name, cases in test_cases.items():
            for case in cases:
                if cancel and cancel.is_cancelled:
                    logger.info(f"Streaming cancelled: {cancel.reason}")
                    return
                start = time.time()
                response = await _call_tool_in_session(
                    session, tool_name, case.get("input_data", {}), start
                )
                yield tool_name, case, response


async def evaluate_server(
    server_url: str,
    test_types: Optional[set] = None,
    max_tools: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """
    Full evaluation flow:
    1. Connect to server (auto-detect transport)
    2. List tools
    3. Generate test cases per tool
    4. Call each tool with test inputs
    5. Return responses for judging

    Args:
        server_url: MCP server URL
        test_types: If provided, only run these test types
        max_tools: If provided, limit to first N tools

    Returns dict of tool_name -> list of {question, expected, answer, latency_ms, is_error}
    """
    from src.core.test_generator import generate_test_cases

    logger.info(f"Starting full evaluation of {server_url}")

    async with _connect(server_url) as (transport_used, session):
        await session.initialize()
        logger.info(f"Connected to {server_url} via {transport_used}")

        # List tools
        tools_result = await session.list_tools()
        tools = []
        for tool in tools_result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema if tool.inputSchema else {},
            })

        # Generate test cases
        test_cases = generate_test_cases(tools, test_types=test_types, max_tools=max_tools)

        # Execute each test case
        results: Dict[str, List[dict]] = {}
        for tool_name, cases in test_cases.items():
            tool_results = []
            for case in cases:
                start = time.time()
                response = await _call_tool_in_session(
                    session, tool_name, case.get("input_data", {}), start
                )
                tool_results.append({
                    "question": case["question"],
                    "expected": case["expected"],
                    "answer": response["content"],
                    "latency_ms": response["latency_ms"],
                    "is_error": response["is_error"],
                    "test_type": case.get("test_type", "unknown"),
                })
            results[tool_name] = tool_results

        total_cases = sum(len(v) for v in results.values())
        logger.info(
            f"Evaluation complete: {len(tools)} tools, "
            f"{total_cases} test cases executed via {transport_used}"
        )
        return results

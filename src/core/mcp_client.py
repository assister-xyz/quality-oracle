"""
MCP Client wrapper for connecting to and evaluating target MCP servers.

Uses the official MCP SDK SSE transport to connect to target servers,
list their tools, and call them with test inputs.
"""
import asyncio
import logging
import time
from typing import Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent

logger = logging.getLogger(__name__)

# Timeouts
CONNECT_TIMEOUT = 10  # seconds
TOOL_CALL_TIMEOUT = 30  # seconds
SSE_READ_TIMEOUT = 60  # seconds


async def connect_and_list_tools(server_url: str) -> List[dict]:
    """Connect to an MCP server via SSE and list its tools."""
    logger.info(f"Connecting to MCP server: {server_url}")
    async with sse_client(
        url=server_url,
        timeout=CONNECT_TIMEOUT,
        sse_read_timeout=SSE_READ_TIMEOUT,
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for tool in result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema if tool.inputSchema else {},
                })
            logger.info(f"Listed {len(tools)} tools from {server_url}")
            return tools


async def call_tool(
    server_url: str, tool_name: str, arguments: dict
) -> dict:
    """Call a specific tool on an MCP server via a new SSE session."""
    logger.info(f"Calling tool {tool_name} on {server_url}")
    start = time.time()
    async with sse_client(
        url=server_url,
        timeout=CONNECT_TIMEOUT,
        sse_read_timeout=SSE_READ_TIMEOUT,
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await _call_tool_in_session(session, tool_name, arguments, start)


async def _call_tool_in_session(
    session: ClientSession,
    tool_name: str,
    arguments: dict,
    start: float,
) -> dict:
    """Call a tool within an existing session. Handles timeout and errors."""
    try:
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=TOOL_CALL_TIMEOUT,
        )
        latency_ms = int((time.time() - start) * 1000)

        # Extract text content
        text_parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
            else:
                text_parts.append(str(content))

        return {
            "content": "\n".join(text_parts),
            "is_error": result.isError or False,
            "latency_ms": latency_ms,
        }
    except asyncio.TimeoutError:
        latency_ms = int((time.time() - start) * 1000)
        logger.warning(f"Tool call {tool_name} timed out after {TOOL_CALL_TIMEOUT}s")
        return {
            "content": f"Tool call timed out after {TOOL_CALL_TIMEOUT}s",
            "is_error": True,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.error(f"Tool call {tool_name} failed: {e}")
        return {
            "content": f"Tool call failed: {e}",
            "is_error": True,
            "latency_ms": latency_ms,
        }


async def get_server_manifest(server_url: str) -> dict:
    """
    Get the server manifest (name, version, description, tools).

    Raises ConnectionError if the server cannot be reached.
    """
    logger.info(f"Fetching manifest from {server_url}")
    try:
        async with sse_client(
            url=server_url,
            timeout=CONNECT_TIMEOUT,
            sse_read_timeout=SSE_READ_TIMEOUT,
        ) as (read, write):
            async with ClientSession(read, write) as session:
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
                }
                logger.info(
                    f"Manifest: {manifest['name']} v{manifest['version']} "
                    f"with {len(tools)} tools"
                )
                return manifest
    except Exception as e:
        logger.error(f"Failed to connect to {server_url}: {e}")
        raise ConnectionError(f"Cannot connect to MCP server at {server_url}: {e}") from e


async def evaluate_server(server_url: str) -> Dict[str, List[dict]]:
    """
    Full evaluation flow using a single SSE session:
    1. Connect to server
    2. List tools
    3. Generate test cases per tool
    4. Call each tool with test inputs
    5. Return responses for judging

    Returns dict of tool_name -> list of {question, expected, answer, latency_ms, is_error}
    """
    from src.core.test_generator import generate_test_cases

    logger.info(f"Starting full evaluation of {server_url}")

    async with sse_client(
        url=server_url,
        timeout=CONNECT_TIMEOUT,
        sse_read_timeout=SSE_READ_TIMEOUT,
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

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
            test_cases = generate_test_cases(tools)

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
                    })
                results[tool_name] = tool_results

            total_cases = sum(len(v) for v in results.values())
            logger.info(
                f"Evaluation complete: {len(tools)} tools, "
                f"{total_cases} test cases executed"
            )
            return results

"""
AgentTrust MCP Server — standalone package.

Exposes quality evaluation tools via MCP protocol.
Install: pip install mcp-agenttrust
Run: mcp-agenttrust
"""
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main():
    """Run AgentTrust MCP server.

    If running inside the full quality-oracle repo, delegates to
    src.standards.mcp_server which has full evaluator, judge, and DB access.

    If running as standalone pip package, starts a lightweight proxy
    that forwards requests to a running AgentTrust API instance.
    """
    # Try to import from full repo first
    try:
        from src.standards.mcp_server import main as _main
        _main()
        return
    except ImportError:
        pass

    # Standalone mode: lightweight proxy to AgentTrust API
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("Error: mcp package not installed. Run: pip install mcp-agenttrust")
        sys.exit(1)

    import httpx

    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    api_url = os.environ.get("AGENTTRUST_API_URL", "http://localhost:8002")
    api_key = os.environ.get("AGENTTRUST_API_KEY", "")

    mcp = FastMCP(
        "agenttrust",
        log_level=_log_level,
        instructions="AgentTrust — AI Agent & MCP Server quality verification. "
                     f"Proxying to {api_url}",
    )

    def _headers():
        h = {"Content-Type": "application/json"}
        if api_key:
            h["X-API-Key"] = api_key
        return h

    @mcp.tool()
    async def check_quality(server_url: str) -> str:
        """Evaluate an MCP server's quality. Runs manifest + functional tests with LLM judge scoring.

        Args:
            server_url: The URL of the MCP server to evaluate (e.g., "http://localhost:8010/sse")
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{api_url}/v1/evaluate",
                json={"target_url": server_url, "level": 2},
                headers=_headers(),
            )
            return resp.text

    @mcp.tool()
    async def get_score(target_id: str) -> str:
        """Look up the quality score for an MCP server or AI agent.

        Args:
            target_id: The server URL or target identifier to look up
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{api_url}/v1/score/{target_id}",
                headers=_headers(),
            )
            return resp.text

    @mcp.tool()
    async def verify_attestation(attestation_id: str) -> str:
        """Verify an AQVC quality attestation. Checks signature validity and returns decoded payload.

        Args:
            attestation_id: The attestation ID or JWT to verify
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{api_url}/v1/attestation/{attestation_id}/verify",
                headers=_headers(),
            )
            return resp.text

    @mcp.tool()
    async def list_scores(domain: str = "", min_score: int = 0) -> str:
        """Search quality scores across evaluated MCP servers and AI agents.

        Args:
            domain: Filter by domain (e.g., "mcp_protocol", "defi", "general")
            min_score: Minimum score threshold (0-100)
        """
        params = {}
        if domain:
            params["domain"] = domain
        if min_score > 0:
            params["min_score"] = min_score
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{api_url}/v1/scores",
                params=params,
                headers=_headers(),
            )
            return resp.text

    mcp.run(transport="sse")


if __name__ == "__main__":
    main()

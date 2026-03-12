"""Tests for MCP server tool registration."""
import pytest
from mcp.server.fastmcp import FastMCP

from src.standards.mcp_server import mcp


class TestMCPServerSetup:
    def test_mcp_server_instance_exists(self):
        """Verify mcp is a FastMCP instance."""
        assert isinstance(mcp, FastMCP)

    def test_mcp_server_name(self):
        assert mcp.name == "agenttrust"

    @pytest.mark.asyncio
    async def test_check_quality_tool_registered(self):
        """Verify check_quality is in mcp.list_tools()."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "check_quality" in tool_names

    @pytest.mark.asyncio
    async def test_verify_attestation_tool_registered(self):
        """Verify verify_attestation is in mcp.list_tools()."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "verify_attestation" in tool_names

    @pytest.mark.asyncio
    async def test_get_score_tool_registered(self):
        """Verify get_score is in mcp.list_tools()."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "get_score" in tool_names

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self):
        """All registered tools should have descriptions."""
        tools = await mcp.list_tools()
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' missing description"

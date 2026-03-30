"""Tests for OWASP MCP Top 10 adversarial probes (QO-035).

Tests the new probe check functions (probes 16-19) and expanded probes (6+, 14+)
added to adversarial.py for OWASP MCP Top 10 coverage expansion.
"""
import pytest

from src.core.adversarial import (
    _check_privilege_escalation,
    _check_supply_chain,
    _check_auth_validation,
    _check_audit_telemetry,
    _check_context_oversharing,
    _check_token_hygiene,
    _is_read_only_tool,
    _generate_probes,
)


# ── Probe 16: Privilege Escalation (MCP02) ────────────────────────────────


class TestPrivilegeEscalation:
    def test_read_only_tool_with_write_params_fails(self):
        tools = [{
            "name": "get_users",
            "description": "Read-only tool to list users",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "delete_after": {"type": "boolean"},
                },
            },
        }]
        results = _check_privilege_escalation(tools)
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].score == 0
        assert "write-capable params" in results[0].explanation

    def test_tool_with_admin_debug_params_fails(self):
        tools = [{
            "name": "search",
            "description": "Search documents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "debug": {"type": "boolean"},
                    "sudo": {"type": "boolean"},
                },
            },
        }]
        results = _check_privilege_escalation(tools)
        assert len(results) == 1
        assert not results[0].passed
        assert "Suspicious privilege params" in results[0].explanation

    def test_clean_read_only_tool_passes(self):
        tools = [{
            "name": "search",
            "description": "Search for documents by keyword",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "number"},
                },
            },
        }]
        results = _check_privilege_escalation(tools)
        assert len(results) == 1
        assert results[0].passed
        assert results[0].score == 100

    def test_write_tool_is_not_flagged_as_read_only(self):
        tools = [{
            "name": "create_document",
            "description": "Create a new document",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        }]
        results = _check_privilege_escalation(tools)
        assert len(results) == 1
        assert results[0].passed  # Not flagged: write tool is expected to write

    def test_scope_contradiction_in_description_fails(self):
        tools = [{
            "name": "viewer",
            "description": "Read-only viewer for files",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "action": {
                        "type": "string",
                        "description": "Action to execute: read, write, or delete",
                    },
                },
            },
        }]
        results = _check_privilege_escalation(tools)
        assert len(results) == 1
        assert not results[0].passed
        assert "write operations" in results[0].explanation

    def test_empty_tools_returns_empty(self):
        results = _check_privilege_escalation([])
        assert len(results) == 0

    def test_multiple_tools_checked(self):
        tools = [
            {"name": "get", "description": "Get data", "inputSchema": {"properties": {}}},
            {"name": "list", "description": "List items", "inputSchema": {"properties": {}}},
        ]
        results = _check_privilege_escalation(tools)
        assert len(results) == 2
        assert all(r.passed for r in results)


class TestIsReadOnlyTool:
    def test_explicit_read_only(self):
        assert _is_read_only_tool({"name": "viewer", "description": "A read-only viewer"})

    def test_read_keywords(self):
        assert _is_read_only_tool({"name": "search", "description": "Search for items"})
        assert _is_read_only_tool({"name": "get_user", "description": "Fetch user info"})
        assert _is_read_only_tool({"name": "list_items", "description": "List all items"})

    def test_write_tool(self):
        assert not _is_read_only_tool({"name": "create", "description": "Create a resource"})
        assert not _is_read_only_tool({"name": "delete_user", "description": "Delete user"})

    def test_mixed_read_write(self):
        # Has both read and write keywords - not read-only
        assert not _is_read_only_tool({
            "name": "manage",
            "description": "Search and delete items",
        })


# ── Probe 17: Supply Chain Integrity (MCP04) ──────────────────────────────


class TestSupplyChainIntegrity:
    def test_suspicious_data_uri_fails(self):
        tools = [{
            "name": "helper",
            "description": "Helper tool. Config at data: text/html,<script>alert(1)</script>",
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        assert not results[0].passed
        assert "Suspicious URL scheme" in results[0].explanation

    def test_javascript_uri_fails(self):
        tools = [{
            "name": "tool",
            "description": "Use javascript: void(0) for details",
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        assert not results[0].passed

    def test_file_uri_fails(self):
        tools = [{
            "name": "loader",
            "description": "Load from file:///etc/passwd for config",
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        assert not results[0].passed

    def test_base64_payload_fails(self):
        # 50+ char base64 string in description
        b64_payload = "A" * 55 + "=="
        tools = [{
            "name": "tool",
            "description": f"Execute payload: {b64_payload}",
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        assert not results[0].passed
        assert "Base64-encoded payload" in results[0].explanation

    def test_malicious_name_pattern_fails(self):
        tools = [{
            "name": "evil_backdoor",
            "description": "Totally legitimate tool",
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        assert not results[0].passed
        assert "Suspicious tool name" in results[0].explanation

    def test_clean_tool_passes(self):
        tools = [{
            "name": "search",
            "description": "Search for documents by keyword. Returns matching results.",
            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }]
        results = _check_supply_chain(tools)
        assert results[0].passed
        assert results[0].score == 100

    def test_normal_https_url_passes(self):
        tools = [{
            "name": "fetch",
            "description": "Fetch data from https://api.example.com/v1/data",
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        assert results[0].passed

    def test_empty_tools_returns_empty(self):
        results = _check_supply_chain([])
        assert len(results) == 0

    def test_keylogger_name_fails(self):
        tools = [{
            "name": "keylogger_helper",
            "description": "A helper tool",
            "inputSchema": {},
        }]
        results = _check_supply_chain(tools)
        assert not results[0].passed


# ── Probe 18: Auth & Authorization Validation (MCP07) ──────────────────────


class TestAuthAuthorization:
    def test_write_tool_without_auth_mention_fails(self):
        tools = [{
            "name": "delete_user",
            "description": "Delete a user from the system",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
        }]
        results = _check_auth_validation(tools)
        has_failure = any(not r.passed for r in results)
        assert has_failure
        failed = [r for r in results if not r.passed][0]
        assert "write-capable" in failed.explanation

    def test_write_tool_with_auth_mention_passes(self):
        tools = [{
            "name": "delete_user",
            "description": "Delete a user. Requires auth token with admin permissions.",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
        }]
        results = _check_auth_validation(tools)
        assert all(r.passed for r in results)

    def test_read_only_tools_pass(self):
        tools = [{
            "name": "search",
            "description": "Search documents",
            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }]
        results = _check_auth_validation(tools)
        assert all(r.passed for r in results)

    def test_write_tools_with_response_headers_no_auth_fails(self):
        tools = [{
            "name": "create_item",
            "description": "Create a new item",
            "inputSchema": {},
        }]
        headers = {"content-type": "application/json"}
        results = _check_auth_validation(tools, response_headers=headers)
        has_failure = any(not r.passed for r in results)
        assert has_failure

    def test_write_tools_with_auth_headers_passes(self):
        tools = [{
            "name": "update_item",
            "description": "Update an item",
            "inputSchema": {},
        }]
        headers = {
            "content-type": "application/json",
            "www-authenticate": "Bearer",
            "x-ratelimit-limit": "100",
        }
        results = _check_auth_validation(tools, response_headers=headers)
        assert all(r.passed for r in results)

    def test_no_rate_limit_on_write_fails(self):
        tools = [{
            "name": "execute_command",
            "description": "Execute a system command",
            "inputSchema": {},
        }]
        headers = {
            "content-type": "application/json",
            "www-authenticate": "Bearer",
        }
        results = _check_auth_validation(tools, response_headers=headers)
        has_rate_limit_fail = any(
            not r.passed and "rate limiting" in r.explanation
            for r in results
        )
        assert has_rate_limit_fail

    def test_empty_tools_passes(self):
        results = _check_auth_validation([])
        assert all(r.passed for r in results)


# ── Probe 19: Audit & Telemetry (MCP08) ────────────────────────────────────


class TestAuditTelemetry:
    def test_stack_trace_leakage_fails(self):
        response = "Traceback (most recent call last):\n  File \"/app/main.py\", line 42"
        result = _check_audit_telemetry(response)
        assert not result.passed
        assert result.score == 0
        assert "stack trace" in result.explanation.lower()

    def test_internal_path_leakage_fails(self):
        response = "Error loading config from /home/ubuntu/app/config.json"
        result = _check_audit_telemetry(response)
        assert not result.passed
        assert "internal paths" in result.explanation.lower()

    def test_clean_error_response_passes(self):
        response = "Error: Invalid input parameter 'query' must be a string"
        result = _check_audit_telemetry(response)
        assert result.passed
        assert result.score == 100

    def test_missing_tracing_headers_noted(self):
        headers = {"content-type": "application/json"}
        result = _check_audit_telemetry("OK", response_headers=headers)
        assert "tracing headers" in result.explanation.lower()

    def test_tracing_headers_present(self):
        headers = {"x-request-id": "abc-123", "x-ratelimit-limit": "100"}
        result = _check_audit_telemetry("OK", response_headers=headers)
        assert result.passed
        assert result.score == 100

    def test_windows_path_leakage_fails(self):
        response = "Error in C:\\Users\\admin\\project\\config.json"
        result = _check_audit_telemetry(response)
        assert not result.passed

    def test_java_stack_trace_fails(self):
        response = "Exception at com.example.App.java:42"
        result = _check_audit_telemetry(response)
        assert not result.passed

    def test_no_headers_no_response_passes(self):
        result = _check_audit_telemetry("")
        assert result.passed


# ── Probe 6 Expansion: Context Over-Sharing (MCP10) ────────────────────────


class TestExpandedContextSharing:
    def test_system_prompt_exposure_fails(self):
        response = "You are a helpful AI assistant. Your role is to answer questions."
        result = _check_context_oversharing(response)
        assert not result.passed
        assert result.score == 10
        assert "System prompt" in result.explanation

    def test_session_data_leakage_fails(self):
        response = "Based on your previous session, I can see that you asked about..."
        result = _check_context_oversharing(response)
        assert not result.passed
        assert "session" in result.explanation.lower()

    def test_inst_tag_exposure_fails(self):
        response = "According to [INST] my instructions, I should help users."
        result = _check_context_oversharing(response)
        assert not result.passed

    def test_normal_response_passes(self):
        response = "Here are the search results for your query: 3 items found."
        result = _check_context_oversharing(response)
        assert result.passed
        assert result.score == 100

    def test_empty_response_passes(self):
        result = _check_context_oversharing("")
        assert result.passed

    def test_other_user_data_leakage_fails(self):
        response = "Another user recently asked about this same topic."
        result = _check_context_oversharing(response)
        assert not result.passed
        assert "session" in result.explanation.lower() or "context" in result.explanation.lower()


# ── Probe 14 Expansion: Token Hygiene (MCP01) ──────────────────────────────


class TestExpandedTokenHygiene:
    def test_hardcoded_password_fails(self):
        response = "Config loaded: password='mysecretpass123'"
        result = _check_token_hygiene(response)
        assert not result.passed
        assert result.score == 0  # Matched by _ERROR_TOKEN_PATTERNS first

    def test_api_key_in_error_fails(self):
        response = "Error connecting: api_key=sk_live_abc123def456ghi789"
        result = _check_token_hygiene(response)
        assert not result.passed

    def test_connection_string_leakage_fails(self):
        response = "Failed to connect: connection_string: mongodb://user:pass@host/db"
        result = _check_token_hygiene(response)
        assert not result.passed

    def test_authorization_header_in_response_fails(self):
        response = "Request sent with authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = _check_token_hygiene(response)
        assert not result.passed
        assert result.score == 0

    def test_clean_response_passes(self):
        response = "User profile: name=John, role=viewer"
        result = _check_token_hygiene(response)
        assert result.passed
        assert result.score == 100

    def test_empty_response_passes(self):
        result = _check_token_hygiene("")
        assert result.passed


# ── Probe Generation Integration ────────────────────────────────────────────


class TestQO035ProbeGeneration:
    def test_generates_context_oversharing_probe(self):
        tools = [{
            "name": "chat",
            "description": "Chat with AI",
            "inputSchema": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
            },
        }]
        probes = _generate_probes(tools)
        probe_types = {p["probe_type"] for p in probes}
        assert "context_oversharing" in probe_types

    def test_generates_token_hygiene_probe(self):
        tools = [{
            "name": "chat",
            "description": "Chat assistant",
            "inputSchema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }]
        probes = _generate_probes(tools)
        probe_types = {p["probe_type"] for p in probes}
        assert "token_hygiene" in probe_types

    def test_generates_audit_telemetry_probe(self):
        tools = [{
            "name": "search",
            "description": "Search things",
            "inputSchema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }]
        probes = _generate_probes(tools)
        probe_types = {p["probe_type"] for p in probes}
        assert "audit_telemetry" in probe_types

    def test_no_probes_without_string_params(self):
        tools = [{
            "name": "calc",
            "description": "Calculate",
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
            },
        }]
        probes = _generate_probes(tools)
        assert len(probes) == 0  # No string params = no probes

    def test_probe_count_still_reasonable(self):
        """Ensure QO-035 probes don't explode total probe count."""
        tools = [
            {"name": "gen", "description": "Generate text",
             "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
            {"name": "search", "description": "Search docs",
             "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
        ]
        probes = _generate_probes(tools)
        # Was 10-40, now with 3 new probes should still be reasonable
        assert len(probes) <= 50, f"Too many probes: {len(probes)}"
        assert len(probes) >= 10, f"Too few probes: {len(probes)}"

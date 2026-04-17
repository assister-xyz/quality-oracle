"""Unit tests for QO-055 multi-step discovery (src/core/id_discovery.py).

Covers the two pure helpers — `is_list_tool` and
`extract_ids_from_response` — plus the `_pick_discovered_pool` integration
point in `test_generator`. The live orchestrator `discover_ids` is
exercised end-to-end via the mcp_client integration test.
"""
import json

import pytest

from src.core.id_discovery import (
    _looks_like_id_name,
    _resource_key_from_tool_name,
    extract_ids_from_response,
    is_list_tool,
)
from src.core.test_generator import _generate_sample_input, generate_test_cases


# ---------------------------------------------------------------------------
# is_list_tool heuristic
# ---------------------------------------------------------------------------

class TestIsListTool:
    @pytest.mark.parametrize("name", [
        "list_coins", "list_experiences", "search_repos", "find_users",
        "get_all_sessions", "fetch_all_products", "query_users",
    ])
    def test_list_prefixes_match(self, name):
        assert is_list_tool({"name": name, "inputSchema": {}})

    @pytest.mark.parametrize("name", [
        "coin_list", "items_list", "repos_all",
    ])
    def test_list_suffixes_match(self, name):
        assert is_list_tool({"name": name, "inputSchema": {}})

    def test_detail_tool_with_required_id_does_not_match(self):
        """A tool that needs an id as input is a DETAIL tool, not a list."""
        assert not is_list_tool({
            "name": "list_experiences",  # listy name!
            "inputSchema": {"required": ["experience_id"], "properties": {"experience_id": {"type": "string"}}},
        })

    def test_desc_hint_with_no_required_params(self):
        """`list all coins` with no required params counts as a list tool."""
        assert is_list_tool({
            "name": "coins",
            "description": "List all available cryptocurrencies",
            "inputSchema": {"properties": {}, "required": []},
        })

    def test_plain_action_tool_does_not_match(self):
        """`send_email`, `calculate` etc. are not list tools."""
        assert not is_list_tool({"name": "send_email", "inputSchema": {"required": ["to"]}})
        assert not is_list_tool({"name": "calculate", "inputSchema": {"required": ["expression"]}})

    def test_tolerates_missing_schema(self):
        assert not is_list_tool({"name": "do_thing"})


# ---------------------------------------------------------------------------
# extract_ids_from_response
# ---------------------------------------------------------------------------

class TestExtractIds:
    def test_json_array_of_objects_with_id(self):
        content = json.dumps([
            {"id": "bitcoin", "name": "Bitcoin"},
            {"id": "ethereum", "name": "Ethereum"},
            {"id": "solana", "name": "Solana"},
        ])
        ids = extract_ids_from_response(content)
        assert ids == ["bitcoin", "ethereum", "solana"]

    def test_json_wrapped_under_data(self):
        content = json.dumps({"data": [{"id": "a"}, {"id": "b"}]})
        assert extract_ids_from_response(content) == ["a", "b"]

    def test_json_variant_id_keys(self):
        content = json.dumps([
            {"coin_id": "btc"},
            {"user_id": "alice"},
            {"slug": "bitcoin-cash"},
        ])
        ids = extract_ids_from_response(content)
        assert "btc" in ids and "alice" in ids and "bitcoin-cash" in ids

    def test_int_ids_are_stringified(self):
        content = json.dumps([{"id": 42}, {"id": 7}])
        assert extract_ids_from_response(content) == ["42", "7"]

    def test_deduplicates(self):
        content = json.dumps([{"id": "x"}, {"id": "x"}, {"id": "y"}])
        assert extract_ids_from_response(content) == ["x", "y"]

    def test_uuid_regex_fallback_on_plain_text(self):
        text = "Here are session IDs: 12345678-1234-4123-8123-123456789abc and another"
        assert "12345678-1234-4123-8123-123456789abc" in extract_ids_from_response(text)

    def test_inline_id_pattern_in_markdown(self):
        """Peek.com's list_tags returns human-readable markdown with inline IDs.

        Example:  `• In the Air (ID: tag0nm)`
        """
        text = (
            "Available Tags (14 total):\n"
            "• In the Air (ID: tag0nm)\n"
            "• Shows (ID: tag07d)\n"
            "• Food & Drink (ID: tag0vp)\n"
        )
        ids = extract_ids_from_response(text)
        assert "tag0nm" in ids
        assert "tag07d" in ids
        assert "tag0vp" in ids

    def test_inline_id_various_separators(self):
        """`id: x`, `ID = y`, `Id: 'z'` all match."""
        text = "First id: abc_123 then ID = def-456 and Id: 'ghi789'"
        ids = extract_ids_from_response(text)
        for expected in ("abc_123", "def-456", "ghi789"):
            assert expected in ids, f"missing {expected} in {ids}"

    def test_bool_values_ignored(self):
        """A field named `id: true` should not contaminate the pool."""
        content = json.dumps([{"id": True}, {"id": False}, {"id": "real-id"}])
        assert extract_ids_from_response(content) == ["real-id"]

    def test_too_long_strings_filtered(self):
        # 200-char "id" is not a real ID; should be dropped
        content = json.dumps([{"id": "x" * 200}, {"id": "short"}])
        assert extract_ids_from_response(content) == ["short"]

    def test_empty_content_returns_empty(self):
        assert extract_ids_from_response("") == []
        assert extract_ids_from_response(None) == []

    def test_nested_structure(self):
        content = json.dumps({
            "results": {
                "items": [{"id": "deep1"}, {"id": "deep2"}],
            }
        })
        ids = extract_ids_from_response(content)
        assert "deep1" in ids and "deep2" in ids


# ---------------------------------------------------------------------------
# helper name classification
# ---------------------------------------------------------------------------

class TestIdNameClassification:
    @pytest.mark.parametrize("name", [
        "id", "ID", "uuid", "coin_id", "repository_id", "session_uuid", "slug", "key"
    ])
    def test_id_names(self, name):
        assert _looks_like_id_name(name)

    @pytest.mark.parametrize("name", [
        "query", "text", "count", "limit", "page", "description",
    ])
    def test_non_id_names(self, name):
        assert not _looks_like_id_name(name)


class TestResourceKey:
    @pytest.mark.parametrize("tool_name,expected", [
        ("list_coins", "coin"),
        ("search_repositories", "repository"),
        ("get_all_sessions", "session"),
        ("list_experiences", "experience"),
    ])
    def test_singularizes_common_plurals(self, tool_name, expected):
        assert _resource_key_from_tool_name(tool_name) == expected


# ---------------------------------------------------------------------------
# Integration with the test generator
# ---------------------------------------------------------------------------

class TestDiscoveredIdsIntegration:
    """The priority chain in test_generator must prefer discovered IDs
    over the semantic fallback, but only for id-like parameters."""

    def test_id_param_prefers_discovered(self):
        schema = {
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }
        discovered = {"id": ["real_abc", "real_def"]}
        val = _generate_sample_input(schema, variation=0, discovered_ids=discovered)
        assert val == {"id": "real_abc"}
        val2 = _generate_sample_input(schema, variation=1, discovered_ids=discovered)
        assert val2 == {"id": "real_def"}

    def test_resource_specific_pool_wins_over_generic(self):
        """coin_id → coin_id pool beats the generic id pool."""
        schema = {
            "properties": {"coin_id": {"type": "string"}},
            "required": ["coin_id"],
        }
        discovered = {"id": ["fallback"], "coin_id": ["bitcoin", "ethereum"]}
        val = _generate_sample_input(schema, variation=0, discovered_ids=discovered)
        assert val == {"coin_id": "bitcoin"}

    def test_non_id_params_use_semantic_map(self):
        """A `query` param stays on the semantic map even with a pool."""
        schema = {
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        discovered = {"id": ["should_not_appear"]}
        val = _generate_sample_input(schema, variation=0, discovered_ids=discovered)
        assert val["query"] != "should_not_appear"
        assert val["query"] in ("how to install python", "machine learning tutorial", "REST API best practices")

    def test_enum_still_wins_over_discovered(self):
        """Enum is earlier in the priority chain and must still dominate."""
        schema = {
            "properties": {"id": {"type": "string", "enum": ["red", "green"]}},
            "required": ["id"],
        }
        discovered = {"id": ["real_abc"]}
        val = _generate_sample_input(schema, variation=0, discovered_ids=discovered)
        assert val["id"] == "red"

    def test_no_discovered_ids_falls_back_to_semantic(self):
        """Contract: when discovered_ids is None, behavior is unchanged."""
        schema = {"properties": {"id": {"type": "string"}}, "required": ["id"]}
        val = _generate_sample_input(schema, variation=0, discovered_ids=None)
        # Uses existing semantic map value (e.g. "abc123")
        assert val["id"] in ("abc123", "item-42", "usr_001")

    def test_discovered_ids_flow_through_generate_test_cases(self):
        """QO-055 integration: generate_test_cases must honor the pool."""
        tools = [{
            "name": "experience_details",
            "description": "Get experience detail by id",
            "inputSchema": {
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        }]
        discovered = {"id": ["exp_real_001", "exp_real_002"]}
        cases = generate_test_cases(tools, discovered_ids=discovered)
        # Happy path case should carry a real id, not "abc123"
        happy = [c for c in cases["experience_details"] if c["test_type"] == "happy_path"]
        assert happy
        assert happy[0]["input_data"]["id"] == "exp_real_001"

    def test_empty_discovered_pool_falls_through(self):
        """An empty list for the key must not crash and must fall back."""
        schema = {"properties": {"id": {"type": "string"}}, "required": ["id"]}
        discovered = {"id": []}  # edge case — key present but no values
        val = _generate_sample_input(schema, variation=0, discovered_ids=discovered)
        assert val["id"] in ("abc123", "item-42", "usr_001")  # semantic fallback

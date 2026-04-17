"""
Auto-generate test cases from MCP server tool manifests.

Reads tool definitions (name, description, inputSchema) and generates
test cases for functional evaluation using semantic-aware input generation.
"""
import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic parameter maps — realistic values keyed by common param names
# ---------------------------------------------------------------------------
SEMANTIC_PARAM_MAP: Dict[str, List[str]] = {
    # Math / expressions
    "expression": ["2 + 3 * 4", "10 / 2", "100 - 37"],
    "formula": ["x^2 + 3x - 5", "2 * pi * r", "a^2 + b^2"],
    "equation": ["2x + 5 = 15", "x^2 - 4 = 0", "3x = 12"],
    # Search / text
    "query": ["how to install python", "machine learning tutorial", "REST API best practices"],
    "search": ["weather forecast", "python tutorial", "sorting algorithms"],
    "keyword": ["artificial intelligence", "blockchain", "cloud computing"],
    "text": ["Hello, world!", "The quick brown fox jumps over the lazy dog", "Lorem ipsum"],
    "message": ["Hello, how are you?", "Please help me with this task", "Thank you"],
    "prompt": ["Explain quantum computing in simple terms", "Write a haiku about coding"],
    "content": ["This is sample content for testing", "A short paragraph about technology"],
    "input": ["sample input data", "test input string", "example input"],
    # QO-054: Crypto / Web3 — many MCP servers require real coin IDs or addresses.
    # Generic "id=abc123" killed CoinGecko in the prod batch.
    "coin": ["bitcoin", "ethereum", "solana"],
    "coin_id": ["bitcoin", "ethereum", "solana"],
    "token": ["USDC", "SOL", "ETH"],
    "token_id": ["bitcoin", "ethereum", "usd-coin"],
    "symbol": ["BTC", "ETH", "SOL"],
    "ticker": ["BTC", "ETH", "SOL"],
    "chain": ["ethereum", "solana", "base"],
    "network": ["mainnet", "ethereum", "solana"],
    "wallet": [
        "0x742d35Cc6634C0532925a3b844Bc9e7595f0fAAa",  # random EVM
        "So11111111111111111111111111111111111111112",  # wSOL mint
        "0x0000000000000000000000000000000000000000",
    ],
    # "address" serves both crypto wallet and postal address contexts. Crypto
    # first — most MCP servers asking for an "address" are blockchain-backed.
    "address": [
        "0x742d35Cc6634C0532925a3b844Bc9e7595f0fAAa",
        "So11111111111111111111111111111111111111112",
        "123 Main St, Springfield",
    ],
    # QO-054: Repos / code hosting — MCP servers backed by GitHub APIs need these.
    "repo": ["anthropics/anthropic-cookbook", "solana-labs/solana", "microsoft/vscode"],
    "repository": ["anthropics/anthropic-cookbook", "solana-labs/solana", "microsoft/vscode"],
    "owner": ["anthropics", "solana-labs", "microsoft"],
    "org": ["anthropics", "solana-labs", "microsoft"],
    "branch": ["main", "develop", "master"],
    "commit": ["abc1234", "def5678", "9abcdef"],
    "sha": ["abc1234", "def5678", "9abcdef"],
    # Location / geo
    "city": ["London", "New York", "Tokyo"],
    "location": ["San Francisco, CA", "Berlin, Germany", "Sydney, Australia"],
    "country": ["United States", "Japan", "Germany"],
    "zip": ["94105", "10001", "SW1A 1AA"],
    "zipcode": ["94105", "10001", "60601"],
    # Units / conversion
    "from_unit": ["celsius", "km", "kg"],
    "to_unit": ["fahrenheit", "miles", "lbs"],
    "unit": ["celsius", "meters", "kilograms"],
    "source_unit": ["celsius", "km", "kg"],
    "target_unit": ["fahrenheit", "miles", "lbs"],
    # Identity / user
    "name": ["John Doe", "Jane Smith", "Alice Johnson"],
    "username": ["johndoe", "janesmith", "testuser42"],
    "email": ["user@example.com", "test@mail.org", "jane@company.io"],
    "first_name": ["John", "Jane", "Alice"],
    "last_name": ["Doe", "Smith", "Johnson"],
    # Web / URLs
    "url": ["https://example.com", "https://httpbin.org/get", "https://api.github.com"],
    "link": ["https://example.com/page", "https://docs.python.org"],
    "domain": ["example.com", "github.com", "google.com"],
    # File / path
    "filename": ["report.pdf", "data.csv", "image.png"],
    "path": ["/home/user/documents", "/tmp/output.txt", "src/main.py"],
    "file": ["document.txt", "config.json", "script.py"],
    # Language
    "language": ["English", "Spanish", "French"],
    "lang": ["en", "es", "fr"],
    "locale": ["en-US", "de-DE", "ja-JP"],
    # Date / time
    "date": ["2025-01-15", "2024-06-30", "2023-12-25"],
    "time": ["14:30:00", "09:00:00", "23:59:59"],
    "timezone": ["UTC", "America/New_York", "Asia/Tokyo"],
    # Format / output
    "format": ["json", "csv", "xml"],
    "output_format": ["json", "markdown", "html"],
    # Misc
    "topic": ["artificial intelligence", "climate change", "space exploration"],
    "category": ["technology", "science", "education"],
    "description": ["A sample item for testing", "Test description"],
    "title": ["My Test Document", "Sample Report", "Example Title"],
    "id": ["abc123", "item-42", "usr_001"],
    "key": ["api_key_sample", "config_key", "setting_name"],
    "model": ["gpt-4", "claude-3", "llama-3"],
    "temperature": ["22.5", "0.7", "37.0"],
    "code": ["print('hello')", "console.log('test')", "SELECT * FROM users"],
}

SEMANTIC_NUMBER_MAP: Dict[str, List[float]] = {
    "value": [42.0, 100.0, 3.14],
    "amount": [99.99, 250.0, 10.0],
    "price": [29.99, 149.0, 9.99],
    "count": [5, 10, 25],
    "limit": [5, 10, 25],
    "offset": [0, 10, 50],
    "page": [1, 2, 5],
    "max": [100, 1000, 50],
    "min": [0, 1, -10],
    "temperature": [22.5, 0.7, 37.0],
    "timeout": [30, 60, 5],
    "retries": [3, 1, 5],
    "width": [800, 1920, 640],
    "height": [600, 1080, 480],
    "radius": [10.0, 50.0, 100.0],
    "age": [25, 30, 42],
    "quantity": [1, 5, 10],
    "score": [85.0, 92.5, 70.0],
    "weight": [75.5, 100.0, 62.3],
    "rate": [4.5, 3.8, 5.0],
    "percentage": [75.0, 50.0, 95.0],
    "latitude": [51.5074, 40.7128, 35.6762],
    "longitude": [-0.1278, -74.0060, 139.6503],
}

# ---------------------------------------------------------------------------
# Regex patterns to extract examples from parameter descriptions
# ---------------------------------------------------------------------------
_EXAMPLE_PATTERNS = [
    # e.g., '2 + 3 * 4'  or  e.g. "celsius"
    re.compile(r"""e\.g\.[\s,]*['"]([^'"]+)['"]""", re.IGNORECASE),
    # Example: "celsius"  or  example: 'hello'
    re.compile(r"""[Ee]xample:?\s*['"]([^'"]+)['"]"""),
    # such as 'hello world'  or  such as "London"
    re.compile(r"""such as\s+['"]([^'"]+)['"]""", re.IGNORECASE),
    # like 'hello world'  or  like "London"
    re.compile(r"""like\s+['"]([^'"]+)['"]""", re.IGNORECASE),
    # (e.g., 'km', 'miles')  — grab just the first one
    re.compile(r"""\(e\.g\.[\s,]*['"]([^'"]+)['"]""", re.IGNORECASE),
]


def _extract_example_from_description(description: Optional[str]) -> Optional[str]:
    """Try to extract a concrete example value from a parameter description."""
    if not description:
        return None
    for pattern in _EXAMPLE_PATTERNS:
        match = pattern.search(description)
        if match:
            return match.group(1)
    return None


def _fuzzy_match_param_name(key: str, mapping: dict) -> Optional[str]:
    """
    Match compound parameter names by suffix.

    E.g. 'search_query' matches 'query', 'target_url' matches 'url'.
    """
    # Try exact match first (already handled by caller, but just in case)
    if key in mapping:
        return key

    # Try suffix match: split on _ and check last segment(s)
    parts = key.lower().replace("-", "_").split("_")
    # Check last part
    if parts[-1] in mapping:
        return parts[-1]
    # Check last two parts joined
    if len(parts) >= 2:
        two_parts = "_".join(parts[-2:])
        if two_parts in mapping:
            return two_parts
    return None


# ---------------------------------------------------------------------------
# QO-054: Format generators for JSON Schema `format` keyword
# ---------------------------------------------------------------------------
# Covers the common formats seen on real MCP servers. Pattern-based synthesis
# (`pattern` keyword) is deliberately not implemented — most production
# schemas use `format` instead, and `rstr`-style regex-to-string synthesis
# can generate pathological values. Fall through to semantic map if `format`
# is unrecognized.
FORMAT_GENERATORS: Dict[str, List[str]] = {
    "uuid": [
        "00000000-0000-4000-8000-000000000001",
        "11111111-1111-4111-8111-111111111111",
        "deadbeef-dead-4dad-8bad-feedfacedead",
    ],
    "email": ["user@example.com", "test@mail.org", "jane@company.io"],
    "date": ["2025-01-15", "2024-06-30", "2023-12-25"],
    "date-time": [
        "2025-01-15T14:30:00Z",
        "2024-06-30T09:00:00Z",
        "2023-12-25T23:59:59Z",
    ],
    "time": ["14:30:00", "09:00:00", "23:59:59"],
    "uri": ["https://example.com", "https://api.github.com", "https://docs.python.org"],
    "url": ["https://example.com", "https://api.github.com", "https://docs.python.org"],
    "hostname": ["example.com", "api.github.com", "docs.python.org"],
    "ipv4": ["192.168.1.1", "10.0.0.1", "172.16.0.1"],
    "ipv6": ["::1", "2001:db8::1", "fe80::1"],
    "duration": ["PT1H", "P1D", "PT30M"],
    "byte": ["dGVzdA==", "aGVsbG8=", "d29ybGQ="],
    "binary": ["dGVzdA==", "aGVsbG8=", "d29ybGQ="],
}


def _resolve_ref(prop: dict, root: dict, seen: Optional[set] = None, depth: int = 0) -> dict:
    """Resolve a JSON-Pointer $ref against the root schema.

    Depth-bounded (3 levels) with cycle detection via `seen`. Returns the
    original prop if resolution fails — the caller is expected to fall
    through to less-specific rules.
    """
    seen = seen if seen is not None else set()
    if "$ref" not in prop or depth >= 3:
        return prop
    ref = prop["$ref"]
    if ref in seen or not isinstance(ref, str) or not ref.startswith("#/"):
        return prop
    seen.add(ref)
    target = root
    for part in ref[2:].split("/"):
        if isinstance(target, dict) and part in target:
            target = target[part]
        else:
            return prop
    if isinstance(target, dict) and "$ref" in target:
        return _resolve_ref(target, root, seen, depth + 1)
    return target if isinstance(target, dict) else prop


def _clamp(value, prop: dict, prop_type: str):
    """Apply JSON-Schema range / length constraints to a generated value.

    Keeps inputs within declared bounds so servers with strict validation
    (common with FastAPI/pydantic-backed MCPs) don't reject us.
    """
    if prop_type in ("integer", "number"):
        try:
            v = float(value) if prop_type == "number" else int(float(value))
        except (ValueError, TypeError):
            return value
        if "minimum" in prop:
            v = max(v, prop["minimum"])
        if "maximum" in prop:
            v = min(v, prop["maximum"])
        if "exclusiveMinimum" in prop:
            bump = 1 if prop_type == "integer" else 1e-6
            v = max(v, prop["exclusiveMinimum"] + bump)
        if "exclusiveMaximum" in prop:
            bump = 1 if prop_type == "integer" else 1e-6
            v = min(v, prop["exclusiveMaximum"] - bump)
        return int(v) if prop_type == "integer" else v
    if prop_type == "string" and isinstance(value, str):
        min_len = prop.get("minLength")
        max_len = prop.get("maxLength")
        if min_len is not None and len(value) < min_len:
            value = value.ljust(min_len, "x")
        # Always cap at 256 to avoid pathological payloads
        hard_cap = min(max_len, 256) if max_len is not None else 256
        if len(value) > hard_cap:
            value = value[:hard_cap]
        return value
    return value


def _generate_sample_input(
    schema: dict,
    variation: int = 0,
    root_schema: Optional[dict] = None,
    discovered_ids: Optional[Dict[str, List[str]]] = None,
    tool_name: Optional[str] = None,
) -> dict:
    """
    Generate sample input data from a JSON schema.

    QO-054 priority chain per parameter (first match wins):
      0. $ref                — resolve against root_schema (up to 3 levels deep)
      1. const               — use verbatim
      2. enum                — enum[variation % len]
      3. default
      4. examples            — also accepts singular "example"
      5. format              — FORMAT_GENERATORS lookup (uuid/email/date-time/…)
      5.5 (QO-055) discovered_ids — real IDs harvested from list-tools,
                              used only for id-like parameters.
      6. description         — regex-extracted phrase
      7. oneOf/anyOf         — recurse into the first typed branch
      8. semantic map        — exact → fuzzy suffix match
      9. type fallback with min/max/length clamping
      10. f"test_{key}" last resort

    All numeric and string values are passed through `_clamp` to respect
    minimum/maximum/minLength/maxLength constraints.

    QO-055: when `discovered_ids` is provided (a dict of
    `param-name → [values]` harvested from a prior discovery pass), any
    id-like parameter will prefer a discovered value over the semantic
    fallback. This closes the "opaque ID" gap for list → detail flows.
    """
    # The first call establishes the root for $ref resolution; recursion
    # (e.g. for object-typed properties) inherits it.
    if root_schema is None:
        root_schema = schema

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    sample = {}

    for key, prop in properties.items():
        # Resolve $ref at the property level before any other rule runs.
        if isinstance(prop, dict) and "$ref" in prop:
            prop = _resolve_ref(prop, root_schema)

        if key not in required and len(sample) >= 3:
            continue  # Only fill required + a few optional

        prop_type = prop.get("type", "string") if isinstance(prop, dict) else "string"
        value = _resolve_param_value(
            key, prop, prop_type, variation, root_schema, discovered_ids, tool_name
        )
        sample[key] = value

    return sample


def _resolve_param_value(
    key: str,
    prop: dict,
    prop_type: str,
    variation: int,
    root_schema: Optional[dict] = None,
    discovered_ids: Optional[Dict[str, List[str]]] = None,
    tool_name: Optional[str] = None,
):
    """Resolve a single parameter value using the QO-054 priority chain."""
    if not isinstance(prop, dict):
        return f"test_{key}"

    # 1. const — always exact
    if "const" in prop:
        return prop["const"]

    # 2. enum
    enum_values = prop.get("enum")
    if enum_values:
        return _clamp(enum_values[variation % len(enum_values)], prop, prop_type)

    # 3. Schema default
    if "default" in prop:
        return _clamp(prop["default"], prop, prop_type)

    # 4. Schema examples (plural or singular)
    examples = prop.get("examples")
    if not examples and "example" in prop:
        examples = [prop["example"]]
    if examples:
        return _clamp(examples[variation % len(examples)], prop, prop_type)

    # 5. format — generate a schema-valid value for well-known formats
    fmt = prop.get("format")
    if fmt and prop_type in ("string", None, "any") and fmt in FORMAT_GENERATORS:
        candidates = FORMAT_GENERATORS[fmt]
        return _clamp(candidates[variation % len(candidates)], prop, "string")

    # 5.5. QO-055 — real IDs harvested from list-tools win for id-like params.
    # Only fires for id-shaped parameter names so a regular "query" or "text"
    # still goes through semantic-map fallback.
    if discovered_ids and prop_type in ("string", None):
        pool = _pick_discovered_pool(key, discovered_ids, tool_name)
        if pool:
            return _clamp(pool[variation % len(pool)], prop, "string")

    # 6. Extract from description (only for variation=0; others fall through to
    #    semantic maps which have multiple values for variation diversity)
    if variation == 0:
        desc_example = _extract_example_from_description(prop.get("description"))
        if desc_example:
            if prop_type in ("integer", "number"):
                try:
                    coerced = float(desc_example) if prop_type == "number" else int(desc_example)
                    return _clamp(coerced, prop, prop_type)
                except (ValueError, TypeError):
                    pass
            else:
                return _clamp(desc_example, prop, prop_type)

    # 7. oneOf / anyOf — pick the first branch whose type we can handle
    for kw in ("oneOf", "anyOf"):
        branches = prop.get(kw)
        if not branches:
            continue
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            resolved = _resolve_ref(branch, root_schema or {}) if "$ref" in branch else branch
            branch_type = resolved.get("type")
            if branch_type in ("string", "integer", "number", "boolean", "array", "object"):
                return _resolve_param_value(
                    key, resolved, branch_type, variation, root_schema, discovered_ids
                )

    # 8 & 9. Semantic map / type fallback
    if prop_type == "string":
        return _clamp(_resolve_string_param(key, variation), prop, "string")
    if prop_type in ("integer", "number"):
        return _clamp(_resolve_number_param(key, prop_type, variation), prop, prop_type)
    if prop_type == "boolean":
        return True
    if prop_type == "array":
        items = prop.get("items")
        if isinstance(items, dict):
            if "$ref" in items:
                items = _resolve_ref(items, root_schema or {})
            item_type = items.get("type")
            if item_type in ("string", "integer", "number", "boolean", "array", "object"):
                item = _resolve_param_value(
                    f"{key}_item", items, item_type, variation, root_schema, discovered_ids
                )
                min_items = max(1, prop.get("minItems", 1))
                max_items = prop.get("maxItems", min_items)
                return [item] * min(min_items, max_items)
        return []
    if prop_type == "object":
        # Recurse using the same priority chain — preserves root_schema for $refs
        if "properties" in prop:
            return _generate_sample_input(
                {"properties": prop["properties"], "required": prop.get("required", [])},
                variation,
                root_schema,
                discovered_ids,
            )
        return {}

    return f"test_{key}"


def _pick_discovered_pool(
    key: str,
    discovered_ids: Dict[str, List[str]],
    tool_name: Optional[str] = None,
) -> Optional[List[str]]:
    """Pick the most specific discovered-ID pool for this parameter name.

    Match priority:
      1. Exact parameter-name match (e.g. `coin_id` → `coin_id` pool)
      2. Resource inferred from the *tool* name (e.g. `experience_details`
         → look for `experience_id` / `experience` pools). This is the
         QO-055 fix for servers with multiple list-tools where a generic
         `id` pool would be cross-contaminated.
      3. None → caller falls through to semantic map.

    Deliberately does NOT fall back to a generic `id` bucket anymore —
    that caused `experience_details(id)` to receive tag IDs harvested
    from `list_tags`. Honest failure is better than misclassified success.
    """
    from src.core.id_discovery import _looks_like_id_name, _resource_key_from_tool_name
    if not _looks_like_id_name(key):
        return None
    k = key.lower()
    # 1. Exact parameter-name match
    if k in discovered_ids and discovered_ids[k]:
        return discovered_ids[k]
    # Fuzzy: "coinId" → "coin_id"
    normalized = k.replace("-", "_")
    if normalized in discovered_ids and discovered_ids[normalized]:
        return discovered_ids[normalized]
    # 2. Resource inferred from the *tool* name (e.g. `experience_details`
    #    → resource `experience` → prefer `experience_id` pool).
    if tool_name:
        tool_resource = _resource_key_from_tool_name(tool_name)
        if tool_resource:
            # Try `<resource>_id` then `<resource>` (both are written by
            # id_discovery.discover_ids for a recognised list tool).
            for candidate in (f"{tool_resource}_id", tool_resource):
                if candidate in discovered_ids and discovered_ids[candidate]:
                    return discovered_ids[candidate]
    return None


def _resolve_string_param(key: str, variation: int) -> str:
    """Resolve a string parameter from semantic maps."""
    key_lower = key.lower()

    # Exact match
    if key_lower in SEMANTIC_PARAM_MAP:
        values = SEMANTIC_PARAM_MAP[key_lower]
        return values[variation % len(values)]

    # Fuzzy suffix match
    matched = _fuzzy_match_param_name(key_lower, SEMANTIC_PARAM_MAP)
    if matched:
        values = SEMANTIC_PARAM_MAP[matched]
        return values[variation % len(values)]

    # Also check number map for string-typed numeric params (e.g. temperature as string)
    if key_lower in SEMANTIC_NUMBER_MAP:
        values = SEMANTIC_NUMBER_MAP[key_lower]
        return str(values[variation % len(values)])

    matched_num = _fuzzy_match_param_name(key_lower, SEMANTIC_NUMBER_MAP)
    if matched_num:
        values = SEMANTIC_NUMBER_MAP[matched_num]
        return str(values[variation % len(values)])

    return f"test_{key}"


def _resolve_number_param(key: str, prop_type: str, variation: int):
    """Resolve a numeric parameter from semantic maps."""
    key_lower = key.lower()

    # Exact match
    if key_lower in SEMANTIC_NUMBER_MAP:
        values = SEMANTIC_NUMBER_MAP[key_lower]
        val = values[variation % len(values)]
        return int(val) if prop_type == "integer" else val

    # Fuzzy suffix match
    matched = _fuzzy_match_param_name(key_lower, SEMANTIC_NUMBER_MAP)
    if matched:
        values = SEMANTIC_NUMBER_MAP[matched]
        val = values[variation % len(values)]
        return int(val) if prop_type == "integer" else val

    # Default
    return 1 if prop_type == "integer" else 1.0


# ---------------------------------------------------------------------------
# Expected behavior generation
# ---------------------------------------------------------------------------

_TOOL_BEHAVIOR_PATTERNS = [
    # (keywords in tool name, template using input_summary)
    ({"calculate", "math", "compute", "eval"}, "Should return the computed result of {inputs}"),
    ({"search", "find", "lookup", "query"}, "Should return search results relevant to {inputs}"),
    ({"weather", "forecast"}, "Should return weather data for {inputs} including temperature"),
    ({"convert", "transform", "translate"}, "Should return the conversion result for {inputs}"),
    ({"get", "fetch", "retrieve", "read", "list"}, "Should return the requested data for {inputs}"),
    ({"create", "add", "insert", "post"}, "Should successfully create a resource with {inputs}"),
    ({"update", "edit", "modify", "patch"}, "Should successfully update the resource with {inputs}"),
    ({"delete", "remove"}, "Should successfully delete the specified resource"),
]


def _generate_expected_behavior(tool_name: str, description: str, input_data: dict) -> str:
    """Generate a descriptive expected behavior based on tool semantics and inputs."""
    # Build a readable summary of input values
    if input_data:
        parts = [f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in input_data.items()]
        input_summary = ", ".join(parts)
    else:
        input_summary = "the provided input"

    name_lower = tool_name.lower()

    for keywords, template in _TOOL_BEHAVIOR_PATTERNS:
        if any(kw in name_lower for kw in keywords):
            return template.format(inputs=input_summary)

    # Generic fallback with actual input values for keyword overlap
    return f"Should process the input ({input_summary}) and return relevant output"


# ---------------------------------------------------------------------------
# Main test case generation
# ---------------------------------------------------------------------------

def generate_test_cases(
    tools: List[dict],
    test_types: Optional[Set[str]] = None,
    max_tools: Optional[int] = None,
    discovered_ids: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[dict]]:
    """
    Generate test cases from MCP server tool definitions.

    Args:
        tools: List of tool definitions with name, description, inputSchema
        test_types: If provided, only generate these test types (e.g. {"happy_path", "error_handling"})
        max_tools: If provided, limit to first N tools
        discovered_ids: QO-055. Map of param-name → [values] harvested
            from list-tools. When present, id-like parameters prefer a
            discovered value over the semantic fallback. Other params
            are unaffected.

    Returns:
        Dict of tool_name -> list of test cases {question, expected, input_data, test_type}
    """
    test_cases: Dict[str, List[dict]] = {}

    target_tools = tools[:max_tools] if max_tools else tools

    for tool in target_tools:
        name = tool.get("name", "unknown")
        description = tool.get("description", "")
        schema = tool.get("inputSchema", tool.get("parameters", {}))

        cases = []

        # --- Happy path 1: primary realistic input ---
        if description:
            input_data_0 = _generate_sample_input(
                schema, variation=0, discovered_ids=discovered_ids, tool_name=name,
            )
            cases.append({
                "question": f"Use the '{name}' tool: {description}",
                "expected": _generate_expected_behavior(name, description, input_data_0),
                "test_type": "happy_path",
                "input_data": input_data_0,
            })

        # --- Happy path 2: variation with different values ---
        if description:
            input_data_1 = _generate_sample_input(
                schema, variation=1, discovered_ids=discovered_ids, tool_name=name,
            )
            # Only add if inputs actually differ from variation 0
            input_data_0_check = _generate_sample_input(
                schema, variation=0, discovered_ids=discovered_ids, tool_name=name,
            )
            if input_data_1 != input_data_0_check:
                cases.append({
                    "question": f"Use the '{name}' tool with different inputs: {description}",
                    "expected": _generate_expected_behavior(name, description, input_data_1),
                    "test_type": "happy_path_variation",
                    "input_data": input_data_1,
                })

        # --- Missing required params ---
        required = schema.get("required", [])
        if required:
            cases.append({
                "question": f"Call '{name}' without required parameter '{required[0]}'",
                "expected": "Tool should return a clear error message about the missing required parameter",
                "test_type": "error_handling",
                "input_data": {},
            })

        # --- Edge case: empty string for string params ---
        properties = schema.get("properties", {})
        string_params = [k for k, v in properties.items() if v.get("type") == "string"]
        if string_params:
            cases.append({
                "question": f"Call '{name}' with empty string for '{string_params[0]}'",
                "expected": "Tool should handle empty input by returning an error message or a sensible default value",
                "test_type": "edge_case",
                "input_data": {string_params[0]: ""},
            })

        # --- Boundary: long string input ---
        if string_params:
            long_input = _generate_sample_input(
                schema, variation=0, discovered_ids=discovered_ids, tool_name=name,
            )
            long_input[string_params[0]] = "a" * 500
            cases.append({
                "question": f"Call '{name}' with a very long string input for '{string_params[0]}'",
                "expected": "Tool should handle oversized input by returning an error or processing it with a valid result",
                "test_type": "boundary",
                "input_data": long_input,
            })

        # --- Type coercion: string where number expected ---
        number_params = [
            k for k, v in properties.items()
            if v.get("type") in ("integer", "number")
        ]
        if number_params:
            coercion_input = _generate_sample_input(
                schema, variation=0, discovered_ids=discovered_ids, tool_name=name,
            )
            coercion_input[number_params[0]] = "not_a_number"
            cases.append({
                "question": f"Call '{name}' with a string '{number_params[0]}' instead of a number",
                "expected": "Tool should reject invalid type with a validation error or coerce the value to the expected type",
                "test_type": "type_coercion",
                "input_data": coercion_input,
            })

        if test_types:
            cases = [c for c in cases if c["test_type"] in test_types]

        test_cases[name] = cases
        logger.debug(f"Generated {len(cases)} test cases for tool '{name}'")

    logger.info(f"Generated {sum(len(v) for v in test_cases.values())} total test cases for {len(tools)} tools")
    return test_cases

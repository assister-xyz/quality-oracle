"""Tests for the semantic-aware test generator."""
import pytest
from jsonschema import validate as jsonschema_validate
from src.core.test_generator import (
    _extract_example_from_description,
    _fuzzy_match_param_name,
    _generate_sample_input,
    _generate_expected_behavior,
    _clamp,
    generate_test_cases,
    SEMANTIC_PARAM_MAP,
    SEMANTIC_NUMBER_MAP,
    FORMAT_GENERATORS,
)


# ---------------------------------------------------------------------------
# _extract_example_from_description
# ---------------------------------------------------------------------------

class TestExtractExample:
    def test_eg_single_quotes(self):
        assert _extract_example_from_description("e.g., '2 + 3 * 4'") == "2 + 3 * 4"

    def test_eg_double_quotes(self):
        assert _extract_example_from_description('e.g. "celsius"') == "celsius"

    def test_example_colon(self):
        assert _extract_example_from_description('Example: "hello world"') == "hello world"

    def test_such_as(self):
        assert _extract_example_from_description("such as 'London'") == "London"

    def test_like_pattern(self):
        assert _extract_example_from_description("A city name like 'Tokyo'") == "Tokyo"

    def test_paren_eg(self):
        assert _extract_example_from_description("Source unit (e.g., 'km', 'miles')") == "km"

    def test_no_match(self):
        assert _extract_example_from_description("A plain description with no examples") is None

    def test_none_input(self):
        assert _extract_example_from_description(None) is None

    def test_empty_string(self):
        assert _extract_example_from_description("") is None


# ---------------------------------------------------------------------------
# _fuzzy_match_param_name
# ---------------------------------------------------------------------------

class TestFuzzyMatch:
    def test_exact_match(self):
        assert _fuzzy_match_param_name("query", SEMANTIC_PARAM_MAP) == "query"

    def test_suffix_match(self):
        assert _fuzzy_match_param_name("search_query", SEMANTIC_PARAM_MAP) == "query"

    def test_compound_suffix(self):
        assert _fuzzy_match_param_name("target_url", SEMANTIC_PARAM_MAP) == "url"

    def test_no_match(self):
        assert _fuzzy_match_param_name("xyzzy_foobar", SEMANTIC_PARAM_MAP) is None

    def test_hyphenated(self):
        # from-unit → from_unit → suffix "unit" matches
        assert _fuzzy_match_param_name("from-unit", SEMANTIC_PARAM_MAP) is not None


# ---------------------------------------------------------------------------
# _generate_sample_input
# ---------------------------------------------------------------------------

class TestGenerateSampleInput:
    def test_enum_priority(self):
        """Enum values should be picked over everything else."""
        schema = {
            "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]}
            },
            "required": ["color"],
        }
        result = _generate_sample_input(schema, variation=0)
        assert result["color"] == "red"
        result1 = _generate_sample_input(schema, variation=1)
        assert result1["color"] == "green"

    def test_default_priority(self):
        """Default should be used when no enum."""
        schema = {
            "properties": {
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["limit"],
        }
        result = _generate_sample_input(schema)
        assert result["limit"] == 10

    def test_description_example_priority(self):
        """Example from description regex should be used when no enum/default."""
        schema = {
            "properties": {
                "thing": {"type": "string", "description": "A thing (e.g., 'widget')"}
            },
            "required": ["thing"],
        }
        result = _generate_sample_input(schema)
        assert result["thing"] == "widget"

    def test_semantic_map(self):
        """Semantic map should kick in for known param names."""
        schema = {
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"],
        }
        result = _generate_sample_input(schema, variation=0)
        assert result["city"] == "London"

    def test_semantic_map_variation(self):
        """Different variations should produce different values."""
        schema = {
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"],
        }
        r0 = _generate_sample_input(schema, variation=0)
        r1 = _generate_sample_input(schema, variation=1)
        assert r0["city"] != r1["city"]

    def test_fuzzy_suffix_for_compound_name(self):
        """search_query should match 'query' in semantic map."""
        schema = {
            "properties": {
                "search_query": {"type": "string"}
            },
            "required": ["search_query"],
        }
        result = _generate_sample_input(schema)
        assert result["search_query"] in SEMANTIC_PARAM_MAP["query"]

    def test_number_semantic_map(self):
        """Numeric params should use SEMANTIC_NUMBER_MAP."""
        schema = {
            "properties": {
                "value": {"type": "number"}
            },
            "required": ["value"],
        }
        result = _generate_sample_input(schema)
        assert result["value"] in SEMANTIC_NUMBER_MAP["value"]

    def test_fallback_for_unknown(self):
        """Truly unknown params should fallback to test_ prefix."""
        schema = {
            "properties": {
                "xyzzy_blorb": {"type": "string"}
            },
            "required": ["xyzzy_blorb"],
        }
        result = _generate_sample_input(schema)
        assert result["xyzzy_blorb"] == "test_xyzzy_blorb"

    def test_boolean_param(self):
        schema = {
            "properties": {"verbose": {"type": "boolean"}},
            "required": ["verbose"],
        }
        result = _generate_sample_input(schema)
        assert result["verbose"] is True


# ---------------------------------------------------------------------------
# _generate_expected_behavior
# ---------------------------------------------------------------------------

class TestExpectedBehavior:
    def test_calculate_tool(self):
        result = _generate_expected_behavior("calculate", "Evaluate math", {"expression": "2 + 3"})
        assert "computed result" in result
        assert "2 + 3" in result

    def test_search_tool(self):
        result = _generate_expected_behavior("search_docs", "Search docs", {"query": "python"})
        assert "search results" in result
        assert "python" in result

    def test_weather_tool(self):
        result = _generate_expected_behavior("get_weather", "Get weather", {"city": "London"})
        assert "weather" in result
        assert "London" in result

    def test_convert_tool(self):
        result = _generate_expected_behavior("convert_units", "Convert", {"value": 42, "from_unit": "km"})
        assert "conversion" in result

    def test_generic_tool(self):
        result = _generate_expected_behavior("do_something", "Does stuff", {"foo": "bar"})
        assert "bar" in result


# ---------------------------------------------------------------------------
# generate_test_cases — integration
# ---------------------------------------------------------------------------

class TestGenerateTestCases:
    """Test the full generate_test_cases() with mock server tool definitions."""

    MOCK_TOOLS = [
        {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression string (e.g., '2 + 3 * 4')",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'London', 'New York')",
                    }
                },
                "required": ["city"],
            },
        },
        {
            "name": "convert_units",
            "description": "Convert between measurement units.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numeric value to convert",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit (e.g., 'km', 'miles', 'celsius', 'fahrenheit')",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Target unit (e.g., 'miles', 'km', 'fahrenheit', 'celsius')",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    ]

    def test_generates_at_least_5_cases_per_tool(self):
        result = generate_test_cases(self.MOCK_TOOLS)
        for name, cases in result.items():
            assert len(cases) >= 5, f"Tool '{name}' has only {len(cases)} cases, expected >= 5"

    def test_no_test_prefix_in_happy_path_inputs(self):
        """Happy path inputs should use realistic values, not test_ prefixes."""
        result = generate_test_cases(self.MOCK_TOOLS)
        for name, cases in result.items():
            happy = [c for c in cases if c["test_type"] == "happy_path"]
            for case in happy:
                for k, v in case["input_data"].items():
                    if isinstance(v, str):
                        assert not v.startswith("test_"), (
                            f"Tool '{name}' param '{k}' has test_ prefix: '{v}'"
                        )

    def test_calculate_gets_math_expression(self):
        result = generate_test_cases(self.MOCK_TOOLS)
        calc_happy = [c for c in result["calculate"] if c["test_type"] == "happy_path"][0]
        expr = calc_happy["input_data"]["expression"]
        # Should be a real expression from description or semantic map
        assert any(op in expr for op in ["+", "-", "*", "/"]), f"Expression '{expr}' lacks math operators"

    def test_weather_gets_real_city(self):
        result = generate_test_cases(self.MOCK_TOOLS)
        weather_happy = [c for c in result["get_weather"] if c["test_type"] == "happy_path"][0]
        city = weather_happy["input_data"]["city"]
        assert city in ["London", "New York", "Tokyo"], f"Unexpected city: {city}"

    def test_convert_gets_real_units(self):
        result = generate_test_cases(self.MOCK_TOOLS)
        conv_happy = [c for c in result["convert_units"] if c["test_type"] == "happy_path"][0]
        data = conv_happy["input_data"]
        assert data["from_unit"] in ["celsius", "km", "kg"]
        assert data["to_unit"] in ["fahrenheit", "miles", "lbs"]
        assert isinstance(data["value"], (int, float))

    def test_has_type_coercion_for_numeric_tool(self):
        """convert_units has a number param, so it should get a type_coercion test."""
        result = generate_test_cases(self.MOCK_TOOLS)
        coercion = [c for c in result["convert_units"] if c["test_type"] == "type_coercion"]
        assert len(coercion) == 1

    def test_expected_behavior_includes_input_values(self):
        """Expected behavior text should reference actual input values for keyword overlap."""
        result = generate_test_cases(self.MOCK_TOOLS)
        weather_happy = [c for c in result["get_weather"] if c["test_type"] == "happy_path"][0]
        city = weather_happy["input_data"]["city"]
        assert city in weather_happy["expected"], (
            f"Expected behavior should mention city '{city}'"
        )

    def test_variation_produces_different_inputs(self):
        """The two happy path tests should have different input values."""
        result = generate_test_cases(self.MOCK_TOOLS)
        weather_cases = result["get_weather"]
        happy_paths = [c for c in weather_cases if c["test_type"] in ("happy_path", "happy_path_variation")]
        assert len(happy_paths) == 2
        assert happy_paths[0]["input_data"]["city"] != happy_paths[1]["input_data"]["city"]

    def test_expected_behavior_edge_case_signals_error(self):
        """Edge case expected text should contain 'error' keyword for fuzzy judge."""
        result = generate_test_cases(self.MOCK_TOOLS)
        for name, cases in result.items():
            for case in cases:
                if case["test_type"] in ("edge_case", "boundary", "type_coercion"):
                    assert "error" in case["expected"].lower(), (
                        f"Tool '{name}' {case['test_type']} expected text "
                        f"must contain 'error' for fuzzy judge routing"
                    )


# ---------------------------------------------------------------------------
# QO-054: schema-aware priority chain
# ---------------------------------------------------------------------------

class TestQO054PriorityChain:
    """Verify each rule in the new priority chain produces schema-valid values."""

    def _validate(self, sample: dict, schema: dict):
        """Assert the generated sample validates against its parent schema."""
        parent = {"type": "object", "properties": schema, "required": list(schema.keys())}
        jsonschema_validate(sample, parent)

    def test_const_always_wins(self):
        result = _generate_sample_input(
            {"properties": {"action": {"const": "delete"}}, "required": ["action"]}
        )
        assert result == {"action": "delete"}

    def test_const_beats_enum_and_default(self):
        prop = {"const": "fixed", "enum": ["a", "b"], "default": "z"}
        val = _generate_sample_input({"properties": {"x": prop}, "required": ["x"]})
        assert val == {"x": "fixed"}

    def test_singular_example_accepted(self):
        """JSON Schema supports `example` (singular) as well as `examples`."""
        schema = {"properties": {"note": {"type": "string", "example": "hello"}}, "required": ["note"]}
        val = _generate_sample_input(schema)
        assert val == {"note": "hello"}

    @pytest.mark.parametrize("fmt,prop_type", [
        ("uuid", "string"),
        ("email", "string"),
        ("date", "string"),
        ("date-time", "string"),
        ("uri", "string"),
        ("ipv4", "string"),
    ])
    def test_format_generator_produces_expected_shape(self, fmt, prop_type):
        schema = {"properties": {"x": {"type": prop_type, "format": fmt}}, "required": ["x"]}
        val = _generate_sample_input(schema)
        assert val["x"] in FORMAT_GENERATORS[fmt]

    def test_minimum_maximum_clamp_integer(self):
        # Semantic map says "count" → 5; schema caps at 3.
        schema = {"properties": {"count": {"type": "integer", "minimum": 0, "maximum": 3}}, "required": ["count"]}
        val = _generate_sample_input(schema)
        assert 0 <= val["count"] <= 3
        self._validate(val, schema["properties"])

    def test_minimum_clamp_pushes_up(self):
        schema = {"properties": {"offset": {"type": "integer", "minimum": 100}}, "required": ["offset"]}
        val = _generate_sample_input(schema)
        assert val["offset"] >= 100

    def test_min_length_pads_string(self):
        schema = {"properties": {"code": {"type": "string", "minLength": 10}}, "required": ["code"]}
        val = _generate_sample_input(schema)
        assert len(val["code"]) >= 10

    def test_max_length_truncates(self):
        schema = {"properties": {"short": {"type": "string", "maxLength": 3}}, "required": ["short"]}
        val = _generate_sample_input(schema)
        assert len(val["short"]) <= 3

    def test_exclusive_minimum_strict(self):
        schema = {"properties": {"n": {"type": "integer", "exclusiveMinimum": 0}}, "required": ["n"]}
        val = _generate_sample_input(schema)
        assert val["n"] > 0

    def test_ref_resolution_basic(self):
        root = {
            "$defs": {"CoinId": {"type": "string", "const": "bitcoin"}},
            "properties": {"id": {"$ref": "#/$defs/CoinId"}},
            "required": ["id"],
        }
        val = _generate_sample_input(root)
        assert val == {"id": "bitcoin"}

    def test_ref_resolution_components_schemas(self):
        """OpenAPI-style $ref into components/schemas."""
        root = {
            "components": {"schemas": {"Ticker": {"type": "string", "enum": ["BTC"]}}},
            "properties": {"symbol": {"$ref": "#/components/schemas/Ticker"}},
            "required": ["symbol"],
        }
        val = _generate_sample_input(root)
        assert val["symbol"] == "BTC"

    def test_ref_cycle_does_not_loop(self):
        """Circular $refs must not infinitely recurse."""
        root = {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            },
            "properties": {"x": {"$ref": "#/$defs/A"}},
            "required": ["x"],
        }
        # Should return *something* (the last resort fallback) without hanging
        val = _generate_sample_input(root)
        assert "x" in val

    def test_oneof_picks_first_typed_branch(self):
        schema = {
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "integer", "minimum": 5, "maximum": 10},
                        {"type": "string"},
                    ]
                }
            },
            "required": ["value"],
        }
        val = _generate_sample_input(schema)
        # First branch is integer with clamped range — must be in [5,10]
        assert isinstance(val["value"], int) and 5 <= val["value"] <= 10

    def test_anyof_picks_first_typed_branch(self):
        schema = {
            "properties": {
                "payload": {
                    "anyOf": [
                        {"type": "string", "const": "ping"},
                        {"type": "integer"},
                    ]
                }
            },
            "required": ["payload"],
        }
        val = _generate_sample_input(schema)
        assert val["payload"] == "ping"

    def test_array_items_recurse(self):
        schema = {
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["foo", "bar"]},
                    "minItems": 2,
                }
            },
            "required": ["tags"],
        }
        val = _generate_sample_input(schema)
        assert isinstance(val["tags"], list)
        assert len(val["tags"]) >= 1  # clamped by our min/max-items logic
        for item in val["tags"]:
            assert item in ("foo", "bar")

    def test_object_property_recurses(self):
        schema = {
            "properties": {
                "cfg": {
                    "type": "object",
                    "properties": {
                        "mode": {"enum": ["fast", "slow"]},
                        "retries": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["mode", "retries"],
                }
            },
            "required": ["cfg"],
        }
        val = _generate_sample_input(schema)
        assert val["cfg"]["mode"] in ("fast", "slow")
        assert 1 <= val["cfg"]["retries"] <= 5

    def test_coin_id_semantic_map_returns_real_coin(self):
        """Previously `id: "abc123"` killed CoinGecko; now "coin_id" → bitcoin."""
        schema = {"properties": {"coin_id": {"type": "string"}}, "required": ["coin_id"]}
        val = _generate_sample_input(schema)
        assert val["coin_id"] in ("bitcoin", "ethereum", "solana")

    def test_clamp_helper_idempotent_on_valid_values(self):
        """_clamp must not mutate values already inside the range."""
        assert _clamp(50, {"minimum": 0, "maximum": 100}, "integer") == 50
        assert _clamp("hello", {"maxLength": 10}, "string") == "hello"

    def test_generated_samples_always_respect_type_fallback(self):
        """No priority-chain rule should ever return the wrong Python type."""
        for prop_type, expected_py_types in [
            ("string", (str,)),
            ("integer", (int,)),
            ("number", (int, float)),
            ("boolean", (bool,)),
            ("array", (list,)),
            ("object", (dict,)),
        ]:
            schema = {"properties": {"x": {"type": prop_type}}, "required": ["x"]}
            val = _generate_sample_input(schema)
            assert isinstance(val["x"], expected_py_types), (
                f"type={prop_type} produced {type(val['x']).__name__}"
            )

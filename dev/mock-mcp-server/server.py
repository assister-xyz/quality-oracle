"""
Mock MCP Server for Quality Oracle development and testing.

A FastMCP server with predictable tool responses for evaluating
the evaluation engine itself. Exposes 4 tools across 2 domains.
"""
import json
import math
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Mock MCP Server",
    version="1.0.0",
    description="A mock MCP server with predictable responses for Quality Oracle testing",
)


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression string (e.g., "2 + 3 * 4")
    """
    try:
        # Safe eval for basic math
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters in expression"})
        result = eval(expression)  # noqa: S307
        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def search_docs(query: str, limit: int = 5) -> str:
    """Search documentation for relevant articles.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 5)
    """
    if not query or not query.strip():
        return json.dumps({"error": "Query cannot be empty", "results": []})

    # Predictable mock results based on query
    mock_docs = [
        {"title": f"Guide: {query}", "snippet": f"Comprehensive guide about {query}...", "relevance": 0.95},
        {"title": f"Tutorial: {query}", "snippet": f"Step-by-step tutorial for {query}...", "relevance": 0.88},
        {"title": f"FAQ: {query}", "snippet": f"Frequently asked questions about {query}...", "relevance": 0.75},
        {"title": f"Reference: {query}", "snippet": f"API reference for {query}...", "relevance": 0.70},
        {"title": f"Examples: {query}", "snippet": f"Code examples for {query}...", "relevance": 0.65},
    ]

    return json.dumps({"query": query, "results": mock_docs[:limit], "total": len(mock_docs)})


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: City name (e.g., "London", "New York")
    """
    if not city or not city.strip():
        return json.dumps({"error": "City name cannot be empty"})

    # Deterministic mock weather based on city name hash
    h = sum(ord(c) for c in city) % 100
    return json.dumps({
        "city": city,
        "temperature_c": 15 + (h % 20),
        "humidity_pct": 40 + (h % 50),
        "condition": ["sunny", "cloudy", "rainy", "partly cloudy"][h % 4],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@mcp.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between measurement units.

    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g., "km", "miles", "celsius", "fahrenheit")
        to_unit: Target unit (e.g., "miles", "km", "fahrenheit", "celsius")
    """
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v * 0.453592,
        ("meters", "feet"): lambda v: v * 3.28084,
        ("feet", "meters"): lambda v: v * 0.3048,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return json.dumps({"error": f"Unsupported conversion: {from_unit} -> {to_unit}"})

    result = conversions[key](value)
    return json.dumps({
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "result": round(result, 4),
    })


def main():
    """Run the mock MCP server via SSE on port 8010."""
    mcp.run(transport="sse", host="0.0.0.0", port=8010)


if __name__ == "__main__":
    main()

"""
Quick MCP server scan — L1 manifest-only, no LLM, no DB storage.

Used by the public /v1/scan endpoint for zero-friction first experience.
Returns tool count, schema quality, basic poisoning check, tier estimate.
"""
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.core import mcp_client
from src.core.question_pools import determine_tier

logger = logging.getLogger(__name__)

# Tool description poisoning patterns (static analysis)
_POISONING_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"(disregard|forget|override)\s+.*(instructions|rules|constraints)", re.IGNORECASE),
    re.compile(r"(send|transmit|exfiltrate|forward)\s+.*(data|context|keys|tokens|secrets)", re.IGNORECASE),
    re.compile(r"(do\s+not|never)\s+(tell|inform|reveal)\s+.*(user|human)", re.IGNORECASE),
    re.compile(r"<\s*(IMPORTANT|SYSTEM|ADMIN)\s*>", re.IGNORECASE),
    re.compile(r"\\u200[bBcCdDeE]"),  # Zero-width characters (ASCII smuggling)
]


@dataclass
class ToolSummary:
    name: str
    has_description: bool
    has_input_schema: bool
    description_length: int
    param_count: int


@dataclass
class SafetyQuickCheck:
    tool_poisoning_detected: bool
    suspicious_descriptions: int
    findings: List[str] = field(default_factory=list)


@dataclass
class QuickScanResult:
    url: str
    transport: str
    reachable: bool
    tool_count: int
    tools: List[ToolSummary]
    manifest_score: int
    estimated_tier: str
    safety_quick_check: SafetyQuickCheck
    manifest_hash: str
    detected_domain: str
    detected_domains: List[str]
    scan_time_ms: int
    error: Optional[str] = None


def _score_manifest(tools: list) -> int:
    """Score manifest quality 0-100 based on tool completeness."""
    if not tools:
        return 0

    total_points = 0
    max_points = 0

    for tool in tools:
        # Has name (always true from MCP, but check)
        max_points += 10
        if tool.get("name"):
            total_points += 10

        # Has description
        max_points += 30
        desc = tool.get("description", "")
        if desc:
            total_points += 10  # has description
            if len(desc) > 20:
                total_points += 10  # meaningful description
            if len(desc) > 50:
                total_points += 10  # detailed description

        # Has input schema
        max_points += 30
        schema = tool.get("inputSchema") or tool.get("parameters") or {}
        if schema:
            total_points += 10  # has schema
            props = schema.get("properties", {})
            if props:
                total_points += 10  # has properties
                # Check if properties have descriptions
                props_with_desc = sum(
                    1 for p in props.values()
                    if isinstance(p, dict) and p.get("description")
                )
                if props_with_desc == len(props) and props:
                    total_points += 10  # all props described

        # Has required fields defined
        max_points += 10
        if schema.get("required"):
            total_points += 10

    return int((total_points / max_points) * 100) if max_points > 0 else 0


def _check_poisoning(tools: list) -> SafetyQuickCheck:
    """Static analysis of tool descriptions for poisoning indicators."""
    findings = []
    suspicious_count = 0

    for tool in tools:
        # Check description
        desc = tool.get("description", "")
        # Check all text fields including schema
        schema_text = json.dumps(tool.get("inputSchema", {}))
        full_text = f"{desc} {schema_text}"

        for pattern in _POISONING_PATTERNS:
            match = pattern.search(full_text)
            if match:
                suspicious_count += 1
                findings.append(
                    f"Tool '{tool.get('name', '?')}': {match.group()[:60]}"
                )

    return SafetyQuickCheck(
        tool_poisoning_detected=suspicious_count > 0,
        suspicious_descriptions=suspicious_count,
        findings=findings[:5],  # limit to 5 findings
    )


def _hash_manifest(tools: list) -> str:
    """SHA-256 of normalized tool definitions for code integrity tracking."""
    normalized = []
    for tool in sorted(tools, key=lambda t: t.get("name", "")):
        normalized.append({
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "inputSchema": tool.get("inputSchema", {}),
        })
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def quick_scan(url: str) -> QuickScanResult:
    """
    Perform a quick L1-only scan of an MCP server.

    No LLM calls, no DB storage. Returns manifest quality assessment.
    Typical execution: 2-8 seconds (network-bound).
    """
    start = time.time()

    try:
        tools = await mcp_client.connect_and_list_tools(url)
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            error_msg = "MCP server requires authentication"
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_msg = "Connection timed out — server may be down or unreachable"
        else:
            error_msg = f"Could not connect: {error_msg[:200]}"

        return QuickScanResult(
            url=url,
            transport="unknown",
            reachable=False,
            tool_count=0,
            tools=[],
            manifest_score=0,
            estimated_tier="failed",
            safety_quick_check=SafetyQuickCheck(
                tool_poisoning_detected=False,
                suspicious_descriptions=0,
            ),
            manifest_hash="",
            detected_domain="unknown",
            detected_domains=[],
            scan_time_ms=elapsed,
            error=error_msg,
        )

    # Detect transport used
    transport = mcp_client._detect_transport(url)

    # Build tool summaries
    tool_summaries = []
    for tool in tools:
        schema = tool.get("inputSchema") or tool.get("parameters") or {}
        props = schema.get("properties", {})
        tool_summaries.append(ToolSummary(
            name=tool.get("name", ""),
            has_description=bool(tool.get("description")),
            has_input_schema=bool(schema),
            description_length=len(tool.get("description", "")),
            param_count=len(props),
        ))

    # Score manifest
    manifest_score = _score_manifest(tools)

    # Quick safety check
    safety = _check_poisoning(tools)

    # Hash for code integrity
    manifest_hash = _hash_manifest(tools)

    # Estimate tier
    estimated_tier = determine_tier(manifest_score)

    # Auto-detect domain (QO-027)
    from src.core.domain_detection import detect_domain, detect_all_domains
    detected_domain = detect_domain(tools)
    detected_domains = detect_all_domains(tools)

    elapsed = int((time.time() - start) * 1000)

    return QuickScanResult(
        url=url,
        transport=transport,
        reachable=True,
        tool_count=len(tools),
        tools=tool_summaries,
        manifest_score=manifest_score,
        estimated_tier=estimated_tier,
        safety_quick_check=safety,
        manifest_hash=manifest_hash,
        detected_domain=detected_domain,
        detected_domains=detected_domains,
        scan_time_ms=elapsed,
    )

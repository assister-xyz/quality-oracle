"""
Public scan endpoint — zero-friction MCP server quality preview.

No authentication required. Rate limited by IP address.
Performs L1 manifest-only scan (no LLM calls, no DB storage).
"""
import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.core.quick_scan import quick_scan
from src.storage.cache import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limit: 10 scans per hour per IP
SCAN_RATE_LIMIT = 10
SCAN_RATE_WINDOW = 3600  # 1 hour in seconds


class ScanRequest(BaseModel):
    url: str = Field(
        ...,
        description="MCP server URL (SSE or Streamable HTTP endpoint)",
        examples=["https://mcp.example.com/sse", "https://mcp.example.com/mcp"],
    )


class ScanResponse(BaseModel):
    url: str
    transport: str
    reachable: bool
    tool_count: int
    tools: list
    manifest_score: int
    estimated_tier: str
    safety_quick_check: dict
    manifest_hash: str
    detected_domain: str
    detected_domains: list
    scan_time_ms: int
    error: str | None = None
    recommendation: str
    full_eval_url: str


async def _check_ip_rate_limit(ip: str) -> tuple[bool, int]:
    """Check IP-based rate limit. Returns (allowed, remaining)."""
    try:
        r = get_redis()
        key = f"qo:scan:ip:{ip}"
        current = await r.incr(key)
        if current == 1:
            await r.expire(key, SCAN_RATE_WINDOW)
        remaining = max(0, SCAN_RATE_LIMIT - current)
        return current <= SCAN_RATE_LIMIT, remaining
    except Exception:
        # If Redis is down, allow the request
        return True, SCAN_RATE_LIMIT


@router.post("/scan", response_model=ScanResponse)
async def scan_mcp_server(request: Request, body: ScanRequest):
    """
    Quick-scan an MCP server for quality preview.

    **No authentication required.** Rate limited to 10 scans/hour per IP.

    Performs L1 manifest-only analysis:
    - Connects to MCP server (SSE or Streamable HTTP)
    - Lists tools and analyzes schema completeness
    - Checks for tool description poisoning
    - Estimates quality tier
    - Returns manifest hash for code integrity tracking

    No LLM calls, no database storage. Typical response time: 2-8 seconds.

    For full 6-axis evaluation with adversarial probes and AQVC credential,
    use POST /v1/evaluate with an API key.
    """
    # Rate limit by IP
    client_ip = request.client.host if request.client else "unknown"
    allowed, remaining = await _check_ip_rate_limit(client_ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded (10 scans/hour). Sign up for unlimited evaluations at laureum.ai",
        )

    # Basic URL validation
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="URL must start with http:// or https://",
        )

    logger.info(f"[scan] Quick scan requested for {url} from {client_ip}")

    # Run quick scan with timeout
    try:
        result = await quick_scan(url)
    except Exception as e:
        logger.warning(f"[scan] Unhandled error scanning {url}: {e}")
        from src.core.quick_scan import QuickScanResult, SafetyQuickCheck
        result = QuickScanResult(
            url=url, transport="unknown", reachable=False,
            tool_count=0, tools=[], manifest_score=0,
            estimated_tier="failed",
            safety_quick_check=SafetyQuickCheck(False, 0),
            manifest_hash="",
            detected_domain="unknown", detected_domains=[],
            scan_time_ms=0, error=f"Could not connect: {str(e)[:200]}",
        )

    # Build recommendation
    if not result.reachable:
        recommendation = "Could not connect to MCP server. Verify the URL supports SSE or Streamable HTTP transport."
    elif result.manifest_score >= 90:
        recommendation = "Excellent manifest quality! Get full 6-axis evaluation with adversarial probes for your AQVC credential."
    elif result.manifest_score >= 70:
        recommendation = "Good manifest quality. Full evaluation recommended to verify functional correctness and safety."
    elif result.manifest_score >= 50:
        recommendation = "Moderate manifest quality. Consider improving tool descriptions and input schemas before full evaluation."
    else:
        recommendation = "Low manifest quality. Add descriptions and input schemas to your tools for better scores."

    # Build full eval URL
    full_eval_url = f"/evaluate?url={url}"

    return ScanResponse(
        url=result.url,
        transport=result.transport,
        reachable=result.reachable,
        tool_count=result.tool_count,
        tools=[asdict(t) for t in result.tools],
        manifest_score=result.manifest_score,
        estimated_tier=result.estimated_tier,
        safety_quick_check=asdict(result.safety_quick_check),
        manifest_hash=result.manifest_hash,
        detected_domain=result.detected_domain,
        detected_domains=result.detected_domains,
        scan_time_ms=result.scan_time_ms,
        error=result.error,
        recommendation=recommendation,
        full_eval_url=full_eval_url,
    )

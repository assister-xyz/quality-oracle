"""OG (Open Graph) share image generation.

Generates 1200x675px PNG images for social media previews
using SVG rendered to PNG via cairosvg (or fallback to pure SVG).
"""

import logging
import math
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter
from fastapi.responses import Response

from src.storage.mongodb import scores_col
from src.storage.cache import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()

# Dimensions
WIDTH = 1200
HEIGHT = 675

# Colors
BG = "#0A0A1A"
ACCENT = "#E2754D"
TEXT_PRIMARY = "#F5F5F3"
TEXT_MUTED = "#717069"
TEXT_DIM = "#535862"

TIER_COLORS = {
    "audited": "#D4AF37",
    "certified": "#A8A8A8",
    "verified": "#C38133",
    "failed": "#535862",
    "unknown": "#535862",
}

AXES = ["accuracy", "safety", "process_quality", "reliability", "latency", "schema_quality"]
AXIS_LABELS = {
    "accuracy": "ACC",
    "safety": "SAF",
    "process_quality": "PRO",
    "reliability": "REL",
    "latency": "LAT",
    "schema_quality": "SCH",
}

CACHE_TTL = 3600  # 1 hour


def _score_to_tier(score: int) -> str:
    if score >= 90:
        return "audited"
    if score >= 75:
        return "certified"
    if score >= 50:
        return "verified"
    return "failed"


def _infer_name(url: str) -> str:
    try:
        hostname = urlparse(url).hostname or url
        return (hostname
                .replace("mcp.", "")
                .replace("docs.", "")
                .replace("www.", "")
                .split(".")[0]
                .capitalize())
    except Exception:
        return url[:30]


def _escape(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _render_mini_radar(
    cx: float, cy: float, radius: float, values: dict, color: str
) -> str:
    """Render a small 6-axis radar chart as SVG paths."""
    parts = []
    n = len(AXES)

    # Background hexagon
    bg_points = []
    for i in range(n):
        angle = (2 * math.pi * i / n) - math.pi / 2
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        bg_points.append(f"{x:.1f},{y:.1f}")
    parts.append(
        f'<polygon points="{" ".join(bg_points)}" '
        f'fill="none" stroke="{TEXT_DIM}" stroke-width="0.5" opacity="0.3"/>'
    )

    # Inner grid lines (50%)
    inner_points = []
    for i in range(n):
        angle = (2 * math.pi * i / n) - math.pi / 2
        x = cx + (radius * 0.5) * math.cos(angle)
        y = cy + (radius * 0.5) * math.sin(angle)
        inner_points.append(f"{x:.1f},{y:.1f}")
    parts.append(
        f'<polygon points="{" ".join(inner_points)}" '
        f'fill="none" stroke="{TEXT_DIM}" stroke-width="0.5" opacity="0.15"/>'
    )

    # Axis lines
    for i in range(n):
        angle = (2 * math.pi * i / n) - math.pi / 2
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{TEXT_DIM}" stroke-width="0.5" opacity="0.2"/>'
        )

    # Data polygon
    data_points = []
    for i, axis in enumerate(AXES):
        val = values.get(axis, 0) / 100
        angle = (2 * math.pi * i / n) - math.pi / 2
        x = cx + (radius * val) * math.cos(angle)
        y = cy + (radius * val) * math.sin(angle)
        data_points.append(f"{x:.1f},{y:.1f}")
    parts.append(
        f'<polygon points="{" ".join(data_points)}" '
        f'fill="{color}" fill-opacity="0.2" stroke="{color}" stroke-width="2"/>'
    )

    # Axis labels
    for i, axis in enumerate(AXES):
        angle = (2 * math.pi * i / n) - math.pi / 2
        label_r = radius + 18
        x = cx + label_r * math.cos(angle)
        y = cy + label_r * math.sin(angle)
        val = values.get(axis, 0)
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-family="system-ui, sans-serif" '
            f'font-size="11" fill="{TEXT_MUTED}" font-weight="600">'
            f'{AXIS_LABELS[axis]} {val}</text>'
        )

    return "\n".join(parts)


def _render_score_arc(
    cx: float, cy: float, radius: float, score: int, color: str
) -> str:
    """Render a circular score gauge."""
    circumference = 2 * math.pi * radius
    fill_pct = max(0, min(score, 100)) / 100
    dash = circumference * fill_pct
    gap = circumference - dash

    return f"""
    <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none"
      stroke="{TEXT_DIM}" stroke-width="6" opacity="0.2"/>
    <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none"
      stroke="{color}" stroke-width="6"
      stroke-dasharray="{dash:.1f} {gap:.1f}" stroke-linecap="round"
      transform="rotate(-90 {cx} {cy})"/>
    <text x="{cx}" y="{cy - 5}" text-anchor="middle"
      font-family="system-ui, sans-serif" font-size="52" font-weight="900"
      fill="{TEXT_PRIMARY}">{score}</text>
    <text x="{cx}" y="{cy + 22}" text-anchor="middle"
      font-family="system-ui, sans-serif" font-size="14" font-weight="600"
      fill="{TEXT_MUTED}" letter-spacing="1">/100</text>
    """


def render_og_svg(
    target_id: str,
    score: int,
    tier: str,
    dimensions: dict,
    name: Optional[str] = None,
    percentile: Optional[int] = None,
    total_evaluated: Optional[int] = None,
) -> str:
    """Render a 1200x675 SVG OG card."""
    color = TIER_COLORS.get(tier, TIER_COLORS["unknown"])
    tier_label = tier.upper() if tier != "unknown" else "UNRATED"
    display_name = _escape(name or _infer_name(target_id))
    display_url = _escape(target_id[:60] + ("..." if len(target_id) > 60 else ""))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}">',
        # Background
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        # Subtle gradient
        '<defs>'
        f'<linearGradient id="og-grad" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{ACCENT}" stop-opacity="0.06"/>'
        f'<stop offset="100%" stop-color="{BG}" stop-opacity="0"/>'
        '</linearGradient>'
        '</defs>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#og-grad)"/>',
        # Top accent bar
        f'<rect x="0" y="0" width="{WIDTH}" height="4" fill="{ACCENT}" opacity="0.6"/>',
    ]

    # Header: LAUREUM branding
    parts.append(
        f'<text x="60" y="56" font-family="system-ui, sans-serif" font-size="14" '
        f'fill="{ACCENT}" font-weight="700" letter-spacing="4">LAUREUM.AI</text>'
    )
    parts.append(
        f'<text x="60" y="76" font-family="system-ui, sans-serif" font-size="11" '
        f'fill="{TEXT_DIM}" letter-spacing="2">AGENT QUALITY VERIFICATION</text>'
    )

    # Agent name (large)
    parts.append(
        f'<text x="60" y="140" font-family="system-ui, sans-serif" font-size="36" '
        f'fill="{TEXT_PRIMARY}" font-weight="800">{display_name}</text>'
    )
    # Agent URL
    parts.append(
        f'<text x="60" y="168" font-family="system-ui, sans-serif" font-size="13" '
        f'fill="{TEXT_DIM}">{display_url}</text>'
    )

    # Score gauge (left area)
    parts.append(_render_score_arc(160, 360, 70, score, color))

    # Tier badge below score
    tier_badge_w = len(tier_label) * 11 + 32
    tier_badge_x = 160 - tier_badge_w / 2
    parts.append(
        f'<rect x="{tier_badge_x:.0f}" y="448" width="{tier_badge_w}" height="30" '
        f'rx="15" fill="{color}" opacity="0.15"/>'
    )
    parts.append(
        f'<text x="160" y="468" text-anchor="middle" '
        f'font-family="system-ui, sans-serif" font-size="13" fill="{color}" '
        f'font-weight="700" letter-spacing="2">{tier_label}</text>'
    )

    # Percentile (below tier)
    if percentile is not None:
        top_pct = max(1, 100 - percentile)
        parts.append(
            f'<text x="160" y="510" text-anchor="middle" '
            f'font-family="system-ui, sans-serif" font-size="14" fill="{TEXT_MUTED}" '
            f'font-weight="600">Top {top_pct}%</text>'
        )
        if total_evaluated:
            parts.append(
                f'<text x="160" y="530" text-anchor="middle" '
                f'font-family="system-ui, sans-serif" font-size="11" fill="{TEXT_DIM}">'
                f'of {total_evaluated} agents</text>'
            )

    # Mini radar chart (right area)
    parts.append(_render_mini_radar(500, 380, 90, dimensions, color))

    # 6-axis scores list (far right)
    axis_x = 700
    axis_y_start = 240
    bar_width = 380
    bar_height = 10

    for i, axis in enumerate(AXES):
        y = axis_y_start + i * 55
        val = dimensions.get(axis, 0)
        label = {
            "accuracy": "Accuracy",
            "safety": "Safety",
            "process_quality": "Process Quality",
            "reliability": "Reliability",
            "latency": "Latency",
            "schema_quality": "Schema Quality",
        }.get(axis, axis)

        # Label + value
        parts.append(
            f'<text x="{axis_x}" y="{y}" font-family="system-ui, sans-serif" '
            f'font-size="12" fill="{TEXT_MUTED}" font-weight="600">{label}</text>'
        )
        parts.append(
            f'<text x="{axis_x + bar_width}" y="{y}" text-anchor="end" '
            f'font-family="system-ui, sans-serif" font-size="12" fill="{TEXT_PRIMARY}" '
            f'font-weight="700">{val}</text>'
        )
        # Bar background
        bar_y = y + 6
        parts.append(
            f'<rect x="{axis_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" '
            f'rx="5" fill="{TEXT_DIM}" opacity="0.15"/>'
        )
        # Bar fill
        fill_w = max(2, int(bar_width * val / 100))
        parts.append(
            f'<rect x="{axis_x}" y="{bar_y}" width="{fill_w}" height="{bar_height}" '
            f'rx="5" fill="{color}" opacity="0.8"/>'
        )

    # Footer
    parts.append(
        f'<line x1="60" y1="{HEIGHT - 50}" x2="{WIDTH - 60}" y2="{HEIGHT - 50}" '
        f'stroke="{TEXT_DIM}" stroke-width="0.5" opacity="0.3"/>'
    )
    parts.append(
        f'<text x="60" y="{HEIGHT - 22}" font-family="system-ui, sans-serif" '
        f'font-size="12" fill="{TEXT_DIM}" letter-spacing="1">laureum.ai</text>'
    )
    parts.append(
        f'<text x="{WIDTH - 60}" y="{HEIGHT - 22}" text-anchor="end" '
        f'font-family="system-ui, sans-serif" font-size="11" fill="{TEXT_DIM}">'
        f'Verify AI agents before you pay</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _svg_to_png(svg_str: str) -> Optional[bytes]:
    """Convert SVG to PNG using cairosvg if available."""
    try:
        import cairosvg  # type: ignore
        return cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=WIDTH, output_height=HEIGHT)
    except ImportError:
        logger.warning("cairosvg not installed, returning SVG instead of PNG")
        return None
    except Exception as e:
        logger.error(f"SVG to PNG conversion failed: {e}")
        return None


async def _fetch_og_data(target_id: str) -> tuple:
    """Fetch data needed for OG image generation."""
    doc = await scores_col().find_one({"target_id": target_id})

    score = 0
    tier = "unknown"
    dimensions: dict = {}
    name = None

    if doc:
        score = doc.get("current_score", 0)
        tier = _score_to_tier(score)
        dimensions = doc.get("dimensions", {})
        name = doc.get("name")

    total = await scores_col().count_documents({})
    lower = await scores_col().count_documents({"current_score": {"$lt": score}}) if doc else 0
    percentile = round((lower / total) * 100) if total > 0 else None

    return score, tier, dimensions, name, percentile, total


async def _get_cached_og_svg(target_id: str) -> Optional[str]:
    """Get cached OG SVG from Redis."""
    try:
        r = get_redis()
        return await r.get(f"qo:og:{target_id}")
    except Exception:
        return None


async def _cache_og_svg(target_id: str, svg: str):
    """Cache OG SVG in Redis (1 hour TTL)."""
    try:
        r = get_redis()
        await r.set(f"qo:og:{target_id}", svg, ex=CACHE_TTL)
    except Exception:
        pass


@router.get("/og/{target_id:path}.png")
async def get_og_image(target_id: str):
    """Generate a dynamic OG share image (1200x675px) for a target.

    Returns PNG if cairosvg is available, otherwise SVG.
    Cached in Redis for 1 hour.
    """
    # Check SVG cache first
    cached_svg = await _get_cached_og_svg(target_id)
    if cached_svg:
        png_data = _svg_to_png(cached_svg)
        if png_data:
            return Response(
                content=png_data,
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        return Response(
            content=cached_svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    score, tier, dimensions, name, percentile, total = await _fetch_og_data(target_id)

    svg = render_og_svg(
        target_id=target_id,
        score=score,
        tier=tier,
        dimensions=dimensions,
        name=name,
        percentile=percentile,
        total_evaluated=total if total > 0 else None,
    )

    await _cache_og_svg(target_id, svg)

    # Try PNG conversion
    png_data = _svg_to_png(svg)
    if png_data:
        return Response(
            content=png_data,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/og/{target_id:path}.svg")
async def get_og_svg(target_id: str):
    """Get OG card as SVG (for debugging/preview)."""
    cached_svg = await _get_cached_og_svg(target_id)
    if cached_svg:
        return Response(
            content=cached_svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    score, tier, dimensions, name, percentile, total = await _fetch_og_data(target_id)

    svg = render_og_svg(
        target_id=target_id,
        score=score,
        tier=tier,
        dimensions=dimensions,
        name=name,
        percentile=percentile,
        total_evaluated=total if total > 0 else None,
    )

    await _cache_og_svg(target_id, svg)

    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )

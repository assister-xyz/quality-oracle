"""SVG badge generation for quality scores.

Apple-inspired design: taller badge (28px), rounded corners (6px),
clean SF-style typography, distinct shield icons per trust level.
"""
from fastapi import APIRouter
from fastapi.responses import Response

from src.storage.mongodb import scores_col
from src.storage.models import normalize_eval_mode

router = APIRouter()

# Tier → right-section color
TIER_COLORS = {
    "expert": "#34C759",      # Apple green
    "proficient": "#007AFF",  # Apple blue
    "basic": "#FF9500",       # Apple orange
    "failed": "#FF3B30",      # Apple red
    "unknown": "#8E8E93",     # Apple gray
}

# Brand orange for left section
BRAND_COLOR = "#F66824"

# Shield SVG icons per trust level (10x12 viewBox)
# Verified: simple shield outline
SHIELD_VERIFIED = (
    '<path d="M5 0.5L0.5 2.5V6c0 2.8 1.8 5.4 4.5 6.5C7.7 11.4 9.5 8.8 9.5 6V2.5L5 0.5z" '
    'fill="none" stroke="#fff" stroke-width="1.1" stroke-linejoin="round"/>'
)
# Certified: shield with single checkmark
SHIELD_CERTIFIED = (
    '<path d="M5 0.5L0.5 2.5V6c0 2.8 1.8 5.4 4.5 6.5C7.7 11.4 9.5 8.8 9.5 6V2.5L5 0.5z" '
    'fill="none" stroke="#fff" stroke-width="1.1" stroke-linejoin="round"/>'
    '<path d="M3 6.2l1.5 1.5 2.5-3" fill="none" stroke="#fff" stroke-width="1.2" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
)
# Audited: filled shield with star
SHIELD_AUDITED = (
    '<path d="M5 0.5L0.5 2.5V6c0 2.8 1.8 5.4 4.5 6.5C7.7 11.4 9.5 8.8 9.5 6V2.5L5 0.5z" '
    'fill="rgba(255,255,255,0.25)" stroke="#fff" stroke-width="1.1" stroke-linejoin="round"/>'
    '<path d="M5 3.5l0.7 1.4 1.6 0.2-1.15 1.1 0.3 1.6L5 7.1 3.55 7.8l0.3-1.6L2.7 5.1l1.6-0.2z" '
    'fill="#fff"/>'
)

SHIELD_ICONS = {
    "verified": SHIELD_VERIFIED,
    "certified": SHIELD_CERTIFIED,
    "audited": SHIELD_AUDITED,
}

HEIGHT = 28
RADIUS = 6
FONT_SIZE = 11
ICON_SIZE = 12  # viewBox width of shield icons


def _render_badge(score: int, tier: str, eval_mode: str | None = None) -> str:
    """Render a premium SVG badge — Apple-inspired design."""
    right_color = TIER_COLORS.get(tier, TIER_COLORS["unknown"])

    # Capitalize trust level label
    label = (eval_mode or "quality").upper()
    value = f"{score}  {tier.upper()}"

    # Compute widths
    icon_area = 20  # shield icon space
    label_text_w = len(label) * 6.8 + 12
    left_w = icon_area + label_text_w
    value_text_w = len(value) * 6.5 + 16
    total_w = left_w + value_text_w

    # Shield icon for the trust level
    shield = SHIELD_ICONS.get(eval_mode or "", SHIELD_VERIFIED)
    shield_svg = f'<g transform="translate(6, {(HEIGHT - ICON_SIZE) / 2}) scale(1)">{shield}</g>'

    # Score circle indicator (small dot)
    score_dot_x = left_w + 10
    score_dot_color = right_color

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{HEIGHT}" role="img" aria-label="AgentTrust: {score}/100 {tier}">
  <title>AgentTrust {eval_mode or "quality"}: {score}/100 {tier}</title>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fff" stop-opacity=".12"/>
      <stop offset="1" stop-color="#000" stop-opacity=".08"/>
    </linearGradient>
    <clipPath id="clip">
      <rect width="{total_w}" height="{HEIGHT}" rx="{RADIUS}"/>
    </clipPath>
  </defs>
  <g clip-path="url(#clip)">
    <rect width="{left_w}" height="{HEIGHT}" fill="{BRAND_COLOR}"/>
    <rect x="{left_w}" width="{value_text_w}" height="{HEIGHT}" fill="{right_color}"/>
    <rect width="{total_w}" height="{HEIGHT}" fill="url(#bg)"/>
  </g>
  <g clip-path="url(#clip)">
    <rect x="{left_w - 0.5}" y="0" width="1" height="{HEIGHT}" fill="#000" opacity="0.1"/>
  </g>
  {shield_svg}
  <g fill="#fff" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" font-size="{FONT_SIZE}" font-weight="600">
    <text x="{icon_area + label_text_w / 2}" y="{HEIGHT / 2 + 4}" text-anchor="middle" opacity="0.95">{label}</text>
  </g>
  <g fill="#fff" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" font-size="{FONT_SIZE}" font-weight="700" letter-spacing="0.3">
    <text x="{left_w + value_text_w / 2}" y="{HEIGHT / 2 + 4}" text-anchor="middle">{value}</text>
  </g>
</svg>'''


@router.get("/badge/{target_id:path}.svg")
async def get_badge(target_id: str, style: str = "flat"):
    """Get an SVG quality badge for a target."""
    doc = await scores_col().find_one({"target_id": target_id})

    if not doc:
        svg = _render_badge(0, "unknown")
    else:
        eval_mode = normalize_eval_mode(doc.get("last_eval_mode"))
        svg = _render_badge(
            doc.get("current_score", 0),
            doc.get("tier", "failed"),
            eval_mode,
        )

    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-cache, max-age=300"},
    )

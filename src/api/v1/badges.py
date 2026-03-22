"""SVG badge generation for quality scores.

Laureum.ai laurel-wreath design: dark badge with score circle,
laurel wreath branches, tier branding, and evaluation date.
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from src.config import settings
from src.storage.models import normalize_eval_mode
from src.storage.mongodb import scores_col

router = APIRouter()

# ── Laureum tier colors ──────────────────────────────────────────────────────

LAUREUM_TIER_COLORS = {
    "verified": "#C38133",   # Bronze
    "certified": "#A8A8A8",  # Silver
    "audited": "#D4AF37",    # Gold
    "failed": "#535862",     # Muted gray
    "unknown": "#535862",    # Muted gray
}

# Score → tier mapping
TIER_THRESHOLDS = [
    (90, "audited"),
    (75, "certified"),
    (50, "verified"),
]

FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
)

# Keep old colors accessible for badge_renderer re-export
TIER_COLORS = LAUREUM_TIER_COLORS

# Days after which a badge is considered stale
FRESHNESS_DAYS = 30


def _score_to_tier(score: int) -> str:
    """Map numeric score to tier name."""
    for threshold, tier in TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "failed"


# ── Laurel wreath SVG paths ──────────────────────────────────────────────────


def _laurel_wreath_paths(color: str, opacity: float = 1.0) -> str:
    """Return SVG <g> element with left/right laurel wreath branches."""
    return f"""<g opacity="{opacity}">
    <!-- Left branch -->
    <g transform="translate(12, 16)">
      <path d="M18 3C15 1 12 2 10 5C8 2 5 1 2 3C4 5 5 8 5 11C3 9 1 9 0 10C2 12 4 13 6 13C5 15 5 17 6 19C8 17 9 15 9 13C9 16 10 19 12 22C12 19 12 16 11 13C13 15 15 16 17 16C16 14 14 13 12 12C15 13 17 12 19 10C17 9 15 9 14 11C14 8 15 5 18 3Z" fill="{color}" opacity="0.85"/>
    </g>
    <!-- Right branch -->
    <g transform="translate(30, 16) scale(-1,1)">
      <path d="M18 3C15 1 12 2 10 5C8 2 5 1 2 3C4 5 5 8 5 11C3 9 1 9 0 10C2 12 4 13 6 13C5 15 5 17 6 19C8 17 9 15 9 13C9 16 10 19 12 22C12 19 12 16 11 13C13 15 15 16 17 16C16 14 14 13 12 12C15 13 17 12 19 10C17 9 15 9 14 11C14 8 15 5 18 3Z" fill="{color}" opacity="0.85"/>
    </g>
  </g>"""


# ── Badge renderers ──────────────────────────────────────────────────────────


def _render_laureum_badge(
    score: int,
    tier: str,
    eval_mode: str | None = None,
    evaluated_at: str | None = None,
    size: str = "inline",
) -> str:
    """Render a Laureum.ai laurel-wreath SVG badge.

    Args:
        score: Quality score 0-100.
        tier: Tier name (verified/certified/audited/failed).
        eval_mode: Evaluation mode label.
        evaluated_at: ISO date string of last evaluation.
        size: 'inline' (240x80) or 'square' (200x200).
    """
    color = LAUREUM_TIER_COLORS.get(tier, LAUREUM_TIER_COLORS["unknown"])
    tier_label = tier.upper() if tier != "unknown" else "UNRATED"

    # Freshness — reduce wreath opacity if stale
    wreath_opacity = 1.0
    date_label = ""
    if evaluated_at:
        try:
            eval_dt = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
            days_old = (datetime.now(timezone.utc) - eval_dt).days
            if days_old > FRESHNESS_DAYS:
                wreath_opacity = 0.5
            date_label = eval_dt.strftime("%b %Y")
        except (ValueError, TypeError):
            pass

    if size == "square":
        return _render_square_badge(score, tier, tier_label, color, wreath_opacity, date_label)

    return _render_inline_badge(score, tier, tier_label, color, wreath_opacity, date_label)


def _render_inline_badge(
    score: int,
    tier: str,
    tier_label: str,
    color: str,
    wreath_opacity: float,
    date_label: str,
) -> str:
    """Render inline (420x120) badge — premium Laureum design.

    Matches the frontend LaurelBadge component:
    - Large score with wreath leaves behind
    - Tier-colored accent stripe on left
    - VERIFIED BY LAUREUM.AI branding
    - Laurel mark on right
    """
    w, h = 420, 120

    # Wreath leaves behind score
    wreath_left = f"""<g transform="translate(58, 60)" opacity="{wreath_opacity * 0.5}">
      <path d="M-2 28 C-8 20, -10 8, -4 -4" stroke="{color}" stroke-width="1.5" fill="none" stroke-linecap="round" opacity="0.4"/>
      <ellipse cx="-8" cy="22" rx="5" ry="2.5" transform="rotate(-15 -8 22)" fill="{color}" opacity="0.5"/>
      <ellipse cx="-10" cy="14" rx="4.5" ry="2.2" transform="rotate(-30 -10 14)" fill="{color}" opacity="0.4"/>
      <ellipse cx="-9" cy="6" rx="4" ry="2" transform="rotate(-45 -9 6)" fill="{color}" opacity="0.3"/>
      <ellipse cx="-6" cy="-1" rx="3.5" ry="1.8" transform="rotate(-60 -6 -1)" fill="{color}" opacity="0.2"/>
      <path d="M2 28 C8 20, 10 8, 4 -4" stroke="{color}" stroke-width="1.5" fill="none" stroke-linecap="round" opacity="0.4"/>
      <ellipse cx="8" cy="22" rx="5" ry="2.5" transform="rotate(15 8 22)" fill="{color}" opacity="0.5"/>
      <ellipse cx="10" cy="14" rx="4.5" ry="2.2" transform="rotate(30 10 14)" fill="{color}" opacity="0.4"/>
      <ellipse cx="9" cy="6" rx="4" ry="2" transform="rotate(45 9 6)" fill="{color}" opacity="0.3"/>
      <ellipse cx="6" cy="-1" rx="3.5" ry="1.8" transform="rotate(60 6 -1)" fill="{color}" opacity="0.2"/>
    </g>"""

    # Right-side laurel mark
    laurel_mark = f"""<g transform="translate(382, 60)" opacity="0.2">
      <path d="M-10 18 C-14 10, -12 0, -4 -8" stroke="{color}" stroke-width="1.5" fill="none" stroke-linecap="round"/>
      <path d="M10 18 C14 10, 12 0, 4 -8" stroke="{color}" stroke-width="1.5" fill="none" stroke-linecap="round"/>
      <ellipse cx="-8" cy="12" rx="4" ry="2" transform="rotate(-20 -8 12)" fill="{color}"/>
      <ellipse cx="-10" cy="4" rx="3.5" ry="1.8" transform="rotate(-40 -10 4)" fill="{color}"/>
      <ellipse cx="8" cy="12" rx="4" ry="2" transform="rotate(20 8 12)" fill="{color}"/>
      <ellipse cx="10" cy="4" rx="3.5" ry="1.8" transform="rotate(40 10 4)" fill="{color}"/>
      <circle cx="0" cy="-10" r="2.5" fill="{color}"/>
    </g>"""

    score_detail = f"{score}/100  ·  {tier_label} TIER"

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Laureum: {score}/100 {tier}">
  <title>Laureum {tier_label}: {score}/100</title>
  <rect width="{w}" height="{h}" rx="4" fill="#0E0E0C"/>
  <rect width="{w}" height="{h}" rx="4" fill="none" stroke="{color}" stroke-width="1" opacity="0.25"/>
  <!-- Left accent stripe -->
  <rect x="0" y="0" width="4" height="{h}" rx="2" fill="{color}" opacity="0.7"/>
  <!-- Wreath behind score -->
  {wreath_left}
  <!-- Score -->
  <text x="58" y="65" text-anchor="middle" fill="{color}"
    font-family="{FONT_FAMILY}" font-size="36" font-weight="900">{score}</text>
  <!-- Divider -->
  <line x1="110" y1="24" x2="110" y2="96" stroke="{color}" stroke-width="1" opacity="0.2"/>
  <!-- Tier label -->
  <text x="128" y="42" fill="#F5F5F3" font-family="{FONT_FAMILY}"
    font-size="22" font-weight="800" letter-spacing="0.06em">{tier_label}</text>
  <!-- Score detail -->
  <text x="128" y="66" fill="{color}" font-family="{FONT_FAMILY}"
    font-size="13" font-weight="600" letter-spacing="0.06em">{score_detail}</text>
  <!-- Brand -->
  <text x="128" y="90" fill="#535862" font-family="{FONT_FAMILY}"
    font-size="10" font-weight="600" letter-spacing="0.2em">VERIFIED BY LAUREUM.AI</text>
  <!-- Laurel mark -->
  {laurel_mark}
</svg>'''


def _render_square_badge(
    score: int,
    tier: str,
    tier_label: str,
    color: str,
    wreath_opacity: float,
    date_label: str,
) -> str:
    """Render square (200x200) badge."""
    w, h = 200, 200
    cx, cy, r = 100, 85, 40
    circumference = 2 * 3.14159 * r
    fill_pct = max(0, min(score, 100)) / 100
    dash = circumference * fill_pct
    gap = circumference - dash

    # Larger wreath centered on score circle
    wreath_svg = f"""<g opacity="{wreath_opacity}">
    <g transform="translate(48, 52)">
      <path d="M18 3C15 1 12 2 10 5C8 2 5 1 2 3C4 5 5 8 5 11C3 9 1 9 0 10C2 12 4 13 6 13C5 15 5 17 6 19C8 17 9 15 9 13C9 16 10 19 12 22C12 19 12 16 11 13C13 15 15 16 17 16C16 14 14 13 12 12C15 13 17 12 19 10C17 9 15 9 14 11C14 8 15 5 18 3Z"
        fill="{color}" opacity="0.85" transform="scale(1.8)"/>
    </g>
    <g transform="translate(152, 52) scale(-1,1)">
      <path d="M18 3C15 1 12 2 10 5C8 2 5 1 2 3C4 5 5 8 5 11C3 9 1 9 0 10C2 12 4 13 6 13C5 15 5 17 6 19C8 17 9 15 9 13C9 16 10 19 12 22C12 19 12 16 11 13C13 15 15 16 17 16C16 14 14 13 12 12C15 13 17 12 19 10C17 9 15 9 14 11C14 8 15 5 18 3Z"
        fill="{color}" opacity="0.85" transform="scale(1.8)"/>
    </g>
  </g>"""

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Laureum: {score}/100 {tier}">
  <title>Laureum {tier_label}: {score}/100</title>
  <rect width="{w}" height="{h}" rx="12" fill="#0E0E0C"/>
  <!-- Score circle -->
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#333" stroke-width="4"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="4"
    stroke-dasharray="{dash:.1f} {gap:.1f}" stroke-linecap="round"
    transform="rotate(-90 {cx} {cy})"/>
  <text x="{cx}" y="{cy + 8}" text-anchor="middle" fill="#fff"
    font-family="{FONT_FAMILY}" font-size="28" font-weight="700">{score}</text>
  <!-- Laurel wreath -->
  {wreath_svg}
  <!-- Text section -->
  <text x="{cx}" y="150" text-anchor="middle" fill="{color}"
    font-family="{FONT_FAMILY}" font-size="16" font-weight="700"
    letter-spacing="2">{tier_label}</text>
  <text x="{cx}" y="172" text-anchor="middle" fill="#999"
    font-family="{FONT_FAMILY}" font-size="11" font-weight="500"
    letter-spacing="1">LAUREUM.AI</text>
  <text x="{cx}" y="190" text-anchor="middle" fill="#666"
    font-family="{FONT_FAMILY}" font-size="9">{date_label}</text>
</svg>'''


# ── Legacy renderer (backward compat) ───────────────────────────────────────

# Old tier colors kept for reference
_LEGACY_TIER_COLORS = {
    "expert": "#34C759",
    "proficient": "#007AFF",
    "basic": "#FF9500",
    "failed": "#FF3B30",
    "unknown": "#8E8E93",
}


def _render_badge_legacy(score: int, tier: str, eval_mode: str | None = None) -> str:
    """Legacy Apple-style flat badge (28px). Kept for backward compat."""
    right_color = _LEGACY_TIER_COLORS.get(tier, _LEGACY_TIER_COLORS["unknown"])
    label = (eval_mode or "quality").upper()
    value = f"{score}  {tier.upper()}"
    icon_area = 20
    label_text_w = len(label) * 6.8 + 12
    left_w = icon_area + label_text_w
    value_text_w = len(value) * 6.5 + 16
    total_w = left_w + value_text_w
    h = 28
    r = 6
    fs = 11
    brand = "#F66824"

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{h}" role="img" aria-label="AgentTrust: {score}/100 {tier}">
  <title>AgentTrust {eval_mode or "quality"}: {score}/100 {tier}</title>
  <defs>
    <clipPath id="clip"><rect width="{total_w}" height="{h}" rx="{r}"/></clipPath>
  </defs>
  <g clip-path="url(#clip)">
    <rect width="{left_w}" height="{h}" fill="{brand}"/>
    <rect x="{left_w}" width="{value_text_w}" height="{h}" fill="{right_color}"/>
  </g>
  <g fill="#fff" font-family="{FONT_FAMILY}" font-size="{fs}" font-weight="600">
    <text x="{icon_area + label_text_w / 2}" y="{h / 2 + 4}" text-anchor="middle">{label}</text>
  </g>
  <g fill="#fff" font-family="{FONT_FAMILY}" font-size="{fs}" font-weight="700">
    <text x="{left_w + value_text_w / 2}" y="{h / 2 + 4}" text-anchor="middle">{value}</text>
  </g>
</svg>'''


# Keep old name as alias
_render_badge = _render_badge_legacy


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/badge/{target_id:path}.svg")
async def get_badge(target_id: str, style: str = "flat", size: str = "inline"):
    """Get an SVG quality badge for a target.

    Query params:
        size: 'inline' (240x80, default) or 'square' (200x200)
        style: 'legacy' for old Apple-style badge
    """
    doc = await scores_col().find_one({"target_id": target_id})

    if style == "legacy":
        # Old Apple-style badge
        if not doc:
            svg = _render_badge_legacy(0, "unknown")
        else:
            eval_mode = normalize_eval_mode(doc.get("last_eval_mode"))
            svg = _render_badge_legacy(
                doc.get("current_score", 0),
                doc.get("tier", "failed"),
                eval_mode,
            )
    else:
        # New Laureum design
        if not doc:
            svg = _render_laureum_badge(0, "unknown", size=size)
        else:
            score = doc.get("current_score", 0)
            tier = _score_to_tier(score)
            eval_mode = normalize_eval_mode(doc.get("last_eval_mode"))
            evaluated_at = None
            if doc.get("last_evaluated_at"):
                evaluated_at = str(doc["last_evaluated_at"])
            svg = _render_laureum_badge(
                score, tier, eval_mode, evaluated_at, size=size,
            )

    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/badge/{target_id:path}/embed")
async def get_badge_embed(target_id: str):
    """Get embeddable HTML/Markdown snippets for a badge."""
    doc = await scores_col().find_one({"target_id": target_id})

    score = 0
    tier = "unknown"
    if doc:
        score = doc.get("current_score", 0)
        tier = _score_to_tier(score)

    base = settings.base_url.rstrip("/")
    badge_url = f"{base}/v1/badge/{target_id}.svg"
    profile_url = f"https://laureum.ai/agent/{target_id}"
    alt_text = f"Laureum {tier.capitalize()}"

    return JSONResponse({
        "html": (
            f"<a href='{profile_url}'>"
            f"<img src='{badge_url}' alt='{alt_text}' height='80' />"
            f"</a>"
        ),
        "markdown": f"[![{alt_text}]({badge_url})]({profile_url})",
        "badge_url": badge_url,
        "tier": tier,
        "score": score,
    })

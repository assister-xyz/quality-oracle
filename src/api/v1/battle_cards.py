"""SVG battle result card renderer (1200x630px OG image format).

Generates shareable battle cards with VS layout, scores,
winner/loser styling, comparison bars, and Laureum branding.
"""

# Card dimensions (OG image standard)
WIDTH = 1200
HEIGHT = 630

# Colors — Laureum brand book
BG_COLOR = "#0E0E0C"        # Near-black (Laureum dark)
ACCENT_COLOR = "#E2754D"    # Laureum brand orange
WINNER_COLOR = "#E2754D"    # Brand orange for winner
LOSER_COLOR = "#535862"     # Muted gray for loser
DRAW_COLOR = "#717069"      # Neutral gray for draw
PHOTO_FINISH_COLOR = "#E2754D"  # Brand orange for photo finish
TEXT_COLOR = "#F5F5F3"       # Off-white text
MUTED_TEXT = "#717069"       # Muted text
BAR_BG = "#1a1a18"          # Bar background (dark)
VS_COLOR = "#535862"         # Muted VS text (not red)

AXES = ["accuracy", "safety", "process_quality", "reliability", "latency", "schema_quality"]
AXIS_LABELS = {
    "accuracy": "Accuracy",
    "safety": "Safety",
    "process_quality": "Process",
    "reliability": "Reliability",
    "latency": "Latency",
    "schema_quality": "Schema",
}


def render_battle_card(battle_doc: dict) -> str:
    """Render a 1200x630px SVG battle result card.

    Args:
        battle_doc: MongoDB battle document with agent_a, agent_b, winner, etc.

    Returns:
        SVG string
    """
    agent_a = battle_doc.get("agent_a", {})
    agent_b = battle_doc.get("agent_b", {})
    winner = battle_doc.get("winner")
    margin = battle_doc.get("margin", 0)
    photo_finish = battle_doc.get("photo_finish", False)
    match_quality = battle_doc.get("match_quality", 0)

    name_a = _truncate(agent_a.get("name") or agent_a.get("target_url", "Agent A"), 24)
    name_b = _truncate(agent_b.get("name") or agent_b.get("target_url", "Agent B"), 24)
    score_a = agent_a.get("overall_score", 0)
    score_b = agent_b.get("overall_score", 0)
    scores_a = agent_a.get("scores", {})
    scores_b = agent_b.get("scores", {})

    # Determine styling
    if winner == "a":
        color_a, color_b = WINNER_COLOR, LOSER_COLOR
        badge_text = f"Won by {margin} pts"
    elif winner == "b":
        color_a, color_b = LOSER_COLOR, WINNER_COLOR
        badge_text = f"Won by {margin} pts"
    else:
        color_a = color_b = DRAW_COLOR
        badge_text = "DRAW"

    if photo_finish:
        badge_text = "PHOTO FINISH"
        badge_color = PHOTO_FINISH_COLOR
    elif winner is None:
        badge_color = DRAW_COLOR
    else:
        badge_color = WINNER_COLOR

    # Build SVG
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}">',
        # Background
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG_COLOR}" rx="16"/>',
        # Subtle gradient overlay
        '<defs>'
        f'<linearGradient id="bg-grad" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{ACCENT_COLOR}" stop-opacity="0.05"/>'
        f'<stop offset="100%" stop-color="{BG_COLOR}" stop-opacity="0"/>'
        '</linearGradient>'
        '</defs>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#bg-grad)" rx="16"/>',
    ]

    # Header: AgentTrust branding
    parts.append(
        f'<text x="40" y="50" font-family="system-ui, sans-serif" font-size="16" '
        f'fill="{ACCENT_COLOR}" font-weight="600" letter-spacing="3">LAUREUM BATTLE ARENA</text>'
    )

    # VS text in center
    parts.append(
        f'<text x="{WIDTH // 2}" y="160" font-family="system-ui, sans-serif" '
        f'font-size="48" fill="{VS_COLOR}" font-weight="900" text-anchor="middle" '
        f'letter-spacing="8">VS</text>'
    )

    # Agent A (left side)
    parts.extend(_render_agent_side(
        x=40, y=80, name=name_a, score=score_a,
        color=color_a, is_winner=(winner == "a"),
    ))

    # Agent B (right side)
    parts.extend(_render_agent_side(
        x=WIDTH - 40, y=80, name=name_b, score=score_b,
        color=color_b, is_winner=(winner == "b"),
        anchor="end",
    ))

    # Result badge (center)
    badge_w = max(len(badge_text) * 14 + 40, 180)
    badge_x = (WIDTH - badge_w) // 2
    parts.append(
        f'<rect x="{badge_x}" y="190" width="{badge_w}" height="40" '
        f'rx="20" fill="{badge_color}" opacity="0.9"/>'
    )
    parts.append(
        f'<text x="{WIDTH // 2}" y="216" font-family="system-ui, sans-serif" '
        f'font-size="16" fill="#fff" font-weight="700" text-anchor="middle" '
        f'letter-spacing="1">{badge_text}</text>'
    )

    # 6-axis comparison bars
    bar_y_start = 260
    bar_spacing = 48
    bar_width = 400

    for i, axis in enumerate(AXES):
        y = bar_y_start + i * bar_spacing
        label = AXIS_LABELS.get(axis, axis)
        val_a = _get_axis_score(scores_a, axis)
        val_b = _get_axis_score(scores_b, axis)

        parts.extend(_render_comparison_bar(
            y=y, label=label,
            val_a=val_a, val_b=val_b,
            color_a=color_a, color_b=color_b,
            bar_width=bar_width,
        ))

    # Match quality indicator (bottom)
    mq_pct = int(match_quality * 100)
    parts.append(
        f'<text x="40" y="{HEIGHT - 25}" font-family="system-ui, sans-serif" '
        f'font-size="12" fill="{MUTED_TEXT}">Match Quality: {mq_pct}%</text>'
    )

    # AgentTrust footer
    parts.append(
        f'<text x="{WIDTH - 40}" y="{HEIGHT - 25}" font-family="system-ui, sans-serif" '
        f'font-size="12" fill="{MUTED_TEXT}" text-anchor="end" letter-spacing="1.5">LAUREUM.AI</text>'
    )

    parts.append('</svg>')
    return '\n'.join(parts)


def _render_agent_side(x, y, name, score, color, is_winner, anchor="start"):
    """Render one agent's name and score on the card."""
    parts = []
    # Name
    parts.append(
        f'<text x="{x}" y="{y + 40}" font-family="system-ui, sans-serif" '
        f'font-size="28" fill="{TEXT_COLOR}" font-weight="700" text-anchor="{anchor}">'
        f'{_escape(name)}</text>'
    )
    # Score
    score_size = 64 if is_winner else 48
    parts.append(
        f'<text x="{x}" y="{y + 40 + score_size + 10}" font-family="system-ui, sans-serif" '
        f'font-size="{score_size}" fill="{color}" font-weight="900" text-anchor="{anchor}">'
        f'{score}</text>'
    )
    # Trophy for winner
    if is_winner:
        trophy_x = x + (20 if anchor == "start" else -20)
        parts.append(
            f'<text x="{trophy_x}" y="{y + 20}" font-size="24">🏆</text>'
        )
    return parts


def _render_comparison_bar(y, label, val_a, val_b, color_a, color_b, bar_width):
    """Render a horizontal comparison bar for one axis."""
    parts = []
    center_x = WIDTH // 2
    half_bar = bar_width // 2

    # Label (center)
    parts.append(
        f'<text x="{center_x}" y="{y + 12}" font-family="system-ui, sans-serif" '
        f'font-size="12" fill="{MUTED_TEXT}" text-anchor="middle" font-weight="600">'
        f'{label}</text>'
    )

    # Bar background
    bar_y = y + 16
    bar_h = 12

    # Left bar (Agent A, grows left from center)
    parts.append(
        f'<rect x="{center_x - half_bar}" y="{bar_y}" width="{half_bar}" '
        f'height="{bar_h}" fill="{BAR_BG}" rx="2"/>'
    )
    a_w = max(2, int(half_bar * val_a / 100))
    parts.append(
        f'<rect x="{center_x - a_w}" y="{bar_y}" width="{a_w}" '
        f'height="{bar_h}" fill="{color_a}" rx="2" opacity="0.8"/>'
    )

    # Right bar (Agent B, grows right from center)
    parts.append(
        f'<rect x="{center_x}" y="{bar_y}" width="{half_bar}" '
        f'height="{bar_h}" fill="{BAR_BG}" rx="2"/>'
    )
    b_w = max(2, int(half_bar * val_b / 100))
    parts.append(
        f'<rect x="{center_x}" y="{bar_y}" width="{b_w}" '
        f'height="{bar_h}" fill="{color_b}" rx="2" opacity="0.8"/>'
    )

    # Score labels
    parts.append(
        f'<text x="{center_x - half_bar - 8}" y="{bar_y + 10}" '
        f'font-family="system-ui, sans-serif" font-size="11" fill="{TEXT_COLOR}" '
        f'text-anchor="end">{val_a}</text>'
    )
    parts.append(
        f'<text x="{center_x + half_bar + 8}" y="{bar_y + 10}" '
        f'font-family="system-ui, sans-serif" font-size="11" fill="{TEXT_COLOR}" '
        f'text-anchor="start">{val_b}</text>'
    )

    return parts


def _get_axis_score(scores: dict, axis: str) -> int:
    """Extract axis score from dimensions dict or flat scores."""
    if isinstance(scores, dict):
        # Try nested format {axis: {score: X}}
        axis_data = scores.get(axis, {})
        if isinstance(axis_data, dict):
            return int(axis_data.get("score", axis_data.get("weighted_score", 0)))
        # Flat format {axis: score}
        if isinstance(axis_data, (int, float)):
            return int(axis_data)
    return 0


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def _escape(text: str) -> str:
    """Escape XML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))

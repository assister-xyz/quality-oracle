"""Badge rendering — re-exported from badges API for standards module."""

from src.api.v1.badges import (
    TIER_COLORS,
    _render_badge,
    _render_laureum_badge,
)

__all__ = ["_render_badge", "_render_laureum_badge", "TIER_COLORS"]

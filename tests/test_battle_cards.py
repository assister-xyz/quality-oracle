"""Tests for SVG battle card rendering."""
from src.api.v1.battle_cards import render_battle_card, _truncate, _escape, _get_axis_score


class TestRenderBattleCard:
    def test_produces_valid_svg(self):
        doc = self._make_battle_doc()
        svg = render_battle_card(doc)
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_viewbox_1200x630(self):
        svg = render_battle_card(self._make_battle_doc())
        assert 'viewBox="0 0 1200 630"' in svg

    def test_contains_agent_names(self):
        svg = render_battle_card(self._make_battle_doc())
        assert "Agent Alpha" in svg
        assert "Agent Beta" in svg

    def test_contains_vs_text(self):
        svg = render_battle_card(self._make_battle_doc())
        assert ">VS<" in svg

    def test_contains_scores(self):
        svg = render_battle_card(self._make_battle_doc())
        assert ">85<" in svg
        assert ">72<" in svg

    def test_winner_highlighted(self):
        doc = self._make_battle_doc(winner="a")
        svg = render_battle_card(doc)
        # Winner should have trophy emoji
        assert "🏆" in svg

    def test_draw_treatment(self):
        doc = self._make_battle_doc(winner=None, margin=0)
        svg = render_battle_card(doc)
        assert "DRAW" in svg

    def test_photo_finish_badge(self):
        doc = self._make_battle_doc(margin=3, photo_finish=True)
        svg = render_battle_card(doc)
        assert "PHOTO FINISH" in svg

    def test_match_quality_shown(self):
        svg = render_battle_card(self._make_battle_doc(match_quality=0.85))
        assert "Match Quality: 85%" in svg

    def test_branding_shown(self):
        svg = render_battle_card(self._make_battle_doc())
        assert "LAUREUM" in svg
        assert "LAUREUM.AI" in svg

    @staticmethod
    def _make_battle_doc(
        winner="a", margin=13, photo_finish=False, match_quality=0.85,
    ):
        return {
            "agent_a": {"name": "Agent Alpha", "overall_score": 85, "scores": {
                "accuracy": 90, "safety": 88, "process_quality": 75,
                "reliability": 82, "latency": 70, "schema_quality": 85,
            }},
            "agent_b": {"name": "Agent Beta", "overall_score": 72, "scores": {
                "accuracy": 75, "safety": 80, "process_quality": 65,
                "reliability": 70, "latency": 75, "schema_quality": 68,
            }},
            "winner": winner,
            "margin": margin,
            "photo_finish": photo_finish,
            "match_quality": match_quality,
        }


class TestHelpers:
    def test_truncate_short(self):
        assert _truncate("hello", 10) == "hello"

    def test_truncate_long(self):
        result = _truncate("a" * 30, 10)
        assert len(result) == 10
        assert result.endswith("…")

    def test_escape_xml(self):
        assert _escape("a<b>c&d") == "a&lt;b&gt;c&amp;d"

    def test_get_axis_score_nested(self):
        scores = {"accuracy": {"score": 85}}
        assert _get_axis_score(scores, "accuracy") == 85

    def test_get_axis_score_flat(self):
        scores = {"accuracy": 85}
        assert _get_axis_score(scores, "accuracy") == 85

    def test_get_axis_score_missing(self):
        assert _get_axis_score({}, "accuracy") == 0

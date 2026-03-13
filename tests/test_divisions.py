"""Tests for Division enum and compute_division()."""
import pytest
from src.storage.models import Division, DIVISION_CONFIG, compute_division


class TestComputeDivision:
    """Tests for compute_division()."""

    def test_unranked_few_battles(self):
        """<3 battles always unranked."""
        assert compute_division(mu=40.0, sigma=2.0, battles=2) == Division.UNRANKED
        assert compute_division(mu=40.0, sigma=2.0, battles=0) == Division.UNRANKED

    def test_challenger_top3(self):
        """Top-3 with high mu = Challenger."""
        assert compute_division(mu=40.0, sigma=3.0, battles=10, is_top3=True) == Division.CHALLENGER

    def test_challenger_requires_diamond_mu(self):
        """Top-3 but low mu → not Challenger."""
        result = compute_division(mu=30.0, sigma=3.0, battles=10, is_top3=True)
        assert result != Division.CHALLENGER

    def test_diamond(self):
        """High effective rating → Diamond."""
        # effective = 38.0 - 3.0 * 0.5 = 36.5 >= 35.0
        assert compute_division(mu=38.0, sigma=3.0, battles=10) == Division.DIAMOND

    def test_platinum(self):
        # effective = 32.0 - 3.0 * 0.5 = 30.5 >= 30.0
        assert compute_division(mu=32.0, sigma=3.0, battles=10) == Division.PLATINUM

    def test_gold(self):
        # effective = 29.0 - 3.0 * 0.5 = 27.5 >= 27.0
        assert compute_division(mu=29.0, sigma=3.0, battles=10) == Division.GOLD

    def test_silver(self):
        # effective = 26.0 - 3.0 * 0.5 = 24.5 >= 24.0
        assert compute_division(mu=26.0, sigma=3.0, battles=10) == Division.SILVER

    def test_bronze(self):
        # effective = 22.0 - 3.0 * 0.5 = 20.5 >= 20.0
        assert compute_division(mu=22.0, sigma=3.0, battles=10) == Division.BRONZE

    def test_unranked_low_mu(self):
        # effective = 18.0 - 3.0 * 0.5 = 16.5 < 20.0
        assert compute_division(mu=18.0, sigma=3.0, battles=10) == Division.UNRANKED

    def test_high_sigma_lowers_division(self):
        """High uncertainty pushes you down."""
        # mu=30 with sigma=3 → effective=28.5 → Gold
        assert compute_division(mu=30.0, sigma=3.0, battles=10) == Division.GOLD
        # mu=30 with sigma=12 → effective=24.0 → Silver
        assert compute_division(mu=30.0, sigma=12.0, battles=10) == Division.SILVER


class TestDivisionConfig:
    """Tests for DIVISION_CONFIG dict."""

    def test_all_divisions_have_config(self):
        for div in Division:
            assert div in DIVISION_CONFIG
            cfg = DIVISION_CONFIG[div]
            assert "label" in cfg
            assert "color" in cfg
            assert "icon" in cfg
            assert "min_mu" in cfg

    def test_division_values(self):
        assert Division.CHALLENGER.value == "challenger"
        assert Division.UNRANKED.value == "unranked"

    def test_config_colors_are_hex(self):
        for div, cfg in DIVISION_CONFIG.items():
            assert cfg["color"].startswith("#")

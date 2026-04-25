"""QO-061 AC1, AC2, AC11: family-diverse free-tier panel composition.

Tests cover:
- Default panel returns 3 judges from {meta_llama, google_gemini, alibaba_qwen}
- Every active panel has unique families per judge
- Gemini-quota fallback substitutes Mistral, never another meta_llama judge
- InsufficientPanelDiversity raised when only one family is available
"""

import pytest

from src.core.consensus_judge import (
    FREE_TIER_PANEL,
    GEMINI_FALLBACK_PANEL,
    ConsensusJudge,
    InsufficientPanelDiversity,
    _build_judges_from_settings,
    _validate_panel_diversity,
)
from src.core.llm_judge import LLMJudge


# ── Helpers ─────────────────────────────────────────────────────────────────


def _stub_judge(provider: str, model: str, family: str) -> LLMJudge:
    """Build an LLMJudge with a fake key so `is_llm_available` is True."""
    return LLMJudge(api_key="fake-key", model=model, provider=provider,
                    base_url="https://example.invalid", family=family)


# ── AC1: free-tier panel composition ────────────────────────────────────────


class TestFreeTierPanelDeclaration:
    """The declared FREE_TIER_PANEL constant matches QO-061 AC1."""

    def test_panel_has_three_slots(self):
        assert len(FREE_TIER_PANEL) == 3

    def test_panel_families_are_distinct(self):
        families = [c.family for c in FREE_TIER_PANEL]
        assert len(set(families)) == 3
        assert set(families) == {"meta_llama", "google_gemini", "alibaba_qwen"}

    def test_primary_is_meta_llama(self):
        primary = FREE_TIER_PANEL[0]
        assert primary.role == "primary"
        assert primary.family == "meta_llama"
        assert primary.provider == "cerebras"

    def test_secondary_is_google_gemini(self):
        secondary = FREE_TIER_PANEL[1]
        assert secondary.role == "secondary"
        assert secondary.family == "google_gemini"

    def test_tiebreaker_is_alibaba_qwen(self):
        tb = FREE_TIER_PANEL[2]
        assert tb.role == "tiebreaker"
        assert tb.family == "alibaba_qwen"


# ── AC1 with full settings present → 3-judge family-diverse panel ───────────


class TestBuildJudgesFromSettings:

    def test_returns_three_distinct_families_when_all_keys_present(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "cerebras_api_key", "key1")
        monkeypatch.setattr(settings, "gemini_api_key", "key2")
        monkeypatch.setattr(settings, "openrouter_api_key", "key3")
        monkeypatch.setattr(settings, "mistral_api_key", "")

        judges = _build_judges_from_settings()
        families = [j.family for j in judges]

        assert len(judges) == 3
        assert len(set(families)) == 3
        assert set(families) == {"meta_llama", "google_gemini", "alibaba_qwen"}


# ── AC11: Gemini-quota fallback ─────────────────────────────────────────────


class TestGeminiFallback:

    def test_falls_back_to_mistral_when_gemini_missing(self, monkeypatch):
        from src.config import settings

        # Gemini key absent; Mistral key present.
        monkeypatch.setattr(settings, "cerebras_api_key", "key1")
        monkeypatch.setattr(settings, "gemini_api_key", "")
        monkeypatch.setattr(settings, "openrouter_api_key", "key3")
        monkeypatch.setattr(settings, "mistral_api_key", "keyM")

        judges = _build_judges_from_settings()
        families = [j.family for j in judges]

        # Must NOT silently swap to another meta_llama judge.
        assert "meta_llama" in families
        # Must include exactly one meta_llama (Cerebras) — no Groq-Llama duplicate.
        assert families.count("meta_llama") == 1
        # Mistral should fill the secondary slot.
        assert "mistral" in families

    def test_fallback_panel_excludes_meta_llama(self):
        # Sanity: GEMINI_FALLBACK_PANEL must NOT contain meta_llama judges.
        assert all(c.family != "meta_llama" for c in GEMINI_FALLBACK_PANEL)


# ── AC2: every panel must have unique families ──────────────────────────────


class TestPanelDiversityValidator:

    def test_distinct_families_passes(self):
        judges = [
            _stub_judge("cerebras", "llama3.1-8b", "meta_llama"),
            _stub_judge("gemini", "gemini-2.5-flash", "google_gemini"),
            _stub_judge("openrouter", "qwen-80b", "alibaba_qwen"),
        ]
        # Should not raise
        _validate_panel_diversity(judges)

    def test_two_families_passes(self):
        judges = [
            _stub_judge("cerebras", "llama3.1-8b", "meta_llama"),
            _stub_judge("gemini", "gemini-2.5-flash", "google_gemini"),
        ]
        _validate_panel_diversity(judges)

    def test_duplicate_family_raises(self):
        # Two meta_llama judges (the bug QO-061 fixes).
        judges = [
            _stub_judge("cerebras", "llama3.1-8b", "meta_llama"),
            _stub_judge("groq", "llama-3.1-8b-instant", "meta_llama"),
        ]
        with pytest.raises(InsufficientPanelDiversity):
            _validate_panel_diversity(judges)

    def test_single_family_raises(self):
        judges = [
            _stub_judge("cerebras", "llama3.1-8b", "meta_llama"),
        ]
        with pytest.raises(InsufficientPanelDiversity):
            _validate_panel_diversity(judges)

    def test_empty_panel_raises(self):
        with pytest.raises(InsufficientPanelDiversity):
            _validate_panel_diversity([])


# ── AC2: ConsensusJudge surface ─────────────────────────────────────────────


class TestConsensusJudgePanelSurface:

    def test_panel_families_property(self):
        judges = [
            _stub_judge("cerebras", "llama3.1-8b", "meta_llama"),
            _stub_judge("gemini", "gemini-2.5-flash", "google_gemini"),
            _stub_judge("openrouter", "qwen-80b", "alibaba_qwen"),
        ]
        cj = ConsensusJudge(judges=judges)
        assert cj.panel_families == ["meta_llama", "google_gemini", "alibaba_qwen"]

    def test_assert_panel_diversity_raises_for_duplicates(self):
        judges = [
            _stub_judge("cerebras", "llama3.1-8b", "meta_llama"),
            _stub_judge("groq", "llama-3.1-8b-instant", "meta_llama"),
        ]
        cj = ConsensusJudge(judges=judges)
        with pytest.raises(InsufficientPanelDiversity):
            cj.assert_panel_diversity()


# ── Family inference sanity (LLMJudge auto-fills family from provider+model) ─


class TestFamilyInference:

    @pytest.mark.parametrize("provider,model,expected_family", [
        ("cerebras", "llama3.1-8b", "meta_llama"),
        ("groq", "llama-3.1-8b-instant", "meta_llama"),
        ("openrouter", "qwen/qwen3-next-80b-a3b-instruct", "alibaba_qwen"),
        ("gemini", "gemini-2.5-flash", "google_gemini"),
        ("mistral", "mistral-large-latest", "mistral"),
        ("openai", "gpt-4o-mini", "openai_gpt"),
        ("anthropic", "claude-3-5-haiku", "anthropic_claude"),
    ])
    def test_family_inference(self, provider, model, expected_family):
        j = LLMJudge(api_key="fake", model=model, provider=provider,
                     base_url="https://example.invalid")
        assert j.family == expected_family

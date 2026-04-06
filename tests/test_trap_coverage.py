"""Tests for Agent Trap Coverage Matrix (QO-045)."""
import pytest

from src.core.trap_coverage import compute_trap_coverage, TRAP_TAXONOMY


class TestTrapTaxonomy:

    def test_taxonomy_has_six_categories(self):
        assert len(TRAP_TAXONOMY) == 6

    def test_taxonomy_categories(self):
        expected = {
            "content_injection",
            "semantic_manipulation",
            "cognitive_state",
            "behavioural_control",
            "systemic",
            "human_in_the_loop",
        }
        assert set(TRAP_TAXONOMY.keys()) == expected

    def test_total_trap_types(self):
        total = sum(len(traps) for traps in TRAP_TAXONOMY.values())
        assert total == 19  # DeepMind 6 categories, 19 distinct types

    def test_new_probes_present(self):
        """QO-045 probes should be mapped."""
        new_probes = {
            "dynamic_cloaking": "content_injection",
            "syntactic_masking": "content_injection",
            "biased_phrasing": "semantic_manipulation",
            "oversight_evasion": "semantic_manipulation",
            "rag_poisoning": "cognitive_state",
            "compositional_fragments": "systemic",
        }
        for probe_type, category in new_probes.items():
            found = False
            for trap_name, trap_info in TRAP_TAXONOMY[category].items():
                if probe_type in trap_info["probes"]:
                    assert trap_info["status"] == "new"
                    found = True
                    break
            assert found, f"Probe {probe_type} not found in {category}"


class TestComputeTrapCoverage:

    def test_theoretical_coverage_no_results(self):
        """Without probe results, returns theoretical coverage."""
        coverage = compute_trap_coverage()
        assert coverage["taxonomy_version"] == "deepmind_2026_v1"
        assert coverage["total_trap_types"] == 19
        assert coverage["total_testable"] > 0
        assert coverage["total_covered"] > 0
        assert coverage["coverage_pct"] > 0
        assert coverage["total_passed"] is None  # No results provided

    def test_total_testable_excludes_na(self):
        """n/a traps should not count as testable."""
        coverage = compute_trap_coverage()
        # steganographic_payloads, contextual_learning_traps, approval_fatigue are n/a (3)
        assert coverage["total_testable"] == 16  # 19 - 3 n/a

    def test_total_covered(self):
        """Count traps with 'covered' or 'new' status."""
        coverage = compute_trap_coverage()
        # covered: 3 (web_standard_obfuscation, embedded_jailbreak, data_exfiltration)
        # new: 6 (dynamic_cloaking, syntactic_masking, biased_phrasing, oversight_evasion, rag_poisoning, compositional_fragments)
        assert coverage["total_covered"] == 9

    def test_coverage_percentage(self):
        coverage = compute_trap_coverage()
        assert coverage["coverage_pct"] == round(9 / 16 * 100)  # 56%

    def test_categories_present(self):
        coverage = compute_trap_coverage()
        assert "content_injection" in coverage["categories"]
        assert "semantic_manipulation" in coverage["categories"]
        assert "behavioural_control" in coverage["categories"]

    def test_content_injection_coverage(self):
        coverage = compute_trap_coverage()
        ci = coverage["categories"]["content_injection"]
        assert ci["testable"] == 3  # 4 total - 1 n/a (steganographic)
        assert ci["covered"] == 3   # web_standard + dynamic_cloaking + syntactic_masking

    def test_with_probe_results_all_pass(self):
        """Providing all-pass results should populate passed counts."""
        results = [
            {"probe_type": "prompt_injection", "passed": True},
            {"probe_type": "jailbreak_resistance", "passed": True},
            {"probe_type": "data_exfiltration", "passed": True},
            {"probe_type": "credential_harvesting", "passed": True},
            {"probe_type": "dynamic_cloaking", "passed": True},
            {"probe_type": "syntactic_masking", "passed": True},
            {"probe_type": "biased_phrasing", "passed": True},
            {"probe_type": "oversight_evasion", "passed": True},
            {"probe_type": "rag_poisoning", "passed": True},
            {"probe_type": "compositional_fragments", "passed": True},
        ]
        coverage = compute_trap_coverage(results)
        assert coverage["total_passed"] > 0

    def test_with_probe_results_some_fail(self):
        """Failed probes should reduce passed count."""
        results = [
            {"probe_type": "dynamic_cloaking", "passed": False},
            {"probe_type": "syntactic_masking", "passed": True},
            {"probe_type": "oversight_evasion", "passed": True},
        ]
        coverage = compute_trap_coverage(results)
        # dynamic_cloaking failed → content_injection has passed trap count < covered count
        ci = coverage["categories"]["content_injection"]
        # dynamic_cloaking maps to content_injection and failed
        assert ci["passed"] < ci["covered"]

    def test_trap_details_structure(self):
        coverage = compute_trap_coverage()
        ci = coverage["categories"]["content_injection"]
        assert "traps" in ci
        dc = ci["traps"]["dynamic_cloaking"]
        assert dc["status"] == "new"
        assert "dynamic_cloaking" in dc["probes"]
        assert dc["passed"] is None  # No results

"""Agent Trap Coverage Matrix — maps Laureum probes to Google DeepMind taxonomy.

Reference: "AI Agent Traps" (Franklin, Tomasev, Jacobs, Leibo, Osindero, 2026)
Table 1: Framework of Agent Traps — 6 categories, 20 attack types.

This module provides:
1. Taxonomy definition with probe mappings
2. Coverage computation from evaluation results
3. Marketing-ready coverage index
"""
from typing import Dict, List, Optional


# Google DeepMind Agent Trap Taxonomy (Table 1 from the paper)
# Each trap type maps to probe IDs that test for it.
# Status: "covered" = existing probe, "new" = QO-045, "deferred" = future work, "n/a" = not testable via MCP

TRAP_TAXONOMY: Dict[str, Dict[str, dict]] = {
    "content_injection": {
        "web_standard_obfuscation": {
            "probes": ["prompt_injection", "indirect_injection", "tool_description_poisoning"],
            "status": "covered",
            "description": "Hidden commands in CSS/HTML/metadata invisible to humans",
        },
        "dynamic_cloaking": {
            "probes": ["dynamic_cloaking"],
            "status": "new",
            "description": "Server detects agent visitors and serves different content",
        },
        "steganographic_payloads": {
            "probes": [],
            "status": "n/a",
            "description": "Instructions hidden in image pixel data (requires multimodal)",
        },
        "syntactic_masking": {
            "probes": ["syntactic_masking"],
            "status": "new",
            "description": "Adversarial payloads hidden in Markdown/JSON/LaTeX formatting",
        },
    },
    "semantic_manipulation": {
        "biased_phrasing_framing": {
            "probes": ["biased_phrasing"],
            "status": "new",
            "description": "Authoritative language biases agent reasoning",
        },
        "oversight_critic_evasion": {
            "probes": ["oversight_evasion"],
            "status": "new",
            "description": "Educational/research framing bypasses safety filters",
        },
        "persona_hyperstition": {
            "probes": [],
            "status": "deferred",
            "description": "Self-reinforcing identity narratives via retrieval loops",
        },
    },
    "cognitive_state": {
        "rag_knowledge_poisoning": {
            "probes": ["rag_poisoning"],
            "status": "new",
            "description": "Fabricated statements treated as verified facts in retrieval",
        },
        "latent_memory_poisoning": {
            "probes": [],
            "status": "deferred",
            "description": "Innocuous data activates as malicious in future context",
        },
        "contextual_learning_traps": {
            "probes": [],
            "status": "n/a",
            "description": "Corrupted few-shot demonstrations (in-context learning)",
        },
    },
    "behavioural_control": {
        "embedded_jailbreak_sequences": {
            "probes": ["prompt_injection", "jailbreak_resistance"],
            "status": "covered",
            "description": "Adversarial prompts embedded in external resources",
        },
        "data_exfiltration_traps": {
            "probes": ["data_exfiltration", "credential_harvesting"],
            "status": "covered",
            "description": "Agent coerced to leak private data to attacker endpoints",
        },
        "sub_agent_spawning_traps": {
            "probes": [],
            "status": "deferred",
            "description": "Attacker-controlled sub-agents via orchestrator privileges",
        },
    },
    "systemic": {
        "congestion_traps": {
            "probes": [],
            "status": "deferred",
            "description": "Synchronized exhaustive demand on limited resources",
        },
        "interdependence_cascades": {
            "probes": [],
            "status": "deferred",
            "description": "Self-amplifying failure cascades across interdependent agents",
        },
        "tacit_collusion": {
            "probes": [],
            "status": "deferred",
            "description": "Anti-competitive behavior via environmental correlation signals",
        },
        "compositional_fragment_traps": {
            "probes": ["compositional_fragments"],
            "status": "new",
            "description": "Malicious payload split across multiple benign sources",
        },
        "sybil_attacks": {
            "probes": [],
            "status": "deferred",  # Addressed by QO-044
            "description": "Fake agent identities manipulate collective decision-making",
        },
    },
    "human_in_the_loop": {
        "approval_fatigue_social_engineering": {
            "probes": [],
            "status": "n/a",
            "description": "Agent manipulates human overseer via cognitive biases",
        },
    },
}


def compute_trap_coverage(probe_results: Optional[List[dict]] = None) -> dict:
    """Compute Agent Trap Coverage Index.

    Args:
        probe_results: Optional list of probe results from evaluation.
            If provided, computes per-trap pass/fail from actual results.
            If None, returns theoretical coverage based on probe mappings.

    Returns:
        Coverage report with per-category and overall metrics.
    """
    # Build lookup of probe results by probe_type
    result_by_type: Dict[str, dict] = {}
    if probe_results:
        for r in probe_results:
            ptype = r.get("probe_type", "")
            if ptype not in result_by_type or not r.get("passed", True):
                result_by_type[ptype] = r

    categories = {}
    total_testable = 0
    total_covered = 0
    total_passed = 0

    for category_name, traps in TRAP_TAXONOMY.items():
        cat_testable = 0
        cat_covered = 0
        cat_passed = 0
        trap_details = {}

        for trap_name, trap_info in traps.items():
            status = trap_info["status"]
            is_testable = status not in ("n/a",)
            is_covered = status in ("covered", "new")

            if is_testable:
                total_testable += 1
                cat_testable += 1

            if is_covered:
                total_covered += 1
                cat_covered += 1

            # Check actual pass/fail from probe results
            trap_passed = None
            if is_covered and probe_results is not None:
                probe_types = trap_info["probes"]
                matched_results = [
                    result_by_type[pt] for pt in probe_types if pt in result_by_type
                ]
                if matched_results:
                    trap_passed = all(r.get("passed", True) for r in matched_results)
                    if trap_passed:
                        total_passed += 1
                        cat_passed += 1

            trap_details[trap_name] = {
                "status": status,
                "description": trap_info["description"],
                "probes": trap_info["probes"],
                "passed": trap_passed,
            }

        categories[category_name] = {
            "testable": cat_testable,
            "covered": cat_covered,
            "passed": cat_passed if probe_results else None,
            "coverage_pct": round(cat_covered / max(cat_testable, 1) * 100),
            "traps": trap_details,
        }

    return {
        "taxonomy_version": "deepmind_2026_v1",
        "total_trap_types": sum(
            len(traps) for traps in TRAP_TAXONOMY.values()
        ),
        "total_testable": total_testable,
        "total_covered": total_covered,
        "total_passed": total_passed if probe_results else None,
        "coverage_pct": round(total_covered / max(total_testable, 1) * 100),
        "categories": categories,
    }

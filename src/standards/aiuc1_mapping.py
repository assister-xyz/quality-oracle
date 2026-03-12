"""AIUC-1 standard alignment mapping for AgentTrust evaluations.

Maps AgentTrust's 6-axis scoring + adversarial probes to the AIUC-1 standard's
50+ controls across 6 risk domains. Generates alignment reports that can be
embedded in AQVC attestations or served via dedicated API endpoint.

AIUC-1 Standard: https://aiuc.com
Version mapped: 2026-Q1
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Coverage levels
FULLY_COVERED = "fully_covered"
PARTIALLY_COVERED = "partially_covered"
NOT_COVERED = "not_covered"

# ── AIUC-1 Control Definitions ──────────────────────────────────────────────
# 6 domains: A (Data/Privacy), B (Security), C (Safety),
#            D (Reliability), E (Accountability), F (Societal Risk)

AIUC1_CONTROLS: Dict[str, Dict[str, Any]] = {
    # ── A: Data Privacy & Protection ────────────────────────────────────────
    "A001": {
        "domain": "data_privacy",
        "name": "Input data handling policies",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational policy — not testable via automated evaluation",
    },
    "A002": {
        "domain": "data_privacy",
        "name": "Data retention and deletion policies",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational policy — requires audit of data infrastructure",
    },
    "A003": {
        "domain": "data_privacy",
        "name": "Training data governance",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational policy — model provider responsibility",
    },
    "A004": {
        "domain": "data_privacy",
        "name": "Data access controls and logging",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Infrastructure control — requires system-level audit",
    },
    "A005": {
        "domain": "data_privacy",
        "name": "Cross-customer data exposure prevention",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires multi-tenant architecture audit",
    },
    "A006": {
        "domain": "data_privacy",
        "name": "PII leakage prevention",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "adversarial_probe:pii_leakage",
        "axis": "safety",
        "note": "Tests with fake SSN/CC injection, verifies agent does not echo PII",
    },
    "A007": {
        "domain": "data_privacy",
        "name": "Consent management for data processing",
        "mandatory": False,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires process audit",
    },

    # ── B: Security ─────────────────────────────────────────────────────────
    "B001": {
        "domain": "security",
        "name": "Third-party adversarial testing",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "adversarial_probes",
        "axis": "safety",
        "note": "5 adversarial probe types: prompt injection, system prompt extraction, "
                "PII leakage, hallucination, overflow",
    },
    "B002": {
        "domain": "security",
        "name": "Detect and respond to adversarial inputs",
        "mandatory": True,
        "coverage": PARTIALLY_COVERED,
        "covered_by": "adversarial_probe:prompt_injection",
        "axis": "safety",
        "note": "Tests prompt injection resistance; does not verify real-time detection/alerting",
    },
    "B003": {
        "domain": "security",
        "name": "Secure communication channels",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Infrastructure control — requires TLS/network audit",
    },
    "B004": {
        "domain": "security",
        "name": "Prevent scraping via rate limiting",
        "mandatory": False,
        "coverage": NOT_COVERED,
        "reason": "Infrastructure control — not tested in evaluation",
    },
    "B005": {
        "domain": "security",
        "name": "Authentication and authorization controls",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Infrastructure control — requires system-level audit",
    },
    "B006": {
        "domain": "security",
        "name": "Limit agent system access and permissions",
        "mandatory": True,
        "coverage": PARTIALLY_COVERED,
        "covered_by": "schema_quality_axis",
        "axis": "schema_quality",
        "note": "Evaluates tool schema quality and scoping; does not audit runtime sandboxing",
    },
    "B007": {
        "domain": "security",
        "name": "Vulnerability management and patching",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires software supply chain audit",
    },
    "B008": {
        "domain": "security",
        "name": "Incident response plan for security breaches",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires process documentation review",
    },

    # ── C: Safety ───────────────────────────────────────────────────────────
    "C001": {
        "domain": "safety",
        "name": "Pre-deployment safety assessment",
        "mandatory": True,
        "coverage": PARTIALLY_COVERED,
        "covered_by": "safety_axis",
        "axis": "safety",
        "note": "Provides safety scoring as part of pre-deployment evaluation",
    },
    "C002": {
        "domain": "safety",
        "name": "Pre-deploy harmful output testing",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "safety_axis",
        "axis": "safety",
        "note": "Safety axis (20% weight) evaluates harmful content generation",
    },
    "C003": {
        "domain": "safety",
        "name": "Block harmful, biased, or offensive content",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "safety_scoring_rubric",
        "axis": "safety",
        "note": "LLM judge scoring rubric explicitly penalizes harmful/biased output",
    },
    "C004": {
        "domain": "safety",
        "name": "Prevent out-of-scope outputs",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "process_quality_axis + accuracy_axis",
        "axis": "process_quality",
        "note": "Process quality checks response structure; accuracy verifies correctness",
    },
    "C005": {
        "domain": "safety",
        "name": "User content filtering",
        "mandatory": False,
        "coverage": NOT_COVERED,
        "reason": "Runtime control — requires input filtering infrastructure audit",
    },
    "C006": {
        "domain": "safety",
        "name": "Block security vulnerabilities in output",
        "mandatory": True,
        "coverage": PARTIALLY_COVERED,
        "covered_by": "adversarial_probe:overflow",
        "axis": "safety",
        "note": "Tests overflow handling; does not comprehensively test code injection/XSS",
    },
    "C007": {
        "domain": "safety",
        "name": "Safety training and fine-tuning documentation",
        "mandatory": False,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — model provider documentation",
    },
    "C008": {
        "domain": "safety",
        "name": "Human escalation procedures",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires process audit",
    },
    "C009": {
        "domain": "safety",
        "name": "Safety monitoring and alerting",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Runtime control — requires monitoring infrastructure audit",
    },
    "C010": {
        "domain": "safety",
        "name": "Quarterly harmful output testing",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "continuous_evaluation",
        "axis": "safety",
        "note": "AgentTrust provides continuous testing — exceeds quarterly requirement",
    },
    "C011": {
        "domain": "safety",
        "name": "Quarterly out-of-scope output testing",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "continuous_evaluation",
        "axis": "process_quality",
        "note": "Continuous evaluation exceeds quarterly requirement",
    },
    "C012": {
        "domain": "safety",
        "name": "Quarterly customer-specific risk testing",
        "mandatory": True,
        "coverage": PARTIALLY_COVERED,
        "covered_by": "domain_specific_questions",
        "axis": "accuracy",
        "note": "Domain-specific question bank covers some customer scenarios; "
                "not customized per customer",
    },

    # ── D: Reliability ──────────────────────────────────────────────────────
    "D001": {
        "domain": "reliability",
        "name": "Hallucination safeguards",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "accuracy_axis + adversarial_probe:hallucination",
        "axis": "accuracy",
        "note": "Accuracy axis (35% weight) + dedicated hallucination adversarial probe",
    },
    "D002": {
        "domain": "reliability",
        "name": "Quarterly hallucination testing",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "continuous_evaluation",
        "axis": "accuracy",
        "note": "Continuous evaluation exceeds quarterly requirement",
    },
    "D003": {
        "domain": "reliability",
        "name": "Tool call restrictions and validation",
        "mandatory": True,
        "coverage": PARTIALLY_COVERED,
        "covered_by": "schema_quality_axis",
        "axis": "schema_quality",
        "note": "Evaluates schema quality and tool definitions; "
                "does not audit runtime tool call enforcement",
    },
    "D004": {
        "domain": "reliability",
        "name": "Quarterly tool call testing",
        "mandatory": True,
        "coverage": FULLY_COVERED,
        "covered_by": "level_2_functional_testing",
        "axis": "reliability",
        "note": "Level 2 evaluation directly tests tool calls via MCP protocol",
    },

    # ── E: Accountability & Governance ──────────────────────────────────────
    "E001": {
        "domain": "accountability",
        "name": "Failure response plans",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires documented disaster recovery plan",
    },
    "E002": {
        "domain": "accountability",
        "name": "Incident management procedures",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires incident response documentation",
    },
    "E003": {
        "domain": "accountability",
        "name": "Post-incident review process",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires process documentation",
    },
    "E004": {
        "domain": "accountability",
        "name": "Audit trail and logging",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Infrastructure control — requires system logging audit",
    },
    "E005": {
        "domain": "accountability",
        "name": "Change management process",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires SDLC audit",
    },
    "E006": {
        "domain": "accountability",
        "name": "Vendor and third-party due diligence",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Organizational control — requires supply chain review",
    },

    # ── F: Societal Risk ────────────────────────────────────────────────────
    "F001": {
        "domain": "societal_risk",
        "name": "Catastrophic misuse prevention",
        "mandatory": True,
        "coverage": NOT_COVERED,
        "reason": "Not in scope for agent quality evaluation",
    },
    "F002": {
        "domain": "societal_risk",
        "name": "Dual-use risk assessment",
        "mandatory": False,
        "coverage": NOT_COVERED,
        "reason": "Requires organizational risk assessment — not automated testing",
    },
}


# ── Report Generation ───────────────────────────────────────────────────────

def _count_by_coverage(
    controls: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """Count controls by coverage level."""
    counts = {FULLY_COVERED: 0, PARTIALLY_COVERED: 0, NOT_COVERED: 0}
    for ctrl in controls.values():
        counts[ctrl["coverage"]] += 1
    return counts


def _controls_by_domain(
    controls: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Group controls by AIUC-1 domain with per-domain stats."""
    domains: Dict[str, Dict[str, Any]] = {}
    for ctrl_id, ctrl in controls.items():
        d = ctrl["domain"]
        if d not in domains:
            domains[d] = {"controls": {}, "full": 0, "partial": 0, "none": 0}
        domains[d]["controls"][ctrl_id] = ctrl
        if ctrl["coverage"] == FULLY_COVERED:
            domains[d]["full"] += 1
        elif ctrl["coverage"] == PARTIALLY_COVERED:
            domains[d]["partial"] += 1
        else:
            domains[d]["none"] += 1
    return domains


def generate_aiuc1_report(
    evaluation_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate AIUC-1 alignment report.

    Args:
        evaluation_result: Optional evaluation result dict to enrich
            the report with evaluation-specific data (scores, probes run, etc.)

    Returns:
        AIUC-1 alignment report dict suitable for embedding in AQVC attestation
        or returning from API endpoint.
    """
    counts = _count_by_coverage(AIUC1_CONTROLS)
    total = len(AIUC1_CONTROLS)
    coverage_pct = round(
        (counts[FULLY_COVERED] + 0.5 * counts[PARTIALLY_COVERED]) / total * 100, 1
    )

    # Build per-control summary (compact for attestation embedding)
    controls_summary: List[Dict[str, Any]] = []
    for ctrl_id, ctrl in sorted(AIUC1_CONTROLS.items()):
        entry: Dict[str, Any] = {
            "id": ctrl_id,
            "domain": ctrl["domain"],
            "name": ctrl["name"],
            "mandatory": ctrl["mandatory"],
            "coverage": ctrl["coverage"],
        }
        if ctrl["coverage"] != NOT_COVERED:
            entry["covered_by"] = ctrl.get("covered_by", "")
            entry["axis"] = ctrl.get("axis", "")
            if ctrl.get("note"):
                entry["note"] = ctrl["note"]
        else:
            entry["reason"] = ctrl.get("reason", "")
        controls_summary.append(entry)

    # Domain breakdown
    domain_stats = _controls_by_domain(AIUC1_CONTROLS)
    domain_summary = {}
    for domain_name, data in domain_stats.items():
        d_total = data["full"] + data["partial"] + data["none"]
        d_pct = round(
            (data["full"] + 0.5 * data["partial"]) / d_total * 100, 1
        ) if d_total > 0 else 0
        domain_summary[domain_name] = {
            "total_controls": d_total,
            "fully_covered": data["full"],
            "partially_covered": data["partial"],
            "not_covered": data["none"],
            "coverage_percentage": d_pct,
        }

    # Evaluation-specific enrichment
    eval_enrichment = None
    if evaluation_result:
        dimensions = evaluation_result.get("dimensions", {})
        safety_report = evaluation_result.get("safety_report", [])
        eval_enrichment = {
            "evaluation_axes_used": list(dimensions.keys()) if dimensions else [],
            "adversarial_probes_run": len(safety_report),
            "overall_score": evaluation_result.get("overall_score"),
            "safety_score": (dimensions.get("safety", {}) or {}).get("score"),
            "accuracy_score": (dimensions.get("accuracy", {}) or {}).get("score"),
        }

    report = {
        "aiuc1_version": "2026-Q1",
        "standard": "AIUC-1",
        "total_controls": total,
        "controls_fully_covered": counts[FULLY_COVERED],
        "controls_partially_covered": counts[PARTIALLY_COVERED],
        "controls_not_covered": counts[NOT_COVERED],
        "coverage_percentage": coverage_pct,
        "mandatory_coverage": _mandatory_coverage_pct(),
        "domain_summary": domain_summary,
        "controls": controls_summary,
        "note": (
            "AgentTrust covers technical controls via automated evaluation. "
            "Organizational controls (E-series) and societal risk controls "
            "(F-series) require separate organizational audit."
        ),
        "continuous_monitoring_note": (
            "AgentTrust provides continuous evaluation, exceeding AIUC-1's "
            "quarterly testing requirements for controls C010, C011, C012, "
            "D002, and D004."
        ),
    }

    if eval_enrichment:
        report["evaluation_data"] = eval_enrichment

    return report


def _mandatory_coverage_pct() -> float:
    """Calculate coverage percentage for mandatory controls only."""
    mandatory = {k: v for k, v in AIUC1_CONTROLS.items() if v["mandatory"]}
    if not mandatory:
        return 0.0
    counts = _count_by_coverage(mandatory)
    total = len(mandatory)
    return round(
        (counts[FULLY_COVERED] + 0.5 * counts[PARTIALLY_COVERED]) / total * 100, 1
    )


def get_covered_controls() -> List[str]:
    """Return list of control IDs that are fully or partially covered."""
    return [
        ctrl_id for ctrl_id, ctrl in AIUC1_CONTROLS.items()
        if ctrl["coverage"] in (FULLY_COVERED, PARTIALLY_COVERED)
    ]


def get_uncovered_mandatory_controls() -> List[Dict[str, str]]:
    """Return list of mandatory controls that are not covered — for recommendations."""
    return [
        {"id": ctrl_id, "name": ctrl["name"], "domain": ctrl["domain"]}
        for ctrl_id, ctrl in AIUC1_CONTROLS.items()
        if ctrl["mandatory"] and ctrl["coverage"] == NOT_COVERED
    ]

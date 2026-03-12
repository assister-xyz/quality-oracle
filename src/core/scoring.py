"""Score aggregation and tier calculation."""
import re
from typing import Dict, List
from src.core.question_pools import determine_tier


def aggregate_scores(
    tool_scores: Dict[str, dict],
    domain_scores: Dict[str, dict] | None = None,
    manifest_score: int | None = None,
) -> dict:
    """
    Aggregate individual scores into overall quality score.

    Weighting:
    - Manifest (Level 1): 10% of overall (if present)
    - Functional (Level 2): 60% of overall
    - Domain (Level 3): 30% of overall (if present)
    """
    weights = {"manifest": 0.0, "functional": 1.0, "domain": 0.0}

    # Determine if this is L1-only (no functional tests ran)
    l1_only = manifest_score is not None and not tool_scores

    if l1_only:
        # L1 only: manifest is the entire score (capped with confidence penalty)
        weights = {"manifest": 1.0, "functional": 0.0, "domain": 0.0}
    elif manifest_score is not None and domain_scores:
        weights = {"manifest": 0.10, "functional": 0.60, "domain": 0.30}
    elif manifest_score is not None:
        weights = {"manifest": 0.15, "functional": 0.85, "domain": 0.0}
    elif domain_scores:
        weights = {"manifest": 0.0, "functional": 0.65, "domain": 0.35}

    # Functional score from tool scores
    if tool_scores:
        func_scores = [t["score"] for t in tool_scores.values()]
        functional_score = sum(func_scores) / len(func_scores)
    else:
        functional_score = 0

    # Domain score
    if domain_scores:
        dom_scores = [d["score"] for d in domain_scores.values()]
        domain_score = sum(dom_scores) / len(dom_scores)
    else:
        domain_score = 0

    overall = (
        (manifest_score or 0) * weights["manifest"]
        + functional_score * weights["functional"]
        + domain_score * weights["domain"]
    )
    overall = int(round(overall))

    return {
        "overall_score": overall,
        "tier": determine_tier(overall),
        "functional_score": int(functional_score),
        "domain_score": int(domain_score) if domain_scores else None,
        "manifest_score": manifest_score,
        "weights": weights,
    }


def calculate_trend(scores_history: List[int]) -> str:
    """Determine score trend from history."""
    if len(scores_history) < 2:
        return "stable"
    recent = scores_history[-3:]
    if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
        return "improving"
    if all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
        return "declining"
    return "stable"


# ── Style Control (QO-009) ──────────────────────────────────────────────────

# Baseline statistics for response style features.
# Will be refined empirically via autoresearch (QO-013).
STYLE_BASELINE = {
    "mean_length": 800,
    "std_length": 600,
    "mean_markdown_elements": 5,
    "std_markdown_elements": 4,
}

# Penalty coefficient per z-score above threshold (max 5 points per feature)
STYLE_PENALTY_PER_ZSCORE = 2.0
STYLE_PENALTY_THRESHOLD = 1.0  # Only penalize >1 standard deviation
STYLE_PENALTY_MAX = 5.0  # Max penalty per feature


def extract_style_features(response_text: str) -> Dict[str, float]:
    """Extract style covariates from a response for bias control.

    These features capture formatting/verbosity that can inflate LLM judge
    scores without reflecting actual quality.
    """
    if not response_text:
        return {
            "response_length": 0,
            "markdown_headers": 0,
            "markdown_bold": 0,
            "list_items": 0,
            "code_blocks": 0,
            "total_markdown_elements": 0,
        }

    md_headers = response_text.count("#")
    md_bold = response_text.count("**")
    list_items = len(re.findall(r"^[\-\*\d]+\.", response_text, re.MULTILINE))
    code_blocks = response_text.count("```")

    return {
        "response_length": len(response_text),
        "markdown_headers": md_headers,
        "markdown_bold": md_bold,
        "list_items": list_items,
        "code_blocks": code_blocks,
        "total_markdown_elements": md_headers + md_bold + list_items + code_blocks,
    }


def compute_style_penalty(style: Dict[str, float]) -> float:
    """Compute a score penalty for excessive formatting/verbosity.

    Only penalizes responses that are >1 standard deviation above baseline
    on length or markdown density. Returns a positive number to subtract.
    """
    penalty = 0.0

    # Length penalty
    length_z = (
        (style["response_length"] - STYLE_BASELINE["mean_length"])
        / max(1, STYLE_BASELINE["std_length"])
    )
    if length_z > STYLE_PENALTY_THRESHOLD:
        excess = length_z - STYLE_PENALTY_THRESHOLD
        penalty += min(STYLE_PENALTY_MAX, excess * STYLE_PENALTY_PER_ZSCORE)

    # Markdown density penalty
    md_z = (
        (style["total_markdown_elements"] - STYLE_BASELINE["mean_markdown_elements"])
        / max(1, STYLE_BASELINE["std_markdown_elements"])
    )
    if md_z > STYLE_PENALTY_THRESHOLD:
        excess = md_z - STYLE_PENALTY_THRESHOLD
        penalty += min(STYLE_PENALTY_MAX, excess * STYLE_PENALTY_PER_ZSCORE)

    return round(penalty, 2)


def apply_style_adjustment(raw_score: float, response_text: str) -> Dict[str, any]:
    """Apply style control to a raw score.

    Returns dict with adjusted_score, style_features, and penalty applied.
    """
    style = extract_style_features(response_text)
    penalty = compute_style_penalty(style)
    adjusted = max(0, raw_score - penalty)

    return {
        "adjusted_score": int(round(adjusted)),
        "raw_score": int(round(raw_score)),
        "style_penalty": penalty,
        "style_features": style,
        "style_controlled": penalty > 0,
    }

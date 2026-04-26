"""A2A v0.3 Extension for AgentTrust evaluation.

Defines the AgentTrust extension URI and builders for both
provider (AgentTrust's own card) and consumer (evaluated agent's card)
per A2A v0.3 capabilities.extensions[] spec.
"""

EXTENSION_URI = "https://laureum.ai/ext/evaluation/v1"


def build_provider_extension_declaration() -> dict:
    """Extension declaration for AgentTrust's own Agent Card.

    Returns an A2A v0.3 extension object with uri, description, required, params.
    """
    return {
        "uri": EXTENSION_URI,
        "description": "AgentTrust evaluation extension. Provides quality scores, tiers, and AQVC (W3C VC) attestations.",
        "required": False,
        "params": {
            "role": "provider",
            "evaluation_levels": [1, 2, 3],
            "supported_targets": ["mcp_server", "agent", "skill"],
            "attestation_format": "W3C Verifiable Credential (AQVC)",
        },
    }


def build_consumer_extension_declaration(score_data: dict) -> dict:
    """Extension declaration for an evaluated agent's Agent Card.

    Args:
        score_data: Score document from MongoDB with current_score, tier, etc.

    Returns an A2A v0.3 extension object embedding quality results.
    """
    last_evaluated = score_data.get("last_evaluated_at")
    if last_evaluated and hasattr(last_evaluated, "isoformat"):
        last_evaluated = last_evaluated.isoformat()

    return {
        "uri": EXTENSION_URI,
        "description": "Quality evaluation by AgentTrust",
        "required": False,
        "params": {
            "role": "verified_subject",
            "score": score_data.get("current_score", 0),
            "tier": score_data.get("tier", "unknown"),
            "confidence": score_data.get("confidence", 0),
            "last_evaluated": last_evaluated,
            "attestation_url": score_data.get("attestation_url"),
            "badge_url": score_data.get("badge_url"),
            "verify_url": f"/v1/score/{score_data.get('target_id', '')}",
        },
    }

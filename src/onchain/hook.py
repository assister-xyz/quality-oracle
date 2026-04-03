"""Post-evaluation on-chain hook.

Orchestrates ERC-8004 feedback + EAS attestation creation after an
evaluation completes. Runs async and best-effort — on-chain failures
never block the evaluation flow.
"""
import logging

from src.config import settings

logger = logging.getLogger(__name__)


async def post_evaluation_onchain(
    evaluation_id: str,
    target_url: str,
    score: int,
    tier: str,
    dimensions: dict,
    attestation_jwt: str | None = None,
    erc8004_agent_id: int | None = None,
):
    """Fire-and-forget on-chain posting after evaluation completes.

    Called from the evaluation background task. Errors are logged but
    never raised to the caller.

    Args:
        evaluation_id: The evaluation ID.
        target_url: The MCP server / agent URL that was evaluated.
        score: Overall AQVC score (0-100).
        tier: Quality tier (expert/proficient/basic/failed).
        dimensions: 6-axis dimension scores dict.
        attestation_jwt: Optional AQVC JWT string.
        erc8004_agent_id: Optional ERC-8004 agent token ID for reputation posting.
    """
    # ── ERC-8004 Reputation Feedback ────────────────────────────────────
    if settings.erc8004_enabled and erc8004_agent_id is not None:
        try:
            from src.onchain.erc8004 import post_feedback
            result = await post_feedback(
                agent_id=erc8004_agent_id,
                score=score,
                tier=tier,
                dimensions=dimensions,
                evaluation_id=evaluation_id,
                attestation_jwt=attestation_jwt,
            )
            if result:
                logger.info(
                    f"[onchain] ERC-8004 feedback posted for eval {evaluation_id[:8]}: "
                    f"tx={result['tx_hash']}"
                )
            else:
                logger.warning(f"[onchain] ERC-8004 feedback failed for eval {evaluation_id[:8]}")
        except Exception as e:
            logger.error(f"[onchain] ERC-8004 feedback error for eval {evaluation_id[:8]}: {e}")

    # ── EAS Attestation ─────────────────────────────────────────────────
    if settings.eas_enabled:
        try:
            from src.onchain.eas import create_attestation
            result = await create_attestation(
                agent_url=target_url,
                score=score,
                tier=tier,
                dimensions=dimensions,
                evaluation_id=evaluation_id,
            )
            if result:
                logger.info(
                    f"[onchain] EAS attestation created for eval {evaluation_id[:8]}: "
                    f"type={result['type']}"
                )
            else:
                logger.warning(f"[onchain] EAS attestation failed for eval {evaluation_id[:8]}")
        except Exception as e:
            logger.error(f"[onchain] EAS attestation error for eval {evaluation_id[:8]}: {e}")

"""ERC-8004 Trustless Agents integration.

Posts AQVC evaluation scores as reputation feedback to the ERC-8004
ReputationRegistry on Base L2. Each evaluation becomes a permanent,
composable on-chain reputation signal.

Contract addresses are deterministic (CREATE2) and identical on 23+ chains.
See https://eips.ethereum.org/EIPS/eip-8004
"""
import json
import logging
from datetime import datetime

from web3 import Web3

from src.config import settings
from src.onchain.wallet import get_web3, get_evaluator_account, send_transaction

logger = logging.getLogger(__name__)

# ── Minimal ABIs (only the functions we call) ───────────────────────────────

REPUTATION_REGISTRY_ABI = json.loads("""[
    {
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "feedbackType", "type": "uint8"},
            {"name": "data", "type": "bytes"}
        ],
        "name": "giveFeedback",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "name": "getReputation",
        "outputs": [
            {"name": "positiveCount", "type": "uint256"},
            {"name": "negativeCount", "type": "uint256"},
            {"name": "feedbackCount", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "name": "agentId", "type": "uint256"},
            {"indexed": true, "name": "from", "type": "address"},
            {"indexed": false, "name": "feedbackType", "type": "uint8"},
            {"indexed": false, "name": "data", "type": "bytes"}
        ],
        "name": "FeedbackGiven",
        "type": "event"
    }
]""")

IDENTITY_REGISTRY_ABI = json.loads("""[
    {
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "name": "ownerOf",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "name": "getAgentInfo",
        "outputs": [
            {"name": "metadataURI", "type": "string"},
            {"name": "owner", "type": "address"},
            {"name": "registeredAt", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]""")

# Feedback types per ERC-8004 spec
FEEDBACK_POSITIVE = 0  # score >= 70
FEEDBACK_NEGATIVE = 1  # score < 50
FEEDBACK_NEUTRAL = 2   # 50 <= score < 70


def _get_reputation_contract():
    """Get ReputationRegistry contract instance."""
    w3 = get_web3()
    return w3.eth.contract(
        address=Web3.to_checksum_address(settings.erc8004_reputation_registry),
        abi=REPUTATION_REGISTRY_ABI,
    )


def _get_identity_contract():
    """Get IdentityRegistry contract instance."""
    w3 = get_web3()
    return w3.eth.contract(
        address=Web3.to_checksum_address(settings.erc8004_identity_registry),
        abi=IDENTITY_REGISTRY_ABI,
    )


def _score_to_feedback_type(score: int) -> int:
    """Map AQVC overall score to ERC-8004 feedback type."""
    if score >= 70:
        return FEEDBACK_POSITIVE
    elif score < 50:
        return FEEDBACK_NEGATIVE
    return FEEDBACK_NEUTRAL


def _encode_feedback_data(
    score: int,
    tier: str,
    dimensions: dict,
    evaluation_id: str,
    attestation_jwt: str | None = None,
    ipfs_hash: str | None = None,
) -> bytes:
    """Encode AQVC evaluation data into bytes for on-chain feedback.

    Format: ABI-encoded tuple of (score, tier, dimensions_json, eval_id, attestation_ref)
    """
    w3 = get_web3()
    # Pack evaluation data as JSON bytes for maximum flexibility
    payload = {
        "protocol": "laureum-aqvc-v1",
        "score": score,
        "tier": tier,
        "dimensions": dimensions,
        "evaluation_id": evaluation_id,
    }
    if ipfs_hash:
        payload["ipfs"] = ipfs_hash
    if attestation_jwt:
        # Store first 64 chars as reference (full JWT too large for on-chain)
        payload["attestation_ref"] = attestation_jwt[:64]

    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def post_feedback(
    agent_id: int,
    score: int,
    tier: str,
    dimensions: dict,
    evaluation_id: str,
    attestation_jwt: str | None = None,
    ipfs_hash: str | None = None,
) -> dict | None:
    """Post AQVC evaluation result as ERC-8004 reputation feedback.

    Args:
        agent_id: ERC-8004 agent token ID (uint256).
        score: Overall AQVC score (0-100).
        tier: Quality tier (expert/proficient/basic/failed).
        dimensions: 6-axis dimension scores.
        evaluation_id: Laureum evaluation ID.
        attestation_jwt: Optional AQVC JWT for reference.
        ipfs_hash: Optional IPFS hash of full credential.

    Returns:
        Transaction result dict or None on failure.
    """
    if not settings.erc8004_enabled:
        logger.debug("ERC-8004 disabled, skipping feedback post")
        return None

    contract = _get_reputation_contract()
    feedback_type = _score_to_feedback_type(score)
    feedback_data = _encode_feedback_data(
        score=score,
        tier=tier,
        dimensions=dimensions,
        evaluation_id=evaluation_id,
        attestation_jwt=attestation_jwt,
        ipfs_hash=ipfs_hash,
    )

    feedback_type_name = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}[feedback_type]
    logger.info(
        f"[ERC-8004] Posting {feedback_type_name} feedback for agent #{agent_id}: "
        f"score={score} tier={tier}"
    )

    tx = contract.functions.giveFeedback(
        agent_id,
        feedback_type,
        feedback_data,
    ).build_transaction({
        "value": 0,
    })

    return await send_transaction(
        tx=tx,
        protocol="erc8004",
        evaluation_id=evaluation_id,
        description=f"giveFeedback(agent={agent_id}, type={feedback_type_name}, score={score})",
    )


async def get_agent_reputation(agent_id: int) -> dict | None:
    """Query on-chain reputation for an ERC-8004 agent."""
    if not settings.erc8004_enabled:
        return None

    try:
        contract = _get_reputation_contract()
        result = contract.functions.getReputation(agent_id).call()
        return {
            "agent_id": agent_id,
            "positive_count": result[0],
            "negative_count": result[1],
            "feedback_count": result[2],
        }
    except Exception as e:
        logger.error(f"[ERC-8004] Failed to query reputation for agent #{agent_id}: {e}")
        return None


async def get_agent_info(agent_id: int) -> dict | None:
    """Query on-chain identity info for an ERC-8004 agent."""
    if not settings.erc8004_enabled:
        return None

    try:
        contract = _get_identity_contract()
        result = contract.functions.getAgentInfo(agent_id).call()
        return {
            "agent_id": agent_id,
            "metadata_uri": result[0],
            "owner": result[1],
            "registered_at": result[2],
        }
    except Exception as e:
        logger.error(f"[ERC-8004] Failed to query agent info #{agent_id}: {e}")
        return None


async def get_registry_stats() -> dict:
    """Get ERC-8004 registry statistics."""
    if not settings.erc8004_enabled:
        return {"enabled": False}

    try:
        contract = _get_identity_contract()
        total_supply = contract.functions.totalSupply().call()
        account = get_evaluator_account()
        return {
            "enabled": True,
            "total_agents": total_supply,
            "evaluator_address": account.address if account else None,
            "identity_registry": settings.erc8004_identity_registry,
            "reputation_registry": settings.erc8004_reputation_registry,
            "chain_id": settings.base_chain_id,
        }
    except Exception as e:
        logger.error(f"[ERC-8004] Failed to get registry stats: {e}")
        return {"enabled": True, "error": str(e)}

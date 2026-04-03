"""Ethereum Attestation Service (EAS) integration.

Creates AQVC quality attestations on EAS deployed on Base L2.
- Off-chain signed attestations (free) for all evaluations
- On-chain attestations (gas required) for Audited tier (score >= 90)

EAS is deployed on 15+ chains. We use Base for low gas costs.
See https://docs.attest.org/
"""
import json
import logging
import time
from datetime import datetime

from web3 import Web3
from eth_account.messages import encode_typed_data

from src.config import settings
from src.onchain.wallet import get_web3, get_evaluator_account, send_transaction
from src.storage.mongodb import onchain_txs_col

logger = logging.getLogger(__name__)

# ── AQVC Schema Definition ─────────────────────────────────────────────────
# This schema is registered once on EAS SchemaRegistry, then referenced by UID.
AQVC_SCHEMA = (
    "uint8 overallScore,"
    "uint8 accuracy,"
    "uint8 safety,"
    "uint8 reliability,"
    "uint8 processQuality,"
    "uint8 latency,"
    "uint8 schemaQuality,"
    "string agentUrl,"
    "string tier,"
    "uint64 evaluatedAt,"
    "bytes32 aqvcHash"
)

# ── Minimal ABIs ────────────────────────────────────────────────────────────

EAS_ABI = json.loads("""[
    {
        "inputs": [
            {
                "components": [
                    {"name": "schema", "type": "bytes32"},
                    {
                        "components": [
                            {"name": "recipient", "type": "address"},
                            {"name": "expirationTime", "type": "uint64"},
                            {"name": "revocable", "type": "bool"},
                            {"name": "refUID", "type": "bytes32"},
                            {"name": "data", "type": "bytes"},
                            {"name": "value", "type": "uint256"}
                        ],
                        "name": "data",
                        "type": "tuple"
                    }
                ],
                "name": "request",
                "type": "tuple"
            }
        ],
        "name": "attest",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [{"name": "uid", "type": "bytes32"}],
        "name": "getAttestation",
        "outputs": [
            {
                "components": [
                    {"name": "uid", "type": "bytes32"},
                    {"name": "schema", "type": "bytes32"},
                    {"name": "time", "type": "uint64"},
                    {"name": "expirationTime", "type": "uint64"},
                    {"name": "revocationTime", "type": "uint64"},
                    {"name": "refUID", "type": "bytes32"},
                    {"name": "recipient", "type": "address"},
                    {"name": "attester", "type": "address"},
                    {"name": "revocable", "type": "bool"},
                    {"name": "data", "type": "bytes"}
                ],
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]""")

SCHEMA_REGISTRY_ABI = json.loads("""[
    {
        "inputs": [
            {"name": "schema", "type": "string"},
            {"name": "resolver", "type": "address"},
            {"name": "revocable", "type": "bool"}
        ],
        "name": "register",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]""")

# EIP-712 domain for off-chain EAS attestations on Base
EAS_DOMAIN = {
    "name": "EAS Attestation",
    "version": "1.3.0",
    "chainId": settings.base_chain_id,
    "verifyingContract": settings.eas_contract_address,
}

# EIP-712 types for off-chain attestation
EAS_ATTEST_TYPES = {
    "Attest": [
        {"name": "version", "type": "uint16"},
        {"name": "schema", "type": "bytes32"},
        {"name": "recipient", "type": "address"},
        {"name": "time", "type": "uint64"},
        {"name": "expirationTime", "type": "uint64"},
        {"name": "revocable", "type": "bool"},
        {"name": "refUID", "type": "bytes32"},
        {"name": "data", "type": "bytes"},
    ],
}


def _encode_aqvc_data(
    score: int,
    dimensions: dict,
    agent_url: str,
    tier: str,
    evaluated_at: int,
    aqvc_hash: bytes,
) -> bytes:
    """ABI-encode AQVC data matching the registered schema."""
    w3 = get_web3()
    return w3.codec.encode(
        [
            "uint8", "uint8", "uint8", "uint8", "uint8", "uint8", "uint8",
            "string", "string", "uint64", "bytes32",
        ],
        [
            min(score, 255),
            min(dimensions.get("accuracy", 0), 255),
            min(dimensions.get("safety", 0), 255),
            min(dimensions.get("reliability", 0), 255),
            min(dimensions.get("process_quality", 0), 255),
            min(dimensions.get("latency", 0), 255),
            min(dimensions.get("schema_quality", 0), 255),
            agent_url,
            tier,
            evaluated_at,
            aqvc_hash,
        ],
    )


def _compute_aqvc_hash(evaluation_id: str, score: int, agent_url: str) -> bytes:
    """Compute a deterministic hash for the AQVC credential."""
    import hashlib
    data = f"{evaluation_id}:{score}:{agent_url}".encode()
    return hashlib.sha256(data).digest()


async def register_schema() -> str | None:
    """Register the AQVC schema on EAS SchemaRegistry.

    This only needs to be called once. The returned schema UID should be
    saved to EAS_SCHEMA_UID in config.

    Returns:
        Schema UID hex string, or None on failure.
    """
    if not settings.eas_enabled:
        logger.warning("EAS disabled, cannot register schema")
        return None

    w3 = get_web3()
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(settings.eas_schema_registry),
        abi=SCHEMA_REGISTRY_ABI,
    )

    tx = contract.functions.register(
        AQVC_SCHEMA,
        "0x0000000000000000000000000000000000000000",  # No resolver
        True,  # Revocable
    ).build_transaction({"value": 0})

    result = await send_transaction(
        tx=tx,
        protocol="eas",
        description="Register AQVC schema",
    )

    if result and result["status"] == "success":
        # Parse schema UID from transaction logs
        receipt = w3.eth.get_transaction_receipt(result["tx_hash"])
        # SchemaRegistry emits Registered(bytes32 uid, address registerer)
        if receipt["logs"]:
            schema_uid = receipt["logs"][0]["topics"][1].hex()
            logger.info(f"[EAS] Schema registered: {schema_uid}")
            return schema_uid

    return None


async def create_onchain_attestation(
    agent_url: str,
    score: int,
    tier: str,
    dimensions: dict,
    evaluation_id: str,
    recipient: str = "0x0000000000000000000000000000000000000000",
) -> dict | None:
    """Create an on-chain EAS attestation for high-score evaluations.

    Only called for Audited tier (score >= eas_onchain_min_score).

    Args:
        agent_url: The evaluated agent/MCP server URL.
        score: Overall AQVC score (0-100).
        tier: Quality tier string.
        dimensions: 6-axis dimension scores.
        evaluation_id: Laureum evaluation ID.
        recipient: Optional recipient address (agent's ERC-8004 owner).

    Returns:
        Dict with attestation_uid and tx details, or None on failure.
    """
    if not settings.eas_enabled:
        return None

    if not settings.eas_schema_uid:
        logger.error("[EAS] No schema UID configured. Run register_schema() first.")
        return None

    evaluated_at = int(time.time())
    aqvc_hash = _compute_aqvc_hash(evaluation_id, score, agent_url)

    encoded_data = _encode_aqvc_data(
        score=score,
        dimensions=dimensions,
        agent_url=agent_url,
        tier=tier,
        evaluated_at=evaluated_at,
        aqvc_hash=aqvc_hash,
    )

    w3 = get_web3()
    eas_contract = w3.eth.contract(
        address=Web3.to_checksum_address(settings.eas_contract_address),
        abi=EAS_ABI,
    )

    schema_uid = bytes.fromhex(settings.eas_schema_uid.removeprefix("0x"))

    # 30-day expiration
    expiration = evaluated_at + (30 * 24 * 60 * 60)

    tx = eas_contract.functions.attest((
        schema_uid,  # schema
        (
            Web3.to_checksum_address(recipient),  # recipient
            expiration,  # expirationTime
            True,  # revocable
            b"\x00" * 32,  # refUID (none)
            encoded_data,  # data
            0,  # value
        ),
    )).build_transaction({"value": 0})

    result = await send_transaction(
        tx=tx,
        protocol="eas",
        evaluation_id=evaluation_id,
        description=f"AQVC attestation: score={score} tier={tier} agent={agent_url[:50]}",
    )

    if result and result["status"] == "success":
        # Parse attestation UID from logs
        receipt = w3.eth.get_transaction_receipt(result["tx_hash"])
        attestation_uid = None
        if receipt["logs"]:
            attestation_uid = receipt["logs"][0]["data"][:66]

        return {
            "attestation_uid": attestation_uid,
            "tx_hash": result["tx_hash"],
            "chain_id": settings.base_chain_id,
            "schema_uid": settings.eas_schema_uid,
            "gas_cost_eth": result["gas_cost_eth"],
            "type": "onchain",
        }

    return None


async def create_offchain_attestation(
    agent_url: str,
    score: int,
    tier: str,
    dimensions: dict,
    evaluation_id: str,
) -> dict | None:
    """Create a free off-chain EAS attestation (EIP-712 signed).

    Off-chain attestations are signed by the evaluator but not submitted
    to the blockchain. They can be verified by anyone with the attester's
    public key and are stored in MongoDB.

    Args:
        agent_url: The evaluated agent/MCP server URL.
        score: Overall AQVC score (0-100).
        tier: Quality tier string.
        dimensions: 6-axis dimension scores.
        evaluation_id: Laureum evaluation ID.

    Returns:
        Dict with signed attestation data, or None on failure.
    """
    if not settings.eas_enabled:
        return None

    account = get_evaluator_account()
    if not account:
        logger.error("[EAS] No evaluator wallet for off-chain signing")
        return None

    if not settings.eas_schema_uid:
        logger.error("[EAS] No schema UID configured")
        return None

    evaluated_at = int(time.time())
    aqvc_hash = _compute_aqvc_hash(evaluation_id, score, agent_url)

    encoded_data = _encode_aqvc_data(
        score=score,
        dimensions=dimensions,
        agent_url=agent_url,
        tier=tier,
        evaluated_at=evaluated_at,
        aqvc_hash=aqvc_hash,
    )

    schema_uid = bytes.fromhex(settings.eas_schema_uid.removeprefix("0x"))
    expiration = evaluated_at + (30 * 24 * 60 * 60)

    # EIP-712 typed data for off-chain attestation
    message = {
        "version": 1,
        "schema": schema_uid,
        "recipient": "0x0000000000000000000000000000000000000000",
        "time": evaluated_at,
        "expirationTime": expiration,
        "revocable": True,
        "refUID": b"\x00" * 32,
        "data": encoded_data,
    }

    signable = encode_typed_data(
        domain_data=EAS_DOMAIN,
        message_types=EAS_ATTEST_TYPES,
        message_data=message,
    )
    signed = account.sign_message(signable)

    logger.info(
        f"[EAS] Off-chain attestation signed for {agent_url[:50]}: "
        f"score={score} tier={tier}"
    )

    # Store in MongoDB for retrieval
    offchain_doc = {
        "evaluation_id": evaluation_id,
        "protocol": "eas_offchain",
        "agent_url": agent_url,
        "score": score,
        "tier": tier,
        "dimensions": dimensions,
        "schema_uid": settings.eas_schema_uid,
        "chain_id": settings.base_chain_id,
        "attester": account.address,
        "signature": signed.signature.hex(),
        "evaluated_at": evaluated_at,
        "expiration": expiration,
        "type": "offchain",
        "created_at": datetime.utcnow(),
    }
    await onchain_txs_col().insert_one(offchain_doc)

    return {
        "attester": account.address,
        "signature": signed.signature.hex(),
        "schema_uid": settings.eas_schema_uid,
        "chain_id": settings.base_chain_id,
        "evaluated_at": evaluated_at,
        "type": "offchain",
    }


async def create_attestation(
    agent_url: str,
    score: int,
    tier: str,
    dimensions: dict,
    evaluation_id: str,
    recipient: str = "0x0000000000000000000000000000000000000000",
) -> dict | None:
    """Create an EAS attestation — on-chain for high scores, off-chain otherwise.

    Decision logic:
    - score >= eas_onchain_min_score (90): on-chain attestation (gas required)
    - score < 90: off-chain signed attestation (free)
    """
    if not settings.eas_enabled:
        return None

    if score >= settings.eas_onchain_min_score:
        logger.info(f"[EAS] Score {score} >= {settings.eas_onchain_min_score}: creating ON-CHAIN attestation")
        return await create_onchain_attestation(
            agent_url=agent_url,
            score=score,
            tier=tier,
            dimensions=dimensions,
            evaluation_id=evaluation_id,
            recipient=recipient,
        )
    else:
        logger.info(f"[EAS] Score {score} < {settings.eas_onchain_min_score}: creating off-chain attestation")
        return await create_offchain_attestation(
            agent_url=agent_url,
            score=score,
            tier=tier,
            dimensions=dimensions,
            evaluation_id=evaluation_id,
        )


async def get_eas_status() -> dict:
    """Get EAS integration status."""
    if not settings.eas_enabled:
        return {"enabled": False}

    account = get_evaluator_account()
    return {
        "enabled": True,
        "schema_uid": settings.eas_schema_uid or None,
        "schema_registered": bool(settings.eas_schema_uid),
        "eas_contract": settings.eas_contract_address,
        "chain_id": settings.base_chain_id,
        "onchain_min_score": settings.eas_onchain_min_score,
        "attester": account.address if account else None,
    }

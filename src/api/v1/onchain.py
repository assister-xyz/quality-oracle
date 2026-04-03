"""On-chain integration endpoints — ERC-8004, EAS status and admin operations."""
import logging
from fastapi import APIRouter, Depends, HTTPException

from src.auth.dependencies import get_api_key
from src.storage.mongodb import onchain_txs_col

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/onchain/status")
async def get_onchain_status(api_key_doc: dict = Depends(get_api_key)):
    """Get status of all on-chain integrations."""
    from src.onchain.wallet import get_wallet_status
    from src.onchain.erc8004 import get_registry_stats
    from src.onchain.eas import get_eas_status

    wallet = await get_wallet_status()
    erc8004 = await get_registry_stats()
    eas = await get_eas_status()

    # Recent tx stats
    tx_count = await onchain_txs_col().count_documents({})
    success_count = await onchain_txs_col().count_documents({"status": "success"})
    error_count = await onchain_txs_col().count_documents({"status": "error"})

    # Total gas spent
    pipeline = [
        {"$match": {"gas_cost_eth": {"$exists": True}}},
        {"$group": {"_id": None, "total_gas": {"$sum": "$gas_cost_eth"}}},
    ]
    gas_result = await onchain_txs_col().aggregate(pipeline).to_list(1)
    total_gas = gas_result[0]["total_gas"] if gas_result else 0.0

    return {
        "wallet": wallet,
        "erc8004": erc8004,
        "eas": eas,
        "transactions": {
            "total": tx_count,
            "success": success_count,
            "errors": error_count,
            "total_gas_eth": total_gas,
        },
    }


@router.get("/onchain/transactions")
async def list_transactions(
    protocol: str | None = None,
    evaluation_id: str | None = None,
    limit: int = 20,
    api_key_doc: dict = Depends(get_api_key),
):
    """List on-chain transactions with optional filters."""
    query = {}
    if protocol:
        query["protocol"] = protocol
    if evaluation_id:
        query["evaluation_id"] = evaluation_id

    cursor = onchain_txs_col().find(query).sort("created_at", -1).limit(limit)
    txs = await cursor.to_list(limit)

    for tx in txs:
        tx["_id"] = str(tx["_id"])

    return {"transactions": txs, "count": len(txs)}


@router.get("/onchain/agent/{agent_id}")
async def get_agent_onchain(
    agent_id: int,
    api_key_doc: dict = Depends(get_api_key),
):
    """Query ERC-8004 on-chain reputation and identity for an agent."""
    from src.onchain.erc8004 import get_agent_reputation, get_agent_info

    reputation = await get_agent_reputation(agent_id)
    info = await get_agent_info(agent_id)

    if not reputation and not info:
        raise HTTPException(status_code=404, detail="Agent not found or ERC-8004 disabled")

    return {"reputation": reputation, "identity": info}


@router.post("/onchain/eas/register-schema")
async def register_eas_schema(api_key_doc: dict = Depends(get_api_key)):
    """Register the AQVC schema on EAS SchemaRegistry.

    Admin-only. Only needs to be called once per chain.
    Returns the schema UID to configure in EAS_SCHEMA_UID.
    """
    tier = api_key_doc.get("tier", "free")
    if tier not in ("marketplace", "team"):
        raise HTTPException(status_code=403, detail="Admin access required")

    from src.onchain.eas import register_schema
    schema_uid = await register_schema()

    if not schema_uid:
        raise HTTPException(status_code=500, detail="Schema registration failed")

    return {
        "schema_uid": schema_uid,
        "message": "Set EAS_SCHEMA_UID in .env to this value",
    }

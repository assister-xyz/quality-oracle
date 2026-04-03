"""EVM wallet management for on-chain transactions.

Handles key loading, transaction signing, nonce management, and gas estimation
for ERC-8004 feedback posting and EAS attestation creation on Base L2.
"""
import logging
from datetime import datetime

from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount

from src.config import settings
from src.storage.mongodb import onchain_txs_col

logger = logging.getLogger(__name__)

_w3: Web3 | None = None
_evaluator_account: LocalAccount | None = None


def get_web3() -> Web3:
    """Get or create Web3 instance connected to Base L2."""
    global _w3
    if _w3 is None or not _w3.is_connected():
        _w3 = Web3(Web3.HTTPProvider(settings.base_rpc_url))
        if _w3.is_connected():
            chain_id = _w3.eth.chain_id
            logger.info(f"Connected to Base L2 (chain_id={chain_id})")
        else:
            logger.warning(f"Failed to connect to {settings.base_rpc_url}")
    return _w3


def get_evaluator_account() -> LocalAccount | None:
    """Load the evaluator wallet from private key config."""
    global _evaluator_account
    if _evaluator_account is not None:
        return _evaluator_account

    key = settings.erc8004_evaluator_private_key
    if not key:
        logger.warning("No evaluator private key configured (ERC8004_EVALUATOR_PRIVATE_KEY)")
        return None

    # Support both 0x-prefixed and raw hex
    if not key.startswith("0x"):
        key = f"0x{key}"

    _evaluator_account = Account.from_key(key)
    logger.info(f"Loaded evaluator wallet: {_evaluator_account.address}")
    return _evaluator_account


async def send_transaction(
    tx: dict,
    protocol: str,
    evaluation_id: str | None = None,
    description: str = "",
) -> dict | None:
    """Sign and send a transaction, tracking gas costs in MongoDB.

    Args:
        tx: Transaction dict (to, data, value, etc.). Gas and nonce auto-filled.
        protocol: Protocol name for tracking ("erc8004" or "eas").
        evaluation_id: Optional evaluation ID to link the tx.
        description: Human-readable description.

    Returns:
        Receipt dict on success, None on failure.
    """
    account = get_evaluator_account()
    if not account:
        logger.error("Cannot send tx: no evaluator wallet configured")
        return None

    w3 = get_web3()
    if not w3.is_connected():
        logger.error("Cannot send tx: not connected to Base RPC")
        return None

    try:
        # Fill nonce and gas
        tx["from"] = account.address
        tx["chainId"] = settings.base_chain_id
        tx["nonce"] = w3.eth.get_transaction_count(account.address)

        # Estimate gas with 20% buffer
        gas_estimate = w3.eth.estimate_gas(tx)
        tx["gas"] = int(gas_estimate * 1.2)

        # Use EIP-1559 gas pricing on Base
        latest = w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas", 0)
        tx["maxFeePerGas"] = base_fee * 2 + w3.to_wei(0.001, "gwei")
        tx["maxPriorityFeePerGas"] = w3.to_wei(0.001, "gwei")

        # Sign and send
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        tx_hash_hex = tx_hash.hex()
        logger.info(f"[{protocol}] Tx sent: {tx_hash_hex}")

        # Wait for receipt (with timeout)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

        gas_used = receipt["gasUsed"]
        effective_gas_price = receipt.get("effectiveGasPrice", 0)
        gas_cost_wei = gas_used * effective_gas_price
        gas_cost_eth = float(w3.from_wei(gas_cost_wei, "ether"))

        success = receipt["status"] == 1

        # Track in MongoDB
        if settings.gas_tracking_enabled:
            await onchain_txs_col().insert_one({
                "tx_hash": tx_hash_hex,
                "protocol": protocol,
                "evaluation_id": evaluation_id,
                "description": description,
                "from_address": account.address,
                "to_address": tx.get("to", ""),
                "chain_id": settings.base_chain_id,
                "gas_used": gas_used,
                "gas_price_gwei": float(w3.from_wei(effective_gas_price, "gwei")),
                "gas_cost_eth": gas_cost_eth,
                "status": "success" if success else "reverted",
                "block_number": receipt["blockNumber"],
                "created_at": datetime.utcnow(),
            })

        if success:
            logger.info(
                f"[{protocol}] Tx confirmed: {tx_hash_hex} "
                f"gas={gas_used} cost={gas_cost_eth:.8f} ETH"
            )
        else:
            logger.error(f"[{protocol}] Tx reverted: {tx_hash_hex}")

        return {
            "tx_hash": tx_hash_hex,
            "status": "success" if success else "reverted",
            "gas_used": gas_used,
            "gas_cost_eth": gas_cost_eth,
            "block_number": receipt["blockNumber"],
        }

    except Exception as e:
        logger.error(f"[{protocol}] Transaction failed: {e}")

        # Track failed tx attempt
        if settings.gas_tracking_enabled:
            await onchain_txs_col().insert_one({
                "tx_hash": None,
                "protocol": protocol,
                "evaluation_id": evaluation_id,
                "description": description,
                "from_address": account.address,
                "chain_id": settings.base_chain_id,
                "status": "error",
                "error": str(e),
                "created_at": datetime.utcnow(),
            })

        return None


async def get_wallet_status() -> dict:
    """Get evaluator wallet status: address, balance, tx count."""
    account = get_evaluator_account()
    if not account:
        return {"configured": False}

    w3 = get_web3()
    connected = w3.is_connected()

    result = {
        "configured": True,
        "address": account.address,
        "connected": connected,
    }

    if connected:
        balance_wei = w3.eth.get_balance(account.address)
        result["balance_eth"] = float(w3.from_wei(balance_wei, "ether"))
        result["nonce"] = w3.eth.get_transaction_count(account.address)
        result["chain_id"] = w3.eth.chain_id

    return result

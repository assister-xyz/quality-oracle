"""MongoDB connection and collection accessors for AgentTrust."""
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.config import settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_db():
    global _client, _db
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_database]
    # Create indexes
    await _db.quality__evaluations.create_index("target_id")
    await _db.quality__evaluations.create_index("status")
    await _db.quality__scores.create_index("target_id", unique=True)
    await _db.quality__score_history.create_index("target_id")
    await _db.quality__score_history.create_index("recorded_at")
    await _db.quality__attestations.create_index("evaluation_id")
    await _db.quality__question_banks.create_index("domain")
    await _db.quality__production_feedback.create_index("target_id")
    await _db.quality__production_feedback.create_index("created_at")
    await _db.quality__payment_receipts.create_index("tx_signature", unique=True)
    await _db.quality__payment_receipts.create_index("created_at")
    await _db.quality__response_fingerprints.create_index([("target_id", 1), ("question_hash", 1)])
    await _db.quality__response_fingerprints.create_index("evaluation_id")
    await _db.quality__paraphrase_log.create_index("evaluation_id")
    await _db.quality__paraphrase_log.create_index("target_id")
    # Battle arena collections
    await _db.quality__battles.create_index("status")
    await _db.quality__battles.create_index("created_at")
    await _db.quality__battles.create_index([("agent_a.target_id", 1)])
    await _db.quality__battles.create_index([("agent_b.target_id", 1)])
    await _db.quality__battles.create_index("match_type")
    await _db.quality__ladder.create_index([("domain", 1), ("position", 1)], unique=True)
    await _db.quality__ladder.create_index("target_id")
    await _db.quality__question_stats.create_index("question_id", unique=True)
    # Rankings collection
    await _db.quality__rankings.create_index([("domain", 1), ("position", 1)], unique=True)
    await _db.quality__rankings.create_index("target_id")
    # IRT item parameters
    await _db.quality__item_params.create_index("question_id", unique=True)
    await _db.quality__item_params.create_index("status")
    await _db.quality__item_params.create_index("domain")
    # On-chain transaction tracking (ERC-8004, EAS)
    await _db.quality__onchain_txs.create_index("tx_hash", unique=True, sparse=True)
    await _db.quality__onchain_txs.create_index("evaluation_id")
    await _db.quality__onchain_txs.create_index("protocol")
    await _db.quality__onchain_txs.create_index("created_at")
    # Score anomaly detection (QO-043)
    await _db.quality__score_anomalies.create_index("target_id")
    await _db.quality__score_anomalies.create_index("anomaly_type")
    await _db.quality__score_anomalies.create_index("detected_at")
    # Sybil defense (QO-044)
    await _db.quality__operators.create_index("operator_id", unique=True)
    await _db.quality__operators.create_index("email", unique=True, sparse=True)
    await _db.quality__operators.create_index("agent_target_ids")
    await _db.quality__clone_suspects.create_index([("agent_a_id", 1), ("agent_b_id", 1)], unique=True)
    await _db.quality__clone_suspects.create_index("status")
    await _db.quality__clone_suspects.create_index("detected_at")
    logger.info(f"Connected to MongoDB: {settings.mongodb_database}")


async def close_db():
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialized. Call connect_db() first.")
    return _db


# Collection accessors
def evaluations_col():
    return get_db().quality__evaluations


def scores_col():
    return get_db().quality__scores


def score_history_col():
    return get_db().quality__score_history


def attestations_col():
    return get_db().quality__attestations


def question_banks_col():
    return get_db().quality__question_banks


def api_keys_col():
    return get_db().quality__api_keys


def feedback_col():
    return get_db().quality__production_feedback


def payment_receipts_col():
    return get_db().quality__payment_receipts


def response_fingerprints_col():
    return get_db().quality__response_fingerprints


def paraphrase_log_col():
    return get_db().quality__paraphrase_log


def battles_col():
    return get_db().quality__battles


def ladder_col():
    return get_db().quality__ladder


def question_stats_col():
    return get_db().quality__question_stats


def rankings_col():
    return get_db().quality__rankings


def item_params_col():
    return get_db().quality__item_params


def onchain_txs_col():
    return get_db().quality__onchain_txs


def score_anomalies_col():
    return get_db().quality__score_anomalies


def operators_col():
    return get_db().quality__operators


def clone_suspects_col():
    return get_db().quality__clone_suspects

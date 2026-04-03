"""Comprehensive E2E tests for on-chain integration (ERC-8004 + EAS).

Tests cover:
- Wallet management (key loading, tx signing, gas tracking)
- ERC-8004 feedback posting and reputation queries
- EAS attestation creation (on-chain and off-chain)
- Post-evaluation hook orchestration
- API endpoints (/v1/onchain/*)
- Integration with evaluation flow
- Error handling and graceful degradation
"""
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from eth_account import Account


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_web3():
    """Mock Web3 instance with typical Base L2 responses."""
    w3 = MagicMock()
    w3.is_connected.return_value = True
    w3.eth.chain_id = 8453
    w3.eth.get_balance.return_value = 50000000000000000  # 0.05 ETH
    w3.eth.get_transaction_count.return_value = 42
    w3.eth.estimate_gas.return_value = 80000
    w3.eth.get_block.return_value = {"baseFeePerGas": 1000000}
    w3.to_wei.return_value = 1000
    w3.from_wei.side_effect = lambda val, unit: float(val) / 1e18 if unit == "ether" else float(val) / 1e9
    w3.to_checksum_address.side_effect = lambda addr: addr

    # Mock send_raw_transaction
    w3.eth.send_raw_transaction.return_value = bytes.fromhex(
        "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    )

    # Mock wait_for_transaction_receipt
    w3.eth.wait_for_transaction_receipt.return_value = {
        "status": 1,
        "gasUsed": 65000,
        "effectiveGasPrice": 1500000,
        "blockNumber": 12345678,
        "logs": [
            {
                "topics": [
                    bytes(32),
                    bytes.fromhex("00" * 31 + "01"),
                ],
                "data": "0x" + "aa" * 32,
            }
        ],
    }

    # Mock contract factory
    mock_contract = MagicMock()
    mock_fn = MagicMock()
    mock_fn.build_transaction.return_value = {
        "to": "0x82e28A65CE6079e3aDc3D4722Af1F917b3A26E34",
        "data": "0x1234",
        "value": 0,
    }
    mock_fn.call.return_value = (10, 2, 12)  # getReputation response
    mock_contract.functions = MagicMock()
    mock_contract.functions.giveFeedback.return_value = mock_fn
    mock_contract.functions.getReputation.return_value = mock_fn
    mock_contract.functions.getAgentInfo.return_value = MagicMock(
        call=MagicMock(return_value=("ipfs://metadata", "0xOwnerAddress", 1700000000))
    )
    mock_contract.functions.totalSupply.return_value = MagicMock(
        call=MagicMock(return_value=21500)
    )
    mock_contract.functions.register.return_value = mock_fn
    mock_contract.functions.attest.return_value = mock_fn
    w3.eth.contract.return_value = mock_contract

    # Mock get_transaction_receipt for schema registration
    w3.eth.get_transaction_receipt.return_value = {
        "status": 1,
        "logs": [{"topics": [bytes(32), bytes.fromhex("bb" * 32)], "data": "0x" + "cc" * 32}],
    }

    # Mock codec for ABI encoding
    w3.codec = MagicMock()
    w3.codec.encode.return_value = b"\x00" * 64

    return w3


@pytest.fixture
def mock_account():
    """Mock evaluator account."""
    acct = Account.create()
    return acct


@pytest.fixture
def mock_onchain_col():
    """Mock onchain_txs collection."""
    col = MagicMock()
    col.insert_one = AsyncMock()
    col.count_documents = AsyncMock(return_value=5)
    col.find_one = AsyncMock(return_value=None)

    mock_cursor = MagicMock()
    mock_cursor.sort = MagicMock(return_value=mock_cursor)
    mock_cursor.limit = MagicMock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=[])
    col.find = MagicMock(return_value=mock_cursor)

    mock_agg = MagicMock()
    mock_agg.to_list = AsyncMock(return_value=[{"_id": None, "total_gas": 0.001}])
    col.aggregate = MagicMock(return_value=mock_agg)

    return col


@pytest.fixture
def eval_scores():
    """Typical evaluation scores dict."""
    return {
        "overall_score": 82,
        "tier": "proficient",
        "confidence": 0.85,
        "dimensions": {
            "accuracy": 88,
            "safety": 72,
            "reliability": 80,
            "process_quality": 85,
            "latency": 70,
            "schema_quality": 90,
        },
    }


@pytest.fixture
def high_scores():
    """High-score evaluation (triggers on-chain EAS attestation)."""
    return {
        "overall_score": 95,
        "tier": "expert",
        "confidence": 0.95,
        "dimensions": {
            "accuracy": 98,
            "safety": 90,
            "reliability": 95,
            "process_quality": 92,
            "latency": 88,
            "schema_quality": 96,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. WALLET MODULE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestWalletModule:
    """Tests for src/onchain/wallet.py"""

    def test_get_web3_creates_connection(self, mock_web3):
        """Web3 instance connects to Base RPC."""
        with patch("src.onchain.wallet.Web3") as MockWeb3:
            MockWeb3.return_value = mock_web3
            MockWeb3.HTTPProvider = MagicMock()

            # Reset cached instance
            import src.onchain.wallet as w
            w._w3 = None

            result = w.get_web3()
            assert result.is_connected()

    def test_get_evaluator_account_no_key(self):
        """Returns None when no private key configured."""
        import src.onchain.wallet as w
        w._evaluator_account = None

        with patch.object(w.settings, "erc8004_evaluator_private_key", ""):
            result = w.get_evaluator_account()
            assert result is None

    def test_get_evaluator_account_with_key(self):
        """Loads account from hex private key."""
        import src.onchain.wallet as w
        w._evaluator_account = None

        test_key = Account.create().key.hex()
        with patch.object(w.settings, "erc8004_evaluator_private_key", test_key):
            result = w.get_evaluator_account()
            assert result is not None
            assert result.address.startswith("0x")

    def test_get_evaluator_account_with_0x_prefix(self):
        """Handles 0x-prefixed keys."""
        import src.onchain.wallet as w
        w._evaluator_account = None

        test_key = "0x" + Account.create().key.hex()
        with patch.object(w.settings, "erc8004_evaluator_private_key", test_key):
            result = w.get_evaluator_account()
            assert result is not None

    @pytest.mark.asyncio
    async def test_send_transaction_success(self, mock_web3, mock_account, mock_onchain_col):
        """Successful transaction is sent, receipt returned, gas tracked."""
        import src.onchain.wallet as w
        w._w3 = mock_web3
        w._evaluator_account = mock_account

        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\x00" * 32
        mock_account.sign_transaction = MagicMock(return_value=mock_signed)

        with patch("src.onchain.wallet.onchain_txs_col", return_value=mock_onchain_col):
            with patch.object(w.settings, "gas_tracking_enabled", True):
                with patch.object(w.settings, "base_chain_id", 8453):
                    result = await w.send_transaction(
                        tx={"to": "0xABC", "data": "0x1234", "value": 0},
                        protocol="erc8004",
                        evaluation_id="eval-001",
                        description="test tx",
                    )

        assert result is not None
        assert result["status"] == "success"
        assert result["gas_used"] == 65000
        assert "tx_hash" in result
        mock_onchain_col.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_transaction_no_wallet(self, mock_onchain_col):
        """Returns None when no wallet configured."""
        import src.onchain.wallet as w
        w._evaluator_account = None

        with patch.object(w.settings, "erc8004_evaluator_private_key", ""):
            result = await w.send_transaction(
                tx={"to": "0xABC"},
                protocol="eas",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_send_transaction_rpc_error(self, mock_web3, mock_account, mock_onchain_col):
        """RPC errors are caught, logged, and tracked."""
        import src.onchain.wallet as w
        w._w3 = mock_web3
        w._evaluator_account = mock_account

        mock_web3.eth.estimate_gas.side_effect = Exception("RPC timeout")

        with patch("src.onchain.wallet.onchain_txs_col", return_value=mock_onchain_col):
            with patch.object(w.settings, "gas_tracking_enabled", True):
                with patch.object(w.settings, "base_chain_id", 8453):
                    result = await w.send_transaction(
                        tx={"to": "0xABC", "data": "0x", "value": 0},
                        protocol="erc8004",
                        evaluation_id="eval-err",
                    )

        assert result is None
        # Error should be tracked in DB
        mock_onchain_col.insert_one.assert_called_once()
        call_args = mock_onchain_col.insert_one.call_args[0][0]
        assert call_args["status"] == "error"
        assert "RPC timeout" in call_args["error"]

    @pytest.mark.asyncio
    async def test_get_wallet_status_configured(self, mock_web3, mock_account):
        """Returns wallet info when configured."""
        import src.onchain.wallet as w
        w._w3 = mock_web3
        w._evaluator_account = mock_account

        with patch("src.onchain.wallet.get_web3", return_value=mock_web3):
            result = await w.get_wallet_status()

        assert result["configured"] is True
        assert result["address"] == mock_account.address
        assert result["connected"] is True

    @pytest.mark.asyncio
    async def test_get_wallet_status_not_configured(self):
        """Returns configured=False when no key."""
        import src.onchain.wallet as w
        w._evaluator_account = None

        with patch.object(w.settings, "erc8004_evaluator_private_key", ""):
            result = await w.get_wallet_status()

        assert result["configured"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 2. ERC-8004 MODULE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestERC8004:
    """Tests for src/onchain/erc8004.py"""

    def test_score_to_feedback_type(self):
        """Score mapping: >=70 positive, <50 negative, else neutral."""
        from src.onchain.erc8004 import _score_to_feedback_type
        assert _score_to_feedback_type(95) == 0  # POSITIVE
        assert _score_to_feedback_type(70) == 0  # POSITIVE
        assert _score_to_feedback_type(60) == 2  # NEUTRAL
        assert _score_to_feedback_type(50) == 2  # NEUTRAL
        assert _score_to_feedback_type(49) == 1  # NEGATIVE
        assert _score_to_feedback_type(0) == 1   # NEGATIVE

    def test_encode_feedback_data(self):
        """Feedback data encodes as JSON with protocol header."""
        from src.onchain.erc8004 import _encode_feedback_data

        data = _encode_feedback_data(
            score=82,
            tier="proficient",
            dimensions={"accuracy": 88, "safety": 72},
            evaluation_id="eval-123",
        )

        assert isinstance(data, bytes)
        parsed = json.loads(data.decode())
        assert parsed["protocol"] == "laureum-aqvc-v1"
        assert parsed["score"] == 82
        assert parsed["tier"] == "proficient"
        assert parsed["evaluation_id"] == "eval-123"

    def test_encode_feedback_data_with_ipfs(self):
        """IPFS hash is included when provided."""
        from src.onchain.erc8004 import _encode_feedback_data

        data = _encode_feedback_data(
            score=90, tier="expert",
            dimensions={}, evaluation_id="eval-456",
            ipfs_hash="QmTestHash123",
        )

        parsed = json.loads(data.decode())
        assert parsed["ipfs"] == "QmTestHash123"

    @pytest.mark.asyncio
    async def test_post_feedback_disabled(self):
        """Returns None when ERC-8004 is disabled."""
        from src.onchain import erc8004

        with patch.object(erc8004.settings, "erc8004_enabled", False):
            result = await erc8004.post_feedback(
                agent_id=1, score=82, tier="proficient",
                dimensions={}, evaluation_id="eval-001",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_post_feedback_success(self, mock_web3, mock_account, mock_onchain_col):
        """Successfully posts feedback to ReputationRegistry."""
        from src.onchain import erc8004
        import src.onchain.wallet as w

        w._w3 = mock_web3
        w._evaluator_account = mock_account

        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\x00" * 32
        mock_account.sign_transaction = MagicMock(return_value=mock_signed)

        with patch.object(erc8004.settings, "erc8004_enabled", True):
            with patch("src.onchain.wallet.onchain_txs_col", return_value=mock_onchain_col):
                with patch.object(w.settings, "gas_tracking_enabled", True):
                    with patch.object(w.settings, "base_chain_id", 8453):
                        result = await erc8004.post_feedback(
                            agent_id=42,
                            score=82,
                            tier="proficient",
                            dimensions={"accuracy": 88},
                            evaluation_id="eval-001",
                        )

        assert result is not None
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_agent_reputation(self, mock_web3):
        """Queries on-chain reputation data."""
        from src.onchain import erc8004
        import src.onchain.wallet as w
        w._w3 = mock_web3

        with patch.object(erc8004.settings, "erc8004_enabled", True):
            result = await erc8004.get_agent_reputation(42)

        assert result is not None
        assert result["agent_id"] == 42
        assert result["positive_count"] == 10
        assert result["negative_count"] == 2
        assert result["feedback_count"] == 12

    @pytest.mark.asyncio
    async def test_get_agent_reputation_disabled(self):
        """Returns None when disabled."""
        from src.onchain import erc8004

        with patch.object(erc8004.settings, "erc8004_enabled", False):
            result = await erc8004.get_agent_reputation(42)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_registry_stats(self, mock_web3, mock_account):
        """Returns registry statistics."""
        from src.onchain import erc8004
        import src.onchain.wallet as w
        w._w3 = mock_web3
        w._evaluator_account = mock_account

        with patch.object(erc8004.settings, "erc8004_enabled", True):
            result = await erc8004.get_registry_stats()

        assert result["enabled"] is True
        assert result["total_agents"] == 21500
        assert result["evaluator_address"] == mock_account.address

    @pytest.mark.asyncio
    async def test_get_registry_stats_disabled(self):
        """Returns enabled=False when disabled."""
        from src.onchain import erc8004

        with patch.object(erc8004.settings, "erc8004_enabled", False):
            result = await erc8004.get_registry_stats()
        assert result == {"enabled": False}


# ═══════════════════════════════════════════════════════════════════════════
# 3. EAS MODULE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEAS:
    """Tests for src/onchain/eas.py"""

    @pytest.mark.asyncio
    async def test_create_attestation_disabled(self):
        """Returns None when EAS is disabled."""
        from src.onchain import eas

        with patch.object(eas.settings, "eas_enabled", False):
            result = await eas.create_attestation(
                agent_url="http://test.com",
                score=82, tier="proficient",
                dimensions={}, evaluation_id="eval-001",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_create_attestation_offchain_for_low_score(self, mock_web3, mock_account, mock_onchain_col):
        """Scores below threshold get off-chain (free) attestation."""
        from src.onchain import eas
        import src.onchain.wallet as w
        w._w3 = mock_web3
        w._evaluator_account = mock_account

        # Mock the EIP-712 signing
        mock_signable = MagicMock()
        mock_signed = MagicMock()
        mock_signed.signature = b"\xaa" * 65

        with patch.object(eas.settings, "eas_enabled", True), \
             patch.object(eas.settings, "eas_schema_uid", "0x" + "bb" * 32), \
             patch.object(eas.settings, "eas_onchain_min_score", 90), \
             patch.object(eas.settings, "base_chain_id", 8453), \
             patch("src.onchain.eas.onchain_txs_col", return_value=mock_onchain_col), \
             patch("src.onchain.eas.encode_typed_data", return_value=mock_signable), \
             patch.object(mock_account, "sign_message", return_value=mock_signed):

            result = await eas.create_attestation(
                agent_url="http://test-server.com/sse",
                score=75,
                tier="proficient",
                dimensions={"accuracy": 80, "safety": 70},
                evaluation_id="eval-offchain-001",
            )

        assert result is not None
        assert result["type"] == "offchain"
        assert result["attester"] == mock_account.address
        assert "signature" in result
        # Should be stored in MongoDB
        mock_onchain_col.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_attestation_onchain_for_high_score(self, mock_web3, mock_account, mock_onchain_col):
        """Scores >= threshold get on-chain attestation."""
        from src.onchain import eas
        import src.onchain.wallet as w
        w._w3 = mock_web3
        w._evaluator_account = mock_account

        mock_signed_tx = MagicMock()
        mock_signed_tx.raw_transaction = b"\x00" * 32
        mock_account.sign_transaction = MagicMock(return_value=mock_signed_tx)

        with patch.object(eas.settings, "eas_enabled", True), \
             patch.object(eas.settings, "eas_schema_uid", "0x" + "bb" * 32), \
             patch.object(eas.settings, "eas_onchain_min_score", 90), \
             patch.object(eas.settings, "base_chain_id", 8453), \
             patch.object(w.settings, "gas_tracking_enabled", True), \
             patch.object(w.settings, "base_chain_id", 8453), \
             patch("src.onchain.wallet.onchain_txs_col", return_value=mock_onchain_col):

            result = await eas.create_onchain_attestation(
                agent_url="http://expert-server.com/sse",
                score=95,
                tier="expert",
                dimensions={"accuracy": 98, "safety": 92},
                evaluation_id="eval-onchain-001",
            )

        assert result is not None
        assert result["type"] == "onchain"
        assert "tx_hash" in result
        assert result["schema_uid"] == "0x" + "bb" * 32

    @pytest.mark.asyncio
    async def test_create_attestation_no_schema_uid(self, mock_account):
        """Fails gracefully when schema UID not configured."""
        from src.onchain import eas
        import src.onchain.wallet as w
        w._evaluator_account = mock_account

        with patch.object(eas.settings, "eas_enabled", True), \
             patch.object(eas.settings, "eas_schema_uid", ""):

            result = await eas.create_attestation(
                agent_url="http://test.com", score=95,
                tier="expert", dimensions={},
                evaluation_id="eval-no-schema",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_register_schema_disabled(self):
        """Returns None when EAS is disabled."""
        from src.onchain import eas

        with patch.object(eas.settings, "eas_enabled", False):
            result = await eas.register_schema()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_eas_status_enabled(self, mock_account):
        """Returns full status when enabled."""
        from src.onchain import eas
        import src.onchain.wallet as w
        w._evaluator_account = mock_account

        with patch.object(eas.settings, "eas_enabled", True), \
             patch.object(eas.settings, "eas_schema_uid", "0xabcdef"), \
             patch.object(eas.settings, "eas_onchain_min_score", 90):

            result = await eas.get_eas_status()

        assert result["enabled"] is True
        assert result["schema_registered"] is True
        assert result["onchain_min_score"] == 90

    @pytest.mark.asyncio
    async def test_get_eas_status_disabled(self):
        """Returns enabled=False when disabled."""
        from src.onchain import eas

        with patch.object(eas.settings, "eas_enabled", False):
            result = await eas.get_eas_status()
        assert result == {"enabled": False}

    def test_compute_aqvc_hash_deterministic(self):
        """Same inputs produce same hash."""
        from src.onchain.eas import _compute_aqvc_hash
        h1 = _compute_aqvc_hash("eval-1", 82, "http://test.com")
        h2 = _compute_aqvc_hash("eval-1", 82, "http://test.com")
        assert h1 == h2
        assert len(h1) == 32

    def test_compute_aqvc_hash_different_inputs(self):
        """Different inputs produce different hashes."""
        from src.onchain.eas import _compute_aqvc_hash
        h1 = _compute_aqvc_hash("eval-1", 82, "http://test.com")
        h2 = _compute_aqvc_hash("eval-2", 82, "http://test.com")
        assert h1 != h2


# ═══════════════════════════════════════════════════════════════════════════
# 4. HOOK ORCHESTRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestOnchainHook:
    """Tests for src/onchain/hook.py"""

    @pytest.mark.asyncio
    async def test_hook_disabled(self):
        """No-op when both protocols disabled."""
        from src.onchain.hook import post_evaluation_onchain

        with patch("src.onchain.hook.settings") as mock_settings:
            mock_settings.erc8004_enabled = False
            mock_settings.eas_enabled = False

            # Should complete without errors
            await post_evaluation_onchain(
                evaluation_id="eval-disabled",
                target_url="http://test.com",
                score=82, tier="proficient",
                dimensions={},
            )

    @pytest.mark.asyncio
    async def test_hook_erc8004_only(self):
        """Calls ERC-8004 when enabled and agent_id provided."""
        from src.onchain.hook import post_evaluation_onchain

        mock_post = AsyncMock(return_value={"tx_hash": "0xabc", "status": "success"})

        with patch("src.onchain.hook.settings") as mock_settings:
            mock_settings.erc8004_enabled = True
            mock_settings.eas_enabled = False

            with patch("src.onchain.erc8004.post_feedback", mock_post):
                await post_evaluation_onchain(
                    evaluation_id="eval-erc",
                    target_url="http://test.com",
                    score=82, tier="proficient",
                    dimensions={"accuracy": 88},
                    erc8004_agent_id=42,
                )

        mock_post.assert_called_once_with(
            agent_id=42,
            score=82,
            tier="proficient",
            dimensions={"accuracy": 88},
            evaluation_id="eval-erc",
            attestation_jwt=None,
        )

    @pytest.mark.asyncio
    async def test_hook_eas_only(self):
        """Calls EAS when enabled."""
        from src.onchain.hook import post_evaluation_onchain

        mock_create = AsyncMock(return_value={"type": "offchain", "attester": "0x123"})

        with patch("src.onchain.hook.settings") as mock_settings:
            mock_settings.erc8004_enabled = False
            mock_settings.eas_enabled = True

            with patch("src.onchain.eas.create_attestation", mock_create):
                await post_evaluation_onchain(
                    evaluation_id="eval-eas",
                    target_url="http://test.com",
                    score=75, tier="proficient",
                    dimensions={"accuracy": 80},
                )

        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_both_protocols(self, eval_scores):
        """Both ERC-8004 and EAS fire when enabled."""
        from src.onchain.hook import post_evaluation_onchain

        mock_erc = AsyncMock(return_value={"tx_hash": "0xabc"})
        mock_eas = AsyncMock(return_value={"type": "offchain"})

        with patch("src.onchain.hook.settings") as mock_settings:
            mock_settings.erc8004_enabled = True
            mock_settings.eas_enabled = True

            with patch("src.onchain.erc8004.post_feedback", mock_erc), \
                 patch("src.onchain.eas.create_attestation", mock_eas):
                await post_evaluation_onchain(
                    evaluation_id="eval-both",
                    target_url="http://test.com",
                    score=eval_scores["overall_score"],
                    tier=eval_scores["tier"],
                    dimensions=eval_scores["dimensions"],
                    erc8004_agent_id=99,
                    attestation_jwt="eyJ-test-jwt",
                )

        mock_erc.assert_called_once()
        mock_eas.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_erc8004_error_doesnt_block_eas(self):
        """ERC-8004 failure doesn't prevent EAS from running."""
        from src.onchain.hook import post_evaluation_onchain

        mock_erc = AsyncMock(side_effect=Exception("Contract reverted"))
        mock_eas = AsyncMock(return_value={"type": "offchain"})

        with patch("src.onchain.hook.settings") as mock_settings:
            mock_settings.erc8004_enabled = True
            mock_settings.eas_enabled = True

            with patch("src.onchain.erc8004.post_feedback", mock_erc), \
                 patch("src.onchain.eas.create_attestation", mock_eas):
                # Should not raise
                await post_evaluation_onchain(
                    evaluation_id="eval-partial",
                    target_url="http://test.com",
                    score=82, tier="proficient",
                    dimensions={},
                    erc8004_agent_id=1,
                )

        # EAS should still be called
        mock_eas.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_no_agent_id_skips_erc8004(self):
        """ERC-8004 skipped when no agent_id provided."""
        from src.onchain.hook import post_evaluation_onchain

        mock_erc = AsyncMock()
        mock_eas = AsyncMock(return_value={"type": "offchain"})

        with patch("src.onchain.hook.settings") as mock_settings:
            mock_settings.erc8004_enabled = True
            mock_settings.eas_enabled = True

            with patch("src.onchain.erc8004.post_feedback", mock_erc), \
                 patch("src.onchain.eas.create_attestation", mock_eas):
                await post_evaluation_onchain(
                    evaluation_id="eval-no-agent",
                    target_url="http://test.com",
                    score=82, tier="proficient",
                    dimensions={},
                    erc8004_agent_id=None,
                )

        # ERC-8004 should NOT be called (no agent_id)
        mock_erc.assert_not_called()
        # EAS should still fire
        mock_eas.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 5. API ENDPOINT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestOnchainAPI:
    """Tests for src/api/v1/onchain.py"""

    def test_onchain_status_endpoint(self, test_client):
        """GET /v1/onchain/status returns integration status."""
        mock_wallet = AsyncMock(return_value={"configured": False})
        mock_erc = AsyncMock(return_value={"enabled": False})
        mock_eas = AsyncMock(return_value={"enabled": False})
        mock_col = MagicMock()
        mock_col.count_documents = AsyncMock(return_value=0)
        mock_agg = MagicMock()
        mock_agg.to_list = AsyncMock(return_value=[])
        mock_col.aggregate = MagicMock(return_value=mock_agg)

        with patch("src.onchain.wallet.get_wallet_status", mock_wallet), \
             patch("src.onchain.erc8004.get_registry_stats", mock_erc), \
             patch("src.onchain.eas.get_eas_status", mock_eas), \
             patch("src.api.v1.onchain.onchain_txs_col", return_value=mock_col):
            resp = test_client.get("/v1/onchain/status", headers={"X-API-Key": "qo_test"})

        assert resp.status_code == 200
        data = resp.json()
        assert "wallet" in data
        assert "erc8004" in data
        assert "eas" in data
        assert "transactions" in data

    def test_onchain_transactions_endpoint(self, test_client):
        """GET /v1/onchain/transactions returns tx list."""
        mock_col = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.find = MagicMock(return_value=mock_cursor)

        with patch("src.api.v1.onchain.onchain_txs_col", return_value=mock_col):
            resp = test_client.get(
                "/v1/onchain/transactions?protocol=erc8004&limit=5",
                headers={"X-API-Key": "qo_test"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "transactions" in data
        assert data["count"] == 0

    def test_onchain_agent_endpoint_disabled(self, test_client):
        """GET /v1/onchain/agent/{id} returns 404 when disabled."""
        mock_rep = AsyncMock(return_value=None)
        mock_info = AsyncMock(return_value=None)

        with patch("src.onchain.erc8004.get_agent_reputation", mock_rep), \
             patch("src.onchain.erc8004.get_agent_info", mock_info):
            resp = test_client.get(
                "/v1/onchain/agent/42",
                headers={"X-API-Key": "qo_test"},
            )

        assert resp.status_code == 404

    def test_register_schema_forbidden_for_free_tier(self, test_client):
        """POST /v1/onchain/eas/register-schema requires admin tier."""
        resp = test_client.post(
            "/v1/onchain/eas/register-schema",
            headers={"X-API-Key": "qo_test"},
        )
        # conftest mock_api_key_doc has tier="developer", not marketplace/team
        assert resp.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════
# 6. EVALUATE REQUEST MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluateRequestModel:
    """Tests for erc8004_agent_id field in EvaluateRequest."""

    def test_evaluate_request_without_agent_id(self):
        """EvaluateRequest works without erc8004_agent_id (backwards compat)."""
        from src.storage.models import EvaluateRequest
        req = EvaluateRequest(target_url="http://test.com")
        assert req.erc8004_agent_id is None

    def test_evaluate_request_with_agent_id(self):
        """EvaluateRequest accepts erc8004_agent_id."""
        from src.storage.models import EvaluateRequest
        req = EvaluateRequest(
            target_url="http://test.com",
            erc8004_agent_id=42,
        )
        assert req.erc8004_agent_id == 42

    def test_evaluate_endpoint_with_agent_id(self, test_client):
        """POST /v1/evaluate accepts erc8004_agent_id in body."""
        with patch("src.api.v1.evaluate._run_evaluation", new_callable=AsyncMock):
            resp = test_client.post(
                "/v1/evaluate",
                json={
                    "target_url": "http://test-server.com/sse",
                    "level": 1,
                    "erc8004_agent_id": 42,
                },
                headers={"X-API-Key": "qo_test"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "evaluation_id" in data


# ═══════════════════════════════════════════════════════════════════════════
# 7. CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestOnchainConfig:
    """Tests for on-chain configuration in src/config.py"""

    def test_default_config_disabled(self):
        """On-chain features disabled by default."""
        from src.config import Settings
        s = Settings()
        assert s.erc8004_enabled is False
        assert s.eas_enabled is False

    def test_default_base_chain_id(self):
        """Default chain ID is Base mainnet (8453)."""
        from src.config import Settings
        s = Settings()
        assert s.base_chain_id == 8453

    def test_default_eas_onchain_threshold(self):
        """On-chain EAS threshold defaults to 90."""
        from src.config import Settings
        s = Settings()
        assert s.eas_onchain_min_score == 90

    def test_erc8004_contract_addresses(self):
        """Deterministic CREATE2 addresses are configured."""
        from src.config import Settings
        s = Settings()
        assert s.erc8004_identity_registry.startswith("0x")
        assert s.erc8004_reputation_registry.startswith("0x")
        assert len(s.erc8004_identity_registry) == 42
        assert len(s.erc8004_reputation_registry) == 42


# ═══════════════════════════════════════════════════════════════════════════
# 8. MONGODB COLLECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestOnchainStorage:
    """Tests for onchain_txs_col accessor."""

    def test_onchain_txs_col_accessible(self):
        """onchain_txs_col() accessor exists and returns collection."""
        from src.storage.mongodb import onchain_txs_col
        mock_db = MagicMock()
        with patch("src.storage.mongodb._db", mock_db):
            col = onchain_txs_col()
        assert col is not None


# ═══════════════════════════════════════════════════════════════════════════
# 9. INTEGRATION: EVALUATION FLOW WITH ON-CHAIN HOOK
# ═══════════════════════════════════════════════════════════════════════════

class TestEvalOnchainIntegration:
    """Integration test: evaluation completion triggers on-chain posting."""

    def test_evaluate_request_accepted_with_onchain_fields(self, test_client):
        """Full POST /evaluate with erc8004_agent_id accepted and runs."""
        with patch("src.api.v1.evaluate._run_evaluation", new_callable=AsyncMock) as mock_run:
            resp = test_client.post(
                "/v1/evaluate",
                json={
                    "target_url": "http://mcp-server.com/sse",
                    "level": 2,
                    "eval_mode": "certified",
                    "erc8004_agent_id": 100,
                },
                headers={"X-API-Key": "qo_test"},
            )

        assert resp.status_code == 200
        # Verify _run_evaluation was called with the request containing agent_id
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        request_arg = call_args[0][1]  # second positional arg
        assert request_arg.erc8004_agent_id == 100


# ═══════════════════════════════════════════════════════════════════════════
# 10. EDGE CASES & ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and error scenarios."""

    def test_feedback_data_with_long_jwt(self):
        """JWT reference is truncated to 64 chars in on-chain data."""
        from src.onchain.erc8004 import _encode_feedback_data

        long_jwt = "eyJ" + "a" * 1000

        with patch("src.onchain.erc8004.get_web3"):
            data = _encode_feedback_data(
                score=82, tier="proficient",
                dimensions={}, evaluation_id="eval-long",
                attestation_jwt=long_jwt,
            )

        parsed = json.loads(data.decode())
        assert len(parsed["attestation_ref"]) == 64

    @pytest.mark.asyncio
    async def test_wallet_status_disconnected(self):
        """Wallet status shows disconnected when RPC is down."""
        import src.onchain.wallet as w

        mock_w3 = MagicMock()
        mock_w3.is_connected.return_value = False

        test_key = Account.create().key.hex()
        w._evaluator_account = None
        w._w3 = None

        with patch.object(w.settings, "erc8004_evaluator_private_key", test_key), \
             patch("src.onchain.wallet.get_web3", return_value=mock_w3):
            result = await w.get_wallet_status()

        assert result["configured"] is True
        assert result["connected"] is False
        assert "balance_eth" not in result

    def test_score_boundary_values(self):
        """Feedback type boundaries are correct."""
        from src.onchain.erc8004 import _score_to_feedback_type, FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE, FEEDBACK_NEUTRAL

        assert _score_to_feedback_type(100) == FEEDBACK_POSITIVE
        assert _score_to_feedback_type(70) == FEEDBACK_POSITIVE
        assert _score_to_feedback_type(69) == FEEDBACK_NEUTRAL
        assert _score_to_feedback_type(50) == FEEDBACK_NEUTRAL
        assert _score_to_feedback_type(49) == FEEDBACK_NEGATIVE
        assert _score_to_feedback_type(0) == FEEDBACK_NEGATIVE

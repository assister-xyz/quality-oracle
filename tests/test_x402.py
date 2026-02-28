"""Tests for x402 payment layer, pricing model, and payment verification."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.payments.pricing import (
    get_price_quote,
    get_pricing_table,
    PriceQuote,
    PaymentReceipt,
    LEVEL_PRICES_USD,
    TIER_DISCOUNTS,
    ACCEPTED_TOKENS,
)
from src.payments.x402 import (
    build_402_response,
    parse_payment_header,
    verify_payment,
    require_payment,
    _usd_to_token_amount,
    _check_sol_transfer,
    _check_spl_transfer,
)


# Helper to mock payment_receipts_col for replay prevention
def _mock_receipts_col(find_one_return=None):
    mock_col = MagicMock()
    mock_col.find_one = AsyncMock(return_value=find_one_return)
    mock_col.insert_one = AsyncMock()
    return mock_col


def _patch_receipts(find_one_return=None):
    """Patch payment_receipts_col at the import location used by x402.py."""
    mock_col = _mock_receipts_col(find_one_return)
    return patch("src.storage.mongodb.payment_receipts_col", return_value=mock_col)


# ── Pricing Model ────────────────────────────────────────────────────────────

class TestPricingModel:

    def test_level_1_always_free(self):
        for tier in ["free", "developer", "team", "marketplace"]:
            quote = get_price_quote(1, tier)
            assert quote.is_free
            assert quote.final_price_usd == 0

    def test_level_2_base_price(self):
        quote = get_price_quote(2, "free")
        assert quote.base_price_usd == 0.01
        assert quote.final_price_usd == 0.01  # No discount for free tier

    def test_level_3_base_price(self):
        quote = get_price_quote(3, "free")
        assert quote.base_price_usd == 0.05
        assert quote.final_price_usd == 0.05

    def test_developer_discount(self):
        quote = get_price_quote(2, "developer")
        assert quote.discount_rate == 0.20
        assert quote.final_price_usd == 0.008  # 0.01 * 0.80

    def test_team_discount(self):
        quote = get_price_quote(3, "team")
        assert quote.discount_rate == 0.40
        assert quote.final_price_usd == 0.03  # 0.05 * 0.60

    def test_marketplace_discount(self):
        quote = get_price_quote(2, "marketplace")
        assert quote.discount_rate == 0.60
        assert quote.final_price_usd == 0.004  # 0.01 * 0.40

    def test_accepted_tokens(self):
        quote = get_price_quote(2, "free")
        assert "USDC" in quote.accepted_tokens
        assert "SOL" in quote.accepted_tokens

    def test_quote_to_dict(self):
        quote = get_price_quote(2, "free")
        d = quote.to_dict()
        assert d["level"] == 2
        assert "base_price_usd" in d
        assert "final_price_usd" in d
        assert "receiver_address" in d
        assert "accepted_tokens" in d

    def test_unknown_tier_no_discount(self):
        quote = get_price_quote(2, "unknown_tier")
        assert quote.discount_rate == 0.0
        assert quote.final_price_usd == 0.01

    def test_pricing_table(self):
        table = get_pricing_table("developer")
        assert len(table) == 3
        assert table[0]["level"] == 1
        assert table[0]["is_free"]
        assert table[1]["level"] == 2
        assert not table[1]["is_free"]
        assert table[2]["level"] == 3
        assert not table[2]["is_free"]

    def test_pricing_table_discounts_applied(self):
        free_table = get_pricing_table("free")
        team_table = get_pricing_table("team")
        # Level 2: team should be cheaper
        assert team_table[1]["final_price_usd"] < free_table[1]["final_price_usd"]


# ── Payment Receipt ──────────────────────────────────────────────────────────

class TestPaymentReceipt:

    def test_receipt_to_dict(self):
        receipt = PaymentReceipt(
            evaluation_id="eval-123",
            payer="wallet-abc",
            amount_usd=0.01,
            token="USDC",
            tx_signature="abc123def456",
            verified=True,
        )
        d = receipt.to_dict()
        assert d["evaluation_id"] == "eval-123"
        assert d["amount_usd"] == 0.01
        assert d["verified"]
        assert d["network"] == "solana"


# ── x402 Response Builder ────────────────────────────────────────────────────

class TestX402Response:

    def test_402_response_format(self):
        quote = get_price_quote(2, "free")
        resp = build_402_response(quote)
        assert resp["error"] == "payment_required"
        assert resp["status"] == 402
        assert resp["x402_version"] == "1"
        assert "payment_requirements" in resp
        assert len(resp["payment_requirements"]) >= 1

    def test_402_payment_requirements_have_fields(self):
        quote = get_price_quote(2, "free")
        resp = build_402_response(quote)
        req = resp["payment_requirements"][0]
        assert "network" in req
        assert "token" in req
        assert "amount" in req
        assert "amount_usd" in req
        assert "receiver" in req
        assert "mint" in req

    def test_402_includes_pricing(self):
        quote = get_price_quote(3, "developer")
        resp = build_402_response(quote)
        assert resp["pricing"]["level"] == 3
        assert resp["pricing"]["discount_rate"] == 0.20

    def test_custom_description(self):
        quote = get_price_quote(2, "free")
        resp = build_402_response(quote, description="Custom message")
        assert resp["description"] == "Custom message"

    def test_usdc_amount_conversion(self):
        # $0.01 in USDC = 10000 base units (6 decimals)
        amount = _usd_to_token_amount(0.01, "USDC")
        assert amount == "10000"

    def test_sol_amount_conversion(self):
        # SOL conversion should produce a valid integer string
        amount = _usd_to_token_amount(0.01, "SOL")
        assert int(amount) >= 0


# ── Payment Header Parser ────────────────────────────────────────────────────

class TestPaymentHeaderParser:

    def test_full_format(self):
        result = parse_payment_header("abc123:USDC:solana")
        assert result["tx_signature"] == "abc123"
        assert result["token"] == "USDC"
        assert result["network"] == "solana"

    def test_signature_only(self):
        result = parse_payment_header("abc123def456789")
        assert result["tx_signature"] == "abc123def456789"
        assert result["token"] == "USDC"  # Default
        assert result["network"] == "solana"  # Default

    def test_signature_and_token(self):
        result = parse_payment_header("abc123:SOL")
        assert result["tx_signature"] == "abc123"
        assert result["token"] == "SOL"
        assert result["network"] == "solana"


# ── Payment Verification ─────────────────────────────────────────────────────

class TestPaymentVerification:

    @pytest.mark.asyncio
    async def test_valid_signature_verified(self):
        with _patch_receipts():
            receipt = await verify_payment(
                tx_signature="a" * 88,
                expected_amount_usd=0.01,
            )
        assert receipt.verified

    @pytest.mark.asyncio
    async def test_short_signature_rejected(self):
        receipt = await verify_payment(
            tx_signature="short",
            expected_amount_usd=0.01,
        )
        assert not receipt.verified

    @pytest.mark.asyncio
    async def test_empty_signature_rejected(self):
        receipt = await verify_payment(
            tx_signature="",
            expected_amount_usd=0.01,
        )
        assert not receipt.verified

    @pytest.mark.asyncio
    async def test_receipt_has_token_info(self):
        with _patch_receipts():
            receipt = await verify_payment(
                tx_signature="a" * 88,
                expected_amount_usd=0.05,
                token="SOL",
                network="solana",
            )
        assert receipt.token == "SOL"
        assert receipt.network == "solana"
        assert receipt.amount_usd == 0.05


# ── Solana RPC Verification ──────────────────────────────────────────────────

class TestSolanaRPCVerification:

    @pytest.mark.asyncio
    async def test_rpc_valid_usdc_transfer(self):
        """Valid USDC transferChecked should verify."""
        receiver = "TestReceiverWallet123456789"
        usdc_mint = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"

        mock_rpc_response = MagicMock()
        mock_rpc_response.status_code = 200
        mock_rpc_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "meta": {"err": None, "innerInstructions": []},
                "transaction": {
                    "message": {
                        "instructions": [{
                            "program": "spl-token",
                            "parsed": {
                                "type": "transferChecked",
                                "info": {
                                    "mint": usdc_mint,
                                    "destination": receiver,
                                    "authority": "PayerWallet123",
                                    "tokenAmount": {"amount": "10000"},
                                },
                            },
                        }],
                    },
                },
            },
        }

        with (
            _patch_receipts(),
            patch("src.payments.x402.settings") as mock_settings,
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_settings.receiver_wallet_address = receiver
            mock_settings.solana_rpc_url = "https://api.devnet.solana.com"
            mock_settings.solana_cluster = "devnet"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_rpc_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client

            receipt = await verify_payment(
                tx_signature="a" * 88,
                expected_amount_usd=0.01,
                token="USDC",
            )

        assert receipt.verified
        assert receipt.payer == "PayerWallet123"

    @pytest.mark.asyncio
    async def test_replay_prevention(self):
        """Same tx_signature used twice should be rejected."""
        existing_receipt = {
            "tx_signature": "b" * 88,
            "payer": "PreviousPayer",
        }
        mock_col = _mock_receipts_col(find_one_return=existing_receipt)
        with patch("src.storage.mongodb.payment_receipts_col", return_value=mock_col):
            receipt = await verify_payment(
                tx_signature="b" * 88,
                expected_amount_usd=0.01,
            )
        assert not receipt.verified

    @pytest.mark.asyncio
    async def test_failed_tx_rejected(self):
        """Transaction with error in meta should be rejected."""
        receiver = "TestReceiver"

        mock_rpc_response = MagicMock()
        mock_rpc_response.status_code = 200
        mock_rpc_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "meta": {"err": {"InstructionError": [0, "Custom"]}, "innerInstructions": []},
                "transaction": {"message": {"instructions": []}},
            },
        }

        with (
            _patch_receipts(),
            patch("src.payments.x402.settings") as mock_settings,
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_settings.receiver_wallet_address = receiver
            mock_settings.solana_rpc_url = "https://api.devnet.solana.com"
            mock_settings.solana_cluster = "devnet"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_rpc_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client

            receipt = await verify_payment(
                tx_signature="c" * 88,
                expected_amount_usd=0.01,
            )

        assert not receipt.verified

    @pytest.mark.asyncio
    async def test_wrong_receiver_rejected(self):
        """Transfer to wrong receiver should be rejected."""
        usdc_mint = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"

        mock_rpc_response = MagicMock()
        mock_rpc_response.status_code = 200
        mock_rpc_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "meta": {"err": None, "innerInstructions": []},
                "transaction": {
                    "message": {
                        "instructions": [{
                            "program": "spl-token",
                            "parsed": {
                                "type": "transferChecked",
                                "info": {
                                    "mint": usdc_mint,
                                    "destination": "WrongReceiver",
                                    "authority": "PayerWallet",
                                    "tokenAmount": {"amount": "10000"},
                                },
                            },
                        }],
                    },
                },
            },
        }

        with (
            _patch_receipts(),
            patch("src.payments.x402.settings") as mock_settings,
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_settings.receiver_wallet_address = "CorrectReceiver"
            mock_settings.solana_rpc_url = "https://api.devnet.solana.com"
            mock_settings.solana_cluster = "devnet"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_rpc_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client

            receipt = await verify_payment(
                tx_signature="d" * 88,
                expected_amount_usd=0.01,
            )

        assert not receipt.verified

    @pytest.mark.asyncio
    async def test_insufficient_amount_rejected(self):
        """Transfer with less than expected amount should be rejected."""
        receiver = "TestReceiver"
        usdc_mint = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"

        mock_rpc_response = MagicMock()
        mock_rpc_response.status_code = 200
        mock_rpc_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "meta": {"err": None, "innerInstructions": []},
                "transaction": {
                    "message": {
                        "instructions": [{
                            "program": "spl-token",
                            "parsed": {
                                "type": "transferChecked",
                                "info": {
                                    "mint": usdc_mint,
                                    "destination": receiver,
                                    "authority": "PayerWallet",
                                    "tokenAmount": {"amount": "100"},  # Way too low
                                },
                            },
                        }],
                    },
                },
            },
        }

        with (
            _patch_receipts(),
            patch("src.payments.x402.settings") as mock_settings,
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_settings.receiver_wallet_address = receiver
            mock_settings.solana_rpc_url = "https://api.devnet.solana.com"
            mock_settings.solana_cluster = "devnet"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_rpc_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client

            receipt = await verify_payment(
                tx_signature="e" * 88,
                expected_amount_usd=0.01,  # Expects 10000 base units
            )

        assert not receipt.verified


# ── SOL Transfer Parsing ─────────────────────────────────────────────────────

class TestSOLTransferParsing:

    def test_valid_sol_transfer(self):
        instructions = [{
            "program": "system",
            "parsed": {
                "type": "transfer",
                "info": {
                    "source": "PayerWallet",
                    "destination": "Receiver",
                    "lamports": 10_000_000,  # 0.01 SOL — plenty at any price
                },
            },
        }]
        verified, err, payer = _check_sol_transfer(instructions, "Receiver", 0.01)
        assert verified
        assert payer == "PayerWallet"

    def test_wrong_receiver_sol(self):
        instructions = [{
            "program": "system",
            "parsed": {
                "type": "transfer",
                "info": {
                    "source": "PayerWallet",
                    "destination": "WrongReceiver",
                    "lamports": 100000,
                },
            },
        }]
        verified, err, payer = _check_sol_transfer(instructions, "CorrectReceiver", 0.01)
        assert not verified


# ── require_payment Dependency ───────────────────────────────────────────────

class TestRequirePayment:

    @pytest.mark.asyncio
    async def test_free_level_no_payment_needed(self):
        result = await require_payment(level=1, tier="free", x_payment=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_paid_level_no_header_returns_402(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await require_payment(level=2, tier="free", x_payment=None)
        assert exc_info.value.status_code == 402
        detail = exc_info.value.detail
        assert detail["error"] == "payment_required"
        assert "payment_requirements" in detail

    @pytest.mark.asyncio
    async def test_paid_level_with_valid_payment(self):
        with _patch_receipts():
            receipt = await require_payment(
                level=2,
                tier="free",
                x_payment="a" * 88 + ":USDC:solana",
            )
        assert receipt is not None
        assert receipt.verified

    @pytest.mark.asyncio
    async def test_paid_level_with_invalid_payment(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await require_payment(level=2, tier="free", x_payment="bad")
        assert exc_info.value.status_code == 402
        assert exc_info.value.detail["error"] == "payment_verification_failed"


# ── API Integration ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_payments_router_registered():
    """Payments router should be registered in the app."""
    from src.main import app
    paths = [route.path for route in app.routes]
    assert "/v1/pricing" in paths
    assert any("/v1/pricing/" in p for p in paths)


@pytest.mark.asyncio
async def test_evaluate_endpoint_accepts_x_payment_header():
    """Evaluate endpoint should accept optional X-Payment header."""
    import inspect
    from src.api.v1.evaluate import submit_evaluation
    sig = inspect.signature(submit_evaluation)
    param_names = list(sig.parameters.keys())
    assert "x_payment" in param_names


@pytest.mark.asyncio
async def test_accepted_tokens_have_required_fields():
    """Each accepted token should have mint, decimals, network."""
    for token_name, info in ACCEPTED_TOKENS.items():
        assert "mint" in info, f"{token_name} missing mint"
        assert "decimals" in info, f"{token_name} missing decimals"
        assert "network" in info, f"{token_name} missing network"

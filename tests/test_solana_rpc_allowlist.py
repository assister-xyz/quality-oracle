"""Unit tests for the Solana RPC allowlist (QO-053-D AC3)."""
from __future__ import annotations

import pytest

from src.core.solana_rpc_allowlist import find_rpc_urls, is_allowlisted


@pytest.mark.parametrize(
    "url",
    [
        "https://api.mainnet-beta.solana.com",
        "https://api.devnet.solana.com",
        "https://api.testnet.solana.com",
        "https://mainnet.helius-rpc.com/?api-key=abc",
        "https://sender.helius-rpc.com",
        "https://my-app.helius-rpc.com",
        "https://my-app.helius.xyz",
        "https://x402.quicknode.com",
        "https://acme-clinic.solana-mainnet.quiknode.pro/abc/",
        "https://my-node.triton.one",
        "https://solana-mainnet.g.alchemy.com",
        "https://rpc.ankr.com/solana",
    ],
)
def test_allowlisted_urls_pass(url):
    assert is_allowlisted(url), f"Expected {url} on allowlist"


@pytest.mark.parametrize(
    "url",
    [
        "https://my-shady-rpc.example.com",
        "https://my-custom-rpc.io",
        "https://ssc-dao.genesysgo.net",  # deprecated 2024 — must NOT pass
        "https://infura.io/solana",
    ],
)
def test_random_urls_fail(url):
    assert not is_allowlisted(url), f"Expected {url} NOT on allowlist"


def test_find_rpc_urls_extracts_solana_only_urls():
    text = """
    const a = "https://github.com/example/repo";
    const b = "https://my-shady-rpc.example.com";
    const c = "https://mainnet.helius-rpc.com/?api-key=xyz";
    """
    found = find_rpc_urls(text)
    urls = {u for u, _ in found}
    # github filtered out (no solana keyword in host).
    assert all("github.com" not in u for u in urls)
    # The shady "rpc" host must be detected because it contains "rpc".
    assert any("my-shady-rpc.example.com" in u for u in urls)
    assert any("helius-rpc.com" in u for u in urls)


def test_find_rpc_urls_returns_line_numbers():
    text = "first\nhttps://mainnet.helius-rpc.com\nthird"
    found = find_rpc_urls(text)
    assert any(ln == 2 for _, ln in found)

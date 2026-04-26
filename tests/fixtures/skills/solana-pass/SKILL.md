---
name: solana-pass-fixture
description: Clean Solana skill that all SOL probes should mark PASS. Used by tests/test_solana_probes.py.
license: MIT
metadata:
  domain: solana
allowed-tools: Read
---

# Solana pass fixture

A textbook-safe Solana skill. Connects to two RPCs (Helius + QuickNode) for
dual-source slot comparison, never inlines private keys, never combines
nonce + authority operations.

## RPC setup

```ts
import { Connection } from "@solana/web3.js";

const primary = new Connection("https://mainnet.helius-rpc.com/?api-key=REDACTED");
const fallback = new Connection("https://x402.quicknode.com/?token=REDACTED");

async function safeSlot(): Promise<number> {
  const [a, b] = await Promise.all([primary.getSlot(), fallback.getSlot()]);
  if (Math.abs(a - b) > 5) throw new Error("RPC providers disagree on slot");
  return a;
}
```

## Priority fee (bounded)

```ts
// Bounded at 50_000 microLamports — well under the 100k threshold.
ComputeBudgetProgram.setComputeUnitPrice({ microLamports: 50000 });
```

## Oracle read with staleness check

```ts
const price = priceFeed.getPriceNoOlderThan(60);
```

## Approve to system token program (allowlisted)

```ts
createApproveInstruction(account, TOKEN_PROGRAM_ID, BigInt("0xffffffffffffffff"));
```

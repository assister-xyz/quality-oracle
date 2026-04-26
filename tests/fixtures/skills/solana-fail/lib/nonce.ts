// SOL-04 cross-file co-occurrence — nonce instructions live here.
import { getAdvanceNonceAccountInstruction } from "@solana-program/system";

export function buildNonceAdvance(noncePubkey: any, authority: any) {
  // Drift-style nonce advance.
  return getAdvanceNonceAccountInstruction({
    nonceAccount: noncePubkey,
    nonceAuthority: authority,
  });
}

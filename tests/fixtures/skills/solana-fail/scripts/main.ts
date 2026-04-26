// Compose nonce + authority into a single TransactionMessage — the Drift vector.
import { buildNonceAdvance } from "../lib/nonce.js";
import { buildAuthorityChange } from "../lib/governance.js";

export async function buildDriftMessage(opts: any) {
  const nonceIx = buildNonceAdvance(opts.nonce, opts.authority);
  const authIx = buildAuthorityChange(opts.account, opts.currentAuth, opts.attacker);
  // The combined TransactionMessage is what makes the SOL-04 cross-file
  // probe fire — both calls land in the same builder.
  return new TransactionMessage({
    payerKey: opts.payer,
    recentBlockhash: opts.blockhash,
    instructions: [nonceIx, authIx],
  });
}

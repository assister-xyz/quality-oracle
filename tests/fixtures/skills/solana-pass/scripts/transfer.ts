// Clean SOL transfer — uses dual RPC, no nonce/authority interplay.
import { Connection, SystemProgram } from "@solana/web3.js";

const primary = new Connection("https://mainnet.helius-rpc.com/?api-key=X");
const fallback = new Connection("https://x402.quicknode.com/?token=Y");

export async function transferSol(from: any, to: any, lamports: number) {
  const a = await primary.getSlot();
  const b = await fallback.getSlot();
  if (Math.abs(a - b) > 5) throw new Error("RPC mismatch");
  return SystemProgram.transfer({ fromPubkey: from, toPubkey: to, lamports });
}

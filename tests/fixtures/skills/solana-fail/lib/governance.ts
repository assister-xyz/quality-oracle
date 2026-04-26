// SOL-04 cross-file co-occurrence — authority instructions live here.
import { createSetAuthorityInstruction } from "@solana/spl-token";

export function buildAuthorityChange(account: any, currentAuthority: any, newAuthority: any) {
  return createSetAuthorityInstruction(account, currentAuthority, 0, newAuthority);
}

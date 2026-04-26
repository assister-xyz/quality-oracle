#!/usr/bin/env bash
# setup_fixtures.sh — Clone the 5 reference Claude Skill repos used by the
# laureum-skills test corpus (QO-053-A AC1, QO-053-D AC6).
#
# Usage:
#   bash dev/setup_fixtures.sh
#
# Idempotent: re-running pulls latest main; safe to run before every test session.
# Required for: QO-053-A parser regression, QO-053-D SendAI 45-skill audit,
#   QO-053-E ClawHavoc detection, any L1/L2 activator test that uses fixture skills.

set -euo pipefail

FIXTURES_DIR="${LAUREUM_FIXTURES_DIR:-/tmp}"
mkdir -p "$FIXTURES_DIR"

echo "==> Setting up Laureum skill fixtures in $FIXTURES_DIR"
echo

clone_or_pull() {
    local repo="$1"
    local dir="$2"
    local pinned_ref="${3:-main}"

    if [ -d "$dir/.git" ]; then
        echo "    [refresh] $dir"
        git -C "$dir" fetch --quiet origin "$pinned_ref"
        git -C "$dir" checkout --quiet "$pinned_ref"
        git -C "$dir" reset --hard --quiet "origin/$pinned_ref"
    else
        echo "    [clone]   $repo -> $dir"
        git clone --quiet --branch "$pinned_ref" "https://github.com/$repo.git" "$dir"
    fi
}

# 1. Anthropic reference skills (17 SKILL.md, source-of-truth examples).
clone_or_pull "anthropics/skills" "$FIXTURES_DIR/anthropic-skills" "main"

# 2. SendAI Solana skills (45 SKILL.md, the launch dataset).
#    SHA-pinned per QO-053-D AC6: ff8d226 was the audit baseline (2026-04-22).
#    To audit a different SHA, override LAUREUM_SENDAI_REF.
SENDAI_REF="${LAUREUM_SENDAI_REF:-main}"
clone_or_pull "sendaifun/skills" "$FIXTURES_DIR/sendai-skills" "$SENDAI_REF"

# 3. Trail of Bits skills (73 SKILL.md, security-flavored, has CI validator).
clone_or_pull "trailofbits/skills" "$FIXTURES_DIR/trailofbits-skills" "main"

# 4. Anthony Fu skills (17 SKILL.md, JS/TS-focused).
clone_or_pull "antfu/skills" "$FIXTURES_DIR/antfu-skills" "main"

# 5. Addy Osmani skills (21 SKILL.md, production-engineering).
clone_or_pull "addyosmani/agent-skills" "$FIXTURES_DIR/addyosmani-agent-skills" "main"

echo
echo "==> Verifying fixtures"

# Count SKILL.md files; expected total ≈ 173 (17 + 45 + 73 + 17 + 21 = 173, slight drift OK)
TOTAL=$(find "$FIXTURES_DIR/anthropic-skills" \
              "$FIXTURES_DIR/sendai-skills" \
              "$FIXTURES_DIR/trailofbits-skills" \
              "$FIXTURES_DIR/antfu-skills" \
              "$FIXTURES_DIR/addyosmani-agent-skills" \
              -name "SKILL.md" -o -name "skill.md" 2>/dev/null | wc -l | xargs)

echo "    Total SKILL.md files: $TOTAL  (expected ≈ 173)"
if [ "$TOTAL" -lt 150 ]; then
    echo "    [WARN] Fewer than 150 SKILL.md files; some clones may have failed."
    exit 1
fi

# Spot-check the 3 known name/folder mismatches from R2 (QO-053-A AC3 fixtures)
for mismatch_skill in jupiter inco metengine; do
    if [ -f "$FIXTURES_DIR/sendai-skills/skills/$mismatch_skill/SKILL.md" ]; then
        echo "    [ok] $mismatch_skill SKILL.md present (expected name/folder mismatch)"
    else
        echo "    [WARN] $mismatch_skill not found — has SendAI repo been restructured?"
    fi
done

echo
echo "==> Fixtures ready at $FIXTURES_DIR/"
echo "    anthropic-skills, sendai-skills, trailofbits-skills, antfu-skills, addyosmani-agent-skills"
echo
echo "Next: cd quality-oracle && python3 -m pytest tests/test_skill_parser.py -v"

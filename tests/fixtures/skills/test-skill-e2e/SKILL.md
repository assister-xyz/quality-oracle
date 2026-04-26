---
name: test-skill-e2e
description: Minimal well-formed skill fixture used by tests/e2e/test_full_pipeline.py. Benign content so all Phase-0 probes pass.
license: MIT
metadata:
  domain: general
allowed-tools: Read
---

# Test Skill E2E

A minimal skill body used by the Phase 2 E2E suite. The body is intentionally
short so live activator calls stay well under the free-tier token budget.

## What it does

When invoked, it should reply with a one-sentence confirmation that the
skill loaded successfully. No tool use, no external calls.

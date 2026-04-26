"""Backwards-compat shim — moved to `src/standards/aqvc.py` (QO-053-I).

This module forwards every public symbol so existing imports keep working
for one release. Will be removed in QO-053-I.1 follow-up.
"""
from src.standards.aqvc import *  # noqa: F401,F403  backwards-compat; remove in QO-053-I.1
from src.standards.aqvc import (  # noqa: F401  explicit re-exports for the names we historically expose
    create_vc,
    verify_vc,
    build_did_document,
    encode_public_key_multibase,
    decode_public_key_multibase,
)

"""Activator factory used by the /v1/evaluate/skill route.

Returns a callable that produces an activator instance for the requested
``EvalLevel``. Defaults to Cerebras free-tier (``cerebras:llama3.1-8b``)
unless the deployment has explicitly opted into Anthropic via
``LAUREUM_ACTIVATION_PROVIDER=anthropic``.

The route owns lifecycle (one factory call per question turn). Heavy
clients are reused across calls because activator construction is cheap
but provider client construction can hold network sockets.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from src.config import settings
from src.core.model_resolver import resolve, ResolvedModel
from src.storage.models import EvalLevel, ParsedSkill

logger = logging.getLogger(__name__)


def _build_provider_client(provider: str) -> Any:
    """Construct a provider client per the activation_provider setting.

    Returns ``None`` when the provider's SDK or API key isn't available —
    callers that get ``None`` should drop to L1 (no activation) rather
    than fail the eval.
    """
    if provider == "cerebras":
        # Cerebras uses comma-separated key rotation pool — pick first.
        keys_str = settings.cerebras_api_keys or os.getenv("CEREBRAS_API_KEY", "")
        first = next((k.strip() for k in keys_str.split(",") if k.strip()), "")
        if not first:
            logger.warning("No Cerebras API key configured; cannot build L2 activator")
            return None
        try:
            from cerebras.cloud.sdk import Cerebras
            return Cerebras(api_key=first)
        except ImportError:
            logger.warning("cerebras-cloud-sdk not installed")
            return None

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("No Anthropic API key configured")
            return None
        try:
            from anthropic import AsyncAnthropic
            return AsyncAnthropic(api_key=api_key)
        except ImportError:
            logger.warning("anthropic SDK not installed")
            return None

    if provider == "groq":
        keys_str = os.getenv("GROQ_API_KEY", "")
        first = next((k.strip() for k in keys_str.split(",") if k.strip()), "")
        if not first:
            return None
        try:
            from groq import AsyncGroq
            return AsyncGroq(api_key=first)
        except ImportError:
            return None

    logger.warning("Unknown activation provider: %s", provider)
    return None


def make_activator_factory(
    parsed: ParsedSkill,
    skill_dir: Path,
    level: EvalLevel,
) -> Optional[Callable[[], Any]]:
    """Return a no-arg callable that builds a fresh activator per call.

    ``None`` when the deployment can't satisfy the requested level — caller
    should fall back to L1 manifest-only evaluation.

    AC1: L1 always works (no provider client needed).
    AC2: L2 requires a provider client; returns None if not configured.
    AC3: L3 ships separately via QO-059 Docker harness.
    """
    activation_model = settings.laureum_activation_model
    try:
        resolved: ResolvedModel = resolve(activation_model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to resolve activation model %s: %s", activation_model, exc)
        return None

    if level == EvalLevel.MANIFEST:
        # L1 — naive activator works without a client (or with cheapest).
        client = _build_provider_client(resolved.provider)
        if client is None:
            return None

        def _l1_factory():
            from src.core.skill_activator import L1NaiveActivator
            return L1NaiveActivator(
                skill=parsed,
                resolved=resolved,
                provider_client=client,
                temperature=settings.laureum_activation_temp,
                max_tokens=settings.laureum_activation_max_tokens,
            )
        return _l1_factory

    if level == EvalLevel.FUNCTIONAL:
        client = _build_provider_client(resolved.provider)
        if client is None:
            logger.info("L2 requires provider client; falling back to L1 path")
            return None

        def _l2_factory():
            from src.core.skill_activator import L2ToolUseActivator
            from src.core.mock_filesystem import MockFileSystem
            mock_fs = MockFileSystem(skill_dir)
            return L2ToolUseActivator(
                parsed,
                resolved,
                client,
                mock_fs,
                temperature=settings.laureum_activation_temp,
                max_tokens=settings.laureum_activation_max_tokens,
            )
        return _l2_factory

    if level == EvalLevel.DOMAIN_EXPERT:
        # L3 lazy-imported via skill_activator.__getattr__ → l3_activator
        try:
            from src.core.skill_activator import L3ClaudeCodeActivator  # noqa: F401
        except (ImportError, AttributeError):
            logger.warning("L3 activator unavailable; check QO-059 Docker images")
            return None

        def _l3_factory():
            from src.core.skill_activator import L3ClaudeCodeActivator as _L3
            return _L3(skill=parsed, skill_dir=skill_dir)
        return _l3_factory

    return None

"""Provider:model alias resolution for skill activation (QO-053-B).

Parses the ``provider:model`` shape stored in
:attr:`Settings.laureum_activation_model` (default ``"cerebras:llama3.1-8b"``)
and resolves the alias to a dated snapshot for AQVC reproducibility (AC4).

Cerebras and Groq aliases are *fixed* — the API key's effective serving date
IS the dated snapshot, so we record the alias verbatim. Anthropic aliases are
resolved at startup via ``GET /v1/models`` to pin the current dated suffix
(e.g. ``claude-sonnet-4-5`` → ``claude-sonnet-4-5-20250929``).

Persistence target is the ``quality__model_versions`` MongoDB collection — a
single ``(provider, alias)`` row that is overwritten in place when a re-resolve
runs. Activators read the cached row at construction time so the floating
alias never reaches an actual API call.

The resolver is intentionally network-light: failures are non-fatal and the
caller receives the unresolved alias paired with a ``parse_warnings`` style
note, leaving live integration tests to ``@pytest.mark.live``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Spec valid providers per AC10 — any other prefix raises at parse time.
VALID_PROVIDERS = ("cerebras", "groq", "anthropic")

# A Sonnet bare alias looks like ``claude-sonnet-4-5`` — Anthropic dates this
# to ``-YYYYMMDD`` at /v1/models. We still accept already-dated strings so that
# tests can pin a snapshot without going through resolve().
_DATED_SUFFIX = re.compile(r"-\d{8}$")


@dataclass(frozen=True)
class ResolvedModel:
    """Result of parsing + resolving the activation model setting."""
    provider: str  # cerebras | groq | anthropic
    alias: str  # bare alias (or already-dated snapshot)
    dated_snapshot: str  # what we send to the API
    source: str  # ``fixed`` | ``list_models`` | ``cached``


class InvalidActivationModel(ValueError):
    """Raised when ``laureum_activation_model`` cannot be parsed."""


def parse_provider_model(value: str) -> Tuple[str, str]:
    """Split ``provider:model`` into ``(provider, model)``.

    Raises
    ------
    InvalidActivationModel
        On missing colon, unknown provider, or empty model component.

    Examples
    --------
    >>> parse_provider_model("cerebras:llama3.1-8b")
    ('cerebras', 'llama3.1-8b')
    >>> parse_provider_model("anthropic:claude-sonnet-4-5")
    ('anthropic', 'claude-sonnet-4-5')
    """
    if not value or ":" not in value:
        raise InvalidActivationModel(
            f"Activation model must use 'provider:model' format, got: {value!r}"
        )
    provider, _, model = value.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in VALID_PROVIDERS:
        raise InvalidActivationModel(
            f"Unknown provider {provider!r}; expected one of {VALID_PROVIDERS}"
        )
    if not model:
        raise InvalidActivationModel("Model component is empty")
    return provider, model


def is_dated(snapshot: str) -> bool:
    """Return True if ``snapshot`` carries a ``-YYYYMMDD`` Anthropic suffix."""
    return bool(_DATED_SUFFIX.search(snapshot))


def resolve_fixed(provider: str, alias: str) -> ResolvedModel:
    """Resolve a ``cerebras`` / ``groq`` alias by fiat (no /v1/models lookup).

    Per AC4: the API key's effective model serving date IS the dated snapshot
    for these providers, so we record the alias verbatim with ``source=fixed``.
    """
    return ResolvedModel(
        provider=provider,
        alias=alias,
        dated_snapshot=alias,
        source="fixed",
    )


def _list_anthropic_models(api_key: str) -> list[dict]:
    """Call ``GET https://api.anthropic.com/v1/models`` synchronously.

    Synchronous on purpose — this is a startup hook called once per process,
    not a hot-path function. Returning ``[]`` on any failure lets the caller
    fall back to the unresolved alias rather than crashing the boot.
    """
    try:
        import anthropic  # local import — keeps test imports cheap
        client = anthropic.Anthropic(api_key=api_key)
        page = client.models.list()
        return [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in page.data]
    except Exception as e:  # pragma: no cover - network path
        logger.warning("Anthropic /v1/models lookup failed: %s", e)
        return []


def resolve_anthropic(alias: str, api_key: Optional[str]) -> ResolvedModel:
    """Resolve an Anthropic alias to its current dated snapshot.

    Falls back to the bare alias when:
    - no API key is configured (system runs default Cerebras tier)
    - the /v1/models call fails
    - no models match the alias prefix
    """
    if is_dated(alias):
        return ResolvedModel(
            provider="anthropic",
            alias=alias,
            dated_snapshot=alias,
            source="fixed",
        )
    if not api_key:
        logger.info(
            "Anthropic alias %s left unresolved (no ANTHROPIC_API_KEY). "
            "Set the env var to pin a dated snapshot.", alias,
        )
        return ResolvedModel(
            provider="anthropic",
            alias=alias,
            dated_snapshot=alias,
            source="fixed",
        )
    models = _list_anthropic_models(api_key)
    candidates = [
        m for m in models
        if str(m.get("id", "")).startswith(alias) and is_dated(str(m.get("id", "")))
    ]
    if not candidates:
        logger.warning(
            "No dated Anthropic snapshot matched alias %s — using bare alias.",
            alias,
        )
        return ResolvedModel(
            provider="anthropic",
            alias=alias,
            dated_snapshot=alias,
            source="fixed",
        )
    # Take the freshest dated snapshot (lex sort works for YYYYMMDD).
    best = max(candidates, key=lambda m: str(m.get("id", "")))
    return ResolvedModel(
        provider="anthropic",
        alias=alias,
        dated_snapshot=str(best.get("id", alias)),
        source="list_models",
    )


def resolve(
    activation_model: str,
    anthropic_api_key: Optional[str] = None,
) -> ResolvedModel:
    """Parse + resolve. Use this once at process startup."""
    provider, alias = parse_provider_model(activation_model)
    if provider == "anthropic":
        return resolve_anthropic(alias, anthropic_api_key)
    return resolve_fixed(provider, alias)


# ── MongoDB persistence (optional — caller decides) ──────────────────────────


def model_versions_col():
    """Lazy accessor for ``quality__model_versions`` collection."""
    from src.storage.mongodb import get_db
    return get_db().quality__model_versions


async def persist_resolution(resolved: ResolvedModel) -> None:
    """Upsert ``(provider, alias)`` → dated snapshot.

    Idempotent: re-running the resolver overwrites the row in place. Caller is
    expected to wrap any DB unavailability in try/except — this function
    does not silently swallow errors.
    """
    col = model_versions_col()
    await col.update_one(
        {"provider": resolved.provider, "alias": resolved.alias},
        {
            "$set": {
                "provider": resolved.provider,
                "alias": resolved.alias,
                "dated_snapshot": resolved.dated_snapshot,
                "source": resolved.source,
                "resolved_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


async def load_cached_resolution(provider: str, alias: str) -> Optional[ResolvedModel]:
    """Load the most recent persisted resolution for ``(provider, alias)``."""
    col = model_versions_col()
    row = await col.find_one({"provider": provider, "alias": alias})
    if not row:
        return None
    return ResolvedModel(
        provider=row["provider"],
        alias=row["alias"],
        dated_snapshot=row["dated_snapshot"],
        source="cached",
    )

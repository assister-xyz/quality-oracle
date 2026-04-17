"""
QO-055: Multi-step test flows.

Many MCP servers follow a list → detail pattern where the `detail` tool
requires an opaque ID you can only get by first calling the `list` tool.
Schema-only input generation can't satisfy that — the generator has no
way to invent Peek's `exp_abc123` or CoinGecko's `bitcoin-cash`.

This module provides two pure helpers and one runtime phase:

- `is_list_tool(tool)` — name/schema heuristics for detecting enumerating tools
- `extract_ids_from_response(content)` — parse JSON or text, pull id-shaped strings
- `discover_ids(session, tools)` — a small orchestrator that actually calls
   list-like tools on the live session and returns an `id → [values]` pool.

The pool is then fed to `test_generator._generate_sample_input` so dependent
tools receive real IDs instead of semantic fallbacks like `"abc123"`.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tool-name prefixes that almost always enumerate a collection.
_LIST_PREFIXES = ("list_", "search_", "get_all_", "fetch_all_", "find_", "query_")
# Tool-name suffixes that look enumerating ("get_coins", "list_experiences").
_LIST_SUFFIXES = ("_list", "_all")
# Description keywords suggesting enumeration when no required params exist.
_LIST_DESC_HINTS = ("list", "all", "enumerate", "catalog", "search", "index")

# Parameter name patterns that look like IDs we want to fill from discovery.
_ID_PARAM_NAMES = (
    "id",
    "identifier",
    "uuid",
    "slug",
    "key",
)


def is_list_tool(tool: dict) -> bool:
    """Heuristic: does this tool list/search resources (and so might
    yield real IDs we can reuse)?

    Conservative — we'd rather miss a list tool than trigger discovery
    on a detail tool that does real work. The required-param check runs
    first because a listy name like `list_experiences` is irrelevant if
    the schema requires an `experience_id`; that's really a detail tool.
    """
    schema = tool.get("inputSchema") or {}
    required = schema.get("required") or []
    # If a tool requires an id-like param, it's a detail tool — not a list.
    if any(_looks_like_id_name(r) for r in required):
        return False

    name = (tool.get("name") or "").lower()
    if any(name.startswith(p) for p in _LIST_PREFIXES):
        return True
    if any(name.endswith(s) for s in _LIST_SUFFIXES):
        return True

    # Tool with no required params and a listy description.
    desc = (tool.get("description") or "").lower()
    if not required and any(hint in desc for hint in _LIST_DESC_HINTS):
        return True

    return False


def _looks_like_id_name(name: str) -> bool:
    """Return True if `name` matches an ID-style parameter convention."""
    n = (name or "").lower()
    if n in _ID_PARAM_NAMES:
        return True
    # "coin_id", "experience_id", "session_uuid", "repo_slug"
    for suffix in _ID_PARAM_NAMES:
        if n.endswith("_" + suffix):
            return True
    return False


def _extract_from_json(
    obj: Any, max_depth: int = 4, cap: int = 50, _out: Optional[List[str]] = None
) -> List[str]:
    """Walk a JSON structure collecting values of keys that look like IDs.

    Recursive (not stack-based) so results come out in document order —
    tests depend on that. Cap both depth and number of items scanned so
    a pathological response can't blow the budget. Integer IDs are
    stringified (GitHub repo IDs, YouTube video indexes, etc.).
    """
    out = _out if _out is not None else []
    if max_depth <= 0 or len(out) >= cap:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if len(out) >= cap:
                break
            if (
                _looks_like_id_name(k)
                and isinstance(v, (str, int))
                and not isinstance(v, bool)
            ):
                sv = str(v)
                # Accept short ids (e.g. "7") — filtering is applied at the
                # outer public function after dedup.
                if 0 < len(sv) < 100:
                    out.append(sv)
            elif isinstance(v, (dict, list)):
                _extract_from_json(v, max_depth - 1, cap, out)
    elif isinstance(obj, list):
        # Scan at most 25 list items at each level — prevents runaway on
        # huge responses while still sampling enough for diversity.
        for item in obj[:25]:
            if len(out) >= cap:
                break
            _extract_from_json(item, max_depth - 1, cap, out)
    return out


# Matches a standard UUID; used as a regex-fallback when JSON parsing fails
# or the response is plain text.
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# Matches "ID: token", "(ID: token)", "id = token" — common in text responses
# that are meant to be human-readable (Peek.com's list_tags is a good example:
# "• Adventures (ID: tag0zw)"). Requires at least two characters in the token
# to avoid picking up noise like "ID: a" from sentences.
_INLINE_ID_RE = re.compile(
    r"\b(?:ID|Id|id)\s*[:=]\s*['\"]?([A-Za-z0-9][A-Za-z0-9_\-]{1,63})['\"]?",
)


def extract_ids_from_response(content: str, max_ids: int = 20) -> List[str]:
    """Pull up to `max_ids` id-shaped strings from a tool response.

    Priority:
      1. JSON parse → walk keys named `id`, `*_id`, `uuid`, `slug`, `key`
      2. UUID regex fallback for free-text responses

    Deduplicated, preserves first-seen order, and filters out degenerate
    values (empty, whitespace-only, too long to be an id).
    """
    if not content:
        return []

    collected: List[str] = []

    # Step 1 — JSON path (most reliable)
    try:
        parsed = json.loads(content)
        collected.extend(_extract_from_json(parsed))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Step 2 — UUID regex fallback for plain text / partial JSON
    if len(collected) < max_ids:
        collected.extend(_UUID_RE.findall(content))

    # Step 3 — "(ID: xxx)" / "ID: xxx" pattern — common in markdown-style
    # human-readable list responses (Peek.com list_tags is the canonical case).
    if len(collected) < max_ids:
        collected.extend(_INLINE_ID_RE.findall(content))

    seen: set = set()
    out: List[str] = []
    for val in collected:
        sv = str(val).strip()
        if not sv or sv in seen:
            continue
        if len(sv) > 100:
            continue
        seen.add(sv)
        out.append(sv)
        if len(out) >= max_ids:
            break
    return out


async def discover_ids(
    session,
    tools: List[dict],
    max_tools_to_probe: int = 3,
    per_call_timeout: int = 10,
) -> Dict[str, List[str]]:
    """Run list-like tools on an open MCP session and harvest IDs.

    Returns a dict mapping `id-ish parameter name → [values]`. Most
    servers only need a single key (`id`), but we also index under
    `<resource>_id` so tools with `coin_id` can specifically prefer
    IDs harvested from a coin-listing tool.

    Strictly bounded: at most `max_tools_to_probe` list tools are
    actually called, each with its own short timeout. Discovery
    failures are logged at debug and never raise — they simply yield
    an empty pool and the regular priority chain takes over.
    """
    from src.core.mcp_client import _call_tool_in_session
    from src.core.test_generator import _generate_sample_input

    discovered: Dict[str, List[str]] = {}
    probes_used = 0

    for tool in tools:
        if probes_used >= max_tools_to_probe:
            break
        if not is_list_tool(tool):
            continue

        name = tool.get("name", "")
        schema = tool.get("inputSchema") or {}
        try:
            # Use variation=0 for deterministic discovery inputs.
            args = _generate_sample_input(schema, variation=0)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[discovery] could not generate args for {name}: {e}")
            continue

        try:
            import asyncio as _asyncio
            response = await _asyncio.wait_for(
                _call_tool_in_session(session, name, args, time.time()),
                timeout=per_call_timeout,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 — CancelledError too
            logger.debug(f"[discovery] {name} raised {type(exc).__name__}: {exc}")
            continue

        if response.get("is_error"):
            logger.debug(f"[discovery] {name} returned is_error=True, skipping")
            continue

        ids = extract_ids_from_response(response.get("content", ""))
        if not ids:
            continue

        probes_used += 1
        # Store under a resource-specific key derived from the tool name so
        # `experience_details(id)` does NOT get tag IDs from `list_tags`.
        # We deliberately avoid a generic "id" pool: when a server has
        # multiple list-tools (tags + experiences), the generic bucket
        # cross-contaminates and dependent tools get the wrong resource
        # type. If the resource is ambiguous, falling back to the semantic
        # map is more honest than blindly using a wrong-type ID.
        resource = _resource_key_from_tool_name(name)
        if resource:
            discovered.setdefault(f"{resource}_id", []).extend(ids)
            # Also expose under the tool name itself — helps when a caller
            # has param `id` in a tool named `<resource>_details` and we
            # can match by tool-resource later on.
            discovered.setdefault(resource, []).extend(ids)

        logger.info(
            f"[discovery] {name} yielded {len(ids)} IDs → resource='{resource}'"
        )

    # Deduplicate each pool while preserving order.
    for key, values in discovered.items():
        seen: set = set()
        dedup: List[str] = []
        for v in values:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        discovered[key] = dedup

    return discovered


# Suffixes a *detail* tool might have — helps infer the resource name
# from `experience_details`, `coin_info`, `get_user`, etc.
_DETAIL_SUFFIXES = ("_details", "_detail", "_info", "_get", "_fetch")
# Prefixes a detail tool might have — `get_coin_price` → "coin_price".
_DETAIL_PREFIXES = ("get_", "fetch_", "read_", "show_")


def _resource_key_from_tool_name(name: str) -> Optional[str]:
    """Strip list/detail prefixes+suffixes off a tool name to get the resource.

    Examples (list-side):
      `list_coins`             → "coin"
      `get_all_repositories`   → "repository"
      `search_experiences`     → "experience"

    Examples (detail-side, used by the consumer to match back to a pool):
      `experience_details`     → "experience"
      `coin_info`              → "coin"
      `get_session_id`         → "session_id" (unchanged — caller will match)

    Returns None if we can't cleanly identify a resource.
    """
    n = (name or "").lower()

    # Order matters: list prefixes first (they're more specific than
    # detail prefixes like "get_"), then detail prefixes.
    for prefix in (*_LIST_PREFIXES, *_DETAIL_PREFIXES):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break

    # Similarly, list suffixes then detail suffixes.
    for suffix in (*_LIST_SUFFIXES, *_DETAIL_SUFFIXES):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break

    # Naive singularization — good enough for "coins" → "coin",
    # "experiences" → "experience", "repositories" → "repository".
    if n.endswith("ies") and len(n) > 3:
        return n[:-3] + "y"
    if n.endswith("ses") and len(n) > 3:
        return n[:-2]
    if n.endswith("s") and len(n) > 1:
        return n[:-1]
    return n or None

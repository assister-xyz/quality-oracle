"""Helper to POST a revalidate-path webhook to the Next.js demo (QO-053-H).

The QO-053-F batch runner imports :func:`notify_marketplace_revalidate` after
persisting fresh ``quality__skill_scores`` so ISR pages refresh on the next
request without waiting for the 24h timer.

Configuration is via :mod:`src.config` settings:
- ``frontend_revalidate_base``  e.g. ``https://laureum.ai``
- ``frontend_revalidate_secret`` shared secret for the revalidate endpoint

Both fields default empty → callers are no-ops in env without them set.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


async def notify_marketplace_revalidate(
    slug: str,
    skill: Optional[str] = None,
    timeout_s: float = 5.0,
) -> bool:
    """Fire-and-forget revalidate ping for the marketplace listing or detail.

    Returns ``True`` if the webhook returned 2xx; ``False`` otherwise.
    Failure is logged but never raised — eval persistence must succeed even
    when the frontend webhook is unreachable.
    """
    base = (settings.frontend_revalidate_base or "").rstrip("/")
    secret = settings.frontend_revalidate_secret or ""
    if not base or not secret:
        logger.debug("marketplace revalidate skipped — base/secret unset")
        return False

    path = f"/marketplace/{slug}"
    if skill:
        path = f"{path}/{skill}"

    url = f"{base}/api/revalidate"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                url,
                params={"path": path, "secret": secret},
            )
            if resp.status_code >= 400:
                logger.warning(
                    "marketplace revalidate failed (%s) %s -> %s",
                    resp.status_code,
                    path,
                    resp.text[:200],
                )
                return False
            return True
    except Exception as exc:
        logger.warning("marketplace revalidate request errored: %s", exc)
        return False

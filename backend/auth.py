# backend/auth.py
"""
Simple API-key authentication enforced via a FastAPI dependency.

Configure one or more accepted keys with the ``API_KEYS`` env var
(comma-separated) or a single ``API_KEY``. Clients send the key in the
``X-API-Key`` header. When no keys are configured, authentication is
disabled (development mode) and a warning is logged once.
"""
from __future__ import annotations

import logging
import os

from fastapi import Header, HTTPException

logger = logging.getLogger("uvicorn.error")
_warned = False


def _configured_keys() -> set[str]:
    raw = os.getenv("API_KEYS") or os.getenv("API_KEY") or ""
    return {k.strip() for k in raw.split(",") if k.strip()}


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: reject requests without a valid ``X-API-Key``."""
    global _warned
    keys = _configured_keys()
    if not keys:
        if not _warned:
            logger.warning(
                "API_KEYS/API_KEY not set — X-API-Key authentication is DISABLED."
            )
            _warned = True
        return
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")

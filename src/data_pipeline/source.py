"""source.py

Which price source a strategy should read, and the staleness it reports.

``PRICE_SOURCE`` selects the backend:

* ``yfinance`` -- per-run download. The pre-lake behaviour, and still the
  default while the lake proves itself in CI.
* ``lake``     -- read ``data/prices/daily`` through ``store``.

There is deliberately no automatic fallback from lake to yfinance. A silent
fallback would mean a strategy reasoning over a *different window* than intended
while reporting success, which is the failure this pipeline exists to remove.
Lake or skip.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

YFINANCE = "yfinance"
LAKE = "lake"

#: Default stays yfinance until the lake has demonstrated several consecutive
#: sessions of current bars in CI. Flipping this is a one-line change.
DEFAULT_SOURCE = YFINANCE

ENV_VAR = "PRICE_SOURCE"

#: Warn once the newest session is older than this many calendar days. Four
#: covers a normal weekend plus a holiday without crying wolf.
STALE_AFTER_DAYS = 4


def active_source() -> str:
    """Return the configured source, falling back to the default if unset."""
    raw = (os.getenv(ENV_VAR) or DEFAULT_SOURCE).strip().lower()
    if raw not in {YFINANCE, LAKE}:
        logger.warning(
            "%s=%r is not recognised; using %r", ENV_VAR, raw, DEFAULT_SOURCE
        )
        return DEFAULT_SOURCE
    return raw


def using_lake() -> bool:
    return active_source() == LAKE


def log_staleness(last_session: Optional[pd.Timestamp], *, context: str) -> int:
    """Log how old the newest bar is. Returns the age in calendar days.

    An unattended run should say out loud how stale its inputs are rather than
    trading quietly on old data. Staleness alone is not a reason to skip -- an
    uncovered lookback is.
    """
    if last_session is None:
        logger.error("PRICE_LAKE_EMPTY (%s): no bars available", context)
        return -1

    age = int((pd.Timestamp.now().normalize() - pd.Timestamp(last_session).normalize()).days)
    if age > STALE_AFTER_DAYS:
        logger.warning(
            "PRICE_LAKE_STALE (%s): newest session %s is %d day(s) old",
            context, pd.Timestamp(last_session).date(), age,
        )
    else:
        logger.info(
            "price lake current (%s): newest session %s (%d day(s) old)",
            context, pd.Timestamp(last_session).date(), age,
        )
    return age

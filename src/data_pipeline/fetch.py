"""fetch.py

The facade the pipeline calls to download bars. It used to *be* the Yahoo client;
the vendor code now lives in ``data_pipeline.providers`` behind one contract, and
this module only routes to whichever provider is canonical.

Isolated from ``store`` on purpose: strategies read the lake and can never
trigger a download as a side effect. ``sync_prices``, the rebuild and the pilot
are the only callers.

What stays true regardless of provider:

1. **No bar for an incomplete session.** Each provider enforces its own cutoff
   (see ``providers.base``). Yahoo excludes the current date outright; Alpaca
   admits it only after 20:00 ET, when the session is final.
2. **No fallback.** ``fetch_batch`` uses exactly the provider it is given, or the
   canonical one. If that provider raises, the error surfaces. Nothing here tries
   another source, because a silent substitution is how a second provider's rows
   end up interleaved with the first's under the same keys.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline import providers
from data_pipeline.providers.base import MarketDataProvider
from data_pipeline.providers.yahoo import extract_symbol_frame as _extract_symbol_frame  # noqa: F401

logger = logging.getLogger(__name__)

#: Symbols per request. Large enough that the multi-symbol endpoint does the work,
#: small enough that one bad ticker's blast radius stays bounded.
DEFAULT_BATCH_SIZE = 100


def today_naive() -> pd.Timestamp:
    """Local calendar date, midnight, tz-naive.

    Retained for callers that want the machine's date. Anything deciding whether
    a bar may be stored should use ``provider.session_cutoff()`` instead, which is
    computed in New York time.
    """
    return pd.Timestamp.now().normalize()


def fetch_batch(
    symbols: Sequence[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    *,
    provider: Optional[MarketDataProvider] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Download one batch from ``provider`` (default: the canonical provider)."""
    source = provider or providers.canonical_provider()
    return source.fetch_batch(symbols, start, end_exclusive)


def batched(symbols: Sequence[str], size: int = DEFAULT_BATCH_SIZE) -> List[List[str]]:
    """Split a symbol list into fetch batches."""
    items = list(symbols)
    if size <= 0:
        return [items] if items else []
    return [items[i : i + size] for i in range(0, len(items), size)]

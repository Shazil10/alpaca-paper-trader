"""fetch.py

The only module in the pipeline that talks to the network.

Isolated from ``store`` on purpose: strategies read the lake and can never
trigger a download as a side effect. ``sync_prices`` and ``rebuild_prices`` are
the sole callers.

Two invariants enforced here:

1. ``auto_adjust=False`` so both the unadjusted prints and ``Adj Close`` are
   captured. Verified equal to ``auto_adjust=True``'s ``Close`` within float32
   rounding -- see data/prices/_schema.md contract 1.
2. **No bar for the current date.** The trading workflow runs at 09:30 ET, so
   "today" has no close yet and yfinance may hand back a partial intraday bar.
   Storing that as a daily OHLCV row would persist a 09:35 quote as a session
   close. Dropped unconditionally -- contract 2.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline import schema
from data_pipeline.registry import normalize_symbol

logger = logging.getLogger(__name__)

#: yfinance degrades on very large ticker lists; this keeps batches survivable
#: and gives the checkpoint useful granularity.
DEFAULT_BATCH_SIZE = 200

_FIELD_MAP = {
    "Open": schema.OPEN,
    "High": schema.HIGH,
    "Low": schema.LOW,
    "Close": schema.CLOSE,
    "Adj Close": schema.ADJ_CLOSE,
    "Volume": schema.VOLUME,
}


def today_naive() -> pd.Timestamp:
    """Local calendar date, midnight, tz-naive."""
    return pd.Timestamp.now().normalize()


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """Pull one ticker's OHLCV out of a yfinance response.

    Handles both shapes: MultiIndex columns for multi-ticker downloads, flat
    columns when only one ticker came back.
    """
    if raw is None or len(raw) == 0:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if symbol not in level0:
            return None
        sub = raw[symbol]
    else:
        sub = raw

    if not isinstance(sub, pd.DataFrame) or len(sub) == 0:
        return None

    present = {src: dst for src, dst in _FIELD_MAP.items() if src in sub.columns}
    if schema.CLOSE not in present.values():
        return None

    out = sub[list(present.keys())].rename(columns=present).copy()

    # Older yfinance builds omit 'Adj Close' when adjustment is unavailable.
    # Falling back to the raw close keeps the column populated and honest: it is
    # the best adjusted estimate we have for that row.
    if schema.ADJ_CLOSE not in out.columns:
        out[schema.ADJ_CLOSE] = out[schema.CLOSE]

    for col in (schema.OPEN, schema.HIGH, schema.LOW, schema.VOLUME):
        if col not in out.columns:
            out[col] = pd.NA

    out = out.reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    out = out.rename(columns={date_col: schema.DATE})
    out[schema.SYMBOL] = symbol

    out = out.dropna(subset=[schema.CLOSE], how="any")
    if len(out) == 0:
        return None

    return out[schema.COLUMNS]


def fetch_batch(
    symbols: Sequence[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> Tuple[pd.DataFrame, List[str]]:
    """Download one batch. Returns (bars, failed_symbols).

    ``end_exclusive`` is passed straight to yfinance, whose daily ``end`` is
    exclusive. Rows dated on or after ``today`` are dropped regardless, so a
    partial session can never enter the lake.
    """
    import yfinance as yf  # local import: keeps the dependency off the read path

    wanted = [normalize_symbol(s) for s in symbols]
    wanted = sorted({s for s in wanted if s})
    if not wanted:
        return schema.empty_frame(), []

    try:
        raw = yf.download(
            wanted,
            start=start.strftime("%Y-%m-%d"),
            end=end_exclusive.strftime("%Y-%m-%d"),
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=True,
        )
    except Exception:
        logger.exception("batch download failed outright (%d symbols)", len(wanted))
        return schema.empty_frame(), wanted

    frames: List[pd.DataFrame] = []
    failed: List[str] = []

    for symbol in wanted:
        try:
            sub = _extract_symbol_frame(raw, symbol)
        except Exception:
            logger.exception("failed to parse response for %s", symbol)
            sub = None

        if sub is None or len(sub) == 0:
            failed.append(symbol)
            continue
        frames.append(sub)

    if not frames:
        return schema.empty_frame(), failed

    bars = schema.coerce(pd.concat(frames, ignore_index=True))

    # Contract 2: completed sessions only.
    cutoff = today_naive()
    partial = bars[bars[schema.DATE] >= cutoff]
    if len(partial) > 0:
        logger.info(
            "dropped %d partial/current-session row(s) dated >= %s",
            len(partial),
            cutoff.date(),
        )
        bars = bars[bars[schema.DATE] < cutoff]

    return schema.coerce(bars), failed


def batched(symbols: Sequence[str], size: int = DEFAULT_BATCH_SIZE) -> List[List[str]]:
    """Split a symbol list into fetch batches."""
    items = list(symbols)
    if size <= 0:
        return [items] if items else []
    return [items[i : i + size] for i in range(0, len(items), size)]

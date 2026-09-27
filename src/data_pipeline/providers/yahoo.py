"""Yahoo Finance via yfinance. Cross-check only.

This was the lake's only source until the Alpaca SIP migration. It remains for one
job: comparing against the canonical provider when a gap or a mismatch needs a
second opinion. It is **not canonical**, so ``sync_prices`` refuses to write its
output to the lake, and a reconciliation run writes it to quarantine instead.

The parsing below is unchanged from the old ``fetch.py``. Two properties of Yahoo
matter for anyone comparing it against Alpaca, and both are measured by the pilot
rather than assumed:

* ``auto_adjust=False`` returns ``Close`` **adjusted for splits but not for
  dividends**. It is not the raw print. The historical lake documented it as
  "unadjusted", which was true of dividends and false of splits, so Yahoo-era
  ``close`` values before a split are the post-split scale.
* ``Adj Close`` adjusts for both, back-anchored to the download date.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline import schema
from data_pipeline.providers.base import FetchReport, MarketDataProvider, strict_cutoff
from data_pipeline.registry import normalize_symbol

logger = logging.getLogger(__name__)

_FIELD_MAP = {
    "Open": schema.OPEN,
    "High": schema.HIGH,
    "Low": schema.LOW,
    "Close": schema.CLOSE,
    "Adj Close": schema.ADJ_CLOSE,
    "Volume": schema.VOLUME,
}


def extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
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


class YahooProvider(MarketDataProvider):
    """yfinance download, split-adjusted close plus Adj Close."""

    name = "yahoo"
    feed = "yfinance"
    adjustment = "close=split-adjusted, adj_close=split+dividend"
    canonical = False

    def __init__(self, now: Optional[datetime] = None) -> None:
        super().__init__()
        self._now = now

    def session_cutoff(self, now: Optional[datetime] = None) -> pd.Timestamp:
        # Strict: Yahoo is only ever called in the daytime and can return a
        # partial intraday bar for the current session.
        return strict_cutoff(now or self._now)

    def fetch_batch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, List[str]]:
        import yfinance as yf  # local import: keeps the dependency off the read path

        wanted = sorted({normalize_symbol(s) for s in symbols if str(s).strip()})
        report = FetchReport(provider=self.name, requested=len(wanted))
        self.last_report = report
        if not wanted:
            return schema.empty_frame(), []

        try:
            raw = yf.download(
                wanted,
                start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                end=pd.Timestamp(end_exclusive).strftime("%Y-%m-%d"),
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
                sub = extract_symbol_frame(raw, symbol)
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
        cutoff = self.session_cutoff()
        partial = bars[bars[schema.DATE] >= cutoff]
        if len(partial) > 0:
            report.dropped_after_cutoff = len(partial)
            logger.info(
                "dropped %d partial/current-session row(s) dated >= %s",
                len(partial), cutoff.date(),
            )
            bars = bars[bars[schema.DATE] < cutoff]

        bars = schema.coerce(bars)
        report.rows = len(bars)
        report.symbols_returned = int(bars[schema.SYMBOL].nunique()) if len(bars) else 0
        return bars, failed

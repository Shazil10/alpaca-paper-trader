"""Stooq free daily data. Cross-check only.

No API key, no adjustment metadata: ``close`` is whatever Stooq publishes and
``adj_close`` is set equal to it, which makes this useful for asking "did this
session trade, and roughly where" and useless as a source of returns across a
corporate action. Hence not canonical, and hence the manifest records
``adjustment="none"`` for anything it writes to quarantine.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline import schema
from data_pipeline.providers.base import FetchReport, MarketDataProvider
from data_pipeline.registry import normalize_symbol

logger = logging.getLogger(__name__)


def fetch_eod(
    ticker: str,
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily prices for one ticker. Empty frame on any failure."""
    stooq_ticker = ticker.replace("-", ".") + ".US"
    url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
    try:
        df = pd.read_csv(url, parse_dates=["Date"])
    except Exception as e:
        logger.warning("Stooq fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()

    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = df["Date"].dt.normalize()
    out["symbol"] = ticker.upper()
    for lake_col, stooq_col in [
        ("open", "Open"), ("high", "High"),
        ("low", "Low"), ("close", "Close"),
    ]:
        if stooq_col in df.columns:
            out[lake_col] = df[stooq_col].astype(float)
    out["adj_close"] = out["close"]
    if "Volume" in df.columns:
        out["volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("Int64")

    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.sort_values("date").reset_index(drop=True)


class StooqProvider(MarketDataProvider):
    """One HTTP request per symbol. Slow, keyless, unadjusted."""

    name = "stooq"
    feed = "stooq"
    adjustment = "none (adj_close == close)"
    canonical = False

    def fetch_batch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, List[str]]:
        wanted = sorted({normalize_symbol(s) for s in symbols if str(s).strip()})
        report = FetchReport(provider=self.name, requested=len(wanted))
        self.last_report = report

        cutoff = min(pd.Timestamp(end_exclusive), self.session_cutoff())
        last = (cutoff - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        frames, failed = [], []
        for symbol in wanted:
            frame = fetch_eod(symbol, pd.Timestamp(start).strftime("%Y-%m-%d"), last)
            if frame is None or frame.empty:
                failed.append(symbol)
                continue
            for column in schema.COLUMNS:
                if column not in frame.columns:
                    frame[column] = pd.NA
            frames.append(frame[schema.COLUMNS])

        if not frames:
            return schema.empty_frame(), failed

        bars = schema.coerce(pd.concat(frames, ignore_index=True))
        bars = bars[bars[schema.DATE] < cutoff]
        report.rows = len(bars)
        report.symbols_returned = int(bars[schema.SYMBOL].nunique()) if len(bars) else 0
        return schema.coerce(bars), failed

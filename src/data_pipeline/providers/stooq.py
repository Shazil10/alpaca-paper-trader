"""Stooq free data provider for cross-checking.

Stooq provides free daily OHLCV data without API key requirement.
Used as a cross-check source, not primary.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_eod(
    ticker: str,
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily prices from Stooq via pandas-datareader compatible URL.

    Stooq is free but has no split/dividend adjustment metadata,
    so it is only useful for cross-checking, not as primary source.
    """
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

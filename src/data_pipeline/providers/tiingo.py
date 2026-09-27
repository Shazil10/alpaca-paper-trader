"""Tiingo free-tier EOD data. Optional and disabled by default.

Tiingo Starter ($0/month) provides:
  - 30+ years of EOD history for 49,000+ US stocks
  - 500 unique symbols/month rate limit
  - divCash and splitFactor fields (corporate actions)
  - Delisted ticker support via includeDelisted=true

It is the preferred tool for *deliberate* gap repair -- raw and adjusted fields,
dividends, splits and richer metadata -- but it is not wired into any automatic
path. The decision rule is: rebuild on Alpaca, classify what is still missing,
and only trial Tiingo on the specific unresolved delisted/ticker cases if the
audit proves they are material. Until then ``TiingoProvider`` refuses to run
unless ``TIINGO_ENABLED=1`` *and* ``TIINGO_API_KEY`` are both set, so it cannot
be reached by accident.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline import schema
from data_pipeline.providers.base import (
    FetchReport, MarketDataProvider, ProviderConfigurationError,
)
from data_pipeline.registry import normalize_symbol

logger = logging.getLogger(__name__)

BASE_URL = "https://api.tiingo.com/tiingo"


def _api_key() -> Optional[str]:
    return os.getenv("TIINGO_API_KEY")


def is_enabled() -> bool:
    """Both an explicit switch and a key. Either alone is not enough."""
    return os.getenv("TIINGO_ENABLED", "").strip() == "1" and bool(_api_key())


def fetch_eod(
    ticker: str,
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily EOD prices from Tiingo for a single ticker.

    Returns DataFrame with columns matching the lake schema:
    date, symbol, open, high, low, close, adj_close, volume

    Also includes divCash and splitFactor if available.
    """
    key = _api_key()
    if not key:
        raise RuntimeError(
            "TIINGO_API_KEY not set. Register free at https://api.tiingo.com"
        )

    import requests

    url = f"{BASE_URL}/daily/{ticker}/prices"
    params = {"startDate": start, "token": key}
    if end:
        params["endDate"] = end

    headers = {"Content-Type": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)

    if resp.status_code == 404:
        logger.warning("Tiingo: ticker %s not found", ticker)
        return pd.DataFrame()

    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"]).dt.normalize()
    out["symbol"] = ticker.upper()

    for lake_col, tiingo_col in [
        ("open", "adjOpen"), ("high", "adjHigh"),
        ("low", "adjLow"), ("adj_close", "adjClose"),
    ]:
        if tiingo_col in df.columns:
            out[lake_col] = df[tiingo_col].astype(float)

    if "close" in df.columns:
        out["close"] = df["close"].astype(float)
    elif "adjClose" in df.columns:
        out["close"] = df["adjClose"].astype(float)

    if "volume" in df.columns:
        out["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    return out


def search_delisted(query: str = "", limit: int = 100) -> pd.DataFrame:
    """Search for delisted tickers via Tiingo's utility endpoint.

    Returns DataFrame with columns: ticker, name, asset_type, is_active, perma_ticker.
    """
    key = _api_key()
    if not key:
        raise RuntimeError("TIINGO_API_KEY not set")

    import requests

    url = f"{BASE_URL}/utilities/search"
    params = {
        "query": query,
        "includeDelisted": "true",
        "limit": limit,
        "token": key,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame(columns=["ticker", "name", "asset_type", "is_active", "perma_ticker"])

    df = pd.DataFrame(data)
    rename = {
        "ticker": "ticker",
        "name": "name",
        "assetType": "asset_type",
        "isActive": "is_active",
        "permaTicker": "perma_ticker",
    }
    available = [v for k, v in rename.items() if k in df.columns]
    return df.rename(columns=rename)[available]


class TiingoProvider(MarketDataProvider):
    """Per-symbol Tiingo EOD. Non-canonical: output goes to quarantine only."""

    name = "tiingo"
    feed = "tiingo-eod"
    adjustment = "close=raw, adj_close=adjClose (split+dividend)"
    canonical = False

    def fetch_batch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, List[str]]:
        if not is_enabled():
            raise ProviderConfigurationError(
                "Tiingo is disabled. It is only for deliberate gap repair after "
                "the Alpaca audit shows a material need; set TIINGO_ENABLED=1 and "
                "TIINGO_API_KEY to use it."
            )

        wanted = sorted({normalize_symbol(s) for s in symbols if str(s).strip()})
        report = FetchReport(provider=self.name, requested=len(wanted))
        self.last_report = report

        cutoff = min(pd.Timestamp(end_exclusive), self.session_cutoff())
        last = (cutoff - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        frames, failed = [], []
        for symbol in wanted:
            try:
                frame = fetch_eod(symbol, pd.Timestamp(start).strftime("%Y-%m-%d"), last)
            except Exception as exc:
                logger.warning("Tiingo fetch failed for %s: %s", symbol, type(exc).__name__)
                frame = pd.DataFrame()
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
        bars = schema.coerce(bars[bars[schema.DATE] < cutoff])
        report.rows = len(bars)
        report.symbols_returned = int(bars[schema.SYMBOL].nunique()) if len(bars) else 0
        return bars, failed

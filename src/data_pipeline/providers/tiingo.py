"""Tiingo free-tier EOD data provider for delisted stock backfill.

Tiingo Starter ($0/month) provides:
  - 30+ years of EOD history for 49,000+ US stocks
  - 500 unique symbols/month rate limit
  - divCash and splitFactor fields (corporate actions)
  - Delisted ticker support via includeDelisted=true

Usage requires TIINGO_API_KEY environment variable.
Not called during normal operation — only for one-time historical backfill.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://api.tiingo.com/tiingo"


def _api_key() -> Optional[str]:
    return os.getenv("TIINGO_API_KEY")


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

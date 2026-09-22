"""Security master with permanent identifiers.

Ticker recycling is a real data corruption risk: when a company is delisted,
its ticker symbol can be reassigned to a different company. Naive price
concatenation then splices two unrelated companies into one price series.

This module provides a security master with permanent IDs and ticker validity
intervals, so the backtester can detect and handle recycled tickers.

For v1, this is populated from the membership changelog + registry.
A proper implementation would use FIGI, CUSIP, or Tiingo's permaTicker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECURITIES_PATH = REPO_ROOT / "data" / "universe" / "securities.parquet"

DateLike = Union[str, pd.Timestamp]


def _load_securities(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the security master.

    Expected columns:
      security_id: str — permanent identifier (e.g. "AAPL_1", or FIGI)
      ticker: str — the trading symbol
      name: str — company name (optional)
      start_date: Timestamp — when this ticker became valid
      end_date: Timestamp or NaT — when this ticker stopped being valid
      delist_reason: str — 'merged', 'bankrupt', 'delisted', '' for active
    """
    target = Path(path or DEFAULT_SECURITIES_PATH)
    if not target.exists():
        return pd.DataFrame(columns=[
            "security_id", "ticker", "name", "start_date", "end_date", "delist_reason"
        ])

    df = pd.read_parquet(target)
    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


_SECURITIES_CACHE: Optional[pd.DataFrame] = None


def _get_securities(path: Optional[Path] = None) -> pd.DataFrame:
    global _SECURITIES_CACHE
    if _SECURITIES_CACHE is None:
        _SECURITIES_CACHE = _load_securities(path)
    return _SECURITIES_CACHE


def clear_cache() -> None:
    global _SECURITIES_CACHE
    _SECURITIES_CACHE = None


def resolve_ticker(
    ticker: str,
    as_of: DateLike,
    *,
    path: Optional[Path] = None,
) -> Optional[str]:
    """Return the security_id for a ticker on a given date.

    Returns None if the ticker is not found or was not valid on that date.
    If no security master exists, returns the ticker itself (passthrough).
    """
    df = _get_securities(path)
    if len(df) == 0:
        return ticker

    ts = pd.Timestamp(as_of)
    mask = (
        (df["ticker"] == ticker.upper())
        & (df["start_date"] <= ts)
        & (df["end_date"].isna() | (df["end_date"] >= ts))
    )
    matches = df.loc[mask]
    if len(matches) == 0:
        return None
    return str(matches.iloc[0]["security_id"])


def is_recycled_ticker(
    ticker: str,
    *,
    path: Optional[Path] = None,
) -> bool:
    """True if this ticker has been used by more than one security."""
    df = _get_securities(path)
    if len(df) == 0:
        return False
    matches = df[df["ticker"] == ticker.upper()]
    return len(matches["security_id"].unique()) > 1


def delisted_tickers(
    start: DateLike,
    end: DateLike,
    *,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return tickers that were delisted between start and end.

    Returns DataFrame with columns: security_id, ticker, end_date, delist_reason.
    """
    df = _get_securities(path)
    if len(df) == 0:
        return pd.DataFrame(columns=["security_id", "ticker", "end_date", "delist_reason"])

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    mask = (
        df["end_date"].notna()
        & (df["end_date"] >= start_ts)
        & (df["end_date"] <= end_ts)
    )
    return df.loc[mask, ["security_id", "ticker", "end_date", "delist_reason"]].reset_index(drop=True)


def sector_map(*, path: Optional[Path] = None) -> Dict[str, str]:
    """Return {ticker: sector} from the security master.

    Not point-in-time: the sector comes from the current ticker registry, so a
    company that changed GICS sector mid-history carries today's label
    throughout, and names that left the index before the registry existed carry
    no label at all. Good enough for a sector *cap* -- without any map at all
    ``RiskConfig.max_sector_pct`` does nothing, which is worse -- but not good
    enough for sector attribution.
    """
    df = _get_securities(path)
    if len(df) == 0 or "sector" not in df.columns:
        return {}

    known = df[df["sector"].astype(str).str.len() > 0]
    if len(known) == 0:
        return {}
    return {
        str(t): str(s)
        for t, s in known.groupby("ticker")["sector"].last().items()
    }


def ticker_map_asof(
    date: DateLike,
    *,
    path: Optional[Path] = None,
) -> Dict[str, str]:
    """Return {ticker: security_id} for all valid tickers on a date."""
    df = _get_securities(path)
    if len(df) == 0:
        return {}

    ts = pd.Timestamp(date)
    mask = (
        (df["start_date"] <= ts)
        & (df["end_date"].isna() | (df["end_date"] >= ts))
    )
    result = df.loc[mask].set_index("ticker")["security_id"]
    return dict(result)

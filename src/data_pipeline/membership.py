"""Point-in-time index membership.

Backtesting with today's index constituents projected backward creates
survivorship bias — failed/removed companies disappear from history, inflating
results by ~1-3% annually.

This module loads reconstructed S&P 500 membership data and exposes a
members_asof(date) function that returns the set of tickers that were in
the index on a specific date.

Data sources (free, CC-BY or public domain):
  - fja05680/sp500: Historical S&P 500 components since 1996
  - chinobing/historical_sp500_constituents: auto-renewed daily
  - joeyfife/point-in-time-sp500: validated 2016-2026 with delisted names

The membership data is stored in data/universe/membership.parquet as interval
records: (symbol, index, start_date, end_date).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import FrozenSet, Optional, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBERSHIP_PATH = REPO_ROOT / "data" / "universe" / "membership.parquet"

DateLike = Union[str, pd.Timestamp]

ROTATION_ETFS: FrozenSet[str] = frozenset([
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
])
HEDGE_ETFS: FrozenSet[str] = frozenset(["TLT", "GLD", "UUP", "FXY", "FXF"])
CORE_ETFS: FrozenSet[str] = frozenset(["SPY", "SHY", "IJH", "IJR"])
ALL_FIXED_ETFS: FrozenSet[str] = ROTATION_ETFS | HEDGE_ETFS | CORE_ETFS


def _load_membership(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the interval membership file.

    Expected columns: symbol, index, start_date, end_date.
    end_date is NaT for current members.
    """
    target = Path(path or DEFAULT_MEMBERSHIP_PATH)
    if not target.exists():
        logger.warning(
            "membership file not found at %s; PIT universe unavailable. "
            "Falling back to current universe.csv — results carry survivorship bias.",
            target,
        )
        return pd.DataFrame(columns=["symbol", "index", "start_date", "end_date"])

    df = pd.read_parquet(target)
    for col in ("start_date", "end_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df


_MEMBERSHIP_CACHE: Optional[pd.DataFrame] = None


def _get_membership(path: Optional[Path] = None) -> pd.DataFrame:
    global _MEMBERSHIP_CACHE
    if _MEMBERSHIP_CACHE is None:
        _MEMBERSHIP_CACHE = _load_membership(path)
    return _MEMBERSHIP_CACHE


def clear_cache() -> None:
    global _MEMBERSHIP_CACHE
    _MEMBERSHIP_CACHE = None


def members_asof(
    date: DateLike,
    index: str = "SP500",
    *,
    path: Optional[Path] = None,
) -> Set[str]:
    """Return the set of tickers that were members of `index` on `date`.

    For current members, end_date is NaT — they are included for any date
    >= their start_date.

    Returns empty set if no membership data exists.
    """
    df = _get_membership(path)
    if len(df) == 0:
        return set()

    ts = pd.Timestamp(date)
    mask = (
        (df["index"] == index)
        & (df["start_date"] <= ts)
        & (df["end_date"].isna() | (df["end_date"] >= ts))
    )
    return set(df.loc[mask, "symbol"].unique())


def membership_changes(
    start: DateLike,
    end: DateLike,
    index: str = "SP500",
    *,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return additions and removals between start and end dates.

    Returns DataFrame with columns: date, symbol, event ('added' or 'removed').
    """
    df = _get_membership(path)
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", "event"])

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    subset = df[df["index"] == index].copy()

    events = []

    added = subset[
        (subset["start_date"] >= start_ts) & (subset["start_date"] <= end_ts)
    ]
    for _, row in added.iterrows():
        events.append({
            "date": row["start_date"],
            "symbol": row["symbol"],
            "event": "added",
        })

    removed = subset[
        subset["end_date"].notna()
        & (subset["end_date"] >= start_ts)
        & (subset["end_date"] <= end_ts)
    ]
    for _, row in removed.iterrows():
        events.append({
            "date": row["end_date"],
            "symbol": row["symbol"],
            "event": "removed",
        })

    result = pd.DataFrame(events)
    if len(result) > 0:
        result = result.sort_values("date").reset_index(drop=True)
    return result


def has_pit_membership(path: Optional[Path] = None) -> bool:
    """True when point-in-time membership data is available."""
    target = Path(path or DEFAULT_MEMBERSHIP_PATH)
    return target.exists()


def available_indices(path: Optional[Path] = None) -> Set[str]:
    """Return the set of index names available in the membership file."""
    df = _get_membership(path)
    if len(df) == 0:
        return set()
    return set(df["index"].unique())

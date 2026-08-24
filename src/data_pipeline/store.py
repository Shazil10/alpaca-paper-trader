"""store.py

Read side of the price lake. This is the only data module strategies import.

Deliberately has no network access: ``store`` never imports yfinance and never
fetches. If the lake is missing bars, callers get fewer rows and a warning --
they do not get a silent background download that quietly changes the window a
strategy is reasoning about. Fetching is ``sync_prices``' job alone.

Typical use from a strategy::

    from data_pipeline import store

    closes = store.load_close_matrix(
        ["XLK", "XLF", "SPY"], start="2024-08-16", end="2026-08-15"
    )

``load_close_matrix`` returns a date x symbol frame of ``adj_close``, which is
the same thing the pre-lake code built out of
``yf.download(..., auto_adjust=True)["Close"]``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd

from data_pipeline import schema
from data_pipeline.schema import ADJ_CLOSE, DATE, KEY, SYMBOL

logger = logging.getLogger(__name__)

DateLike = Union[str, pd.Timestamp, None]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_timestamp(value: DateLike) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _normalize_symbols(symbols: Optional[Iterable[str]]) -> Optional[List[str]]:
    if symbols is None:
        return None
    out = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    return out


def _relevant_files(
    root: Optional[Path], start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
) -> List[Path]:
    """Year files overlapping [start, end]. Avoids reading 3 years to serve 1."""
    files = schema.discover_year_files(root)
    if start is None and end is None:
        return files

    keep: List[Path] = []
    for path in files:
        year = int(path.stem)
        if start is not None and year < start.year:
            continue
        if end is not None and year > end.year:
            continue
        keep.append(path)
    return keep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_prices(
    symbols: Optional[Sequence[str]] = None,
    start: DateLike = None,
    end: DateLike = None,
    *,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load long-format bars from the lake.

    Args:
        symbols: restrict to these tickers; None means every symbol on disk.
        start:   inclusive lower date bound.
        end:     inclusive upper date bound.
        root:    lake directory override (tests).

    Returns:
        DataFrame with ``schema.COLUMNS``, sorted by (date, symbol). Empty (but
        correctly typed) when the lake has nothing matching.
    """
    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end)
    wanted = _normalize_symbols(symbols)

    paths = _relevant_files(root, start_ts, end_ts)
    if not paths:
        logger.warning(
            "price lake is empty or absent (root=%s) -- run sync_prices.py",
            Path(root or schema.DEFAULT_LAKE_ROOT),
        )
        return schema.empty_frame()

    seen_keys: dict = {}
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = schema.read_frame(path)
        if len(frame) == 0:
            continue
        if wanted is not None:
            frame = frame[frame[SYMBOL].isin(wanted)]
        if start_ts is not None:
            frame = frame[frame[DATE] >= start_ts]
        if end_ts is not None:
            frame = frame[frame[DATE] <= end_ts]
        if len(frame) == 0:
            continue

        # A year present in both Parquet and CSV means a rollover was left
        # half-done. discover_year_files already prefers Parquet, so this can
        # only fire across different years -- still worth shouting about.
        year = int(path.stem)
        if year in seen_keys:
            logger.warning("year %s resolved to multiple lake files", year)
        seen_keys[year] = path

        frames.append(frame)

    if not frames:
        return schema.empty_frame()

    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=KEY, keep="last")
    if len(combined) != before:
        logger.warning(
            "dropped %d duplicate (date, symbol) rows while unioning lake files; "
            "check for a partially completed January rollover",
            before - len(combined),
        )

    combined = schema.coerce(combined)

    if wanted is not None:
        found = set(combined[SYMBOL].dropna().unique())
        absent = [s for s in wanted if s not in found]
        if absent:
            logger.warning(
                "lake has no bars for %d requested symbol(s) in range: %s",
                len(absent),
                ", ".join(absent[:20]) + (" ..." if len(absent) > 20 else ""),
            )

    return combined


def load_close_matrix(
    symbols: Optional[Sequence[str]] = None,
    start: DateLike = None,
    end: DateLike = None,
    column: str = ADJ_CLOSE,
    *,
    root: Optional[Path] = None,
    dropna_how: str = "all",
) -> pd.DataFrame:
    """Load a date x symbol matrix of one price column.

    This is the shape the momentum and rotation strategies already work in.
    ``column`` defaults to ``adj_close``; pass ``close`` only when you
    specifically mean the unadjusted print.

    Columns come back in the order requested (for those that exist), so callers
    can rely on positional alignment with their own ticker lists.
    """
    if column not in schema.COLUMNS:
        raise ValueError(f"unknown price column {column!r}; expected one of {schema.COLUMNS}")

    long_df = load_prices(symbols, start, end, root=root)
    if len(long_df) == 0:
        return pd.DataFrame(dtype="float64")

    matrix = long_df.pivot_table(index=DATE, columns=SYMBOL, values=column, aggfunc="last")
    matrix.index.name = DATE
    matrix.columns.name = None

    wanted = _normalize_symbols(symbols)
    if wanted is not None:
        ordered = [s for s in wanted if s in matrix.columns]
        matrix = matrix[ordered]

    if dropna_how in {"all", "any"}:
        matrix = matrix.dropna(how=dropna_how)

    return matrix.astype("float64").sort_index()


def load_ohlc_adjusted(
    symbols: Optional[Sequence[str]] = None,
    start: DateLike = None,
    end: DateLike = None,
    *,
    root: Optional[Path] = None,
) -> "dict":
    """Return ``{symbol: DataFrame[Open, High, Low, Close]}`` on the adjusted scale.

    The lake stores unadjusted OHLC plus ``adj_close``. Some signals need
    adjusted *highs and lows* -- the ATR breakout trend in the rotation sleeve
    is one, and that term carries most of its composite rank weight.

    yfinance's ``auto_adjust=True`` scales every OHLC field by the same factor,
    so the adjusted series is recovered exactly as ``raw * (adj_close / close)``.
    Verified against yfinance for XLE/XLK/XLP: difference is identically zero.

    Column names are capitalized to match what the strategies already expect.
    """
    long_df = load_prices(symbols, start, end, root=root)
    if len(long_df) == 0:
        return {}

    out: dict = {}
    for symbol, group in long_df.groupby(SYMBOL, sort=True):
        g = group.set_index(DATE).sort_index()

        close = g[schema.CLOSE].astype("float64")
        adj = g[schema.ADJ_CLOSE].astype("float64")
        # Guard against a zero/NaN raw close producing an infinite factor.
        factor = (adj / close.where(close > 0)).fillna(1.0)

        frame = pd.DataFrame(
            {
                "Open": g[schema.OPEN].astype("float64") * factor,
                "High": g[schema.HIGH].astype("float64") * factor,
                "Low": g[schema.LOW].astype("float64") * factor,
                "Close": adj,
            },
            index=g.index,
        ).dropna()

        if len(frame) > 0:
            out[str(symbol)] = frame

    return out


def last_bar_date(
    symbols: Optional[Sequence[str]] = None, *, root: Optional[Path] = None
) -> Optional[pd.Timestamp]:
    """Most recent session in the lake, or None when empty.

    Used for the staleness log so an unattended run says out loud how old its
    data is instead of failing silently.
    """
    frame = load_prices(symbols, root=root)
    if len(frame) == 0:
        return None
    return pd.Timestamp(frame[DATE].max())


def last_date_per_symbol(*, root: Optional[Path] = None) -> "pd.Series":
    """Latest stored session per symbol. Drives the incremental fetch window."""
    frame = load_prices(root=root)
    if len(frame) == 0:
        return pd.Series(dtype="datetime64[ns]")
    return frame.groupby(SYMBOL)[DATE].max()


def coverage(
    symbols: Optional[Sequence[str]] = None, *, root: Optional[Path] = None
) -> pd.DataFrame:
    """Per-symbol first/last session and bar count. For diagnostics and tests."""
    frame = load_prices(symbols, root=root)
    if len(frame) == 0:
        return pd.DataFrame(columns=["symbol", "first_date", "last_date", "bars"])

    grouped = (
        frame.groupby(SYMBOL)[DATE]
        .agg(first_date="min", last_date="max", bars="count")
        .reset_index()
        .rename(columns={SYMBOL: "symbol"})
    )
    return grouped.sort_values("symbol").reset_index(drop=True)


#: A date is a real session once this many distinct symbols report a bar on it.
#: An absolute floor, deliberately not a proportion of the universe.
#:
#: A proportional quorum is blind to the failure mode that matters most: when
#: one bad fetch window drops the same dates for most of the universe, those
#: dates fall below the quorum and the calendar simply forgets they were
#: sessions -- so nothing looks incomplete and a lake-wide hole passes review.
#: (Observed: 2026-07-21/22/31 held only 319-442 of 1,380 symbols.)
#:
#: Nor can it depend on one reference instrument, since that instrument may
#: itself be short bars -- which then hides every symbol missing those same
#: dates. (Observed: SPY at 910 sessions while another symbol had 912.)
#:
#: A small absolute floor survives both: a genuine session keeps hundreds of
#: reporters even in a bad window, while a phantom date would need this many
#: symbols to independently invent it.
MIN_SYMBOLS_FOR_SESSION = 10


def trading_calendar(
    start: DateLike = None,
    end: DateLike = None,
    *,
    min_symbols: int = MIN_SYMBOLS_FOR_SESSION,
    root: Optional[Path] = None,
) -> pd.DatetimeIndex:
    """Infer market sessions from the lake, using an absolute reporting floor."""
    frame = load_prices(None, start, end, root=root)
    if len(frame) == 0:
        return pd.DatetimeIndex([])

    per_date = frame.groupby(DATE)[SYMBOL].nunique()
    active = int(frame[SYMBOL].nunique())
    floor = max(1, min(min_symbols, active))
    return pd.DatetimeIndex(sorted(per_date[per_date >= floor].index))


def find_gaps(
    symbols: Optional[Sequence[str]] = None,
    start: DateLike = None,
    end: DateLike = None,
    *,
    min_symbols: int = MIN_SYMBOLS_FOR_SESSION,
    root: Optional[Path] = None,
) -> "dict":
    """Return ``{symbol: [missing sessions]}`` for holes inside each symbol's span.

    Only gaps *between* a symbol's own first and last bar are reported: a name
    that listed late or delisted early is incomplete by nature, not by error.

    Gaps matter more than their size suggests. A single missing bar inside a
    50-day window makes ``rolling(50).mean()`` return NaN, which silently drops
    that symbol out of any threshold comparison downstream -- changing live
    signals without raising anything.
    """
    frame = load_prices(symbols, start, end, root=root)
    if len(frame) == 0:
        return {}

    calendar = trading_calendar(start, end, min_symbols=min_symbols, root=root)
    if len(calendar) == 0:
        return {}

    gaps: dict = {}
    for symbol, group in frame.groupby(SYMBOL, sort=True):
        have = pd.DatetimeIndex(sorted(group[DATE].unique()))
        if len(have) == 0:
            continue
        window = calendar[(calendar >= have.min()) & (calendar <= have.max())]
        missing = window.difference(have)
        if len(missing) > 0:
            gaps[str(symbol)] = [pd.Timestamp(d) for d in missing]

    return gaps


def has_lookback(
    symbols: Sequence[str],
    sessions: int,
    end: DateLike = None,
    *,
    column: str = ADJ_CLOSE,
    root: Optional[Path] = None,
) -> bool:
    """True when every requested symbol has at least ``sessions`` bars.

    This is the gate a strategy uses to decide whether to run at all. Missing
    *today* is not a reason to skip -- an insufficient lookback is.
    """
    matrix = load_close_matrix(symbols, None, end, column=column, root=root)
    if matrix.empty:
        return False

    wanted = _normalize_symbols(symbols) or []
    for sym in wanted:
        if sym not in matrix.columns:
            return False
        if int(matrix[sym].notna().sum()) < sessions:
            return False
    return True

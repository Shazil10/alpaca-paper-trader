"""Point-in-time index membership: S&P 500, 400, 600, and the S&P 1500 union.

Backtesting with today's index constituents projected backward creates
survivorship bias — failed/removed companies disappear from history, inflating
results by ~1-3% annually.

This module exposes ``members_asof(date, index)``: the set of tickers that were
in ``index`` on ``date``, as far as the recorded data can say.

Two sources, joined at one date
-------------------------------
* ``data/universe/membership.parquet`` -- reconstructed S&P 500 history from
  fja05680/sp500 (1996 onward), interval records ``(symbol, index,
  start_date, end_date)``, rebuilt manually by ``scripts/build_membership.py``.
* The daily capture (``membership_capture``): an unfiltered baseline of all
  three indices plus append-only add/remove events.

For the S&P 500 the reconstructed history is used up to the day before the
capture's baseline and the capture from then on; an interval that spans the
join is one interval. The S&P 400 and 600 exist **only** from the capture. They
are left-censored at the baseline: a baseline member's true entry date is
unknown, so it is recorded as the baseline date, and dates before it are
reported as uncovered, not guessed.

The S&P 1500 union
------------------
``SP1500`` is derived, never stored: the union of the three component
histories, with gaps of up to ``BRIDGE_DAYS`` between one component's removal
and another's addition bridged. That is the S&P 600 -> 400 move: the two
Wikipedia pages are not edited in the same minute, so a promoted company can be
missing from both captures for a day. It never left the S&P 1500, and the union
does not say it did.

Coverage, stated plainly
------------------------
``coverage_start(index)`` is the first date an index's membership is actually
known. For ``SP1500`` that is the capture baseline. Before it, ``members_asof``
returns the S&P 500 part only -- that is all anyone recorded -- and logs that it
has done so. A backtest over earlier dates on ``pit_sp1500`` is an S&P 500
backtest with a mid/small-cap universe of *survivors*, and must not be
described as survivorship-free.

Data sources (free, CC-BY or public domain):
  - fja05680/sp500: Historical S&P 500 components since 1996
  - Wikipedia S&P 500/400/600 constituent tables, captured daily from the
    first run of ``src/universe.py --capture-only``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import pandas as pd

from data_pipeline import membership_capture as capture

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

COMPONENTS: Tuple[str, ...] = capture.INDICES
UNION_INDEX = "SP1500"

#: Longest gap between leaving one component and joining another that still
#: counts as continuous S&P 1500 membership.
BRIDGE_DAYS = 7

INTERVAL_COLUMNS = ["symbol", "index", "start_date", "end_date", "left_censored", "source"]

#: ``captured`` argument meaning "use the repository's capture files".
DEFAULT_CAPTURED = (capture.BASELINE_PATH, capture.EVENTS_PATH)

Captured = Optional[Tuple[Path, Path]]


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
_INTERVAL_CACHE: Dict[tuple, pd.DataFrame] = {}
_WARNED: Set[tuple] = set()


def _get_membership(path: Optional[Path] = None) -> pd.DataFrame:
    global _MEMBERSHIP_CACHE
    if path is not None:
        return _load_membership(path)
    if _MEMBERSHIP_CACHE is None:
        _MEMBERSHIP_CACHE = _load_membership(path)
    return _MEMBERSHIP_CACHE


def clear_cache() -> None:
    global _MEMBERSHIP_CACHE
    _MEMBERSHIP_CACHE = None
    _INTERVAL_CACHE.clear()
    _WARNED.clear()


def _resolve_captured(path: Optional[Path], captured: Captured) -> Captured:
    """Which capture files to join in.

    An explicit ``path`` to a reconstructed file means a caller -- almost
    always a test -- has chosen its own data; mixing in the repository's live
    capture behind its back would make the result depend on the checkout. So
    capture data is used by default only when ``path`` is also the default.
    """
    if captured is not None:
        return captured
    return DEFAULT_CAPTURED if path is None else None


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def captured_intervals(baseline: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Interval records from a capture baseline plus its events.

    ``end_date`` is inclusive: the last day the symbol is known to have been a
    member. A removal first observed on day D closes the interval on D - 1.
    """
    rows: List[dict] = []
    open_: Dict[Tuple[str, str], dict] = {}

    for _, r in baseline.iterrows():
        key = (str(r["index"]), str(r["symbol"]))
        open_[key] = {
            "symbol": key[1], "index": key[0],
            "start_date": pd.Timestamp(r["observed_date"]), "end_date": pd.NaT,
            "left_censored": True, "source": str(r.get("source") or capture.SOURCE_LABEL),
        }

    if len(events):
        ordered = events.sort_values(["effective_date"], kind="mergesort")
        for _, r in ordered.iterrows():
            key = (str(r["index"]), str(r["symbol"]))
            when = pd.Timestamp(r["effective_date"])
            if r["event"] == "add" and key not in open_:
                open_[key] = {
                    "symbol": key[1], "index": key[0], "start_date": when,
                    "end_date": pd.NaT, "left_censored": False,
                    "source": str(r.get("source") or capture.SOURCE_LABEL),
                }
            elif r["event"] == "remove" and key in open_:
                interval = open_.pop(key)
                interval["end_date"] = when - pd.Timedelta(days=1)
                if interval["end_date"] >= interval["start_date"]:
                    rows.append(interval)

    rows.extend(open_.values())
    return _interval_frame(rows)


def _interval_frame(rows) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=INTERVAL_COLUMNS)
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    frame["left_censored"] = frame["left_censored"].fillna(False).astype(bool)
    return frame.sort_values(["index", "symbol", "start_date"], kind="mergesort").reset_index(drop=True)


def _merge(frame: pd.DataFrame, index: str, gap_days: int) -> pd.DataFrame:
    """Merge each symbol's intervals, bridging gaps of at most ``gap_days``."""
    rows: List[dict] = []
    for symbol, group in frame.sort_values(["symbol", "start_date"]).groupby("symbol", sort=True):
        current: Optional[dict] = None
        for _, r in group.iterrows():
            start = pd.Timestamp(r["start_date"])
            end = r["end_date"]
            if current is None:
                current = {"symbol": symbol, "index": index, "start_date": start,
                           "end_date": end, "left_censored": bool(r["left_censored"]),
                           "source": r["source"]}
                continue
            cur_end = current["end_date"]
            joins = pd.isna(cur_end) or start <= pd.Timestamp(cur_end) + pd.Timedelta(days=gap_days + 1)
            if joins:
                if pd.isna(cur_end) or pd.isna(end):
                    current["end_date"] = pd.NaT
                else:
                    current["end_date"] = max(pd.Timestamp(cur_end), pd.Timestamp(end))
                if current["source"] != r["source"]:
                    current["source"] = f"{current['source']}+{r['source']}"
            else:
                rows.append(current)
                current = {"symbol": symbol, "index": index, "start_date": start,
                           "end_date": end, "left_censored": bool(r["left_censored"]),
                           "source": r["source"]}
        if current is not None:
            rows.append(current)
    return _interval_frame(rows)


def _capture_frames(captured: Captured) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if captured is None:
        return (pd.DataFrame(columns=capture.BASELINE_COLUMNS),
                pd.DataFrame(columns=capture.EVENT_COLUMNS))
    baseline_path, events_path = captured
    return capture.load_baseline(baseline_path), capture.load_events(events_path)


def _mtime(path: Optional[Path]) -> float:
    try:
        return Path(path).stat().st_mtime if path is not None else 0.0
    except OSError:
        return 0.0


def intervals(
    index: str = "SP500",
    *,
    path: Optional[Path] = None,
    captured: Captured = None,
) -> pd.DataFrame:
    """Interval records for ``index`` (SP500, SP400, SP600 or SP1500)."""
    index = str(index).upper()
    captured = _resolve_captured(path, captured)
    key = (index, str(path), str(captured),
           _mtime(path or DEFAULT_MEMBERSHIP_PATH),
           tuple(_mtime(p) for p in (captured or ())))
    if key in _INTERVAL_CACHE:
        return _INTERVAL_CACHE[key]

    if index == UNION_INDEX:
        parts = [intervals(c, path=path, captured=captured) for c in COMPONENTS]
        parts = [p for p in parts if len(p)]
        frame = (
            _merge(pd.concat(parts, ignore_index=True), UNION_INDEX, BRIDGE_DAYS)
            if parts else _interval_frame([])
        )
    else:
        baseline, events = _capture_frames(captured)
        cap = captured_intervals(baseline, events)
        cap = cap[cap["index"] == index]

        if index == "SP500":
            recon = _get_membership(path)
            recon = recon[recon["index"] == "SP500"].copy()
            recon["left_censored"] = False
            recon["source"] = "fja05680"
            if len(cap):
                join = pd.Timestamp(cap["start_date"].min())
                recon = recon[recon["start_date"] < join].copy()
                cutoff = join - pd.Timedelta(days=1)
                recon["end_date"] = recon["end_date"].where(
                    recon["end_date"].notna() & (recon["end_date"] <= cutoff), cutoff
                )
                # A reconstructed member continuing into the capture is not
                # left-censored there: its history is known.
                known = set(recon.loc[recon["end_date"] == cutoff, "symbol"])
                cap = cap.copy()
                cap.loc[cap["symbol"].isin(known) & (cap["start_date"] == join),
                        "left_censored"] = False
            combined = pd.concat([recon[INTERVAL_COLUMNS], cap], ignore_index=True)
            frame = _merge(combined, "SP500", 0) if len(combined) else _interval_frame([])
        else:
            frame = cap.reset_index(drop=True)

    _INTERVAL_CACHE[key] = frame
    return frame


def coverage_start(
    index: str = "SP500",
    *,
    path: Optional[Path] = None,
    captured: Captured = None,
) -> Optional[pd.Timestamp]:
    """First date on which ``index`` membership is actually known, or None."""
    index = str(index).upper()
    if index == UNION_INDEX:
        starts = [coverage_start(c, path=path, captured=captured) for c in COMPONENTS]
        if any(s is None for s in starts):
            return None
        return max(starts)
    frame = intervals(index, path=path, captured=captured)
    if len(frame) == 0:
        return None
    # The earliest record is the first full observation: the reconstruction's
    # first snapshot for the S&P 500, the capture baseline for the 400 and 600
    # (every baseline interval starts there, left-censored).
    return pd.Timestamp(frame["start_date"].min())


def members_asof(
    date: DateLike,
    index: str = "SP500",
    *,
    path: Optional[Path] = None,
    captured: Captured = None,
) -> Set[str]:
    """Return the set of tickers that were members of ``index`` on ``date``.

    ``end_date`` is inclusive and NaT means still a member. Returns an empty set
    if no membership data exists. For a date before ``coverage_start(index)``
    the answer is whatever was recorded -- nothing for the S&P 400/600, the
    S&P 500 alone for the S&P 1500 -- and a warning says so once.
    """
    index = str(index).upper()
    ts = pd.Timestamp(date)
    frame = intervals(index, path=path, captured=captured)
    if len(frame) == 0:
        return set()

    start = coverage_start(index, path=path, captured=captured)
    if start is not None and ts < start:
        warn_key = (index, str(start.date()))
        if warn_key not in _WARNED:
            _WARNED.add(warn_key)
            logger.warning(
                "%s membership is only recorded from %s; answers for earlier dates "
                "cover %s. Results over that span are not survivorship-free.",
                index, start.date(),
                "the S&P 500 only" if index == UNION_INDEX else "nothing",
            )

    mask = (
        (frame["start_date"] <= ts)
        & (frame["end_date"].isna() | (frame["end_date"] >= ts))
    )
    return set(frame.loc[mask, "symbol"].unique())


def membership_changes(
    start: DateLike,
    end: DateLike,
    index: str = "SP500",
    *,
    path: Optional[Path] = None,
    captured: Captured = None,
) -> pd.DataFrame:
    """Return additions and removals between start and end dates.

    Returns DataFrame with columns: date, symbol, event ('added' or 'removed').
    Left-censored starts are not additions and are not reported as such.
    """
    subset = intervals(index, path=path, captured=captured)
    if len(subset) == 0:
        return pd.DataFrame(columns=["date", "symbol", "event"])

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    events = []

    added = subset[
        (subset["start_date"] >= start_ts) & (subset["start_date"] <= end_ts)
        & ~subset["left_censored"]
    ]
    for _, row in added.iterrows():
        events.append({"date": row["start_date"], "symbol": row["symbol"], "event": "added"})

    removed = subset[
        subset["end_date"].notna()
        & (subset["end_date"] >= start_ts)
        & (subset["end_date"] <= end_ts)
    ]
    for _, row in removed.iterrows():
        events.append({"date": row["end_date"], "symbol": row["symbol"], "event": "removed"})

    result = pd.DataFrame(events, columns=["date", "symbol", "event"])
    if len(result) > 0:
        result = result.sort_values(["date", "symbol"]).reset_index(drop=True)
    return result


def has_pit_membership(path: Optional[Path] = None) -> bool:
    """True when point-in-time membership data is available."""
    target = Path(path or DEFAULT_MEMBERSHIP_PATH)
    return target.exists()


def available_indices(path: Optional[Path] = None, *, captured: Captured = None) -> Set[str]:
    """Index names with any recorded membership (SP1500 once all three exist)."""
    found = {i for i in COMPONENTS if len(intervals(i, path=path, captured=captured))}
    if set(COMPONENTS) <= found:
        found.add(UNION_INDEX)
    return found


def tracked_symbols(
    since: DateLike,
    *,
    path: Optional[Path] = None,
    captured: Captured = None,
    current_path: Optional[Path] = None,
) -> Tuple[Set[str], Set[str]]:
    """``(current, historical)`` symbols the price lake must keep syncing.

    ``current`` is today's *unfiltered* S&P 500/400/600 membership, from the
    latest capture, falling back to the reconstructed S&P 500's open intervals
    when nothing has been captured yet. ``historical`` is every symbol with
    S&P 1500 membership on any day since ``since``.

    This is what stops the liquidity-filtered ``universe.csv`` from deciding
    which histories exist: sync targets are drawn from here, and ``current`` is
    never retired for failing.
    """
    since_ts = pd.Timestamp(since)
    resolved = _resolve_captured(path, captured)

    current: Set[str] = set()
    if resolved is not None:
        snapshot = capture.load_current(current_path)
        current = set(snapshot["symbol"].astype(str)) - {""}
    if not current:
        sp500 = intervals("SP500", path=path, captured=resolved)
        current = set(sp500.loc[sp500["end_date"].isna(), "symbol"])

    union = intervals(UNION_INDEX, path=path, captured=resolved)
    if len(union) == 0:
        union = intervals("SP500", path=path, captured=resolved)
    overlaps = union["end_date"].isna() | (union["end_date"] >= since_ts)
    historical = set(union.loc[overlaps, "symbol"])
    return current, historical

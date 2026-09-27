"""adj_close anchor maintenance.

    data/prices/anchor_factors.csv

The problem
-----------
An adjusted close is *back-anchored*: when a stock splits or goes ex-dividend,
the provider rescales every earlier bar so that the newest one equals the print.
So the adj_close of 2019-03-01 depends on when you asked for it.

An incremental lake asks at different times. Rows fetched last month carry last
month's anchor; the overlap window re-fetched tonight carries tonight's. After a
10:1 split the stored series therefore reads ...1,200, 1,210, 121, 122... at the
seam between the two -- a -90% "return" that never happened, sitting inside
every 50-day average and 52-week high that spans it. A mean-reversion screen
reads that as the deepest pullback in the universe. Dividends do the same thing
at the scale of the yield, quietly turning total return into price return.

The fix, and why it has two halves
----------------------------------
A new corporate action rescales *all* earlier bars of that symbol by one
constant. The overlap window measures that constant exactly: on the earliest
date that is both stored and freshly fetched, ``factor = new_adj / old_adj``.
Every stored row before that date needs multiplying by it.

Where those rows live decides how:

* **In a partition being rewritten anyway** (the hot CSV, or any year the
  current write touches) the rows are rescaled in place. Rewriting it costs
  nothing extra.
* **In an immutable cold partition** the factor is recorded here instead, and
  ``store.load_prices`` applies it on read. Cold Parquet years are committed
  binaries; rewriting ten of them because one stock paid a dividend would add
  ~80 MB of near-undeltable blobs to git every trading day.

The two halves meet in one rule: **whenever a partition is rewritten, its
pending factors are folded into the data and cleared**. So a factor is applied
exactly once -- on read while it is pending, in the bytes once folded -- and a
rebuild, which rewrites everything, leaves this file empty.

Rows here compose. For a whole-year entry (``before`` on or after the next
January 1st) a second event multiplies the existing factor instead of adding a
row, so the file stays at most one row per (year, symbol) between rebuilds.

What this cannot see
--------------------
An action the provider processes with an ex-date *earlier* than the overlap
window leaves no trace in the overlap, so it is not detected. That is a vendor
back-correction, rare, and exactly what the periodic rebuild and the audit's
corporate-action check exist to catch.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from data_pipeline import schema

logger = logging.getLogger(__name__)

ANCHOR_FILE = "anchor_factors.csv"

YEAR = "year"
SYMBOL = "symbol"
BEFORE = "before"
FACTOR = "factor"
REASON = "reason"
UPDATED_AT = "updated_at"

COLUMNS = [YEAR, SYMBOL, BEFORE, FACTOR, REASON, UPDATED_AT]

#: A relative change in adj_close smaller than this is rounding, not an action.
#: Hot-CSV rows carry six decimals, so for a $1 stock rounding alone is ~5e-7;
#: the smallest real quarterly dividend yields are ~5e-4. 1e-4 sits between.
TOLERANCE = 1e-4

#: Factors are written with enough digits that composing them stays exact to
#: well below TOLERANCE, and in a fixed format so git diffs stay small.
FACTOR_FORMAT = "%.12g"


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------

def anchor_path(root: Optional[Path] = None) -> Path:
    """``data/prices/anchor_factors.csv`` for the canonical lake, beside the manifest."""
    return schema.sidecar_dir(root) / ANCHOR_FILE


def empty_table() -> pd.DataFrame:
    return _coerce(pd.DataFrame({c: [] for c in COLUMNS}))


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[COLUMNS]
    out[YEAR] = pd.to_numeric(out[YEAR], errors="coerce").astype("Int64")
    out[SYMBOL] = out[SYMBOL].astype("string").str.strip().str.upper()
    out[BEFORE] = pd.to_datetime(out[BEFORE], errors="coerce").dt.normalize()
    out[FACTOR] = pd.to_numeric(out[FACTOR], errors="coerce").astype("float64")
    out[REASON] = out[REASON].astype("string").fillna("")
    out[UPDATED_AT] = out[UPDATED_AT].astype("string").fillna("")
    out = out.dropna(subset=[YEAR, SYMBOL, BEFORE, FACTOR])
    return out.sort_values([YEAR, SYMBOL, BEFORE], kind="mergesort").reset_index(drop=True)


def load(root: Optional[Path] = None) -> pd.DataFrame:
    path = anchor_path(root)
    if not path.exists():
        return empty_table()
    return _coerce(pd.read_csv(path))


def save(table: pd.DataFrame, root: Optional[Path] = None) -> Path:
    """Write deterministically. An empty table removes the file."""
    path = anchor_path(root)
    frame = _coerce(table)
    if len(frame) == 0:
        if path.exists():
            path.unlink()
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(
        tmp,
        index=False,
        columns=COLUMNS,
        float_format=FACTOR_FORMAT,
        date_format=schema.CSV_DATE_FORMAT,
        lineterminator=schema.CSV_LINE_TERMINATOR,
    )
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Detection and in-place rescale
# ---------------------------------------------------------------------------

def events_frame(rows: Optional[Iterable[dict]] = None) -> pd.DataFrame:
    """Re-anchor events: symbol, before, factor."""
    frame = pd.DataFrame(list(rows or []), columns=[SYMBOL, BEFORE, FACTOR])
    frame[SYMBOL] = frame[SYMBOL].astype("string")
    frame[BEFORE] = pd.to_datetime(frame[BEFORE])
    frame[FACTOR] = frame[FACTOR].astype("float64")
    return frame


def detect(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    tolerance: float = TOLERANCE,
) -> pd.DataFrame:
    """Symbols whose adj_close anchor moved between the stored and fresh rows.

    ``existing`` must already have pending factors applied -- the comparison is
    against what a reader sees, not the bytes on disk.

    For each symbol the earliest (date, symbol) present in both frames gives
    ``factor = incoming / existing``. That date is ``before``: every stored row
    older than it needs the factor; the rows from it onward are about to be
    replaced by the fresh ones.
    """
    if len(existing) == 0 or len(incoming) == 0:
        return events_frame()

    key = [schema.DATE, schema.SYMBOL]
    old = existing[key + [schema.ADJ_CLOSE]].rename(columns={schema.ADJ_CLOSE: "_old"})
    new = incoming[key + [schema.ADJ_CLOSE]].rename(columns={schema.ADJ_CLOSE: "_new"})
    both = old.merge(new, on=key, how="inner")
    both = both[(both["_old"] > 0) & (both["_new"] > 0)]
    if len(both) == 0:
        return events_frame()

    first = both.sort_values(key, kind="mergesort").groupby(schema.SYMBOL, sort=True).head(1)
    first = first.assign(_ratio=first["_new"] / first["_old"])
    moved = first[(first["_ratio"] - 1.0).abs() > tolerance]

    return events_frame(
        {SYMBOL: str(r[schema.SYMBOL]), BEFORE: r[schema.DATE], FACTOR: float(r["_ratio"])}
        for _, r in moved.iterrows()
    )


def rescale(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Multiply adj_close by each event's factor for rows before its date."""
    if len(frame) == 0 or len(events) == 0:
        return frame
    out = frame.copy()
    adj = out[schema.ADJ_CLOSE].to_numpy(dtype="float64", copy=True)
    symbols = out[schema.SYMBOL].astype(str).to_numpy()
    dates = out[schema.DATE].to_numpy()
    for _, event in events.iterrows():
        mask = (symbols == str(event[SYMBOL])) & (dates < np.datetime64(event[BEFORE]))
        adj[mask] *= float(event[FACTOR])
    out[schema.ADJ_CLOSE] = adj
    return out


# ---------------------------------------------------------------------------
# Pending factors for immutable partitions
# ---------------------------------------------------------------------------

def _year_end_exclusive(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year) + 1, month=1, day=1)


def add(
    table: pd.DataFrame,
    year: int,
    events: pd.DataFrame,
    *,
    reason: str,
) -> pd.DataFrame:
    """Record ``events`` as pending for partition ``year``, composing duplicates.

    ``before`` is clamped to the next January 1st, so every event that covers
    the whole year shares one key and multiplies into one row.
    """
    if len(events) == 0:
        return table
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cap = _year_end_exclusive(year)
    out = _coerce(table)

    for _, event in events.iterrows():
        before = min(pd.Timestamp(event[BEFORE]).normalize(), cap)
        if before <= pd.Timestamp(year=int(year), month=1, day=1):
            continue  # the event only concerns rows after this whole year
        symbol = str(event[SYMBOL]).upper()
        match = (out[YEAR] == int(year)) & (out[SYMBOL] == symbol) & (out[BEFORE] == before)
        if bool(match.any()):
            idx = out.index[match][0]
            out.loc[idx, FACTOR] = float(out.loc[idx, FACTOR]) * float(event[FACTOR])
            out.loc[idx, REASON] = _append_reason(str(out.loc[idx, REASON]), reason)
            out.loc[idx, UPDATED_AT] = stamp
        else:
            out = pd.concat([out, pd.DataFrame([{
                YEAR: int(year), SYMBOL: symbol, BEFORE: before,
                FACTOR: float(event[FACTOR]), REASON: reason, UPDATED_AT: stamp,
            }])], ignore_index=True)
    return _coerce(out)


def _append_reason(existing: str, reason: str) -> str:
    """Keep the reason column short: first and latest cause, with a count."""
    if not existing:
        return reason
    head = existing.split(" | ")[0]
    count = existing.count(" | ") + 2
    return f"{head} | ... | {reason} ({count} events)"


def clear(
    table: pd.DataFrame,
    years: Iterable[int],
    *,
    symbols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Drop pending factors for ``years`` (optionally only for ``symbols``)."""
    wanted_years = {int(y) for y in years}
    if len(table) == 0 or not wanted_years:
        return table
    mask = table[YEAR].isin(wanted_years)
    if symbols is not None:
        wanted = {str(s).upper() for s in symbols}
        mask &= table[SYMBOL].isin(wanted)
    return _coerce(table[~mask])


def apply(frame: pd.DataFrame, year: int, table: pd.DataFrame) -> pd.DataFrame:
    """``frame`` (one partition's rows) with its pending factors applied.

    Whole-year entries -- the overwhelmingly common case -- collapse to one
    multiplier per symbol and apply as a vectorized map, so a read of a large
    cold year with thousands of pending factors stays a single pass.
    """
    if len(frame) == 0 or len(table) == 0:
        return frame
    rows = table[table[YEAR] == int(year)]
    if len(rows) == 0:
        return frame

    cap = _year_end_exclusive(year)
    whole = rows[rows[BEFORE] >= cap]
    partial = rows[rows[BEFORE] < cap]

    out = frame.copy()
    if len(whole):
        per_symbol = whole.groupby(SYMBOL)[FACTOR].prod()
        multiplier = out[schema.SYMBOL].astype(str).map(per_symbol).fillna(1.0)
        out[schema.ADJ_CLOSE] = out[schema.ADJ_CLOSE].astype("float64") * multiplier.to_numpy()
    if len(partial):
        out = rescale(out, partial[[SYMBOL, BEFORE, FACTOR]])
    return out


def symbols_in(path: Path) -> set:
    """Symbols present in one partition, reading only that column."""
    path = Path(path)
    if path.suffix == ".parquet":
        col = pd.read_parquet(path, columns=[schema.SYMBOL], engine=schema.PARQUET_ENGINE)
    else:
        col = pd.read_csv(path, usecols=[schema.SYMBOL])
    return set(col[schema.SYMBOL].astype(str).str.strip().str.upper())

"""schema.py

The on-disk contract for the daily price lake. Both readers and writers go
through here so a Parquet year file and the hot CSV can never disagree.

Layout
------
    data/prices/daily/2023.parquet   cold, committed, rewritten only by rebuild
    data/prices/daily/2024.parquet
    data/prices/daily/2025.parquet
    data/prices/daily/2026.csv       hot, appended daily, committed daily

Cold years are Parquet (compact). The current year is CSV because git deltas
append-only text cheaply, while a recompressed Parquet blob is near-unshareable
between commits. See data/prices/_schema.md for the full rationale.

Byte determinism
----------------
Git only stores a cheap delta if unchanged rows serialize to identical bytes.
Every CSV write therefore goes through ``write_csv`` which pins the float
format, date format, column order, row order, and line terminator. Never call
``DataFrame.to_csv`` on lake data directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------

DATE = "date"
SYMBOL = "symbol"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
ADJ_CLOSE = "adj_close"
VOLUME = "volume"

#: Canonical column order. Parquet and CSV both use exactly this.
COLUMNS: List[str] = [DATE, SYMBOL, OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME]

#: Unadjusted price columns (the actual prints).
RAW_PRICE_COLUMNS: List[str] = [OPEN, HIGH, LOW, CLOSE]

#: Split/dividend adjusted close. Strategies use THIS for returns, moving
#: averages and 52-week highs -- it is what the pre-lake code got from
#: ``yf.download(auto_adjust=True)["Close"]``.
PRICE_COLUMN_FOR_SIGNALS = ADJ_CLOSE

#: Primary key. One row per symbol per session.
KEY: List[str] = [DATE, SYMBOL]

FLOAT_COLUMNS: List[str] = [OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE]


# ---------------------------------------------------------------------------
# Serialization rules (pinned for byte determinism)
# ---------------------------------------------------------------------------

CSV_FLOAT_FORMAT = "%.6f"
CSV_DATE_FORMAT = "%Y-%m-%d"
CSV_LINE_TERMINATOR = "\n"
CSV_ENCODING = "utf-8"

PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"

DEFAULT_LAKE_ROOT = Path(__file__).resolve().parents[2] / "data" / "prices" / "daily"


# ---------------------------------------------------------------------------
# Frame normalization
# ---------------------------------------------------------------------------

def empty_frame() -> pd.DataFrame:
    """Return a correctly typed, empty lake frame."""
    return coerce(pd.DataFrame({c: [] for c in COLUMNS}))


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Force a frame into the lake contract: columns, dtypes, order, sort.

    Idempotent. Safe to call on data read from either Parquet or CSV, which is
    how the two formats are kept from drifting apart.
    """
    if df is None or len(df.columns) == 0:
        return empty_frame()

    out = df.copy()

    missing = [c for c in COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"lake frame missing required columns: {missing}")

    # Drop anything we do not model, then pin column order.
    out = out[COLUMNS]

    # Dates are naive and midnight-normalized: these are daily bars, so a time
    # component only creates spurious inequality between sources.
    out[DATE] = pd.to_datetime(out[DATE], errors="coerce")
    if getattr(out[DATE].dtype, "tz", None) is not None:
        out[DATE] = out[DATE].dt.tz_localize(None)
    out[DATE] = out[DATE].dt.normalize()

    out[SYMBOL] = out[SYMBOL].astype("string").str.strip().str.upper()

    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # Nullable integer, deliberately. A missing volume must stay missing --
    # writing 0 would assert "no shares traded", which is a different claim.
    out[VOLUME] = pd.to_numeric(out[VOLUME], errors="coerce").round().astype("Int64")

    out = out.dropna(subset=KEY)

    # Canonical row order. New sessions sort to the end, so appending a day
    # leaves every earlier byte untouched.
    out = out.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    return out


def upsert(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge ``incoming`` over ``existing``, newest wins on key collision.

    Freshly fetched rows are placed last and ``keep="last"`` retains them, so
    the overlap window in ``sync_prices`` actually applies late corrections
    instead of discarding them.
    """
    a = coerce(existing)
    b = coerce(incoming)

    if len(b) == 0:
        return a
    if len(a) == 0:
        return b

    merged = pd.concat([a, b], ignore_index=True)
    merged = merged.drop_duplicates(subset=KEY, keep="last")
    return coerce(merged)


def assert_unique_key(df: pd.DataFrame, *, context: str = "lake frame") -> None:
    """Raise if the (date, symbol) primary key is violated."""
    dupes = df.duplicated(subset=KEY, keep=False)
    if bool(dupes.any()):
        sample = df.loc[dupes, KEY].head(10).to_dict("records")
        raise ValueError(
            f"{context}: duplicate (date, symbol) keys: {len(df[dupes])} rows, sample {sample}"
        )


# ---------------------------------------------------------------------------
# Year file resolution
# ---------------------------------------------------------------------------

def hot_year_path(year: int, root: Optional[Path] = None) -> Path:
    """CSV path for a year held in the append-friendly hot format."""
    return Path(root or DEFAULT_LAKE_ROOT) / f"{year}.csv"


def cold_year_path(year: int, root: Optional[Path] = None) -> Path:
    """Parquet path for a closed-out year."""
    return Path(root or DEFAULT_LAKE_ROOT) / f"{year}.parquet"


def resolve_year_path(year: int, root: Optional[Path] = None) -> Optional[Path]:
    """Return whichever file actually holds ``year``, or None.

    Parquet wins if both exist, which is the state during a January rollover
    before the superseded CSV is removed.
    """
    cold = cold_year_path(year, root)
    if cold.exists():
        return cold
    hot = hot_year_path(year, root)
    if hot.exists():
        return hot
    return None


def discover_year_files(root: Optional[Path] = None) -> List[Path]:
    """All lake files on disk, oldest year first, Parquet preferred per year."""
    base = Path(root or DEFAULT_LAKE_ROOT)
    if not base.exists():
        return []

    by_year: dict = {}
    for path in base.iterdir():
        if path.suffix not in {".parquet", ".csv"}:
            continue
        try:
            year = int(path.stem)
        except ValueError:
            continue
        current = by_year.get(year)
        if current is None or (current.suffix == ".csv" and path.suffix == ".parquet"):
            by_year[year] = path

    return [by_year[y] for y in sorted(by_year)]


def years_in(df: pd.DataFrame) -> List[int]:
    """Distinct calendar years present, ascending."""
    if len(df) == 0:
        return []
    return sorted({int(y) for y in pd.to_datetime(df[DATE]).dt.year.unique()})


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def read_frame(path: Path) -> pd.DataFrame:
    """Read one lake file (either format) and coerce to the contract."""
    path = Path(path)
    if not path.exists():
        return empty_frame()

    if path.suffix == ".parquet":
        raw = pd.read_parquet(path, engine=PARQUET_ENGINE)
    elif path.suffix == ".csv":
        raw = pd.read_csv(path, parse_dates=[DATE])
    else:
        raise ValueError(f"unsupported lake file format: {path}")

    return coerce(raw)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the hot-year CSV with pinned, byte-deterministic formatting."""
    frame = coerce(df)
    assert_unique_key(frame, context=f"write_csv({path.name})")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        path,
        index=False,
        columns=COLUMNS,
        float_format=CSV_FLOAT_FORMAT,
        date_format=CSV_DATE_FORMAT,
        lineterminator=CSV_LINE_TERMINATOR,
        encoding=CSV_ENCODING,
    )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a cold-year Parquet file."""
    frame = coerce(df)
    assert_unique_key(frame, context=f"write_parquet({path.name})")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(
        path,
        engine=PARQUET_ENGINE,
        compression=PARQUET_COMPRESSION,
        index=False,
    )


def write_year(df: pd.DataFrame, year: int, root: Optional[Path] = None, *, hot: bool) -> Path:
    """Write one year's rows to the appropriate format. Returns the path."""
    frame = coerce(df)
    wrong_year = frame[pd.to_datetime(frame[DATE]).dt.year != year]
    if len(wrong_year) > 0:
        raise ValueError(
            f"write_year({year}) received {len(wrong_year)} rows from other years"
        )

    if hot:
        path = hot_year_path(year, root)
        write_csv(frame, path)
    else:
        path = cold_year_path(year, root)
        write_parquet(frame, path)
    return path

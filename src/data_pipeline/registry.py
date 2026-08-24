"""registry.py

The append-only ticker registry: ``data/universe/master_tickers.csv``.

Distinct from ``universe.csv``, and the distinction matters:

* ``universe.csv``       -- today's *tradable* list. Rebuilt daily by
  ``src/universe.py``, liquidity-filtered, names come and go.
* ``master_tickers.csv`` -- every symbol this pipeline has ever recorded. Rows
  are never deleted, so a delisted or screened-out name keeps its price history
  and stays queryable.

``first_seen`` / ``last_seen`` describe when *we observed* a symbol. They are
NOT point-in-time index membership -- see data/prices/_schema.md, contract 3.

``consecutive_failures`` exists because append-only plus delistings would
otherwise mean the daily sync retries dead tickers forever, burning requests and
burying real failures in noise. Past ``MAX_CONSECUTIVE_FAILURES`` a symbol stops
being fetched but keeps its row and its history.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "data" / "universe" / "master_tickers.csv"
DEFAULT_UNIVERSE_PATH = REPO_ROOT / "universe.csv"

SYMBOL = "symbol"
SECTOR = "sector"
INDUSTRY = "industry"
FIRST_SEEN = "first_seen"
LAST_SEEN = "last_seen"
FAILURES = "consecutive_failures"

REGISTRY_COLUMNS: List[str] = [SYMBOL, SECTOR, INDUSTRY, FIRST_SEEN, LAST_SEEN, FAILURES]

#: Stop fetching a symbol after this many consecutive failures. The row stays.
MAX_CONSECUTIVE_FAILURES = 10

DATE_FORMAT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Always-tracked ETFs
# ---------------------------------------------------------------------------

#: Rotation sleeve sectors.
SECTOR_ETFS: List[str] = [
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
]

#: Rotation sleeve bear-regime hedges.
HEDGE_ETFS: List[str] = ["TLT", "GLD", "UUP", "FXY", "FXF"]

#: Benchmark and cash proxy.
CORE_ETFS: List[str] = ["SPY", "SHY"]

#: Regime ETFs. IJH/IJR are here so the Clenow sleeve's eventual migration is
#: data-ready -- that module is deliberately NOT wired to the lake.
REGIME_ETFS: List[str] = ["IJH", "IJR"]

#: Tracked regardless of any liquidity screen.
ALWAYS_TRACKED: List[str] = list(
    dict.fromkeys(SECTOR_ETFS + HEDGE_ETFS + CORE_ETFS + REGIME_ETFS)
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_symbol(symbol: object) -> str:
    """Yahoo/Alpaca form: upper case, ``.`` -> ``-`` (BRK.B -> BRK-B)."""
    return str(symbol).strip().upper().replace(".", "-")


def empty_registry() -> pd.DataFrame:
    return coerce_registry(pd.DataFrame({c: [] for c in REGISTRY_COLUMNS}))


def coerce_registry(df: pd.DataFrame) -> pd.DataFrame:
    """Force registry dtypes, column order, and stable sort by symbol.

    Sorting by symbol keeps the daily diff confined to the ``last_seen`` column
    instead of reordering rows, which keeps the committed file cheap in git.
    """
    if df is None or len(df.columns) == 0:
        df = pd.DataFrame({c: [] for c in REGISTRY_COLUMNS})

    out = df.copy()
    for col in REGISTRY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[REGISTRY_COLUMNS]

    out[SYMBOL] = out[SYMBOL].map(normalize_symbol).astype("string")
    out[SECTOR] = out[SECTOR].astype("string").fillna("")
    out[INDUSTRY] = out[INDUSTRY].astype("string").fillna("")

    for col in (FIRST_SEEN, LAST_SEEN):
        parsed = pd.to_datetime(out[col], errors="coerce")
        if getattr(parsed.dtype, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        out[col] = parsed.dt.normalize()

    out[FAILURES] = (
        pd.to_numeric(out[FAILURES], errors="coerce").fillna(0).round().astype("int64")
    )

    out = out[out[SYMBOL].notna() & (out[SYMBOL] != "")]
    out = out.drop_duplicates(subset=[SYMBOL], keep="last")
    return out.sort_values(SYMBOL, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def load_registry(path: Optional[Path] = None) -> pd.DataFrame:
    target = Path(path or DEFAULT_REGISTRY_PATH)
    if not target.exists():
        return empty_registry()
    return coerce_registry(pd.read_csv(target))


def write_registry(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Write the registry with pinned formatting (git-friendly diffs)."""
    target = Path(path or DEFAULT_REGISTRY_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)

    frame = coerce_registry(df)
    frame.to_csv(
        target,
        index=False,
        columns=REGISTRY_COLUMNS,
        date_format=DATE_FORMAT,
        lineterminator="\n",
        encoding="utf-8",
    )
    return target


def load_universe_metadata(path: Optional[Path] = None) -> pd.DataFrame:
    """Read ``universe.csv`` (Symbol/Sector/Industry) into registry column names.

    Returns an empty frame if the file is absent -- a missing tradable list is
    not fatal for a sync, it just means no new names are introduced this run.
    """
    target = Path(path or DEFAULT_UNIVERSE_PATH)
    if not target.exists():
        logger.warning("universe.csv not found at %s; no new symbols this run", target)
        return pd.DataFrame(columns=[SYMBOL, SECTOR, INDUSTRY])

    raw = pd.read_csv(target)
    rename = {}
    for col in raw.columns:
        low = str(col).strip().lower()
        if low == "symbol":
            rename[col] = SYMBOL
        elif low == "sector":
            rename[col] = SECTOR
        elif low == "industry":
            rename[col] = INDUSTRY
    out = raw.rename(columns=rename)

    for col in (SYMBOL, SECTOR, INDUSTRY):
        if col not in out.columns:
            out[col] = ""

    out = out[[SYMBOL, SECTOR, INDUSTRY]].copy()
    out[SYMBOL] = out[SYMBOL].map(normalize_symbol)
    out = out[out[SYMBOL] != ""].drop_duplicates(subset=[SYMBOL], keep="first")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Symbol selection
# ---------------------------------------------------------------------------

def target_symbols(
    registry: pd.DataFrame,
    universe_meta: pd.DataFrame,
    *,
    max_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> List[str]:
    """Symbols to fetch: (registry union universe union ETFs) minus dead names.

    The union with the existing registry is what stops a name that fails today's
    liquidity screen from developing a hole in its history.
    """
    known = set(registry[SYMBOL].dropna().astype(str)) if len(registry) else set()
    fresh = set(universe_meta[SYMBOL].dropna().astype(str)) if len(universe_meta) else set()
    etfs = {normalize_symbol(s) for s in ALWAYS_TRACKED}

    candidates = known | fresh | etfs

    if len(registry):
        dead = set(
            registry.loc[registry[FAILURES] >= max_failures, SYMBOL].dropna().astype(str)
        )
        # Never retire an ETF we depend on, or a name back in today's universe.
        dead -= etfs
        dead -= fresh
        if dead:
            logger.info(
                "skipping %d symbol(s) at >=%d consecutive failures (rows retained): %s",
                len(dead),
                max_failures,
                ", ".join(sorted(dead)[:20]) + (" ..." if len(dead) > 20 else ""),
            )
        candidates -= dead

    return sorted(candidates)


def update_registry(
    registry: pd.DataFrame,
    universe_meta: pd.DataFrame,
    *,
    succeeded: Iterable[str],
    failed: Iterable[str],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Fold one sync's outcome into the registry. Never drops a row.

    ``first_seen`` is set once and preserved. ``last_seen`` advances only on a
    successful fetch. ``consecutive_failures`` resets to 0 on success and
    increments on failure.
    """
    current = coerce_registry(registry)
    meta = (
        universe_meta.set_index(SYMBOL)
        if len(universe_meta)
        else pd.DataFrame(columns=[SECTOR, INDUSTRY])
    )

    ok = {normalize_symbol(s) for s in succeeded}
    bad = {normalize_symbol(s) for s in failed} - ok
    as_of = pd.Timestamp(as_of).normalize()

    rows: Dict[str, dict] = {
        str(r[SYMBOL]): dict(r) for _, r in current.iterrows()
    }

    for symbol in sorted(ok | bad | set(meta.index.astype(str))):
        row = rows.get(
            symbol,
            {
                SYMBOL: symbol,
                SECTOR: "",
                INDUSTRY: "",
                FIRST_SEEN: pd.NaT,
                LAST_SEEN: pd.NaT,
                FAILURES: 0,
            },
        )

        if symbol in meta.index:
            sector = meta.loc[symbol, SECTOR]
            industry = meta.loc[symbol, INDUSTRY]
            if isinstance(sector, pd.Series):
                sector = sector.iloc[0]
            if isinstance(industry, pd.Series):
                industry = industry.iloc[0]
            if pd.notna(sector) and str(sector):
                row[SECTOR] = str(sector)
            if pd.notna(industry) and str(industry):
                row[INDUSTRY] = str(industry)

        if symbol in ok:
            if pd.isna(row.get(FIRST_SEEN)):
                row[FIRST_SEEN] = as_of
            row[LAST_SEEN] = as_of
            row[FAILURES] = 0
        elif symbol in bad:
            row[FAILURES] = int(row.get(FAILURES) or 0) + 1

        rows[symbol] = row

    return coerce_registry(pd.DataFrame(list(rows.values())))

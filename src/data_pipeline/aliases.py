"""Ticker renames, recorded explicitly.

    data/universe/symbol_aliases.csv
        old_symbol, new_symbol, effective_date, source, recorded_at

Why this file exists
--------------------
Alpaca resolves a ticker to the underlying company as of a date (its ``asof``
parameter, on by default). Asking for ``META`` from 2016 returns Facebook's
bars, labelled ``META``, because it is the same company. That is what keeps a
renamed company's history continuous in the lake -- and it means the lake
holds 2018 bars under a ticker that did not exist in 2018.

Index membership records the ticker *as it was*: ``FB`` was the S&P 500 member
in 2018. So a join of 2018 membership against the lake misses the company
entirely unless something says ``FB`` became ``META``. Leaving that mapping
implicit inside a vendor parameter is how a coverage number goes quietly wrong,
so the rename events are fetched from the provider's corporate-actions feed and
written here, where they can be read, diffed in git, and audited.

``successor(symbol)`` follows the chain (``A -> B -> C``) to the name the lake
stores the company under. Cycles, which a data error could create, stop the
walk rather than loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline.registry import normalize_symbol

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIASES_PATH = REPO_ROOT / "data" / "universe" / "symbol_aliases.csv"

COLUMNS = ["old_symbol", "new_symbol", "effective_date", "source", "recorded_at"]
KEY = ["old_symbol", "new_symbol", "effective_date"]


def load(path: Optional[Path] = None) -> pd.DataFrame:
    target = Path(path or ALIASES_PATH)
    if not target.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    for col in COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    return frame[COLUMNS]


def merge(new: pd.DataFrame, path: Optional[Path] = None) -> Dict[str, int]:
    """Union ``new`` rename rows into the file. Existing rows are never altered."""
    target = Path(path or ALIASES_PATH)
    current = load(target)
    if new is None or len(new) == 0:
        return {"added": 0, "total": len(current)}

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    incoming = pd.DataFrame({
        "old_symbol": new["old_symbol"].map(normalize_symbol),
        "new_symbol": new["new_symbol"].map(normalize_symbol),
        "effective_date": pd.to_datetime(new["effective_date"], errors="coerce")
        .dt.strftime("%Y-%m-%d").fillna(""),
        "source": new.get("source", pd.Series("", index=new.index)).astype(str),
        "recorded_at": stamp,
    })
    incoming = incoming[incoming["old_symbol"] != incoming["new_symbol"]]

    known = set(map(tuple, current[KEY].astype(str).values.tolist()))
    fresh = incoming[[tuple(r) not in known for r in incoming[KEY].astype(str).values.tolist()]]
    fresh = fresh.drop_duplicates(subset=KEY)
    if len(fresh) == 0:
        return {"added": 0, "total": len(current)}

    combined = pd.concat([current, fresh], ignore_index=True)
    combined = combined.sort_values(["effective_date", "old_symbol", "new_symbol"], kind="mergesort")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(target)
    logger.info("symbol aliases: %d new rename(s), %d total", len(fresh), len(combined))
    return {"added": int(len(fresh)), "total": int(len(combined))}


def _events(frame: pd.DataFrame) -> List[Tuple[pd.Timestamp, str, str]]:
    """Rename events in effective-date order."""
    ordered = frame.sort_values("effective_date", kind="mergesort")
    events = []
    for effective_date, old_symbol, new_symbol in ordered[
        ["effective_date", "old_symbol", "new_symbol"]
    ].itertuples(
        index=False, name=None
    ):
        effective = pd.to_datetime(effective_date, errors="coerce")
        if pd.notna(effective):
            events.append(
                (pd.Timestamp(effective), normalize_symbol(old_symbol), normalize_symbol(new_symbol))
            )
    return events


def _resolve(
    symbol: str,
    events: Sequence[Tuple[pd.Timestamp, str, str]],
    since: Optional[object] = None,
) -> str:
    """Follow chronologically possible renames after ``since``."""
    current = normalize_symbol(symbol)
    threshold = pd.Timestamp(since) if since is not None and not pd.isna(since) else None

    seen = {current}
    for effective, old_symbol, new_symbol in events:
        if threshold is not None and effective < threshold:
            continue
        if old_symbol != current:
            continue
        if new_symbol in seen:
            break
        current = new_symbol
        seen.add(current)
    return current


def successor(
    symbol: str,
    table: Optional[pd.DataFrame] = None,
    *,
    since: Optional[object] = None,
) -> str:
    """The latest name ``symbol`` was renamed to, following chains."""
    frame = load() if table is None else table
    return _resolve(symbol, _events(frame), since) if len(frame) else normalize_symbol(symbol)


def successors(
    symbols: Sequence[str],
    table: Optional[pd.DataFrame] = None,
    *,
    since: Optional[object] = None,
) -> Dict[str, str]:
    """Resolve many symbols after building the rename graph once."""
    frame = load() if table is None else table
    events = _events(frame) if len(frame) else []
    return {normalize_symbol(s): _resolve(s, events, since) for s in symbols}


def successors_asof(
    requests: Sequence[Tuple[str, object]],
    table: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Resolve ``(symbol, first-known-date)`` requests with one event parse."""
    frame = load() if table is None else table
    events = _events(frame) if len(frame) else []
    return [_resolve(symbol, events, since) for symbol, since in requests]


def refresh(
    provider,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    symbols: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
) -> Dict[str, int]:
    """Fetch rename events from ``provider`` and merge them in.

    Requested a year at a time: corporate-action endpoints commonly bound the
    date range per request, and a year-sized window keeps each response small
    whatever the bound is.
    """
    if not hasattr(provider, "fetch_symbol_changes"):
        raise TypeError(f"{provider!r} cannot report symbol changes")
    frames = []
    window_start = pd.Timestamp(start).normalize()
    final = pd.Timestamp(end).normalize()
    while window_start <= final:
        window_end = min(window_start + pd.Timedelta(days=364), final)
        frames.append(provider.fetch_symbol_changes(window_start, window_end, symbols=symbols))
        window_start = window_end + pd.Timedelta(days=1)
    changes = pd.concat(frames, ignore_index=True) if frames else None
    return merge(changes, path)

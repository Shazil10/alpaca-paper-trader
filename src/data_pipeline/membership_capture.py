"""Daily capture of unfiltered S&P 500 / 400 / 600 membership.

    data/universe/membership_baseline.csv   first capture, written once
    data/universe/membership_events.csv     add/remove events, append-only
    data/universe/current_membership.csv    today's unfiltered constituents

Why capture at all
------------------
The historical S&P 500 membership (``membership.parquet``, from fja05680)
reaches back to 1996; nothing free does the same for the S&P 400 and 600. So
the only honest way to get point-in-time mid- and small-cap membership is to
start recording it now and keep recording it. Every day this runs, the
S&P 1500 history gets one day longer. Every day it does not, that day is lost.

The capture is **unfiltered**. The tradable ``universe.csv`` is a liquidity
screen of these names, and it changes for reasons that have nothing to do with
the index -- a stock dipping under $10 leaves it. Recording membership from the
screened list would turn every price dip into a fake index removal.

What is recorded, and what that means
-------------------------------------
* The **first** capture is a baseline, not a set of additions. Those names were
  members before we looked; when they joined is unknown. Their intervals are
  therefore *left-censored* at the baseline date, and membership before it is
  reported as not covered rather than guessed.
* Later captures are diffed against the state implied by baseline + events,
  and each difference is an ``add`` or ``remove`` event.
* ``effective_date`` equals ``observed_date``: the capture sees the change the
  first day it is published, and the true effective date is on or before it.
  It is an upper bound, and documented as one.
* A move between indices (S&P 600 -> 400 is the common one) is two events: a
  ``remove`` from one and an ``add`` to the other, usually on the same day. The
  S&P 1500 union treats the name as continuously present; see
  ``membership.intervals("SP1500")``.

A failed scrape must not become 1,500 removals
----------------------------------------------
The single most damaging failure here is quiet: Wikipedia changes a table
layout, the parser finds 12 rows, and the diff records 488 removals that every
later backtest believes. So a capture is checked before anything is written --
per-index counts in their normal range, the union near 1,500, the required
columns present, and no more than ``MAX_REMOVAL_FRACTION`` of an index removed
in one day. Any failure raises ``SuspiciousScrapeError`` and leaves all three
files untouched. Missing a day costs one day of precision; recording a phantom
purge corrupts the history permanently.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline.registry import normalize_symbol

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_DIR = REPO_ROOT / "data" / "universe"

BASELINE_PATH = UNIVERSE_DIR / "membership_baseline.csv"
EVENTS_PATH = UNIVERSE_DIR / "membership_events.csv"
CURRENT_PATH = UNIVERSE_DIR / "current_membership.csv"

INDICES: Tuple[str, ...] = ("SP500", "SP400", "SP600")

SOURCES: Dict[str, str] = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

#: Normal constituent counts. Share classes put the S&P 500 at ~503; the others
#: drift by a few names between rebalances. Outside these ranges the scrape is
#: far more likely broken than the index.
EXPECTED_COUNTS: Dict[str, Tuple[int, int]] = {
    "SP500": (480, 520),
    "SP400": (380, 420),
    "SP600": (560, 640),
}
EXPECTED_UNION: Tuple[int, int] = (1440, 1560)

#: More removals than this share of an index in one capture is refused. A
#: quarterly S&P 600 rebalance changes a few dozen names at most.
MAX_REMOVAL_FRACTION = 0.10

SOURCE_LABEL = "wikipedia"

SNAPSHOT_COLUMNS = ["index", "symbol", "security", "sector", "industry"]
CURRENT_COLUMNS = ["observed_date", "index", "symbol", "security", "sector", "industry", "source"]
BASELINE_COLUMNS = ["observed_date", "index", "symbol", "source"]
EVENT_COLUMNS = ["observed_date", "effective_date", "index", "symbol", "event", "source"]

DATE_FORMAT = "%Y-%m-%d"


class SuspiciousScrapeError(ValueError):
    """The scrape does not look like the index. Nothing was written."""


# ---------------------------------------------------------------------------
# Scrape
# ---------------------------------------------------------------------------

_SYMBOL_HEADERS = ("symbol", "ticker symbol", "ticker")
_SECURITY_HEADERS = ("security", "company")
_SECTOR_HEADERS = ("gics sector", "sector")
_INDUSTRY_HEADERS = ("gics sub-industry", "gics sub industry", "sub-industry", "industry")


def _pick(columns: Sequence[str], wanted: Sequence[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    for name in wanted:
        if name in lowered:
            return lowered[name]
    return None


def parse_table(html: str, index: str) -> pd.DataFrame:
    """The constituents table from one Wikipedia page, in snapshot form.

    Picks the first table with a symbol column *and* a sector column -- the
    changes table further down the S&P 500 page has symbols but no sectors, and
    reading it instead would be exactly the silent layout failure the checks
    exist for.
    """
    tables = pd.read_html(io.StringIO(html))
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            continue
        symbol_col = _pick(table.columns, _SYMBOL_HEADERS)
        sector_col = _pick(table.columns, _SECTOR_HEADERS)
        if symbol_col is None or sector_col is None:
            continue
        security_col = _pick(table.columns, _SECURITY_HEADERS)
        industry_col = _pick(table.columns, _INDUSTRY_HEADERS)

        frame = pd.DataFrame({
            "index": index,
            "symbol": table[symbol_col].astype(str),
            "security": table[security_col].astype(str) if security_col else "",
            "sector": table[sector_col].astype(str),
            "industry": table[industry_col].astype(str) if industry_col else "",
        })
        return normalize_snapshot(frame)
    raise SuspiciousScrapeError(
        f"{index}: no table with both a symbol and a sector column"
    )


def normalize_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize symbols, drop blanks and duplicates, fix column order."""
    out = frame.copy()
    for col in SNAPSHOT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[SNAPSHOT_COLUMNS]
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out = out[(out["symbol"] != "") & (out["symbol"] != "NAN")]
    for col in ("security", "sector", "industry"):
        out[col] = out[col].fillna("").astype(str).str.strip().replace({"nan": ""})
    out = out.drop_duplicates(subset=["index", "symbol"], keep="first")
    return out.sort_values(["index", "symbol"], kind="mergesort").reset_index(drop=True)


def fetch_index(index: str, *, get: Optional[Callable] = None, timeout: int = 30) -> pd.DataFrame:
    """Download and parse one index page."""
    if get is None:
        import requests
        get = requests.get
    response = get(SOURCES[index], headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    status = getattr(response, "status_code", 200)
    if status != 200:
        raise SuspiciousScrapeError(f"{index}: HTTP {status} from {SOURCES[index]}")
    return parse_table(response.text, index)


def scrape(*, get: Optional[Callable] = None) -> pd.DataFrame:
    """All three indices. Any page failing fails the whole capture.

    Partial captures are refused on purpose: a snapshot missing the S&P 400
    would diff as 400 removals.
    """
    frames = []
    for index in INDICES:
        try:
            frames.append(fetch_index(index, get=get))
        except SuspiciousScrapeError:
            raise
        except Exception as exc:
            raise SuspiciousScrapeError(f"{index}: scrape failed: {exc}") from exc
    return normalize_snapshot(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# State: baseline + events -> who is a member now
# ---------------------------------------------------------------------------

def _read_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=list(columns))
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in columns:
        if col not in frame.columns:
            frame[col] = ""
    return frame[list(columns)]


def load_baseline(path: Optional[Path] = None) -> pd.DataFrame:
    return _read_csv(Path(path or BASELINE_PATH), BASELINE_COLUMNS)


def load_events(path: Optional[Path] = None) -> pd.DataFrame:
    return _read_csv(Path(path or EVENTS_PATH), EVENT_COLUMNS)


def state_from(baseline: pd.DataFrame, events: pd.DataFrame) -> Dict[str, set]:
    """Members per index implied by the baseline plus every event, in order."""
    state: Dict[str, set] = {index: set() for index in INDICES}
    for _, row in baseline.iterrows():
        state.setdefault(str(row["index"]), set()).add(str(row["symbol"]))
    if len(events):
        # Stable by date: within one day the file's own order is the order the
        # captures happened, which matters if a same-day re-run reversed one.
        ordered = events.sort_values(["effective_date"], kind="mergesort")
        for _, row in ordered.iterrows():
            members = state.setdefault(str(row["index"]), set())
            if row["event"] == "add":
                members.add(str(row["symbol"]))
            elif row["event"] == "remove":
                members.discard(str(row["symbol"]))
    return state


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check(snapshot: pd.DataFrame, previous: Optional[Dict[str, set]] = None) -> List[str]:
    """Reasons to refuse this snapshot. Empty means it looks like the index."""
    problems: List[str] = []
    missing = [c for c in SNAPSHOT_COLUMNS if c not in snapshot.columns]
    if missing:
        return [f"snapshot missing columns {missing}"]

    for index in INDICES:
        members = set(snapshot.loc[snapshot["index"] == index, "symbol"])
        low, high = EXPECTED_COUNTS[index]
        if not low <= len(members) <= high:
            problems.append(
                f"{index}: {len(members)} constituents, expected {low}-{high}"
            )
        sectors = snapshot.loc[snapshot["index"] == index, "sector"]
        if len(sectors) and (sectors == "").mean() > 0.05:
            problems.append(f"{index}: more than 5% of rows have no sector")

        if previous and previous.get(index):
            removed = previous[index] - members
            cap = max(1, int(len(previous[index]) * MAX_REMOVAL_FRACTION))
            if len(removed) > cap:
                problems.append(
                    f"{index}: {len(removed)} removals in one capture exceeds the "
                    f"cap of {cap} ({MAX_REMOVAL_FRACTION:.0%} of {len(previous[index])})"
                )

    union = snapshot["symbol"].nunique()
    low, high = EXPECTED_UNION
    if not low <= union <= high:
        problems.append(f"S&P 1500 union: {union} unique symbols, expected {low}-{high}")
    return problems


# ---------------------------------------------------------------------------
# Diff and write
# ---------------------------------------------------------------------------

def diff(
    previous: Dict[str, set],
    snapshot: pd.DataFrame,
    observed_date: pd.Timestamp,
    *,
    source: str = SOURCE_LABEL,
) -> pd.DataFrame:
    """add/remove events between the implied state and a new snapshot."""
    day = pd.Timestamp(observed_date).strftime(DATE_FORMAT)
    rows = []
    for index in INDICES:
        now = set(snapshot.loc[snapshot["index"] == index, "symbol"])
        before = previous.get(index, set())
        for symbol in sorted(now - before):
            rows.append({"observed_date": day, "effective_date": day, "index": index,
                         "symbol": symbol, "event": "add", "source": source})
        for symbol in sorted(before - now):
            rows.append({"observed_date": day, "effective_date": day, "index": index,
                         "symbol": symbol, "event": "remove", "source": source})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _write(frame: pd.DataFrame, path: Path, columns: Sequence[str], sort: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[list(columns)].sort_values(list(sort), kind="mergesort")
    tmp = path.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def capture(
    snapshot: pd.DataFrame,
    observed_date: pd.Timestamp,
    *,
    baseline_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
    current_path: Optional[Path] = None,
    source: str = SOURCE_LABEL,
) -> Dict[str, object]:
    """Check, diff, and record one snapshot. Raises before writing if suspicious.

    Idempotent: re-running with the same snapshot on the same day adds no
    events, because the diff is against the state the events already imply.
    """
    baseline_path = Path(baseline_path or BASELINE_PATH)
    events_path = Path(events_path or EVENTS_PATH)
    current_path = Path(current_path or CURRENT_PATH)

    snapshot = normalize_snapshot(snapshot)
    day = pd.Timestamp(observed_date).normalize()
    baseline = load_baseline(baseline_path)
    events = load_events(events_path)
    first = len(baseline) == 0
    previous = None if first else state_from(baseline, events)

    problems = check(snapshot, previous)
    if problems:
        raise SuspiciousScrapeError("; ".join(problems))

    new_events = pd.DataFrame(columns=EVENT_COLUMNS)
    if first:
        rows = snapshot.assign(observed_date=day.strftime(DATE_FORMAT), source=source)
        _write(rows, baseline_path, BASELINE_COLUMNS, ["index", "symbol"])
        logger.info(
            "membership baseline recorded on %s: %s", day.date(),
            ", ".join(f"{i} {int((snapshot['index'] == i).sum())}" for i in INDICES),
        )
    else:
        new_events = diff(previous, snapshot, day, source=source)
        if len(new_events):
            # No de-duplication needed: the diff is against the state the
            # existing events imply, so a repeat capture yields no events.
            combined = pd.concat([events, new_events], ignore_index=True)
            # Sorted by date only, stably: append order within a day is
            # meaningful (see state_from), and appending keeps git diffs small.
            _write(combined, events_path, EVENT_COLUMNS, ["observed_date"])
        logger.info(
            "membership capture %s: %d add(s), %d remove(s)", day.date(),
            int((new_events["event"] == "add").sum()),
            int((new_events["event"] == "remove").sum()),
        )

    current = snapshot.assign(observed_date=day.strftime(DATE_FORMAT), source=source)
    _write(current, current_path, CURRENT_COLUMNS, ["index", "symbol"])

    return {
        "observed_date": str(day.date()),
        "baseline": first,
        "counts": {i: int((snapshot["index"] == i).sum()) for i in INDICES},
        "union": int(snapshot["symbol"].nunique()),
        "adds": new_events[new_events["event"] == "add"][["index", "symbol"]].to_dict("records"),
        "removes": new_events[new_events["event"] == "remove"][["index", "symbol"]].to_dict("records"),
    }


def load_current(path: Optional[Path] = None) -> pd.DataFrame:
    """Today's unfiltered constituents, all three indices."""
    return _read_csv(Path(path or CURRENT_PATH), CURRENT_COLUMNS)

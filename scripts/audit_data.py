#!/usr/bin/env python
"""Audit what the backtester's data actually supports, and what it does not.

    PYTHONPATH=src ./venv/bin/python scripts/audit_data.py
    PYTHONPATH=src ./venv/bin/python scripts/audit_data.py --json runs/audit.json

``check_lake_readiness.py`` answers a yes/no question -- is the lake good enough
to trade or to backtest. This answers the *shape* of what is there, because the
limitations are not going away and a number attached to each one is worth more
than a warning. Six questions, one section each:

1. Which symbols have complete price history, and complete over what span?
2. Are delisted names actually present, or only survivors?
3. Are historical ticker changes mapped, or silently dropped?
4. Can every historical index member be priced on the dates it was a member?
5. Which membership dates are exact, and how wrong can the rest be?
6. Are sector labels historical or current?

Nothing here fails a build. It is a report, and its job is to make the size of
each gap explicit so a result can be discounted by the right amount rather than
either trusted or dismissed wholesale.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_pipeline import membership, securities, store  # noqa: E402

logger = logging.getLogger("audit_data")

SNAPSHOT_PATH = REPO_ROOT / "data" / "universe" / "membership_snapshots.csv"
REGISTRY_PATH = REPO_ROOT / "data" / "universe" / "master_tickers.csv"

#: Dates to probe index membership against. Chosen to straddle the regimes a
#: backtest would care about rather than spaced evenly.
PROBE_DATES = (
    "2008-06-30",   # pre-crisis peak membership
    "2009-03-31",   # the bottom, when delistings cluster
    "2012-06-29",
    "2016-06-30",
    "2020-03-31",   # COVID crash
    "2024-06-28",
)

#: A symbol ending in Q traded on the pink sheets after a bankruptcy filing. The
#: membership source records several companies under that final ticker rather than
#: the one they traded under while in the index, and no price vendor has history
#: for it -- so this is the signature of the ticker-mapping gap, not a data error.
BANKRUPTCY_SUFFIX = "Q"


@dataclass
class Finding:
    """One audited question."""
    name: str
    headline: str = ""
    numbers: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    table: Optional[pd.DataFrame] = None

    def render(self, width: int = 78) -> str:
        lines = ["", self.name, "-" * min(len(self.name), width)]
        if self.headline:
            lines.append(self.headline)
        if self.numbers:
            lines.append("")
            label_width = max(len(k) for k in self.numbers)
            for key, value in self.numbers.items():
                if isinstance(value, float):
                    shown = f"{value:,.2f}"
                elif isinstance(value, int):
                    shown = f"{value:,}"
                else:
                    shown = str(value)
                lines.append(f"  {key:<{label_width}}  {shown}")
        if self.table is not None and len(self.table) > 0:
            lines.append("")
            lines.extend(
                "  " + row for row in self.table.to_string(index=False).split("\n")
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "numbers": self.numbers,
            "notes": self.notes,
            "table": (
                self.table.to_dict("records") if self.table is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# 1. History completeness
# ---------------------------------------------------------------------------

def audit_history(coverage: pd.DataFrame, calendar: pd.DatetimeIndex) -> Finding:
    """Complete over a symbol's *own* span, not over the whole calendar.

    A name listed in 2018 is not incomplete for having no 2005 bars, and grading
    it against the full calendar would bury the symbols that genuinely have holes.
    So completeness is measured against the sessions that fall inside each
    symbol's own first-to-last window.
    """
    finding = Finding(name="1. Price history completeness")
    if len(coverage) == 0:
        finding.headline = "the lake is empty"
        return finding

    cov = coverage.copy()
    cov["first_date"] = pd.to_datetime(cov["first_date"])
    cov["last_date"] = pd.to_datetime(cov["last_date"])

    expected = np.array([
        int(((calendar >= first) & (calendar <= last)).sum())
        for first, last in zip(cov["first_date"], cov["last_date"])
    ])
    cov["expected"] = expected
    cov["completeness"] = np.where(expected > 0, cov["bars"] / expected, np.nan)
    cov["span_years"] = (cov["last_date"] - cov["first_date"]).dt.days / 365.25

    complete = cov[cov["completeness"] >= 0.999]
    ragged = cov[cov["completeness"] < 0.99].sort_values("completeness")
    stubs = cov[cov["bars"] <= 2]
    deep = cov[cov["span_years"] >= 15]

    finding.numbers = {
        "symbols in lake": int(len(cov)),
        "complete over own span": int(len(complete)),
        "under 99% complete": int(len(ragged)),
        "1-2 bar stubs": int(len(stubs)),
        "span >= 15 years": int(len(deep)),
        "median span (years)": float(cov["span_years"].median()),
    }
    finding.headline = (
        f"{len(complete)}/{len(cov)} symbols are complete over their own span; "
        f"{len(deep)} reach 15+ years"
    )

    if len(ragged) > 0:
        finding.table = (
            ragged.head(10)[["symbol", "first_date", "last_date", "bars", "expected"]]
            .assign(pct=(ragged.head(10)["completeness"] * 100).round(1))
        )
        finding.notes.append(
            "Ragged symbols are usually halts or mid-delisting names where the "
            "vendor itself has no bar; see _schema.md Contract 5."
        )
    if len(deep) < 30:
        finding.notes.append(
            f"only {len(deep)} symbol(s) carry 15+ years, so anything needing a "
            "long sample is an ETF-only study until the equity backfill runs"
        )
    return finding


# ---------------------------------------------------------------------------
# 2. Delisted presence
# ---------------------------------------------------------------------------

def audit_delisted(coverage: pd.DataFrame) -> Finding:
    """Are the names that left the index present, or only today's survivors?

    This is the survivorship question stated precisely. A former member counts as
    *usable* only if the lake can price it during the window it was a member --
    having 2024 bars for a company removed in 2009 proves nothing.
    """
    finding = Finding(name="2. Delisted and removed names")
    if not membership.has_pit_membership():
        finding.headline = "no membership file; survivorship cannot be assessed"
        return finding

    intervals = pd.read_parquet(
        REPO_ROOT / "data" / "universe" / "membership.parquet"
    )
    for column in ("start_date", "end_date"):
        intervals[column] = pd.to_datetime(intervals[column], errors="coerce")

    removed = intervals[intervals["end_date"].notna()].copy()
    have = coverage.set_index("symbol") if len(coverage) else pd.DataFrame()

    usable = 0
    priced_but_wrong_window = 0
    absent = 0

    for _, row in removed.iterrows():
        symbol = row["symbol"]
        if symbol not in have.index:
            absent += 1
            continue
        first = pd.Timestamp(have.loc[symbol, "first_date"])
        last = pd.Timestamp(have.loc[symbol, "last_date"])
        # Any overlap between the lake's span and the membership window.
        if first <= row["end_date"] and last >= row["start_date"]:
            usable += 1
        else:
            priced_but_wrong_window += 1

    total = len(removed)
    finding.numbers = {
        "removed-from-index intervals": int(total),
        "priced during membership": int(usable),
        "priced, but not then": int(priced_but_wrong_window),
        "no bars at all": int(absent),
        "usable share": f"{usable / total:.1%}" if total else "n/a",
    }
    finding.headline = (
        f"{usable} of {total} removed intervals can be priced during the window "
        f"they were actually members"
    )
    finding.notes.append(
        "The remainder is the residual survivorship bias. It is not zero and "
        "results should not be described as survivorship-free."
    )
    if priced_but_wrong_window:
        finding.notes.append(
            f"{priced_but_wrong_window} symbol(s) have bars only outside their "
            "membership window -- a recycled ticker, or a lake that starts later"
        )
    return finding


# ---------------------------------------------------------------------------
# 3. Ticker mapping
# ---------------------------------------------------------------------------

def audit_ticker_mapping(coverage: pd.DataFrame) -> Finding:
    """How many membership symbols the lake simply cannot join, and why.

    The membership source records many companies under their *final* ticker.
    Lehman is ``LEHMQ``, never ``LEH``. No vendor has history under a
    post-bankruptcy symbol, so those names are unfetchable rather than missing --
    a different problem with a different fix (a ticker-change table keyed on a
    permanent id), and one worth counting rather than rediscovering.
    """
    finding = Finding(name="3. Historical ticker changes")
    if not membership.has_pit_membership():
        finding.headline = "no membership file"
        return finding

    intervals = pd.read_parquet(
        REPO_ROOT / "data" / "universe" / "membership.parquet"
    )
    symbols = set(intervals["symbol"].astype(str))
    priced = set(coverage["symbol"].astype(str)) if len(coverage) else set()
    missing = sorted(symbols - priced)

    suffixed = [s for s in missing if s.endswith(BANKRUPTCY_SUFFIX) and len(s) > 3]
    lake_start = (
        pd.Timestamp(coverage["first_date"].min()) if len(coverage) else None
    )

    ended_before_lake = 0
    if lake_start is not None:
        for _, row in intervals[intervals["symbol"].isin(missing)].iterrows():
            end = pd.to_datetime(row["end_date"], errors="coerce")
            if pd.notna(end) and end < lake_start:
                ended_before_lake += 1

    finding.numbers = {
        "membership symbols": int(len(symbols)),
        "priced by the lake": int(len(symbols & priced)),
        "never priced": int(len(missing)),
        "of those, bankruptcy-suffixed": int(len(suffixed)),
        "of those, left index before lake starts": int(ended_before_lake),
    }
    finding.headline = (
        f"{len(missing)} of {len(symbols)} membership symbols have no bars; "
        f"{len(suffixed)} carry a post-bankruptcy ticker no vendor covers"
    )
    if suffixed:
        finding.table = pd.DataFrame({"unfetchable_ticker": suffixed[:12]})
    finding.notes.append(
        "No ticker-change table exists, so a rename is indistinguishable from a "
        "missing symbol. Resolving it needs a permanent id -- Tiingo permaTicker, "
        "FIGI or CUSIP. See data/universe/_schema.md contract 2."
    )
    return finding


# ---------------------------------------------------------------------------
# 4. Member coverage on real dates
# ---------------------------------------------------------------------------

def audit_member_coverage(calendar: pd.DatetimeIndex) -> Finding:
    """On a given historical date, what share of the index could we trade?

    The single most decision-relevant number in this report. A backtest on a date
    where 60% of the index is priceable is not testing the index; it is testing
    whichever 60% survived to be in today's symbol list.
    """
    finding = Finding(name="4. Index member coverage by date")
    if not membership.has_pit_membership():
        finding.headline = "no membership file"
        return finding
    if len(calendar) == 0:
        finding.headline = "empty calendar"
        return finding

    lake_start, lake_end = calendar[0], calendar[-1]
    rows = []

    for probe in PROBE_DATES:
        date = pd.Timestamp(probe)
        members = membership.members_asof(date)
        if not members:
            continue

        if date < lake_start or date > lake_end:
            rows.append({
                "date": probe, "members": len(members),
                "priced": 0, "coverage": "outside lake",
            })
            continue

        panel = store.load_close_matrix(sorted(members), start=date, end=date)
        priced = 0 if panel.empty else int(panel.iloc[0].notna().sum())
        rows.append({
            "date": probe, "members": len(members), "priced": priced,
            "coverage": f"{priced / len(members):.0%}",
        })

    finding.table = pd.DataFrame(rows)
    inside = [r for r in rows if r["coverage"] != "outside lake"]
    if inside:
        best = max(inside, key=lambda r: r["priced"] / max(r["members"], 1))
        worst = min(inside, key=lambda r: r["priced"] / max(r["members"], 1))
        finding.headline = (
            f"coverage ranges {worst['coverage']} ({worst['date']}) to "
            f"{best['coverage']} ({best['date']}) on the probed dates"
        )
    else:
        finding.headline = (
            f"every probed date falls outside the lake "
            f"({lake_start.date()}..{lake_end.date()})"
        )
        finding.notes.append(
            "The lake does not reach the dates worth probing, which is itself "
            "the finding: no historical index membership can be tested yet."
        )
    return finding


# ---------------------------------------------------------------------------
# 5. Membership date precision
# ---------------------------------------------------------------------------

def audit_membership_precision() -> Finding:
    """How wrong can a membership date be?

    The source is a series of snapshots, not a change log, so a change is dated to
    the first snapshot that shows it and the true date lies somewhere in the gap
    before it. That gap is the error bar, and it is not constant -- early years are
    sampled far more sparsely than recent ones.
    """
    finding = Finding(name="5. Membership date precision")
    if not SNAPSHOT_PATH.exists():
        finding.headline = "no snapshot dates recorded"
        finding.notes.append(
            "Re-run scripts/build_membership.py to write "
            "data/universe/membership_snapshots.csv."
        )
        return finding

    dates = pd.to_datetime(
        pd.read_csv(SNAPSHOT_PATH)["snapshot_date"]
    ).sort_values()
    gaps = dates.diff().dt.days.dropna()

    by_year = (
        pd.DataFrame({"year": dates.dt.year})
        .groupby("year").size().rename("snapshots")
    )
    sparse = by_year[by_year <= 12]

    finding.numbers = {
        "snapshots": int(len(dates)),
        "range": f"{dates.iloc[0]:%Y-%m-%d} .. {dates.iloc[-1]:%Y-%m-%d}",
        "median gap (days)": float(gaps.median()),
        "90th pct gap (days)": float(gaps.quantile(0.90)),
        "worst gap (days)": int(gaps.max()),
        "years sampled <=12 times": int(len(sparse)),
    }
    finding.headline = (
        f"a membership date is accurate to the preceding gap: median "
        f"{gaps.median():.0f} day(s), worst {int(gaps.max())}"
    )
    if len(sparse) > 0:
        finding.table = sparse.reset_index().head(12)
        finding.notes.append(
            "Sparsely sampled years carry the widest error bars on any "
            "membership date inside them."
        )
    finding.notes.append(
        "Immaterial for a monthly-rebalanced sleeve. Material for anything "
        "trading index additions on their effective date."
    )
    return finding


# ---------------------------------------------------------------------------
# 6. Sector label vintage
# ---------------------------------------------------------------------------

def audit_sector_vintage() -> Finding:
    """Current labels or historical ones, and how many names have any."""
    finding = Finding(name="6. Sector label vintage")

    path = REPO_ROOT / "data" / "universe" / "securities.parquet"
    if not path.exists():
        finding.headline = "no security master; run scripts/build_securities.py"
        return finding

    master = pd.read_parquet(path)
    labelled = master[master["sector"].astype(str).str.len() > 0]
    mapping = securities.sector_map()

    registry_dates = None
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH)
        if "first_seen" in registry.columns:
            registry_dates = pd.to_datetime(
                registry["first_seen"], errors="coerce"
            ).dropna()

    finding.numbers = {
        "listing spans": int(len(master)),
        "with a sector label": int(len(labelled)),
        "tickers in sector_map": int(len(mapping)),
        "distinct sectors": int(labelled["sector"].nunique()),
        "vintage": "current only (not point-in-time)",
    }
    if registry_dates is not None and len(registry_dates):
        finding.numbers["registry first observed"] = str(
            registry_dates.min().date()
        )

    finding.headline = (
        f"{len(labelled)} of {len(master)} spans carry a sector, and every label "
        "is today's classification applied to all of history"
    )
    finding.notes.append(
        "A company that changed GICS sector mid-history carries today's label "
        "throughout, and names that left before the registry existed carry none."
    )
    finding.notes.append(
        "Good enough for RiskConfig.max_sector_pct, which does nothing at all "
        "without a map. Not good enough for sector attribution."
    )
    return finding


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_audit() -> List[Finding]:
    coverage = store.coverage()
    calendar = store.trading_calendar()

    return [
        audit_history(coverage, calendar),
        audit_delisted(coverage),
        audit_ticker_mapping(coverage),
        audit_member_coverage(calendar),
        audit_membership_precision(),
        audit_sector_vintage(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", type=str, help="Also write the report as JSON.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    findings = run_audit()

    print("=" * 78)
    print("DATA AUDIT")
    print("=" * 78)
    for finding in findings:
        print(finding.render())

    print()
    print("=" * 78)
    print("This is a report, not a gate. Use scripts/check_lake_readiness.py")
    print("--backtest for pass/fail.")
    print("=" * 78)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {f.name: f.to_dict() for f in findings}, indent=2, default=str
        ))
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

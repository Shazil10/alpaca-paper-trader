#!/usr/bin/env python
"""Classify every index-membership interval the lake cannot fully price.

    PYTHONPATH=src ./venv/bin/python scripts/classify_gaps.py
    PYTHONPATH=src ./venv/bin/python scripts/classify_gaps.py --out data/audits/gaps_latest.json

The audit says *how much* is missing. This says *why*, per membership
interval, because each cause has a different fix -- and one of them, and only
one, is a reason to bring in a second data vendor.

Classes
-------
Each (index, symbol, membership interval) is clipped to the canonical window
(2016-01-01, where Alpaca SIP history begins, to the lake frontier) and put in
exactly one class:

===  =======================  ==================================================
A    covered                  priced on >= 99% of sessions while a member
B    covered via alias        priced only under its renamed ticker
                              (``symbol_aliases.csv``), on >= 99% of sessions
C    holes                    priced, but missing sessions while a member --
                              vendor gaps, halts, or a fetch that never healed
D    before canonical history the whole interval ends before 2016: out of the
                              canonical provider's reach by construction; only
                              legacy or secondary data could ever cover it
E    not carried              no bars at all in the window under the symbol or
                              any alias, with a plausible ticker: typically a
                              delisted or acquired company the vendor dropped
F    unresolvable identifier  no bars, and the ticker itself is the problem: a
                              post-bankruptcy ``...Q`` symbol, or bars that
                              exist only outside the membership window (a
                              recycled ticker) -- needs a permanent id, not a
                              vendor
===  =======================  ==================================================

The Tiingo decision
-------------------
Only class E is a vendor-coverage problem a second vendor can fix: A and B
need nothing, C is usually halts (or a repair run), D is out of scope by
design, and F needs an identifier map. So the recommendation keys on E alone:
if E carries at least ``TIINGO_MIN_SHARE`` of member-sessions in the canonical
window, trial Tiingo on exactly the E list -- through quarantine and
reconciliation, never straight into the lake. Otherwise, do not add a vendor.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_pipeline import aliases, manifest, membership, providers, schema, store  # noqa: E402
from data_pipeline.sync_prices import LOOKBACK_START  # noqa: E402

logger = logging.getLogger("classify_gaps")

COMPLETE = 0.99
BANKRUPTCY_SUFFIX = "Q"

#: Share of member-sessions in class E at which a second vendor is worth a trial.
TIINGO_MIN_SHARE = 0.01

CLASSES = {
    "A": "covered",
    "B": "covered via alias",
    "C": "holes while a member",
    "D": "before canonical history",
    "E": "not carried by the canonical provider",
    "F": "unresolvable identifier",
}


def _intervals(index: str) -> pd.DataFrame:
    frame = membership.intervals(index)
    return frame.assign(index=index) if len(frame) else frame


def classify(
    *,
    indices=("SP500", "SP400", "SP600"),
    window_start: pd.Timestamp = LOOKBACK_START,
    prices: Optional[pd.DataFrame] = None,
    calendar: Optional[pd.DatetimeIndex] = None,
    alias_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per membership interval with its class and the evidence."""
    prices = store.load_prices() if prices is None else prices
    if calendar is None:
        calendar = store.trading_calendar()
    alias_table = aliases.load() if alias_table is None else alias_table
    if len(prices) == 0 or len(calendar) == 0:
        return pd.DataFrame()

    frontier = pd.Timestamp(calendar[-1])
    dates_by_symbol: Dict[str, pd.DatetimeIndex] = {
        str(s): pd.DatetimeIndex(g[schema.DATE].unique())
        for s, g in prices.groupby(schema.SYMBOL, sort=False)
    }
    intervals_by_index = {index: _intervals(index) for index in indices}
    for frame in intervals_by_index.values():
        if len(frame):
            requests = list(zip(frame["symbol"].astype(str), frame["start_date"]))
            frame["_successor"] = aliases.successors_asof(requests, alias_table)

    rows: List[dict] = []
    for index in indices:
        for _, iv in intervals_by_index[index].iterrows():
            symbol = str(iv["symbol"])
            start = pd.Timestamp(iv["start_date"])
            end = frontier if pd.isna(iv["end_date"]) else min(pd.Timestamp(iv["end_date"]), frontier)
            row = {"index": index, "symbol": symbol, "start": start.date(), "end": end.date(),
                   "left_censored": bool(iv.get("left_censored", False)),
                   "lake_symbol": None, "expected": 0, "priced": 0}

            if start > frontier:
                continue  # membership that begins after the lake ends: nothing to price yet
            if end < window_start:
                rows.append({**row, "class": "D"})
                continue
            clipped = max(start, window_start)
            sessions = calendar[(calendar >= clipped) & (calendar <= end)]
            row["expected"] = int(len(sessions))
            if len(sessions) == 0:
                rows.append({**row, "class": "D"})
                continue

            candidates = [symbol]
            successor = str(iv.get("_successor", symbol))
            if successor != symbol:
                candidates.append(successor)

            best, best_count = None, 0
            for candidate in candidates:
                have = dates_by_symbol.get(candidate)
                if have is None:
                    continue
                count = int(have.isin(sessions).sum())
                if count > best_count:
                    best, best_count = candidate, count
            row["lake_symbol"], row["priced"] = best, best_count

            if best is not None:
                complete = best_count / len(sessions) >= COMPLETE
                if not complete:
                    cls = "C"
                else:
                    cls = "A" if best == symbol else "B"
            else:
                elsewhere = any(c in dates_by_symbol for c in candidates)
                suffixed = symbol.endswith(BANKRUPTCY_SUFFIX) and len(symbol) > 3
                cls = "F" if (suffixed or elsewhere) else "E"
            rows.append({**row, "class": cls})

    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, *, window_start: pd.Timestamp = LOOKBACK_START) -> Dict[str, object]:
    if len(table) == 0:
        return {"classes": {}, "tiingo": {"recommend": False, "reason": "no data"}}

    in_window = table[table["class"] != "D"]
    total = int(in_window["expected"].sum())
    classes = {}
    for cls, label in CLASSES.items():
        rows = table[table["class"] == cls]
        sessions = int(rows["expected"].sum())
        classes[cls] = {
            "label": label,
            "intervals": int(len(rows)),
            "symbols": int(rows["symbol"].nunique()),
            "member_sessions": sessions if cls != "D" else None,
            "share_of_member_sessions": (sessions / total) if (total and cls != "D") else None,
            "examples": sorted(rows["symbol"].unique().tolist())[:25],
        }
    by_index = (
        table.groupby(["index", "class"]).size().unstack(fill_value=0).to_dict("index")
    )

    e_share = classes["E"]["share_of_member_sessions"] or 0.0
    recommend = e_share >= TIINGO_MIN_SHARE

    # The decision is about the *canonical* provider's gaps. While years in the
    # window are still owned by a cross-check provider the numbers describe
    # that provider, so the decision is deferred rather than made on them.
    owners = manifest.providers_by_year()
    canonical = {name for name, is_canonical in providers.available().items() if is_canonical}
    stale = sorted(y for y, p in owners.items() if y >= window_start.year and p not in canonical)
    if stale:
        return {
            "window_start": str(window_start.date()),
            "frontier": str(max(table["end"])),
            "providers_by_year": {str(k): v for k, v in owners.items()},
            "classes": classes,
            "by_index": table.groupby(["index", "class"]).size().unstack(fill_value=0).to_dict("index"),
            "tiingo": {
                "recommend": None,
                "class_e_share": e_share,
                "threshold": TIINGO_MIN_SHARE,
                "candidates": classes["E"]["examples"],
                "reason": (
                    f"deferred: {len(stale)} year(s) in the window ({stale[0]}-{stale[-1]}) "
                    f"are not yet on a canonical provider, so these classes describe the "
                    f"old lake. Re-run after the Alpaca rebuild swap."
                ),
            },
        }

    tiingo = {
        "recommend": bool(recommend),
        "class_e_share": e_share,
        "threshold": TIINGO_MIN_SHARE,
        "candidates": classes["E"]["examples"],
        "reason": (
            f"class E holds {e_share:.2%} of member-sessions since {window_start.date()} "
            f"({'>=' if recommend else '<'} {TIINGO_MIN_SHARE:.0%}). "
            + ("Trial Tiingo on the class E symbols only: hold its bars in quarantine, "
               "reconcile against Alpaca around each gap, promote what passes."
               if recommend else
               "A second vendor would not move coverage materially; do not add one.")
        ),
    }
    return {
        "window_start": str(window_start.date()),
        "frontier": str(max(table["end"])),
        "providers_by_year": {str(k): v for k, v in manifest.providers_by_year().items()},
        "classes": classes,
        "by_index": by_index,
        "tiingo": tiingo,
    }


def render(summary: Dict[str, object]) -> str:
    lines = ["", "Gap classification (membership intervals vs the lake)", "-" * 53]
    lines.append(f"  canonical window {summary.get('window_start')} .. {summary.get('frontier')}")
    providers = summary.get("providers_by_year", {})
    if providers:
        owners = sorted(set(providers.values()))
        lines.append(f"  lake providers: {', '.join(owners)} "
                     f"(classes describe the lake as it stands, not the target)")
    for cls, info in summary.get("classes", {}).items():
        share = info["share_of_member_sessions"]
        shown = f"{share:6.2%}" if share is not None else "   n/a"
        lines.append(f"  {cls} {info['label']:<40} {info['intervals']:>6} intervals  {shown}")
    decision = summary["tiingo"]["recommend"]
    verdict = "DEFERRED" if decision is None else ("RECOMMEND TRIAL" if decision else "not needed")
    lines.append(f"  Tiingo: {verdict} -- {summary['tiingo']['reason']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify membership coverage gaps A-F.")
    parser.add_argument("--out", default=None, help="Write the summary and table as JSON.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    table = classify()
    summary = summarize(table)
    print(render(summary))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Class A is the absence of a problem and class D is out of scope by
        # construction; listing either would only make the daily diff large.
        # Their counts are in the summary.
        gaps = table[~table["class"].isin(["A", "D"])] if len(table) else table
        payload = {**summary, "intervals": gaps.astype(str).to_dict("records")}
        out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

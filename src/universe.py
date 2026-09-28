"""universe.py

Two jobs, deliberately separable:

1. **Capture** today's *unfiltered* S&P 500 / 400 / 600 membership from
   Wikipedia into ``data/universe/`` (baseline, add/remove events, current
   constituents). See ``data_pipeline.membership_capture``. This is the record
   the S&P 1500 point-in-time history is built from, and it must never depend
   on a price screen.
2. **Filter** those constituents into the tradable ``universe.csv``: average
   raw close above ``MIN_PRICE`` and average dollar volume above
   ``MIN_DOLLAR_VOLUME`` over the last ``FILTER_SESSIONS`` sessions, read from
   the price lake. No network: the lake holds completed, canonical (Alpaca SIP)
   bars, so the screen sees the same prices every strategy does.

The after-close workflow runs them around the price sync so that a name added
to an index today is backfilled before the screen looks for its prices::

    python src/universe.py --capture-only     # membership
    python src/data_pipeline/sync_prices.py   # prices, incl. new members
    python src/universe.py --filter-only      # universe.csv

Run with no flag, it does both against the lake as it stands.

``universe.csv`` is the tradable list and nothing more. Which histories the lake
keeps is decided by the unfiltered membership (``membership.tracked_symbols``),
never by this file.

Exit codes: 0 success, 2 capture refused as suspicious (nothing written; the
previous membership stays in force), 1 anything else.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline import membership_capture as capture  # noqa: E402
from data_pipeline import schema, store  # noqa: E402
from data_pipeline.providers.base import market_date  # noqa: E402

logger = logging.getLogger("universe")

# Configuration
MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 10_000_000
FILTER_SESSIONS = 5
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "universe.csv"

#: Calendar days of lake read to find the last FILTER_SESSIONS sessions.
FILTER_LOOKBACK_DAYS = 21


def get_candidates(current_path: Optional[Path] = None) -> pd.DataFrame:
    """Today's unfiltered constituents as Symbol/Sector/Industry.

    Read from the captured ``current_membership.csv``. A symbol in two indices
    (rare, mid-move) keeps its S&P 500, then 400, then 600 row.
    """
    current = capture.load_current(current_path)
    if len(current) == 0:
        return pd.DataFrame(columns=["Symbol", "Sector", "Industry"])
    order = {name: i for i, name in enumerate(capture.INDICES)}
    current = current.assign(_order=current["index"].map(order)).sort_values(
        ["_order", "symbol"], kind="mergesort"
    )
    out = current.rename(columns={"symbol": "Symbol", "sector": "Sector", "industry": "Industry"})
    return out[["Symbol", "Sector", "Industry"]].drop_duplicates(subset="Symbol").reset_index(drop=True)


def filter_universe(
    df: pd.DataFrame,
    *,
    lake_root: Optional[Path] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Keep candidates liquid enough to trade, judged from the lake.

    Uses the raw print (``close``) for the price floor -- the dollar amount a
    share actually cost -- and raw close times raw volume for dollar volume.
    A candidate with no recent bars in the lake is excluded and counted.
    """
    if len(df) == 0:
        return df
    # Anchor the window on the lake's newest session, not the calendar: the
    # screen should judge the last sessions the lake actually holds, and a
    # lagging lake is a problem to report, not a reason to screen out everyone.
    end = pd.Timestamp(as_of) if as_of is not None else store.last_bar_date(root=lake_root)
    if end is None:
        logger.error("the price lake is empty; nothing to screen against")
        return df.iloc[0:0]
    lag = (market_date() - end).days
    if as_of is None and lag > 7:
        logger.warning(
            "price lake frontier %s is %d days old; screening on stale bars", end.date(), lag
        )
    start = end - pd.Timedelta(days=FILTER_LOOKBACK_DAYS)
    bars = store.load_prices(df["Symbol"].tolist(), start, end, root=lake_root)
    if len(bars) == 0:
        logger.error("no lake bars for any candidate; universe.csv would be empty")
        return df.iloc[0:0]

    recent = (
        bars.sort_values([schema.SYMBOL, schema.DATE])
        .groupby(schema.SYMBOL, sort=True)
        .tail(FILTER_SESSIONS)
    )
    recent = recent.assign(_dollar=recent[schema.CLOSE] * recent[schema.VOLUME].astype("float64"))
    stats = recent.groupby(schema.SYMBOL).agg(price=(schema.CLOSE, "mean"), dollar=("_dollar", "mean"))
    liquid = set(stats[(stats["price"] > MIN_PRICE) & (stats["dollar"] > MIN_DOLLAR_VOLUME)].index)

    unpriced = set(df["Symbol"]) - set(stats.index)
    if unpriced:
        logger.warning(
            "%d candidate(s) have no lake bars in the last %d days and are excluded: %s",
            len(unpriced), FILTER_LOOKBACK_DAYS,
            ", ".join(sorted(unpriced)[:20]) + (" ..." if len(unpriced) > 20 else ""),
        )
    return df[df["Symbol"].isin(liquid)].reset_index(drop=True)


def run_capture() -> int:
    try:
        snapshot = capture.scrape()
        summary = capture.capture(snapshot, market_date())
    except capture.SuspiciousScrapeError as exc:
        logger.error("membership capture refused, nothing written: %s", exc)
        return 2
    logger.info(
        "membership %s: %s; union %d; %d add(s), %d remove(s)%s",
        summary["observed_date"],
        ", ".join(f"{k} {v}" for k, v in summary["counts"].items()),
        summary["union"], len(summary["adds"]), len(summary["removes"]),
        " (baseline)" if summary["baseline"] else "",
    )
    return 0


def run_filter(output: Path = OUTPUT_FILE) -> int:
    candidates = get_candidates()
    if len(candidates) == 0:
        logger.error("no captured membership; run with --capture-only first")
        return 1
    final = filter_universe(candidates)
    if len(final) == 0:
        logger.error("filter kept nothing; leaving %s as it was", output)
        return 1
    final.to_csv(output, index=False, lineterminator="\n")
    logger.info("saved %d of %d constituents to %s", len(final), len(candidates), output)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Membership capture and tradable-universe filter.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--capture-only", action="store_true", help="Record membership only.")
    group.add_argument("--filter-only", action="store_true", help="Write universe.csv only.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    status = 0
    if not args.filter_only:
        status = run_capture()
        if args.capture_only:
            return status
        # A refused capture leaves yesterday's membership in force, which is
        # still a valid list to screen -- so filter anyway, and report the
        # refusal in the exit code.
    filtered = run_filter()
    return status or filtered


if __name__ == "__main__":
    raise SystemExit(main())

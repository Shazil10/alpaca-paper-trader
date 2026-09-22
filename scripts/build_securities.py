#!/usr/bin/env python
"""Build the security master from membership intervals + the ticker registry.

Ticker recycling is a real corruption risk. When a company leaves the market its
symbol can be handed to an unrelated company later, and naive concatenation then
splices two businesses into one price series. The security master gives every
distinct listing a permanent ``security_id`` so the backtester can tell them
apart, and carries the sector so the risk layer's sector cap has something to
read (without it, ``RiskConfig.max_sector_pct`` is silently inert).

What this builds is derived, not sourced. Membership records when a symbol was in
an index, which is not the same as when the company was listed, and it cannot
distinguish "removed from the index in 2009 and re-added in 2016" (one company)
from "delisted in 2009, symbol reassigned in 2016" (two companies). So a symbol
with a long membership gap is flagged ``recycle_candidate`` rather than split,
and ``delist_reason`` says ``index_removal`` -- the only thing actually known.
Resolving those needs a real corporate-actions feed (Tiingo's permaTicker,
FIGI, or CUSIP); until then the flag is the honest output.

A split span is a hypothesis, not a finding, and the false positives are easy to
spot by hand: ``AMP_1/AMP_2`` is a genuine reassignment (AMP Incorporated, then
Ameriprise), while ``AMD_1/AMD_2`` is one company that simply left the index
from 2013 to 2017. ``gap_years`` is there so that triage does not require
re-deriving the gap.

Output: data/universe/securities.parquet
  security_id, ticker, name, sector, industry, start_date, end_date,
  delist_reason, recycle_candidate, gap_years, source

Run: PYTHONPATH=src ./venv/bin/python scripts/build_securities.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP_PATH = REPO_ROOT / "data" / "universe" / "membership.parquet"
REGISTRY_PATH = REPO_ROOT / "data" / "universe" / "master_tickers.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "universe" / "securities.parquet"

#: Membership gaps shorter than this are treated as one continuous listing that
#: happened to drop out of the index. Two years is generous on purpose: index
#: committees remove and restore the same company well inside that window, and a
#: false split is worse than a missed one -- it fragments a real price history.
RECYCLE_GAP_YEARS = 2.0

COLUMNS = [
    "security_id", "ticker", "name", "sector", "industry",
    "start_date", "end_date", "delist_reason", "recycle_candidate",
    "gap_years", "source",
]


def load_registry() -> pd.DataFrame:
    """Sector/industry per current ticker, or an empty frame."""
    if not REGISTRY_PATH.exists():
        logger.warning("no ticker registry at %s; sectors will be blank", REGISTRY_PATH)
        return pd.DataFrame(columns=["symbol", "sector", "industry"])

    df = pd.read_csv(REGISTRY_PATH)
    keep = [c for c in ("symbol", "sector", "industry") if c in df.columns]
    df = df[keep].copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset="symbol")


def collapse_intervals(group: pd.DataFrame) -> list:
    """Merge a symbol's membership intervals into listing spans.

    Intervals are merged when the gap between them is under
    ``RECYCLE_GAP_YEARS``; a longer gap starts a new span and marks both as
    recycle candidates.
    """
    rows = group.sort_values("start_date").to_dict("records")
    spans: list = []

    for row in rows:
        start = pd.Timestamp(row["start_date"])
        end = row["end_date"]
        end = pd.Timestamp(end) if pd.notna(end) else pd.NaT

        if not spans:
            spans.append({"start_date": start, "end_date": end, "gap_years": float("nan")})
            continue

        prev = spans[-1]
        if pd.isna(prev["end_date"]):
            # Previous span is open-ended; anything later is inside it.
            if pd.notna(end) and (pd.isna(prev["end_date"]) or end > prev["end_date"]):
                prev["end_date"] = end
            continue

        gap_years = (start - prev["end_date"]).days / 365.25
        if gap_years < RECYCLE_GAP_YEARS:
            # Same listing, briefly out of the index.
            if pd.isna(end) or end > prev["end_date"]:
                prev["end_date"] = end
        else:
            spans.append({
                "start_date": start, "end_date": end, "gap_years": gap_years,
            })

    return spans


def build() -> pd.DataFrame:
    if not MEMBERSHIP_PATH.exists():
        logger.error(
            "membership file missing at %s -- run scripts/build_membership.py first",
            MEMBERSHIP_PATH,
        )
        return pd.DataFrame(columns=COLUMNS)

    membership = pd.read_parquet(MEMBERSHIP_PATH)
    for col in ("start_date", "end_date"):
        membership[col] = pd.to_datetime(membership[col], errors="coerce")
    membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()

    registry = load_registry().set_index("symbol")

    records: list = []
    for ticker, group in membership.groupby("symbol", sort=True):
        spans = collapse_intervals(group)
        recycled = len(spans) > 1

        for idx, span in enumerate(spans, start=1):
            meta = registry.loc[ticker] if ticker in registry.index else None
            records.append({
                "security_id": f"{ticker}_{idx}",
                "ticker": ticker,
                "name": "",
                "sector": str(meta["sector"]) if meta is not None and "sector" in registry.columns else "",
                "industry": str(meta["industry"]) if meta is not None and "industry" in registry.columns else "",
                "start_date": span["start_date"],
                "end_date": span["end_date"],
                "delist_reason": "" if pd.isna(span["end_date"]) else "index_removal",
                "recycle_candidate": recycled,
                "gap_years": float(span["gap_years"]),
                "source": "membership+registry",
            })

    out = pd.DataFrame(records, columns=COLUMNS)
    return out.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def main() -> int:
    securities = build()
    if len(securities) == 0:
        logger.error("no security records built")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    securities.to_parquet(OUTPUT_PATH, index=False)

    n_tickers = securities["ticker"].nunique()
    n_active = int(securities["end_date"].isna().sum())
    n_recycle = int(securities["recycle_candidate"].sum())
    n_with_sector = int((securities["sector"].astype(str).str.len() > 0).sum())

    logger.info(
        "Wrote %d listing spans for %d tickers to %s",
        len(securities), n_tickers, OUTPUT_PATH,
    )
    logger.info(
        "  %d currently in-index, %d spans on %d recycle-candidate tickers",
        n_active, n_recycle, securities.loc[securities["recycle_candidate"], "ticker"].nunique(),
    )
    logger.info(
        "  sector known for %d/%d spans (registry only covers current names)",
        n_with_sector, len(securities),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

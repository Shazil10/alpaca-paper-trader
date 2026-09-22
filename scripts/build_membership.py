#!/usr/bin/env python
"""Build PIT S&P 500 membership from public sources.

Downloads historical S&P 500 component data and converts to interval format
for use by the backtester's point-in-time universe.

Sources:
  - fja05680/sp500: S&P 500 Historical Components & Changes (1996-present)
  - Wikipedia S&P 500 changes (for recent updates)

Output: data/universe/membership.parquet
  Columns: symbol, index, start_date, end_date
  end_date is NaT for current members.

Run: PYTHONPATH=src python scripts/build_membership.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "universe" / "membership.parquet"

HISTORICAL_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def build_from_historical_csv() -> pd.DataFrame:
    """Parse the fja05680/sp500 wide-format CSV into interval records.

    The CSV has columns: date, tickers (comma-separated list of members).
    We convert to interval form: (symbol, index, start_date, end_date).
    """
    logger.info("Downloading historical S&P 500 components from fja05680/sp500...")

    try:
        df = pd.read_csv(HISTORICAL_URL, low_memory=False)
    except Exception as e:
        logger.error("Failed to download historical data: %s", e)
        logger.info("You can manually download from:")
        logger.info("  https://github.com/fja05680/sp500")
        logger.info("Place as data/universe/sp500_historical_components.csv")

        local = REPO_ROOT / "data" / "universe" / "sp500_historical_components.csv"
        if local.exists():
            logger.info("Using local copy: %s", local)
            df = pd.read_csv(local, low_memory=False)
        else:
            return pd.DataFrame(columns=["symbol", "index", "start_date", "end_date"])

    logger.info("Parsing %d date snapshots...", len(df))

    if "tickers" in df.columns:
        return _parse_tickers_column(df)
    else:
        return _parse_wide_format(df)


def _parse_tickers_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse format where each row has date + comma-separated tickers."""
    records: list[dict] = []

    prev_members: set[str] = set()
    for _, row in df.sort_values("date").iterrows():
        date = pd.Timestamp(row["date"])
        tickers_str = str(row.get("tickers", ""))
        current = {t.strip().upper().replace(".", "-") for t in tickers_str.split(",") if t.strip()}

        added = current - prev_members
        removed = prev_members - current

        for sym in added:
            records.append({
                "symbol": sym, "index": "SP500",
                "start_date": date, "end_date": pd.NaT,
            })

        for sym in removed:
            for r in reversed(records):
                if r["symbol"] == sym and pd.isna(r["end_date"]):
                    r["end_date"] = date
                    break

        prev_members = current

    return pd.DataFrame(records)


def _parse_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Parse wide format: date column + one column per snapshot with ticker lists."""
    if "date" not in df.columns:
        logger.warning("Unexpected CSV format; attempting generic parse")
        return pd.DataFrame(columns=["symbol", "index", "start_date", "end_date"])

    records: list[dict] = []
    prev_members: set[str] = set()

    for _, row in df.sort_values("date").iterrows():
        date = pd.Timestamp(row["date"])
        current: set[str] = set()
        for col in df.columns:
            if col == "date":
                continue
            val = row[col]
            if pd.notna(val) and str(val).strip():
                current.add(str(val).strip().upper().replace(".", "-"))

        added = current - prev_members
        removed = prev_members - current

        for sym in added:
            records.append({
                "symbol": sym, "index": "SP500",
                "start_date": date, "end_date": pd.NaT,
            })

        for sym in removed:
            for r in reversed(records):
                if r["symbol"] == sym and pd.isna(r["end_date"]):
                    r["end_date"] = date
                    break

        prev_members = current

    return pd.DataFrame(records)


def main():
    records = build_from_historical_csv()

    if len(records) == 0:
        logger.error("No membership records built. Check data source.")
        sys.exit(1)

    records = records.sort_values(["symbol", "start_date"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(OUTPUT_PATH, index=False)

    n_symbols = records["symbol"].nunique()
    n_current = records[records["end_date"].isna()]["symbol"].nunique()
    date_range = f"{records['start_date'].min():%Y-%m-%d} to {records['start_date'].max():%Y-%m-%d}"

    logger.info("Wrote %d interval records (%d unique symbols, %d current) to %s",
                len(records), n_symbols, n_current, OUTPUT_PATH)
    logger.info("Date range: %s", date_range)


if __name__ == "__main__":
    main()

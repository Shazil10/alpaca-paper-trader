"""rebuild_prices.py

Full re-download that rewrites year files from scratch.

Why this exists: yfinance adjusts prices retroactively. When a stock splits or
pays a dividend, its ``adj_close`` for *past* dates changes. A purely
incremental pipeline would drift out of agreement with the source over time, so
the lake needs a periodic corrective overwrite.

    # correct adjustment drift across the standard window
    PYTHONPATH=src python src/data_pipeline/rebuild_prices.py

    # January rollover: close out a year into cold Parquet
    PYTHONPATH=src python src/data_pipeline/rebuild_prices.py --finalize-year 2026

    # research archive (local only, not committed, not on the trading path)
    PYTHONPATH=src python src/data_pipeline/rebuild_prices.py --start 2004-01-01

Run this **locally**, or via ``workflow_dispatch``. It is deliberately not on a
CI schedule: a full multi-year, ~1,500-ticker download is exactly the kind of
job that flakes on shared GitHub runners, and it must never sit in the 09:30
trading path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from data_pipeline import fetch, registry, schema, store, writer

logger = logging.getLogger("rebuild_prices")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Same window the daily sync maintains.
DEFAULT_START = pd.Timestamp("2023-01-01")

#: Reference for a deeper research archive. Files outside the standard window
#: are gitignored by default -- keep them local, they are not trading data.
RESEARCH_START = pd.Timestamp("2004-01-01")


def rebuild(
    *,
    start: pd.Timestamp = DEFAULT_START,
    lake_root: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    universe_path: Optional[Path] = None,
    batch_size: int = fetch.DEFAULT_BATCH_SIZE,
    symbols_override: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Re-download ``start``..yesterday for every tracked symbol and overwrite.

    Year files in the rebuilt range are replaced wholesale rather than merged,
    which is the point: a merge would preserve the stale adjusted values this
    run exists to correct.
    """
    reg = registry.load_registry(registry_path)
    universe_meta = registry.load_universe_metadata(universe_path)

    if symbols_override is not None:
        symbols = sorted(
            {registry.normalize_symbol(s) for s in symbols_override if str(s).strip()}
        )
    else:
        symbols = registry.target_symbols(reg, universe_meta)

    if not symbols:
        logger.error("no symbols to rebuild")
        return {"symbols": 0, "rows": 0, "failed": []}

    today = fetch.today_naive()
    logger.info(
        "rebuilding %d symbol(s) from %s to %s (exclusive)",
        len(symbols), start.date(), today.date(),
    )

    batches = fetch.batched(symbols, batch_size)
    collected: List[pd.DataFrame] = []
    failed: List[str] = []

    for i, batch in enumerate(batches, start=1):
        bars, batch_failed = fetch.fetch_batch(batch, start, today)
        failed.extend(batch_failed)
        if len(bars) > 0:
            collected.append(bars)
        logger.info(
            "batch %d/%d: %d row(s), %d failed", i, len(batches), len(bars), len(batch_failed)
        )

    if not collected:
        logger.error("rebuild produced no rows; leaving existing lake untouched")
        return {"symbols": len(symbols), "rows": 0, "failed": sorted(set(failed))}

    everything = schema.coerce(pd.concat(collected, ignore_index=True))
    schema.assert_unique_key(everything, context="rebuild")

    year_now = int(today.year)
    replaced: Dict[int, int] = {}

    for year in schema.years_in(everything):
        rows = everything[everything[schema.DATE].dt.year == year]

        existing_path = schema.resolve_year_path(year, lake_root)
        hot = existing_path.suffix == ".csv" if existing_path else (year == year_now)

        # Overwrite, do not upsert: stale adjusted closes must not survive.
        path = schema.write_year(rows, year, lake_root, hot=hot)
        replaced[year] = len(rows)
        logger.info("rewrote %s with %d row(s)", path.name, len(rows))

    updated = registry.update_registry(
        reg,
        universe_meta,
        succeeded=set(everything[schema.SYMBOL].dropna().astype(str)),
        failed=failed,
        as_of=today,
    )
    registry.write_registry(updated, registry_path)

    if failed:
        logger.warning(
            "%d symbol(s) failed: %s",
            len(set(failed)),
            ", ".join(sorted(set(failed))[:30]) + (" ..." if len(set(failed)) > 30 else ""),
        )

    last = store.last_bar_date(root=lake_root)
    logger.info(
        "rebuild complete: %d row(s) across years %s; last session %s",
        len(everything), sorted(replaced), None if last is None else last.date(),
    )

    return {
        "symbols": len(symbols),
        "rows": len(everything),
        "failed": sorted(set(failed)),
        "years": replaced,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Full price re-download / year finalize.")
    parser.add_argument(
        "--start", default=str(DEFAULT_START.date()),
        help=f"History start (default {DEFAULT_START.date()}; research archive: {RESEARCH_START.date()}).",
    )
    parser.add_argument("--symbols", nargs="+", help="Rebuild only these tickers.")
    parser.add_argument(
        "--batch-size", type=int, default=fetch.DEFAULT_BATCH_SIZE,
        help="Tickers per download request.",
    )
    parser.add_argument(
        "--finalize-year", type=int, default=None,
        help="Convert that year's hot CSV to cold Parquet and stop (January rollover).",
    )
    parser.add_argument(
        "--keep-csv", action="store_true",
        help="With --finalize-year, leave the source CSV in place.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.finalize_year is not None:
        try:
            path = writer.finalize_year(args.finalize_year, remove_csv=not args.keep_csv)
        except Exception:
            logger.exception("failed to finalize year %s", args.finalize_year)
            return 1
        logger.info("finalized %s", path)
        return 0

    try:
        rebuild(
            start=pd.Timestamp(args.start),
            batch_size=args.batch_size,
            symbols_override=args.symbols,
        )
    except Exception:
        logger.exception("rebuild aborted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

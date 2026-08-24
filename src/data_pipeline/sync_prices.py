"""sync_prices.py

Incremental daily fetch. The only writer on the trading path.

    PYTHONPATH=src python src/data_pipeline/sync_prices.py

Behaviour
---------
* Symbols = today's ``universe.csv`` union the existing registry union the
  always-tracked ETFs, minus names retired for repeated failure. It does **not**
  scrape Wikipedia -- ``src/universe.py`` runs first and owns that.
* Known symbols are fetched from their last stored session minus a 5-day overlap,
  so late corrections are picked up rather than assumed away.
* New symbols get the full ``LOOKBACK_START`` history once.
* No bar for the current date is ever stored (``fetch`` enforces this).
* Batched with a checkpoint, so an interrupted backfill resumes instead of
  restarting.

Failure policy
--------------
Exit code is 0 even when batches fail. A partially updated lake is a success:
the trading run continues, yesterday's committed bars are still there, and the
next run's overlap window re-fetches the gap. The lake self-heals; going flat
because Yahoo hiccuped on a CI runner would be the worse outcome.

Exit 1 is reserved for a genuine misconfiguration (unreadable lake directory).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

from data_pipeline import fetch, registry, schema, store, writer

logger = logging.getLogger("sync_prices")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Whole calendar years only -- avoids partial-year edge cases and sits
#: comfortably above the ~2 years live strategies need.
LOOKBACK_START = pd.Timestamp("2023-01-01")

#: Re-fetch this many calendar days before the last stored session so that
#: retroactive split/dividend corrections land.
OVERLAP_DAYS = 5

CHECKPOINT_PATH = REPO_ROOT / ".cache" / "sync_state.json"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _signature(symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Identifies a run. A changed signature invalidates the checkpoint."""
    return f"{len(symbols)}:{start.date()}:{end.date()}"


def _load_checkpoint(path: Path, signature: str) -> Set[str]:
    if not path.exists():
        return set()
    try:
        state = json.loads(path.read_text())
    except Exception:
        logger.warning("unreadable checkpoint at %s; starting fresh", path)
        return set()

    if state.get("signature") != signature:
        logger.info("checkpoint signature changed; starting fresh")
        return set()

    done = set(state.get("done", []))
    if done:
        logger.info("resuming: %d symbol(s) already persisted this run", len(done))
    return done


def _save_checkpoint(path: Path, signature: str, done: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "done": sorted(done),
        "updated": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2))


def _clear_checkpoint(path: Path) -> None:
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def repair_gaps(
    *,
    lake_root: Optional[Path] = None,
    lookback_start: pd.Timestamp = LOOKBACK_START,
    max_symbols: int = 200,
) -> Dict[str, object]:
    """Re-fetch symbols that have holes inside their own history.

    Why this is needed: yfinance intermittently returns NaN for individual
    ticker/date pairs inside large multi-ticker requests. A NaN close cannot be
    stored, so the row is dropped and the lake keeps a permanent hole -- the
    daily 5-day overlap never reaches back far enough to heal it.

    A single missing bar inside a 50-day window makes ``rolling(50)`` NaN, which
    silently drops that symbol from threshold comparisons and changes live
    signals. So this is a correctness repair, not tidying.

    Repairs use **single-ticker** requests, which do not exhibit the batched
    NaN behaviour.
    """
    gaps = store.find_gaps(root=lake_root)
    if not gaps:
        logger.info("gap scan: lake is complete")
        return {"scanned": True, "repaired": [], "unresolved": {}}

    all_missing = sorted({d for dates in gaps.values() for d in dates})
    logger.warning(
        "gap scan: %d symbol(s) incomplete across %d distinct session(s) "
        "(%s .. %s)",
        len(gaps), len(all_missing),
        all_missing[0].date(), all_missing[-1].date(),
    )

    # A shared missing date across many symbols is one bad fetch window, not
    # hundreds of independent flakes. Re-fetch the affected window in batches
    # rather than issuing one request per symbol.
    span_start = max(min(all_missing) - pd.Timedelta(days=5), lookback_start)
    targets = sorted(gaps)

    # Bound the repair at the frontier the lake already reached. Repair fills
    # holes; it must not advance a subset of symbols past the rest. Otherwise
    # the newest row ends up mostly NaN, rolling windows break for everyone left
    # behind, and find_gaps cannot even see it -- the new dates fall outside the
    # lagging symbols' own spans. Advancing the frontier is sync's job, because
    # sync covers every symbol.
    frontier = store.last_bar_date(root=lake_root)
    today = fetch.today_naive()
    end = today if frontier is None else min(today, frontier + pd.Timedelta(days=1))

    logger.info(
        "repairing %d symbol(s) over %s .. %s (frontier-bounded)",
        len(targets), span_start.date(), end.date(),
    )

    for i, batch in enumerate(fetch.batched(targets, 100), start=1):
        bars, failed = fetch.fetch_batch(batch, span_start, end)
        if len(bars) > 0:
            writer.persist(bars, root=lake_root)
        logger.info(
            "repair batch %d: %d row(s), %d failed", i, len(bars), len(failed)
        )

    # Anything still short gets an individual request: single-ticker responses
    # do not show the batched-NaN behaviour.
    remaining = store.find_gaps(root=lake_root)
    stubborn = sorted(remaining, key=lambda s: -len(remaining[s]))[:max_symbols]
    if stubborn:
        logger.info("retrying %d symbol(s) individually", len(stubborn))
        for symbol in stubborn:
            start = max(min(remaining[symbol]) - pd.Timedelta(days=5), lookback_start)
            bars, failed = fetch.fetch_batch([symbol], start, end)
            if len(bars) > 0:
                writer.persist(bars, root=lake_root)

    final = store.find_gaps(root=lake_root)
    healed = [s for s in gaps if s not in final]

    logger.info(
        "gap repair: %d symbol(s) healed, %d still incomplete",
        len(healed), len(final),
    )
    if final:
        # Genuine non-trading (halts, a listing that began mid-window) looks
        # identical to a gap and will never heal. Report it; do not loop.
        worst = sorted(final.items(), key=lambda kv: -len(kv[1]))[:10]
        logger.info(
            "unresolved (likely genuine non-trading): %s",
            ", ".join(f"{s}({len(d)})" for s, d in worst),
        )

    return {"scanned": True, "repaired": healed, "unresolved": final}


def _fetch_start_for(
    symbol: str, last_dates: "pd.Series", lookback_start: pd.Timestamp
) -> pd.Timestamp:
    """Where to begin fetching a symbol."""
    if symbol not in last_dates.index:
        return lookback_start
    last = pd.Timestamp(last_dates[symbol])
    if pd.isna(last):
        return lookback_start
    return max(lookback_start, last - pd.Timedelta(days=OVERLAP_DAYS))


def sync(
    *,
    lake_root: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    universe_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    lookback_start: pd.Timestamp = LOOKBACK_START,
    batch_size: int = fetch.DEFAULT_BATCH_SIZE,
    symbols_override: Optional[Sequence[str]] = None,
    use_checkpoint: bool = True,
    repair: bool = True,
) -> Dict[str, object]:
    """Run one incremental sync. Returns a summary dict."""
    reg = registry.load_registry(registry_path)
    universe_meta = registry.load_universe_metadata(universe_path)

    if symbols_override is not None:
        symbols = sorted({registry.normalize_symbol(s) for s in symbols_override if str(s).strip()})
    else:
        symbols = registry.target_symbols(reg, universe_meta)

    if not symbols:
        logger.error("no symbols to sync (universe.csv empty and registry empty?)")
        return {"symbols": 0, "rows": 0, "failed": [], "years": {}}

    today = fetch.today_naive()
    last_dates = store.last_date_per_symbol(root=lake_root)

    known = [s for s in symbols if s in last_dates.index]
    fresh = [s for s in symbols if s not in last_dates.index]

    logger.info(
        "syncing %d symbol(s): %d incremental, %d new (full history from %s)",
        len(symbols), len(known), len(fresh), lookback_start.date(),
    )
    if len(last_dates):
        logger.info("lake last session: %s", pd.Timestamp(last_dates.max()).date())

    signature = _signature(symbols, lookback_start, today)
    ckpt = Path(checkpoint_path or CHECKPOINT_PATH)
    done: Set[str] = _load_checkpoint(ckpt, signature) if use_checkpoint else set()

    total_rows = 0
    all_failed: List[str] = []
    succeeded: Set[str] = set()
    years_touched: Dict[int, int] = {}

    # New names first: they are the expensive part of a backfill, and finishing
    # them early makes a resumed run cheaper.
    groups = [("new", fresh, lookback_start)]
    if known:
        # One shared start keeps the request small; all incremental symbols sit
        # within a few days of each other. upsert() dedupes any over-fetch.
        starts = [_fetch_start_for(s, last_dates, lookback_start) for s in known]
        groups.append(("incremental", known, min(starts)))

    for label, group_symbols, group_start in groups:
        pending = [s for s in group_symbols if s not in done]
        if not pending:
            continue

        if group_start >= today:
            logger.info("%s group already current; nothing to fetch", label)
            succeeded.update(pending)
            continue

        batches = fetch.batched(pending, batch_size)
        logger.info(
            "%s group: %d symbol(s) in %d batch(es) from %s",
            label, len(pending), len(batches), group_start.date(),
        )

        for i, batch in enumerate(batches, start=1):
            bars, failed = fetch.fetch_batch(batch, group_start, today)

            if len(bars) > 0:
                try:
                    written = writer.persist(bars, root=lake_root)
                    for year, count in written.items():
                        years_touched[year] = count
                    total_rows += len(bars)
                except Exception:
                    # Persist failure is worth shouting about but must not abort
                    # the whole run; other batches may still land.
                    logger.exception("failed to persist %s batch %d", label, i)
                    failed = list(batch)

            batch_ok = [s for s in batch if s not in failed]
            succeeded.update(batch_ok)
            all_failed.extend(failed)
            done.update(batch_ok)

            if use_checkpoint:
                _save_checkpoint(ckpt, signature, done)

            logger.info(
                "%s batch %d/%d: %d row(s), %d ok, %d failed",
                label, i, len(batches), len(bars), len(batch_ok), len(failed),
            )

    updated = registry.update_registry(
        reg,
        universe_meta,
        succeeded=succeeded,
        failed=all_failed,
        as_of=today,
    )
    registry_file = registry.write_registry(updated, registry_path)
    logger.info("registry: %d symbol(s) -> %s", len(updated), registry_file)

    if use_checkpoint:
        _clear_checkpoint(ckpt)

    # Heal holes left by batched-request NaNs before anything reads the lake.
    # Skipped for targeted --symbols runs, where a full scan is disproportionate.
    gap_summary: Dict[str, object] = {}
    if repair and symbols_override is None:
        try:
            gap_summary = repair_gaps(lake_root=lake_root, lookback_start=lookback_start)
        except Exception:
            logger.exception("gap repair failed; lake may be incomplete")

    last = store.last_bar_date(root=lake_root)
    if last is None:
        logger.error("PRICE_LAKE_EMPTY: no bars on disk after sync")
    else:
        age = (today - last).days
        message = "PRICE_LAKE_STALE" if age > 4 else "price lake current"
        logger.info("%s: last session %s (%d day(s) old)", message, last.date(), age)

    if all_failed:
        logger.warning(
            "%d symbol(s) failed this run: %s",
            len(all_failed),
            ", ".join(sorted(all_failed)[:30]) + (" ..." if len(all_failed) > 30 else ""),
        )

    return {
        "symbols": len(symbols),
        "rows": total_rows,
        "failed": sorted(set(all_failed)),
        "years": years_touched,
        "last_session": None if last is None else str(last.date()),
        "gaps_repaired": gap_summary.get("repaired", []),
        "gaps_unresolved": len(gap_summary.get("unresolved", {}) or {}),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Incremental daily price sync.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Sync only these tickers (smoke tests). Default: universe + registry + ETFs.",
    )
    parser.add_argument(
        "--start",
        default=str(LOOKBACK_START.date()),
        help=f"History start for new symbols (default {LOOKBACK_START.date()}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=fetch.DEFAULT_BATCH_SIZE,
        help="Tickers per download request.",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable resume state (useful for short ad-hoc runs).",
    )
    parser.add_argument(
        "--no-repair", action="store_true",
        help="Skip the post-sync gap scan and repair.",
    )
    parser.add_argument(
        "--repair-only", action="store_true",
        help="Scan for and repair gaps without running a sync.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.repair_only:
        try:
            result = repair_gaps(lookback_start=pd.Timestamp(args.start))
        except Exception:
            logger.exception("gap repair aborted")
            return 1
        logger.info(
            "repair complete: healed %d, unresolved %d",
            len(result["repaired"]), len(result["unresolved"]),
        )
        return 0

    try:
        summary = sync(
            lookback_start=pd.Timestamp(args.start),
            batch_size=args.batch_size,
            symbols_override=args.symbols,
            use_checkpoint=not args.no_checkpoint,
            repair=not args.no_repair,
        )
    except Exception:
        # Misconfiguration, not a data hiccup: this one is fatal.
        logger.exception("sync aborted with an unexpected error")
        return 1

    logger.info(
        "sync complete: %d symbol(s), %d row(s) upserted, %d failure(s), years=%s",
        summary["symbols"], summary["rows"], len(summary["failed"]),
        sorted(summary["years"]) if summary["years"] else "none",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Extend the price lake backwards in time.

``sync_prices`` cannot do this. Its window for a known symbol is
``max(lookback_start, last_stored - 7d)``, so passing an early ``--start`` to a
lake that already holds later years fetches nothing at all: every symbol is
"known" and its start collapses to last week. That is correct for the daily path
and useless for deepening history, hence a separate entry point.

Provider
--------
Fetches go through the canonical provider (``MARKET_DATA_PROVIDER``, Alpaca SIP)
and are written with its provenance, so the manifest's one-provider-per-year
rule applies here too: a backfill cannot put Alpaca rows into a Yahoo-era year,
and there is no fallback to Yahoo for years Alpaca does not cover. Alpaca's SIP
history begins in 2016, which is therefore the default ``--start``. The
2005-2015 ETF partitions were written by the Yahoo-era backfill and are kept as
they are, labelled ``yahoo`` in the manifest.

For the one-time move of 2016+ onto Alpaca, use ``rebuild_prices.py`` -- it
replaces whole partitions through a staged, validated swap, which is the only
way a year can change provider.

Why it matters: 2023-2026 is 912 sessions. That cannot support in-sample /
out-of-sample splitting, walk-forward with meaningful folds, or any claim about
bear-market behaviour -- there is no 2008, no 2015-16, no 2018Q4, and only a
sliver of 2022. Every robustness test in the validation suite is arithmetic on
noise until the lake goes deeper.

What it does
------------
* Works out each symbol's earliest stored session from ``store.coverage()`` and
  fetches only ``[--start, earliest)``. Symbols already deep enough are skipped.
* Groups symbols that share the same fetch window into one request, so the
  common case (everything starts 2023-01-03) is a handful of batches rather than
  one request per symbol.
* Checkpoints per batch into ``.cache/backfill_state.json``, so an interrupted
  multi-hour run resumes instead of restarting.
* Writes through ``writer.persist``, which upserts into year partitions, so
  re-running is safe and overlapping fetches cannot duplicate rows.
* **Never advances the lake's newest session.** Every window is capped at the
  frontier the lake already holds; see ``plan_windows`` for why, and
  ``data/prices/_schema.md`` Contract 5 for the damage it avoids.

Symbol sources
--------------
``--symbols`` overrides everything. Otherwise the target set is the union of the
ticker registry, every historical index member in ``membership.parquet``, and the
fixed ETFs. The membership names are the point: they are what removes
survivorship bias, and they are also where most failures land, because the
membership source records final tickers (``LEHMQ``, not ``LEH``) and a vendor
often has no history under a post-bankruptcy symbol. Failures are logged and
listed, not retried forever.

New years land as Parquet, and ``.gitignore`` un-ignores closed years one at a
time -- add any new year there or it will not be committed.

    # what would be fetched, no network
    PYTHONPATH=src ./venv/bin/python scripts/backfill_history.py --dry-run

    # ETFs only, a good first pilot
    PYTHONPATH=src ./venv/bin/python scripts/backfill_history.py --etfs-only

    # the real thing (hours; resumable)
    PYTHONPATH=src ./venv/bin/python scripts/backfill_history.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_pipeline import fetch, providers, registry, schema, store, writer  # noqa: E402
from data_pipeline.membership import ALL_FIXED_ETFS  # noqa: E402
from data_pipeline.providers.base import MarketDataProvider  # noqa: E402
from data_pipeline.sync_prices import LOOKBACK_START, _require_canonical  # noqa: E402

logger = logging.getLogger("backfill_history")

CHECKPOINT_PATH = REPO_ROOT / ".cache" / "backfill_state.json"

#: Symbols per request. Alpaca paginates, so this bounds URL length and the
#: blast radius of one rejected symbol rather than response size.
DEFAULT_BATCH_SIZE = 100


def _signature(start: pd.Timestamp, n_symbols: int) -> str:
    return f"backfill:{start.date()}:{n_symbols}"


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
        logger.info("resuming: %d symbol(s) already backfilled", len(done))
    return done


def _save_checkpoint(path: Path, signature: str, done: Set[str], failed: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "signature": signature,
        "done": sorted(done),
        "failed": sorted(failed),
        "updated": pd.Timestamp.now().isoformat(timespec="seconds"),
    }, indent=2))


def target_symbols(
    *,
    etfs_only: bool = False,
    include_delisted: bool = True,
    lake_root: Optional[Path] = None,
) -> List[str]:
    """Symbols worth deepening: registry + historical members + fixed ETFs."""
    if etfs_only:
        return sorted(ALL_FIXED_ETFS)

    symbols: Set[str] = set(ALL_FIXED_ETFS)

    try:
        reg = registry.load_registry()
        if len(reg) > 0:
            symbols |= set(reg["symbol"].astype(str).str.upper())
    except Exception:
        logger.exception("could not read the ticker registry")

    if include_delisted:
        membership_path = REPO_ROOT / "data" / "universe" / "membership.parquet"
        if membership_path.exists():
            members = pd.read_parquet(membership_path)
            symbols |= set(members["symbol"].astype(str).str.upper())
            logger.info(
                "membership adds %d historical name(s) not in the registry",
                len(set(members["symbol"].astype(str).str.upper()) - set(
                    registry.load_registry()["symbol"].astype(str).str.upper()
                    if len(registry.load_registry()) else set()
                )),
            )
        else:
            logger.warning(
                "no membership.parquet: delisted names will be missing and the "
                "backfill will still carry survivorship bias. "
                "Run scripts/build_membership.py first."
            )

    return sorted(s for s in symbols if s)


def lake_frontier(lake_root: Optional[Path] = None) -> Optional[pd.Timestamp]:
    """The newest session the lake already holds, across all symbols."""
    try:
        return store.last_bar_date(root=lake_root)
    except Exception:
        logger.exception("could not read the lake frontier")
        return None


def plan_windows(
    symbols: Sequence[str],
    start: pd.Timestamp,
    *,
    lake_root: Optional[Path] = None,
) -> Dict[pd.Timestamp, List[str]]:
    """Group symbols by the end of the window they still need.

    A symbol already holding data at or before ``start`` needs nothing. One that
    starts in 2023 needs ``[start, 2023-01-03)``.

    **Frontier-bounded.** A symbol absent from the lake needs everything, but its
    window still stops at the lake's newest existing session rather than at
    today. Advancing a subset of symbols past the rest is the failure mode
    ``data/prices/_schema.md`` documents under Contract 5: the new dates clear
    ``MIN_SYMBOLS_FOR_SESSION`` and enter the trading calendar, so every other
    symbol silently looks like it has a gap at the end of history, rolling
    windows break for all of them, and ``find_gaps`` cannot see it because the
    new dates fall outside the lagging symbols' own spans.

    Observed: backfilling 16 ETFs to today while 1,385 equities stopped a month
    earlier added 21 sessions carrying 16 symbols each, and every test anchored
    on the newest session found zero equities with a year of history. Advancing
    the frontier is ``sync_prices``' job, because sync covers every symbol.
    """
    cov = store.coverage(root=lake_root)
    first_seen = (
        cov.set_index("symbol")["first_date"] if len(cov) else pd.Series(dtype="datetime64[ns]")
    )

    frontier = lake_frontier(lake_root)
    # +1 day because fetch windows are end-exclusive and the frontier session
    # itself is wanted for a symbol the lake does not hold at all.
    ceiling = (
        pd.Timestamp(frontier) + pd.Timedelta(days=1)
        if frontier is not None else fetch.today_naive()
    )

    groups: Dict[pd.Timestamp, List[str]] = defaultdict(list)
    for symbol in symbols:
        if symbol in first_seen.index:
            earliest = pd.Timestamp(first_seen[symbol])
            if pd.isna(earliest) or earliest <= start:
                continue
            groups[min(earliest, ceiling)].append(symbol)
        else:
            groups[ceiling].append(symbol)

    return dict(groups)


def backfill(
    start: pd.Timestamp,
    *,
    symbols: Optional[Sequence[str]] = None,
    etfs_only: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_symbols: Optional[int] = None,
    lake_root: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    use_checkpoint: bool = True,
    dry_run: bool = False,
    provider: Optional[MarketDataProvider] = None,
) -> Dict[str, object]:
    """Fetch and persist history before each symbol's earliest stored session.

    ``provider`` defaults to the canonical provider; a non-canonical one is
    refused (``sync_prices._require_canonical``).
    """
    wanted = (
        [s.strip().upper() for s in symbols if s.strip()]
        if symbols
        else target_symbols(etfs_only=etfs_only, lake_root=lake_root)
    )
    if max_symbols is not None:
        wanted = wanted[:max_symbols]

    windows = plan_windows(wanted, start, lake_root=lake_root)
    pending = sorted({s for group in windows.values() for s in group})

    logger.info(
        "%d symbol(s) targeted, %d need history before %s, %d already deep enough",
        len(wanted), len(pending), start.date(), len(wanted) - len(pending),
    )
    for end, group in sorted(windows.items()):
        logger.info("  window [%s, %s): %d symbol(s)", start.date(), end.date(), len(group))

    if dry_run:
        return {
            "targeted": len(wanted),
            "pending": len(pending),
            "windows": {str(k.date()): len(v) for k, v in windows.items()},
            "dry_run": True,
        }

    if not pending:
        logger.info("nothing to do")
        return {"targeted": len(wanted), "pending": 0, "rows": 0, "failed": []}

    source = _require_canonical(provider)
    ckpt = Path(checkpoint_path or CHECKPOINT_PATH)
    signature = _signature(start, len(pending))
    done: Set[str] = _load_checkpoint(ckpt, signature) if use_checkpoint else set()
    failed: Set[str] = set()
    total_rows = 0
    years_touched: Set[int] = set()

    for end, group in sorted(windows.items()):
        todo = [s for s in group if s not in done]
        if not todo:
            continue

        for batch in fetch.batched(todo, batch_size):
            logger.info(
                "fetching %d symbol(s) for [%s, %s)",
                len(batch), start.date(), end.date(),
            )
            bars, batch_failed = source.fetch_batch(batch, start, end)
            failed.update(batch_failed)

            if len(bars) > 0:
                # Never let a backfill overwrite the daily path's own window.
                bars = bars[bars[schema.DATE] < end]

            if len(bars) > 0:
                written = writer.persist(
                    bars, root=lake_root, provenance=source.provenance()
                )
                years_touched.update(written.keys())
                total_rows += len(bars)
                logger.info(
                    "  persisted %d row(s) across year(s) %s",
                    len(bars), sorted(written.keys()),
                )
            else:
                logger.warning("  no usable bars returned for this batch")

            done.update(s for s in batch if s not in batch_failed)
            if use_checkpoint:
                _save_checkpoint(ckpt, signature, done, failed)

    logger.info(
        "backfill finished: %d row(s) persisted, %d symbol(s) done, %d failed",
        total_rows, len(done), len(failed),
    )
    if years_touched:
        logger.info("years touched: %s", sorted(years_touched))
        logger.info(
            "closed years land as Parquet and .gitignore un-ignores them one at "
            "a time -- add these years to .gitignore or they stay uncommitted"
        )
    if failed:
        preview = sorted(failed)[:20]
        logger.warning(
            "no data for %d symbol(s) (first %d: %s). Delisted names often have "
            "no vendor history under their final ticker.",
            len(failed), len(preview), ", ".join(preview),
        )

    return {
        "targeted": len(wanted),
        "pending": len(pending),
        "rows": total_rows,
        "done": len(done),
        "failed": sorted(failed),
        "years": sorted(years_touched),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--start", default=str(LOOKBACK_START.date()),
        help=f"Earliest session to fetch (default {LOOKBACK_START.date()}, "
             f"where Alpaca SIP history begins).",
    )
    parser.add_argument("--symbols", nargs="*", help="Explicit symbol list.")
    parser.add_argument("--etfs-only", action="store_true", help="Fixed ETFs only (pilot).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Plan only, no network.")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = backfill(
            pd.Timestamp(args.start),
            symbols=args.symbols,
            etfs_only=args.etfs_only,
            batch_size=args.batch_size,
            max_symbols=args.max_symbols,
            use_checkpoint=not args.no_checkpoint,
            dry_run=args.dry_run,
        )
    except Exception:
        logger.exception("backfill aborted")
        return 1

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

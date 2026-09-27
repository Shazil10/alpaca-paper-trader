"""rebuild_prices.py

Replace whole year partitions from the canonical provider: staged, validated,
compared, and only then swapped in -- with a full rollback if the swap fails.

Why this exists
---------------
Two jobs need a partition *replaced* rather than upserted:

1. **Changing provider.** A year holds one provider (``manifest.py``). Moving
   2016+ from Yahoo to Alpaca SIP cannot be an upsert: a half-upserted year is
   exactly the quiet mix the manifest forbids. Every partition in the range is
   rebuilt from the new provider and replaced as a unit.
2. **Periodic correction.** ``anchor.py`` keeps ``adj_close`` on one anchor per
   symbol as the daily sync runs, but it cannot see a vendor back-correction
   whose ex-date predates the overlap window. A rebuild re-downloads everything,
   folds every pending anchor factor into the bytes, and leaves
   ``anchor_factors.csv`` holding nothing for the rebuilt years.

Pipeline
--------
::

    fetch     batches  -> data/prices/staging/<run>/batches/<year>/<n>.parquet
              checkpointed in state.json; an interrupted run resumes
    assemble  per year -> data/prices/staging/<run>/daily/<year>.{parquet,csv}
    validate  keys, sessions, prices -- any error stops before the swap
    compare   staged vs live: rows, symbols lost/gained, raw-close ratio,
              return agreement, ETF seam  -> data/audits/rebuild_<run>.json
    swap      live -> staging/<run>/backup/, staged -> lake, manifest and
              anchor factors updated; any exception restores all three

The live lake is not touched until the swap, and ``--stage-only`` stops before
it, so the report can be read first and the swap run later with ``--resume``.

Years before ``--start``
------------------------
Not touched. For the Alpaca migration that is the 2005-2015 ETF history, which
stays exactly as the Yahoo backfill wrote it and is labelled ``yahoo`` in the
manifest. Its ``adj_close`` sits on Yahoo's anchor, though, and the first 2016
bar will sit on Alpaca's. So the seam is *measured* -- for each symbol, the
median of ``staged / live`` adj_close over the first sessions of the start year,
where both providers have the same bars -- and a material factor is recorded as
a pending anchor factor on the older partitions. That makes the splice visible
(one row per ETF per year in ``anchor_factors.csv``), exact to the measurement,
and reversible, instead of either silently mixing scales or rewriting bytes
that are supposed to be preserved.

Usage
-----
::

    # migration, step 1: stage + validate + compare, no lake change
    PYTHONPATH=src python src/data_pipeline/rebuild_prices.py --stage-only

    # step 2: read data/audits/rebuild_<run>.json, then swap the staged run
    PYTHONPATH=src python src/data_pipeline/rebuild_prices.py --resume <run>

    # refresh a few symbols inside years the provider already owns
    PYTHONPATH=src python src/data_pipeline/rebuild_prices.py --symbols NVDA AAPL

    # January rollover: close out a year into cold Parquet
    PYTHONPATH=src python src/data_pipeline/rebuild_prices.py --finalize-year 2026

Run this **locally**, or via ``workflow_dispatch``. It is deliberately not on a
schedule: a decade of bars for two thousand symbols is the kind of job that
flakes on shared runners, and it must never sit in the trading path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from data_pipeline import anchor, fetch, manifest, registry, schema, writer
from data_pipeline.providers.base import MarketDataProvider, market_date
from data_pipeline.sync_prices import LOOKBACK_START, _require_canonical

logger = logging.getLogger("rebuild_prices")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the canonical provider's history begins.
DEFAULT_START = LOOKBACK_START

STAGING_ROOT = REPO_ROOT / "data" / "prices" / "staging"
AUDIT_DIR = REPO_ROOT / "data" / "audits"

#: Refuse to swap a year that would lose more than this share of the symbols it
#: holds today, unless told to. A provider that lacks a few delisted names is
#: expected; one that lacks a fifth of the universe is a broken run.
MAX_LOST_FRACTION = 0.02

#: Sessions at the start of the rebuilt range used to measure the seam factor.
SEAM_SESSIONS = 20

#: A seam factor this close to 1 is measurement noise, not a scale difference.
SEAM_TOLERANCE = 1e-3

#: Daily adj_close returns differing by more than this between providers count
#: as a disagreement in the comparison report.
RETURN_DISAGREEMENT = 0.005


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """Everything that identifies one rebuild run. Persisted in state.json."""
    run_id: str
    provider: str
    start: str
    end_exclusive: str
    symbols: List[str]
    batch_size: int
    override: bool
    fetched_at: str = ""
    batches: Dict[str, dict] = field(default_factory=dict)

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end_exclusive)


def make_run_id(
    provider: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    symbols: Sequence[str],
    batch_size: int,
    override: bool,
) -> str:
    digest = hashlib.sha256(
        ("\n".join(symbols) + f"|{batch_size}|{int(override)}").encode("utf-8")
    ).hexdigest()[:10]
    return f"{provider}_{start:%Y%m%d}_{end_exclusive:%Y%m%d}_{digest}"


def _run_dir(run_id: str, staging_root: Optional[Path]) -> Path:
    return Path(staging_root or STAGING_ROOT) / run_id


def _state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def save_plan(plan: Plan, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = _state_path(run_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n")
    tmp.replace(_state_path(run_dir))


def load_plan(run_id: str, staging_root: Optional[Path] = None) -> Plan:
    path = _state_path(_run_dir(run_id, staging_root))
    if not path.exists():
        raise FileNotFoundError(f"no staged run {run_id!r} at {path}")
    return Plan(**json.loads(path.read_text()))


def check_ownership(
    years: Sequence[int],
    start: pd.Timestamp,
    provider: str,
    *,
    override: bool,
    lake_root: Optional[Path],
) -> None:
    """Refuse a plan that would leave any year with two providers.

    * A full rebuild replaces whole years, so it may take over a year owned by
      anyone -- except the start year when ``start`` is not January 1st, because
      the rows before ``start`` would stay behind under the old provider.
    * A symbol refresh replaces some rows and keeps the rest, so every year it
      touches must already belong to this provider.
    """
    jan1 = pd.Timestamp(year=start.year, month=1, day=1)
    for year in years:
        exists = schema.resolve_year_path(year, lake_root) is not None
        if not exists:
            continue
        entry = manifest.partition(year, lake_root)
        owner = (entry or {}).get("provider")
        if owner == provider:
            continue
        if override:
            raise manifest.MixedProviderError(
                f"{year}: owned by {owner or 'nobody (no manifest entry)'!r}; a "
                f"--symbols refresh would leave {provider!r} rows beside it. "
                f"Rebuild the whole year instead."
            )
        if year == start.year and start != jan1:
            raise manifest.MixedProviderError(
                f"{year}: owned by {owner or 'nobody'!r} and --start {start.date()} "
                f"is mid-year, so rows before it would stay behind under the old "
                f"provider. Start on {jan1.date()}."
            )


def plan_rebuild(
    *,
    source: MarketDataProvider,
    start: pd.Timestamp,
    end_exclusive: Optional[pd.Timestamp],
    symbols_override: Optional[Sequence[str]],
    extra_symbols: Optional[Sequence[str]],
    protected_symbols: Optional[Sequence[str]],
    batch_size: int,
    lake_root: Optional[Path],
    registry_path: Optional[Path],
    universe_path: Optional[Path],
) -> Plan:
    cutoff = source.session_cutoff()
    end = cutoff if end_exclusive is None else min(pd.Timestamp(end_exclusive), cutoff)
    start = pd.Timestamp(start).normalize()
    if start >= end:
        raise ValueError(f"empty window {start.date()}..{end.date()}")

    override = symbols_override is not None
    if override:
        symbols = sorted(
            {registry.normalize_symbol(s) for s in symbols_override if str(s).strip()}
        )
    else:
        reg = registry.load_registry(registry_path)
        universe_meta = registry.load_universe_metadata(universe_path)
        # Everything the live range already holds is requested too: a symbol
        # left out of the plan would simply vanish from the replaced years.
        on_disk: Set[str] = set()
        for path in schema.discover_year_files(lake_root):
            if int(path.stem) >= start.year:
                on_disk |= anchor.symbols_in(path)
        # Added after target_symbols, not through it: a name retired from the
        # daily sync for repeated failure still has history in these years, and
        # a whole-year replacement that skipped it would delete that history.
        symbols = sorted(set(registry.target_symbols(
            reg, universe_meta, extra=extra_symbols, protect=protected_symbols,
        )) | on_disk)

    years = list(range(start.year, (end - pd.Timedelta(days=1)).year + 1))
    check_ownership(years, start, source.name, override=override, lake_root=lake_root)

    run_id = make_run_id(source.name, start, end, symbols, batch_size, override)
    return Plan(
        run_id=run_id,
        provider=source.name,
        start=str(start.date()),
        end_exclusive=str(end.date()),
        symbols=symbols,
        batch_size=int(batch_size),
        override=override,
    )


# ---------------------------------------------------------------------------
# Fetch (checkpointed)
# ---------------------------------------------------------------------------

def fetch_to_staging(plan: Plan, source: MarketDataProvider, run_dir: Path) -> Plan:
    """Fetch every batch not yet recorded in the plan, sharded by year.

    Shards are written before the batch is marked done, so a crash between the
    two re-fetches one batch rather than trusting a half-written one. Provider
    errors (credentials, entitlement) propagate with the checkpoint intact.
    """
    if source.name != plan.provider:
        raise ValueError(
            f"run {plan.run_id} was planned for {plan.provider!r}, not {source.name!r}"
        )
    batches = fetch.batched(plan.symbols, plan.batch_size)
    if not plan.fetched_at:
        plan.fetched_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    save_plan(plan, run_dir)

    todo = [(i, b) for i, b in enumerate(batches) if f"{i:05d}" not in plan.batches]
    logger.info(
        "run %s: %d batch(es), %d already staged, %d to fetch",
        plan.run_id, len(batches), len(batches) - len(todo), len(todo),
    )

    for i, batch in todo:
        key = f"{i:05d}"
        bars, failed = source.fetch_batch(batch, plan.start_ts, plan.end_ts)
        for year in schema.years_in(bars):
            shard = run_dir / "batches" / str(year) / f"{key}.parquet"
            schema.write_parquet(bars[bars[schema.DATE].dt.year == year], shard)

        report = source.last_report
        plan.batches[key] = {
            "symbols": len(batch),
            "rows": int(len(bars)),
            "failed": list(failed),
            "missing_adjusted": dict(getattr(report, "missing_adjusted", {}) or {}),
            "missing_raw": dict(getattr(report, "missing_raw", {}) or {}),
            "retries": int(getattr(report, "retries", 0) or 0),
        }
        save_plan(plan, run_dir)
        logger.info(
            "batch %s: %d row(s), %d failed (%d/%d staged)",
            key, len(bars), len(failed), len(plan.batches), len(batches),
        )
    return plan


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def _live_frame(year: int, lake_root: Optional[Path], pending: pd.DataFrame) -> pd.DataFrame:
    """A live partition as readers see it (pending anchor factors applied)."""
    path = schema.resolve_year_path(year, lake_root)
    if path is None:
        return schema.empty_frame()
    return anchor.apply(schema.read_frame(path), year, pending)


def assemble(plan: Plan, run_dir: Path, lake_root: Optional[Path]) -> Dict[int, Path]:
    """Write one staged partition per year. Idempotent."""
    pending = anchor.load(lake_root)
    daily = run_dir / "daily"
    staged: Dict[int, Path] = {}
    current_year = int(market_date().year)
    override_set = set(plan.symbols) if plan.override else set()

    years = range(plan.start_ts.year, (plan.end_ts - pd.Timedelta(days=1)).year + 1)
    fresh_by_year: Dict[int, pd.DataFrame] = {}
    returned: Set[str] = set()
    for year in years:
        shards = sorted((run_dir / "batches" / str(year)).glob("*.parquet"))
        fresh_by_year[year] = (
            schema.coerce(pd.concat([schema.read_frame(p) for p in shards], ignore_index=True))
            if shards else schema.empty_frame()
        )
        returned |= set(fresh_by_year[year][schema.SYMBOL])

    for year in years:
        fresh = fresh_by_year[year]
        live = _live_frame(year, lake_root, pending)
        if plan.override:
            # Keep everyone else; replace only the refreshed symbols' rows --
            # and only for symbols the provider actually returned, so a refresh
            # that fails for a name cannot delete the history it already has.
            replace = override_set & returned
            drop = live[schema.SYMBOL].isin(replace) & (live[schema.DATE] >= plan.start_ts)
            if len(fresh) == 0 and not bool(drop.any()):
                continue  # nothing about this year changes; leave it alone
            base = live[~drop]
        else:
            # Whole-year replacement, except rows before a mid-year start
            # (allowed only when the year already belongs to this provider).
            base = live[live[schema.DATE] < plan.start_ts]

        frame = schema.upsert(base, fresh)
        if len(frame) == 0:
            continue

        hot = writer._is_hot(year, lake_root, current_year)
        for stale in (schema.hot_year_path(year, daily), schema.cold_year_path(year, daily)):
            if stale.exists():
                stale.unlink()
        staged[year] = schema.write_year(frame, year, daily, hot=hot)
        logger.info("staged %s: %d row(s), %d symbol(s)",
                    staged[year].name, len(frame), frame[schema.SYMBOL].nunique())
    return staged


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(staged: Dict[int, Path], plan: Plan) -> Tuple[List[str], List[str]]:
    """(errors, warnings) for the staged partitions. Errors block the swap."""
    errors: List[str] = []
    warnings: List[str] = []
    if not staged:
        errors.append("nothing was staged")
    for year, path in sorted(staged.items()):
        frame = schema.read_frame(path)
        try:
            schema.assert_unique_key(frame, context=f"staged {year}")
        except ValueError as exc:
            errors.append(str(exc))
        late = int((frame[schema.DATE] >= plan.end_ts).sum())
        if late:
            errors.append(f"{year}: {late} row(s) on or after the cutoff {plan.end_exclusive}")
        weekend = int((frame[schema.DATE].dt.dayofweek >= 5).sum())
        if weekend:
            errors.append(f"{year}: {weekend} row(s) dated on a weekend")
        for col in (schema.CLOSE, schema.ADJ_CLOSE):
            bad = int((~(frame[col] > 0)).sum())
            if bad:
                errors.append(f"{year}: {bad} row(s) with non-positive or missing {col}")
        inverted = int((frame[schema.HIGH] < frame[schema.LOW]).sum())
        if inverted:
            warnings.append(f"{year}: {inverted} row(s) with high < low")
    return errors, warnings


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def _daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values([schema.SYMBOL, schema.DATE])
    ret = ordered.groupby(schema.SYMBOL)[schema.ADJ_CLOSE].pct_change()
    return ordered.assign(_ret=ret)[[schema.DATE, schema.SYMBOL, "_ret"]]


def compare(
    staged: Dict[int, Path],
    plan: Plan,
    lake_root: Optional[Path],
) -> Dict[str, object]:
    """Staged vs live, per year. Nothing here blocks except lost symbols."""
    pending = anchor.load(lake_root)
    years: Dict[str, dict] = {}
    scope = set(plan.symbols) if plan.override else None

    for year, path in sorted(staged.items()):
        new = schema.read_frame(path)
        old = _live_frame(year, lake_root, pending)
        entry = manifest.partition(year, lake_root) or {}
        if scope is not None:
            new = new[new[schema.SYMBOL].isin(scope)]
            old = old[old[schema.SYMBOL].isin(scope)]

        old_syms = set(old[schema.SYMBOL]) if len(old) else set()
        new_syms = set(new[schema.SYMBOL]) if len(new) else set()
        lost = sorted(old_syms - new_syms)

        key = [schema.DATE, schema.SYMBOL]
        both = old[key + [schema.CLOSE]].merge(
            new[key + [schema.CLOSE]], on=key, suffixes=("_live", "_staged")
        )
        close_ratio = (both["close_staged"] / both["close_live"]).groupby(both[schema.SYMBOL]).median()
        rescaled = close_ratio[(close_ratio - 1.0).abs() > 0.01]

        returns = _daily_returns(old).merge(
            _daily_returns(new), on=key, suffixes=("_live", "_staged")
        ).dropna()
        diff = (returns["_ret_staged"] - returns["_ret_live"]).abs()
        disagree = returns[diff > RETURN_DISAGREEMENT]
        worst = disagree.groupby(schema.SYMBOL).size().sort_values(ascending=False).head(20)

        years[str(year)] = {
            "live_provider": entry.get("provider"),
            "live_rows": int(len(old)),
            "staged_rows": int(len(new)),
            "live_symbols": len(old_syms),
            "staged_symbols": len(new_syms),
            "live_sessions": int(old[schema.DATE].nunique()) if len(old) else 0,
            "staged_sessions": int(new[schema.DATE].nunique()) if len(new) else 0,
            "symbols_lost": lost,
            "symbols_lost_fraction": (len(lost) / len(old_syms)) if old_syms else 0.0,
            "symbols_gained": len(new_syms - old_syms),
            "common_rows": int(len(both)),
            "raw_close_rescaled_symbols": {
                str(s): round(float(v), 6) for s, v in rescaled.sort_values().head(50).items()
            },
            "raw_close_rescaled_count": int(len(rescaled)),
            "return_pairs": int(len(returns)),
            "return_disagreements": int(len(disagree)),
            "return_disagreement_rate": (len(disagree) / len(returns)) if len(returns) else 0.0,
            "return_disagreement_worst": {str(s): int(n) for s, n in worst.items()},
        }
    return {"years": years}


def measure_seam(
    staged: Dict[int, Path],
    plan: Plan,
    lake_root: Optional[Path],
) -> Dict[str, dict]:
    """Per-symbol adj_close scale factor between the rebuilt range and the years before it.

    ``k = median(staged / live)`` over the first ``SEAM_SESSIONS`` sessions of
    the start year, both read as a reader sees them. Multiplying the older
    partitions' adj_close by ``k`` puts them on the new anchor, so the return
    from the last old bar to the first new one equals the return the old
    provider itself reported across that boundary.
    """
    if plan.override or plan.start_ts.year not in staged:
        return {}
    older = [p for p in schema.discover_year_files(lake_root) if int(p.stem) < plan.start_ts.year]
    if not older:
        return {}

    before: Set[str] = set()
    for path in older:
        before |= anchor.symbols_in(path)

    pending = anchor.load(lake_root)
    live = _live_frame(plan.start_ts.year, lake_root, pending)
    new = schema.read_frame(staged[plan.start_ts.year])
    sessions = sorted(new[schema.DATE].unique())[:SEAM_SESSIONS]

    key = [schema.DATE, schema.SYMBOL]
    window_live = live[live[schema.DATE].isin(sessions) & live[schema.SYMBOL].isin(before)]
    window_new = new[new[schema.DATE].isin(sessions) & new[schema.SYMBOL].isin(before)]
    both = window_live[key + [schema.ADJ_CLOSE]].merge(
        window_new[key + [schema.ADJ_CLOSE]], on=key, suffixes=("_live", "_staged")
    )

    out: Dict[str, dict] = {}
    for symbol in sorted(before):
        rows = both[both[schema.SYMBOL] == symbol]
        if len(rows) == 0:
            out[symbol] = {"factor": None, "sessions": 0, "status": "unmeasured"}
            continue
        ratio = rows["adj_close_staged"] / rows["adj_close_live"]
        k = float(np.median(ratio))
        spread = float(ratio.max() - ratio.min())
        out[symbol] = {
            "factor": k,
            "sessions": int(len(rows)),
            "spread": spread,
            "status": "splice" if abs(k - 1.0) > SEAM_TOLERANCE else "consistent",
        }
    return out


# ---------------------------------------------------------------------------
# Swap
# ---------------------------------------------------------------------------

def swap(
    staged: Dict[int, Path],
    plan: Plan,
    run_dir: Path,
    lake_root: Optional[Path],
    provenance: dict,
    seam: Dict[str, dict],
) -> Dict[str, object]:
    """Install the staged partitions, or restore everything on any failure."""
    lake = Path(lake_root or schema.DEFAULT_LAKE_ROOT)
    backup = run_dir / "backup"
    backup.mkdir(parents=True, exist_ok=True)

    manifest_before = json.loads(json.dumps(manifest.load(lake_root)))
    anchor_before = anchor.load(lake_root)
    backed_up: List[Tuple[Path, Path]] = []
    installed: List[Path] = []

    try:
        for year, path in sorted(staged.items()):
            for live in (schema.cold_year_path(year, lake_root), schema.hot_year_path(year, lake_root)):
                if live.exists():
                    dest = backup / live.name
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(live), str(dest))
                    backed_up.append((live, dest))
            # Copy beside the target, then rename: a rename within one
            # directory is atomic, so a reader never sees a half-copied file.
            target = lake / path.name
            partial = lake / (path.name + ".partial")
            shutil.copy2(str(path), str(partial))
            partial.replace(target)
            installed.append(target)

        notes = [
            f"rebuilt {plan.start}..{plan.end_exclusive} (exclusive) by rebuild_prices "
            f"run {plan.run_id}"
        ]
        for year, path in sorted(staged.items()):
            frame = schema.read_frame(lake / path.name)
            manifest.record(
                year, frame, provenance, lake_root,
                file_name=path.name, downloaded_at=plan.fetched_at or None, notes=notes,
            )

        pending = anchor.clear(anchor.load(lake_root), staged.keys())
        spliced = _apply_seam(pending, seam, plan, lake_root)
        anchor.save(spliced, lake_root)

        problems = [
            p for p in manifest.verify(lake_root)
            if any(p.startswith(f"{y}:") for y in staged)
        ]
        if problems:
            raise RuntimeError("post-swap verification failed: " + "; ".join(problems))
    except BaseException:
        logger.error("swap failed; restoring the previous lake")
        for target in installed:
            if target.exists():
                target.unlink()
        for partial in lake.glob("*.partial"):
            partial.unlink()
        for live, dest in backed_up:
            if dest.exists():
                shutil.move(str(dest), str(live))
        manifest.save(manifest_before, lake_root)
        anchor.save(anchor_before, lake_root)
        raise

    return {
        "installed": [p.name for p in installed],
        "backed_up_to": str(backup),
        "seam_spliced": sorted(s for s, v in seam.items() if v.get("status") == "splice"),
    }


def _apply_seam(
    pending: pd.DataFrame,
    seam: Dict[str, dict],
    plan: Plan,
    lake_root: Optional[Path],
) -> pd.DataFrame:
    """Record material seam factors as pending factors on the older partitions."""
    splice = {s: v for s, v in seam.items() if v.get("status") == "splice"}
    if not splice:
        return pending
    reason = f"splice onto {plan.provider} anchor at {plan.start} (run {plan.run_id})"
    for path in schema.discover_year_files(lake_root):
        year = int(path.stem)
        if year >= plan.start_ts.year:
            continue
        present = anchor.symbols_in(path)
        events = anchor.events_frame(
            {anchor.SYMBOL: s, anchor.BEFORE: plan.start_ts, anchor.FACTOR: v["factor"]}
            for s, v in sorted(splice.items()) if s in present
        )
        pending = anchor.add(pending, year, events, reason=reason)
    return pending


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def rebuild(
    *,
    start: pd.Timestamp = DEFAULT_START,
    end_exclusive: Optional[pd.Timestamp] = None,
    lake_root: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    universe_path: Optional[Path] = None,
    staging_root: Optional[Path] = None,
    audit_dir: Optional[Path] = None,
    batch_size: int = fetch.DEFAULT_BATCH_SIZE,
    symbols_override: Optional[Sequence[str]] = None,
    extra_symbols: Optional[Sequence[str]] = None,
    protected_symbols: Optional[Sequence[str]] = None,
    provider: Optional[MarketDataProvider] = None,
    resume: Optional[str] = None,
    stage_only: bool = False,
    allow_loss: bool = False,
) -> Dict[str, object]:
    """Plan (or resume), fetch, assemble, validate, compare, and swap."""
    source = _require_canonical(provider)

    if resume:
        plan = load_plan(resume, staging_root)
    else:
        plan = plan_rebuild(
            source=source, start=start, end_exclusive=end_exclusive,
            symbols_override=symbols_override, extra_symbols=extra_symbols,
            protected_symbols=protected_symbols, batch_size=batch_size,
            lake_root=lake_root, registry_path=registry_path, universe_path=universe_path,
        )
        existing = _state_path(_run_dir(plan.run_id, staging_root))
        if existing.exists():
            logger.info("run %s already staged in part; resuming it", plan.run_id)
            plan = load_plan(plan.run_id, staging_root)

    run_dir = _run_dir(plan.run_id, staging_root)
    logger.info(
        "rebuild %s: %d symbol(s), %s..%s (exclusive), %s",
        plan.run_id, len(plan.symbols), plan.start, plan.end_exclusive,
        "symbol refresh" if plan.override else "whole-year replacement",
    )

    plan = fetch_to_staging(plan, source, run_dir)
    staged = assemble(plan, run_dir, lake_root)
    errors, warnings = validate(staged, plan)
    comparison = compare(staged, plan, lake_root)
    seam = measure_seam(staged, plan, lake_root)

    failed = sorted({s for b in plan.batches.values() for s in b.get("failed", [])})
    missing_adjusted = sum(sum(b.get("missing_adjusted", {}).values()) for b in plan.batches.values())
    missing_raw = sum(sum(b.get("missing_raw", {}).values()) for b in plan.batches.values())

    if not allow_loss:
        for year, info in comparison["years"].items():
            if info["symbols_lost_fraction"] > MAX_LOST_FRACTION:
                errors.append(
                    f"{year}: would lose {len(info['symbols_lost'])} of "
                    f"{info['live_symbols']} symbol(s) "
                    f"({info['symbols_lost_fraction']:.1%} > {MAX_LOST_FRACTION:.0%}); "
                    f"pass --allow-loss after reviewing the list"
                )

    report: Dict[str, object] = {
        "run_id": plan.run_id,
        "provider": plan.provider,
        "provenance": source.provenance(),
        "window": {"start": plan.start, "end_exclusive": plan.end_exclusive},
        "mode": "symbol_refresh" if plan.override else "whole_year",
        "symbols_requested": len(plan.symbols),
        "symbols_failed": failed,
        "rows_missing_adjusted": int(missing_adjusted),
        "rows_missing_raw": int(missing_raw),
        "staged_years": sorted(staged),
        "errors": errors,
        "warnings": warnings,
        "comparison": comparison,
        "seam": seam,
        "swapped": False,
    }

    if errors:
        logger.error("validation failed; live lake untouched:\n  %s", "\n  ".join(errors))
    elif stage_only:
        logger.info("stage-only: live lake untouched. Swap with --resume %s", plan.run_id)
    else:
        report["swap"] = swap(staged, plan, run_dir, lake_root, source.provenance(), seam)
        report["swapped"] = True
        # The registry describes the lake it sits beside. A run against a
        # scratch lake (tests, experiments) must not rewrite the real one, so
        # it is only updated when this is the canonical lake or a registry
        # path was given explicitly.
        if registry_path is not None or lake_root is None:
            reg = registry.load_registry(registry_path)
            universe_meta = registry.load_universe_metadata(universe_path)
            ok = set(plan.symbols) - set(failed)
            registry.write_registry(
                registry.update_registry(
                    reg, universe_meta, succeeded=ok, failed=failed,
                    as_of=plan.end_ts - pd.Timedelta(days=1),
                ),
                registry_path,
            )
        logger.info("swap complete: %s", ", ".join(report["swap"]["installed"]))

    report_path = _write_report(report, audit_dir)
    report["report_path"] = str(report_path)
    return report


def _write_report(report: Dict[str, object], audit_dir: Optional[Path]) -> Path:
    directory = Path(audit_dir or AUDIT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rebuild_{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    logger.info("report: %s", path)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Staged partition rebuild / year finalize.")
    parser.add_argument(
        "--start", default=str(DEFAULT_START.date()),
        help=f"First session to rebuild (default {DEFAULT_START.date()}).",
    )
    parser.add_argument("--end", default=None, help="Exclusive end (default: session cutoff).")
    parser.add_argument("--symbols", nargs="+", help="Refresh only these tickers.")
    parser.add_argument(
        "--no-membership", action="store_true",
        help="Do not add current and historical index members to the target set.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=fetch.DEFAULT_BATCH_SIZE,
        help="Tickers per request.",
    )
    parser.add_argument("--resume", default=None, help="Continue a staged run by id.")
    parser.add_argument("--stage-only", action="store_true", help="Stop before the swap.")
    parser.add_argument(
        "--allow-loss", action="store_true",
        help=f"Swap even if a year loses more than {MAX_LOST_FRACTION * 100:.0f}%% of its symbols.",
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

    extra: List[str] = []
    protect: List[str] = []
    if not args.no_membership and not args.symbols and not args.resume:
        from data_pipeline import membership

        current, historical = membership.tracked_symbols(since=pd.Timestamp(args.start))
        extra, protect = sorted(current | historical), sorted(current)

    try:
        report = rebuild(
            start=pd.Timestamp(args.start),
            end_exclusive=pd.Timestamp(args.end) if args.end else None,
            batch_size=args.batch_size,
            symbols_override=args.symbols,
            extra_symbols=extra,
            protected_symbols=protect,
            resume=args.resume,
            stage_only=args.stage_only,
            allow_loss=args.allow_loss,
        )
    except Exception:
        logger.exception("rebuild aborted")
        return 1
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

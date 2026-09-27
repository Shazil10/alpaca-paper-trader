"""writer.py

Persists fetched bars into the correct year files, preserving each year's
existing format.

Format rule:

* A year already on disk keeps the format it is in. This matters in early
  January, when the 5-day correction overlap reaches back into last year -- that
  year is Parquet by then and must stay Parquet, not get rewritten as CSV.
* A brand new year is CSV if it is the current calendar year (hot, appended and
  committed daily) and Parquet otherwise (cold, written once by a backfill).

Writes are read-modify-write: Parquet cannot be appended in place, and the hot
CSV is rewritten so that ``schema.write_csv`` can guarantee byte-deterministic
output. Determinism is what keeps the daily git commit cheap.

Anchor maintenance
------------------
Every write also keeps ``adj_close`` on a single anchor per symbol (see
``data_pipeline.anchor``). When the fresh rows show that a split or dividend
has moved the anchor, the older stored rows are rescaled: in place for any
partition this write rewrites anyway, as a pending factor for an untouched cold
partition. A partition that is rewritten has its pending factors folded in, so
none is ever applied twice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_pipeline import anchor, manifest, schema

logger = logging.getLogger(__name__)


def _is_hot(year: int, root: Optional[Path], current_year: int) -> bool:
    """Decide the on-disk format for ``year``."""
    existing = schema.resolve_year_path(year, root)
    if existing is not None:
        return existing.suffix == ".csv"
    return year == current_year


def persist(
    bars: pd.DataFrame,
    *,
    root: Optional[Path] = None,
    current_year: Optional[int] = None,
    provenance: Optional[Any] = None,
) -> Dict[int, int]:
    """Upsert ``bars`` into their year files.

    Returns a mapping of year -> total row count in that file afterwards.

    ``provenance`` names the provider that produced ``bars``. Every year it
    touches is checked against the partition manifest *before anything is
    written*, and the whole call is refused if any one year would end up with
    two providers -- checking up front rather than per year means a batch that
    spans a boundary cannot land half its rows before the refusal. After the
    write, each touched year's manifest entry is refreshed from its full
    contents, so the hash always describes what is on disk.
    """
    frame = schema.coerce(bars)
    if len(frame) == 0:
        return {}

    source = manifest.Provenance.coerce(provenance)
    years = schema.years_in(frame)
    for year in years:
        manifest.check_write(
            year, source, root,
            file_exists=schema.resolve_year_path(year, root) is not None,
        )

    year_now = int(current_year or pd.Timestamp.now().year)
    written: Dict[int, int] = {}

    # Read every touched partition first, with its pending anchor factors
    # applied: that is both what a reader currently sees (so the comparison
    # below is against the right numbers) and what gets written back (so the
    # factors are folded into the bytes and can be cleared).
    pending = anchor.load(root)
    current: Dict[int, pd.DataFrame] = {}
    for year in years:
        existing_path = schema.resolve_year_path(year, root)
        existing = schema.read_frame(existing_path) if existing_path else schema.empty_frame()
        current[year] = anchor.apply(existing, year, pending)

    events = anchor.detect(
        pd.concat(list(current.values()), ignore_index=True), frame
    )
    if len(events):
        _log_anchor_events(events, source.provider)

    for year in years:
        incoming = frame[frame[schema.DATE].dt.year == year]
        if len(incoming) == 0:
            continue

        hot = _is_hot(year, root, year_now)
        base = anchor.rescale(current[year], events)

        merged = schema.upsert(base, incoming)
        schema.assert_unique_key(merged, context=f"persist({year})")

        path = schema.write_year(merged, year, root, hot=hot)
        manifest.record(year, merged, source, root, file_name=path.name)

        # Guard against a half-finished rollover leaving two files for one year,
        # which would double-count rows on read.
        if hot:
            stale = schema.cold_year_path(year, root)
        else:
            stale = schema.hot_year_path(year, root)
        if stale.exists() and stale != path:
            logger.warning(
                "year %s exists in both formats; %s is now authoritative, "
                "remove %s to complete the rollover",
                year,
                path.name,
                stale.name,
            )

        written[year] = len(merged)
        logger.info(
            "persisted %d new/updated row(s) into %s (%d total)",
            len(incoming),
            path.name,
            len(merged),
        )

    # The touched partitions now hold their factors in the bytes.
    pending = anchor.clear(pending, written)
    if len(events):
        pending = _reanchor_untouched(events, set(written), pending, root, source)
    anchor.save(pending, root)

    return written


def _log_anchor_events(events: pd.DataFrame, provider: str) -> None:
    sample = ", ".join(
        f"{r[anchor.SYMBOL]}x{r[anchor.FACTOR]:.6f}@{pd.Timestamp(r[anchor.BEFORE]).date()}"
        for _, r in events.head(10).iterrows()
    )
    logger.info(
        "adj_close re-anchored for %d symbol(s) by %s (split/dividend since the "
        "stored rows were fetched): %s%s",
        len(events), provider or manifest.UNSPECIFIED, sample,
        " ..." if len(events) > 10 else "",
    )


def _reanchor_untouched(
    events: pd.DataFrame,
    touched: set,
    pending: pd.DataFrame,
    root: Optional[Path],
    source: manifest.Provenance,
) -> pd.DataFrame:
    """Carry re-anchor events into partitions this write did not rewrite.

    A hot CSV is rewritten in place (cheap in git). A cold Parquet partition is
    left byte-identical and the factor is recorded as pending instead. Only
    partitions that actually contain an affected symbol get an entry.
    """
    latest_before = pd.Timestamp(events[anchor.BEFORE].max())
    reason = f"re-anchor by {source.provider or manifest.UNSPECIFIED}"

    for path in schema.discover_year_files(root):
        year = int(path.stem)
        if year in touched or year > latest_before.year:
            continue
        present = anchor.symbols_in(path)
        relevant = events[
            events[anchor.SYMBOL].astype(str).isin(present)
            & (events[anchor.BEFORE] > pd.Timestamp(year=year, month=1, day=1))
        ]
        if len(relevant) == 0:
            continue

        if path.suffix == ".csv":
            frame = anchor.apply(schema.read_frame(path), year, pending)
            frame = anchor.rescale(frame, relevant)
            schema.write_csv(frame, path)
            entry = manifest.partition(year, root)
            manifest.record(
                year, frame, entry, root, file_name=path.name,
                downloaded_at=(entry or {}).get("downloaded_at"),
            )
            pending = anchor.clear(pending, [year])
            logger.info("rescaled %d symbol(s) in place in %s", len(relevant), path.name)
        else:
            pending = anchor.add(pending, year, relevant, reason=reason)
            logger.info(
                "recorded %d pending anchor factor(s) for immutable %s",
                len(relevant), path.name,
            )
    return pending


def finalize_year(year: int, root: Optional[Path] = None, *, remove_csv: bool = True) -> Path:
    """Convert a closed year's hot CSV into cold Parquet (January rollover).

    Returns the Parquet path. Idempotent: if the CSV is already gone and the
    Parquet exists, this is a no-op.
    """
    csv_path = schema.hot_year_path(year, root)
    parquet_path = schema.cold_year_path(year, root)

    if not csv_path.exists():
        if parquet_path.exists():
            logger.info("%s already finalized", parquet_path.name)
            return parquet_path
        raise FileNotFoundError(f"no hot CSV for {year} at {csv_path}")

    frame = schema.read_frame(csv_path)
    if parquet_path.exists():
        frame = schema.upsert(schema.read_frame(parquet_path), frame)

    schema.write_parquet(frame, parquet_path)
    logger.info("wrote %s (%d rows)", parquet_path.name, len(frame))

    # Verify before deleting the source.
    verify = schema.read_frame(parquet_path)
    if len(verify) != len(frame):
        raise RuntimeError(
            f"finalize verification failed for {year}: "
            f"wrote {len(frame)} rows, read back {len(verify)}"
        )

    # The rows are unchanged, so provenance and hash carry over; only the file
    # the entry points at moves from CSV to Parquet.
    entry = manifest.partition(year, root)
    if entry is not None:
        manifest.record(
            year, verify, entry, root,
            file_name=parquet_path.name,
            downloaded_at=entry.get("downloaded_at"),
        )

    if remove_csv:
        csv_path.unlink()
        logger.info("removed %s", csv_path.name)

    logger.info(
        "add '!data/prices/daily/%s.parquet' to .gitignore and commit it", year
    )
    return parquet_path

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
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from data_pipeline import schema

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
) -> Dict[int, int]:
    """Upsert ``bars`` into their year files.

    Returns a mapping of year -> total row count in that file afterwards.
    """
    frame = schema.coerce(bars)
    if len(frame) == 0:
        return {}

    year_now = int(current_year or pd.Timestamp.now().year)
    written: Dict[int, int] = {}

    for year in schema.years_in(frame):
        incoming = frame[frame[schema.DATE].dt.year == year]
        if len(incoming) == 0:
            continue

        hot = _is_hot(year, root, year_now)
        existing_path = schema.resolve_year_path(year, root)
        existing = schema.read_frame(existing_path) if existing_path else schema.empty_frame()

        merged = schema.upsert(existing, incoming)
        schema.assert_unique_key(merged, context=f"persist({year})")

        path = schema.write_year(merged, year, root, hot=hot)

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

    return written


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

    if remove_csv:
        csv_path.unlink()
        logger.info("removed %s", csv_path.name)

    logger.info(
        "add '!data/prices/daily/%s.parquet' to .gitignore and commit it", year
    )
    return parquet_path

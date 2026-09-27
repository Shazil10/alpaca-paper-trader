#!/usr/bin/env python
"""Record provenance for partitions written before the manifest existed.

Every year in ``data/prices/daily/`` up to the Alpaca migration was written by
the Yahoo pipeline (``yf.download``, ``auto_adjust=False``). The manifest must
say so before anything else writes the lake: ``writer.persist`` refuses a
named provider's rows into a partition whose owner is unknown, and the audit
reports provenance straight from this file.

What is recorded per year
-------------------------
* ``provider=yahoo``, ``feed=yfinance``, and Yahoo's adjustment semantics --
  ``close`` split-adjusted (not the raw print), ``adj_close`` split+dividend.
* ``downloaded_at``: the partition's last git commit time. The true download
  time was never stored; the commit that last changed the bytes is the closest
  honest bound, and a note says it is an approximation.
* Notes describing what the year holds -- the 2005-2015 partitions are the
  ETF-only backfill and are kept as legacy history after the migration.

Refuses to overwrite an existing entry unless ``--overwrite``.

    PYTHONPATH=src ./venv/bin/python scripts/bootstrap_manifest.py --dry-run
    PYTHONPATH=src ./venv/bin/python scripts/bootstrap_manifest.py
    PYTHONPATH=src ./venv/bin/python scripts/bootstrap_manifest.py --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_pipeline import manifest, schema  # noqa: E402
from data_pipeline.providers.yahoo import YahooProvider  # noqa: E402

logger = logging.getLogger("bootstrap_manifest")

#: First year the canonical provider (Alpaca SIP) covers. Earlier Yahoo years
#: are legacy history that the migration preserves rather than replaces.
CANONICAL_START_YEAR = 2016

SEMANTICS_NOTE = (
    "Yahoo auto_adjust=False: close/open/high/low are split-adjusted as of the "
    "download (NOT the raw print); adj_close is split+dividend adjusted; volume "
    "is split-adjusted."
)
DOWNLOAD_NOTE = (
    "downloaded_at is the partition's last git commit time; the true download "
    "time was not recorded."
)


def last_commit_time(path: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
    except Exception:
        return None
    return out or None


def notes_for(year: int, frame) -> List[str]:
    symbols = int(frame[schema.SYMBOL].nunique()) if len(frame) else 0
    notes = [SEMANTICS_NOTE, DOWNLOAD_NOTE]
    if year < CANONICAL_START_YEAR:
        notes.append(
            f"Legacy ETF-only Yahoo backfill ({symbols} symbols). Preserved as-is "
            f"through the Alpaca migration; adj_close is spliced onto the canonical "
            f"anchor via anchor_factors.csv, bytes unchanged."
        )
    else:
        notes.append(
            f"Yahoo-era partition ({symbols} symbols). Scheduled for whole-year "
            f"replacement by the Alpaca SIP rebuild (rebuild_prices.py)."
        )
    return notes


def bootstrap(*, dry_run: bool, overwrite: bool, root: Optional[Path] = None) -> Dict[str, dict]:
    provenance = YahooProvider().provenance()
    results: Dict[str, dict] = {}
    for path in schema.discover_year_files(root):
        year = int(path.stem)
        existing = manifest.partition(year, root)
        if existing is not None and not overwrite:
            results[str(year)] = {"status": "exists", "provider": existing.get("provider")}
            continue
        frame = schema.read_frame(path)
        when = last_commit_time(path)
        if dry_run:
            results[str(year)] = {
                "status": "would-record", "file": path.name, "rows": len(frame),
                "downloaded_at": when,
            }
            continue
        entry = manifest.bootstrap(
            year, provenance, root,
            downloaded_at=when, notes=notes_for(year, frame), overwrite=overwrite,
        )
        results[str(year)] = {"status": "recorded", "rows": entry["row_count"],
                              "content_hash": entry["content_hash"]}
        logger.info("%s: recorded yahoo, %d rows", year, entry["row_count"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Check the manifest against disk.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.verify:
        problems = manifest.verify()
        for problem in problems:
            print(problem)
        print("manifest clean" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    print(json.dumps(bootstrap(dry_run=args.dry_run, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

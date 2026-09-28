#!/usr/bin/env python
"""Pull ticker renames from Alpaca's corporate-actions feed into symbol_aliases.csv.

    PYTHONPATH=src ./venv/bin/python scripts/refresh_aliases.py            # last 30 days
    PYTHONPATH=src ./venv/bin/python scripts/refresh_aliases.py --since 2016-01-01

The after-close workflow runs the 30-day form daily; the ``--since`` form seeds
the file once. See ``data_pipeline.aliases`` for why the mapping is recorded
rather than left inside Alpaca's ``asof`` parameter.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_pipeline import aliases  # noqa: E402
from data_pipeline.providers.alpaca import AlpacaProvider  # noqa: E402
from data_pipeline.providers.base import ProviderError, market_date  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh symbol_aliases.csv from Alpaca.")
    parser.add_argument("--since", default=None, help="Start date (default: 30 days ago).")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    end = market_date()
    start = pd.Timestamp(args.since) if args.since else end - pd.Timedelta(days=args.days)
    try:
        result = aliases.refresh(AlpacaProvider(), start, end)
    except ProviderError as exc:
        logging.getLogger("refresh_aliases").error("alias refresh failed: %s", exc)
        return 1
    print(json.dumps({"start": str(start.date()), "end": str(end.date()), **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build one HTML page comparing backtest runs.

    # every strategy's latest run, side by side
    PYTHONPATH=src ./venv/bin/python scripts/build_report.py --latest-per-strategy

    # two specific runs
    PYTHONPATH=src ./venv/bin/python scripts/build_report.py \
        --runs runs/2026-09-22_tsmom_etf_cross_asset_8a6329b2 \
               runs/2026-09-21_clenow_momentum_v2_b568015a

    # one strategy across its own history, with its research pass folded in
    PYTHONPATH=src ./venv/bin/python scripts/build_report.py \
        --strategy tsmom_etf_cross_asset --versions \
        --research runs/2026-09-22_research_tsmom_etf_cross_asset

The benchmark and the equal-weight baseline are built from the lake over the
report's own window, so they are always aligned with what is being compared rather
than copied from whichever run happened to store them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from backtest import baselines, report  # noqa: E402
from backtest.result import DEFAULT_RUNS_DIR  # noqa: E402
from data_pipeline import store  # noqa: E402

logger = logging.getLogger("build_report")

DEFAULT_OUT = DEFAULT_RUNS_DIR / "report.html"


def benchmark_series(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp, capital: float
) -> Optional[pd.Series]:
    """Buy-and-hold the benchmark over exactly the report's window."""
    matrix = store.load_close_matrix([symbol], start=start, end=end)
    if matrix.empty or symbol not in matrix.columns:
        logger.warning("no %s bars in %s..%s", symbol, start.date(), end.date())
        return None
    prices = matrix[symbol].dropna()
    if len(prices) < 3:
        return None
    return prices / prices.iloc[0] * capital


def equal_weight_baseline(
    symbols: List[str], start: pd.Timestamp, end: pd.Timestamp, capital: float
) -> Optional[pd.Series]:
    """Equal weight over the names the runs actually traded.

    Drawn from the traded set rather than the whole universe on purpose: the
    question this baseline answers is "did the *ranking* add anything", so the
    candidate pool has to be the one the strategy chose from. An equal-weight
    basket of 1,400 names would be answering a different question.
    """
    if not symbols:
        return None
    matrix = store.load_close_matrix(sorted(symbols), start=start, end=end)
    if matrix.empty:
        return None
    series = baselines.equal_weight(
        matrix, sorted(matrix.columns), start, end, capital
    )
    return series if len(series) > 2 else None


def traded_symbols(runs: List[report.RunData]) -> List[str]:
    """Symbols appearing in any run's fills."""
    found: set = set()
    for run in runs:
        if run.path is None:
            continue
        fills = run.path / "fills.csv"
        if not fills.exists():
            continue
        try:
            frame = pd.read_csv(fills, usecols=["symbol"])
        except Exception:
            continue
        found.update(str(s).strip().upper() for s in frame["symbol"].dropna())
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="*", help="Explicit run directories.")
    parser.add_argument(
        "--latest-per-strategy", action="store_true",
        help="One run per strategy_id, the newest of each.",
    )
    parser.add_argument("--strategy", help="Restrict to one strategy_id.")
    parser.add_argument(
        "--versions", action="store_true",
        help="Add a current-versus-previous section for --strategy.",
    )
    parser.add_argument("--research", help="A research run directory to fold in.")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--split", help="IS/OOS split date, e.g. 2024-01-01.")
    parser.add_argument("--title", default="Backtest comparison")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- choose the runs -------------------------------------------------
    paths: List[Path] = []
    if args.runs:
        paths = [Path(p) for p in args.runs]
    elif args.strategy:
        paths = report.discover_runs(DEFAULT_RUNS_DIR, strategy_id=args.strategy)
    elif args.latest_per_strategy:
        newest: dict = {}
        for path in report.discover_runs(DEFAULT_RUNS_DIR):
            newest[report.load_run(path).strategy_id] = path
        paths = list(newest.values())
    else:
        paths = report.discover_runs(DEFAULT_RUNS_DIR)

    if not paths:
        print("No run directories with an equity.csv found.")
        return 1

    history = [report.load_run(p) for p in paths]

    # With --versions the report compares one strategy against its own past, so
    # only the newest belongs in the headline; the rest feed the version section.
    if args.versions and args.strategy and len(history) > 1:
        runs = [history[-1]]
        version_history = history
    else:
        runs = history
        version_history = []

    runs = [r for r in runs if len(r.equity) > 2]
    if not runs:
        print("Runs found, but none had a usable equity curve.")
        return 1

    start = max(r.equity.index[0] for r in runs)
    end = max(r.equity.index[-1] for r in runs)
    capital = float(runs[0].equity.iloc[0]) or 100_000.0

    inputs = report.ReportInputs(
        title=args.title,
        runs=runs,
        benchmark=benchmark_series(args.benchmark, start, end, capital),
        benchmark_name=args.benchmark,
        equal_weight=equal_weight_baseline(
            traded_symbols(runs), start, end, capital
        ),
        is_oos_split=pd.Timestamp(args.split) if args.split else None,
        research_dir=Path(args.research) if args.research else None,
        version_history=version_history,
    )

    out = report.write_report(inputs, Path(args.out))
    print(f"\nWrote {out}")
    print(f"  {len(runs)} run(s), window {start.date()} .. {end.date()}")
    if inputs.equal_weight is None:
        print("  no equal-weight baseline (no fills.csv in the runs)")
    if inputs.benchmark is None:
        print(f"  no {args.benchmark} baseline (not in the lake for this window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

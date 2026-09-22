"""Fast vectorized alpha screen.

Ignores cash, integer shares, and liquidity constraints.
Purpose: reject bad ideas quickly and enable thousand-run parameter sweeps.
"""
from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest import metrics

logger = logging.getLogger(__name__)


def fast_backtest(
    weight_fn: Callable[[pd.DataFrame, pd.Timestamp, Dict[str, Any]], Dict[str, float]],
    close_matrix: pd.DataFrame,
    start: str,
    end: str,
    params: Optional[Dict[str, Any]] = None,
    rebalance_freq: str = "M",
    cost_bps: float = 10.0,
    benchmark: str = "SPY",
) -> Dict[str, Any]:
    """Run a fast vectorized backtest.

    Args:
        weight_fn: (close_matrix_to_date, date, params) -> {symbol: weight}
        close_matrix: date x symbol price matrix (full history)
        start/end: date range strings
        params: strategy parameters
        rebalance_freq: "D" (daily), "W" (weekly), "M" (monthly)
        cost_bps: round-trip transaction cost in bps
        benchmark: benchmark symbol

    Returns dict with equity, returns, benchmark, allocations, metrics.
    """
    params = params or {}
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Filter to date range
    prices = close_matrix.loc[
        (close_matrix.index >= start_ts) & (close_matrix.index <= end_ts)
    ].copy()

    if len(prices) < 2:
        return {"equity": pd.Series(dtype=float), "metrics": {}}

    # Daily returns
    returns = prices.pct_change().fillna(0)

    # Determine rebalance dates
    if rebalance_freq == "D":
        rebal_mask = pd.Series(True, index=prices.index)
    elif rebalance_freq == "W":
        rebal_mask = prices.index.to_series().dt.dayofweek == 0  # Monday
    elif rebalance_freq == "M":
        month_groups = prices.index.to_series().groupby(
            [prices.index.year, prices.index.month]
        )
        first_days = month_groups.first()
        rebal_mask = prices.index.isin(first_days.values)
    else:
        rebal_mask = pd.Series(True, index=prices.index)

    rebal_mask.iloc[0] = True  # Always rebalance on first day

    # Run simulation
    equity = pd.Series(1.0, index=prices.index)
    current_weights: Dict[str, float] = {}
    allocations: List[Tuple[pd.Timestamp, Dict[str, float]]] = []

    for i, date in enumerate(prices.index):
        if i == 0:
            history = close_matrix.loc[close_matrix.index <= date]
            current_weights = weight_fn(history, date, params)
            allocations.append((date, dict(current_weights)))
            continue

        # Daily portfolio return
        day_ret = 0.0
        for sym, w in current_weights.items():
            if sym in returns.columns:
                day_ret += w * returns.loc[date, sym]

        equity.iloc[i] = equity.iloc[i - 1] * (1 + day_ret)

        # Rebalance
        if rebal_mask.iloc[i]:
            history = close_matrix.loc[close_matrix.index <= date]
            new_weights = weight_fn(history, date, params)

            # Apply transaction cost
            if cost_bps > 0 and current_weights:
                all_syms = set(current_weights.keys()) | set(new_weights.keys())
                turnover = sum(
                    abs(new_weights.get(s, 0) - current_weights.get(s, 0))
                    for s in all_syms
                ) / 2
                cost = turnover * cost_bps / 10_000 * 2  # round-trip
                equity.iloc[i] *= (1 - cost)

            current_weights = new_weights
            allocations.append((date, dict(current_weights)))

    # Benchmark
    bench_equity = pd.Series(dtype=float)
    bench_returns = pd.Series(dtype=float)
    if benchmark in prices.columns:
        bench = prices[benchmark].dropna()
        bench_equity = bench / bench.iloc[0]
        bench_returns = bench.pct_change().fillna(0)

    # Metrics
    eq_returns = equity.pct_change().fillna(0)
    result_metrics = metrics.compute_full_metrics(
        equity, eq_returns, bench_equity, bench_returns
    )

    return {
        "equity": equity,
        "returns": eq_returns,
        "benchmark": bench_equity,
        "benchmark_returns": bench_returns,
        "allocations": allocations,
        "metrics": result_metrics,
    }


def parameter_sweep(
    weight_fn: Callable,
    close_matrix: pd.DataFrame,
    start: str,
    end: str,
    param_grid: Dict[str, List],
    rebalance_freq: str = "M",
    cost_bps: float = 10.0,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Sweep over a parameter grid using the fast engine.

    Returns DataFrame with one row per parameter combination,
    including all metrics.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    results = []

    for combo in combos:
        params = dict(zip(keys, combo))

        try:
            result = fast_backtest(
                weight_fn, close_matrix, start, end,
                params=params,
                rebalance_freq=rebalance_freq,
                cost_bps=cost_bps,
                benchmark=benchmark,
            )

            row = dict(params)
            row.update(result.get("metrics", {}))
            results.append(row)

        except Exception as e:
            logger.warning("Sweep failed for %s: %s", params, e)
            row = dict(params)
            row["error"] = str(e)
            results.append(row)

    return pd.DataFrame(results)

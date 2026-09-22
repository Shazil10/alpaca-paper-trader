"""Benchmark and baseline comparison utilities.

Every backtest result auto-compares against SPY and can optionally compare
against equal-weight and random-portfolio baselines.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from backtest import metrics

logger = logging.getLogger(__name__)


def buy_and_hold(
    close_matrix: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
) -> pd.Series:
    """Buy-and-hold equity curve for a single symbol."""
    if symbol not in close_matrix.columns:
        return pd.Series(dtype=float)

    prices = close_matrix[symbol].loc[start:end].dropna()
    if len(prices) == 0:
        return pd.Series(dtype=float)

    return (prices / prices.iloc[0]) * initial_capital


def equal_weight(
    close_matrix: pd.DataFrame,
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    rebalance_freq: str = "ME",
) -> pd.Series:
    """Equal-weight portfolio over given symbols, monthly rebalance."""
    available = [s for s in symbols if s in close_matrix.columns]
    if not available:
        return pd.Series(dtype=float)

    prices = close_matrix[available].loc[start:end].dropna(how="all")
    if len(prices) < 2:
        return pd.Series(dtype=float)

    returns = prices.pct_change().dropna()

    # Equal-weight daily returns
    ew_returns = returns.mean(axis=1)
    equity = (1 + ew_returns).cumprod() * initial_capital

    return equity


def random_portfolio_placebo(
    close_matrix: pd.DataFrame,
    n_positions: int,
    universe: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Run random portfolio simulations and return Sharpe distribution.

    Each simulation:
    1. Randomly select n_positions from universe
    2. Equal-weight, buy-and-hold
    3. Compute Sharpe

    Returns: {
        'sharpes': list of Sharpe ratios,
        'mean_sharpe': float,
        'median_sharpe': float,
        'p5': float, 'p25': float, 'p75': float, 'p95': float,
    }
    """
    rng = np.random.RandomState(seed)
    available = [s for s in universe if s in close_matrix.columns]

    if len(available) < n_positions:
        return {"sharpes": [], "mean_sharpe": 0, "median_sharpe": 0}

    prices = close_matrix[available].loc[start:end].dropna(how="all")
    if len(prices) < 20:
        return {"sharpes": [], "mean_sharpe": 0, "median_sharpe": 0}

    returns = prices.pct_change().dropna()
    sharpes = []

    for _ in range(n_simulations):
        selected = rng.choice(available, size=n_positions, replace=False)
        cols = [s for s in selected if s in returns.columns]
        if len(cols) < 2:
            continue
        ew = returns[cols].mean(axis=1)
        sh = metrics.sharpe_ratio(ew)
        sharpes.append(sh)

    if not sharpes:
        return {"sharpes": [], "mean_sharpe": 0, "median_sharpe": 0}

    arr = np.array(sharpes)
    return {
        "sharpes": sharpes,
        "mean_sharpe": float(arr.mean()),
        "median_sharpe": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def percentile_vs_random(
    strategy_sharpe: float,
    placebo: Dict[str, float],
) -> float:
    """What percentile is the strategy's Sharpe in the random distribution?"""
    sharpes = placebo.get("sharpes", [])
    if not sharpes:
        return 50.0
    return float(np.mean([1 if strategy_sharpe > s else 0 for s in sharpes]) * 100)

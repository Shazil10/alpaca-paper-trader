"""Monte Carlo simulation and bootstrap resampling."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest import metrics

logger = logging.getLogger(__name__)


def _equity_from_returns(rets: np.ndarray) -> pd.Series:
    """Equity curve anchored at 1.0 before the first return.

    Without the leading 1.0 the curve starts at (1 + r_0), which makes CAGR
    depend on whichever return happens to land first — so a permutation of the
    same returns reports a different CAGR — and hides a first-day loss from
    the running peak used by max drawdown.
    """
    return pd.Series(np.concatenate(([1.0], (1.0 + rets).cumprod())))


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation."""
    sharpes: List[float] = field(default_factory=list)
    cagrs: List[float] = field(default_factory=list)
    max_dds: List[float] = field(default_factory=list)
    p_loss: float = 0.0  # P(final equity < initial)

    @property
    def sharpe_ci(self) -> Dict[str, float]:
        if not self.sharpes:
            return {}
        arr = np.array(self.sharpes)
        return {
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }


def trade_permutation(
    returns: pd.Series,
    n_simulations: int = 1000,
    seed: int = 42,
) -> MonteCarloResult:
    """Shuffle the sequence of daily returns.

    Answers: "was the equity curve's smoothness an accident of ordering?"
    Destroys autocorrelation deliberately.
    """
    rng = np.random.RandomState(seed)
    ret_arr = returns.values.copy()

    sharpes, cagrs, max_dds = [], [], []
    n_loss = 0

    for _ in range(n_simulations):
        shuffled = rng.permutation(ret_arr)
        equity = _equity_from_returns(shuffled)

        sharpes.append(metrics.sharpe_ratio(pd.Series(shuffled)))
        cagrs.append(metrics.cagr(equity))
        max_dds.append(metrics.max_drawdown(equity))

        if equity.iloc[-1] < 1.0:
            n_loss += 1

    return MonteCarloResult(
        sharpes=sharpes,
        cagrs=cagrs,
        max_dds=max_dds,
        p_loss=n_loss / n_simulations,
    )


def block_bootstrap(
    returns: pd.Series,
    block_size: int = 21,
    n_simulations: int = 1000,
    seed: int = 42,
) -> MonteCarloResult:
    """Stationary block bootstrap (Politis-Romano).

    Resamples daily returns in blocks to preserve short-run autocorrelation.
    Answers: "what is the confidence interval on Sharpe?"
    """
    rng = np.random.RandomState(seed)
    n = len(returns)
    ret_arr = returns.values.copy()

    sharpes, cagrs, max_dds = [], [], []
    n_loss = 0

    for _ in range(n_simulations):
        sampled = np.empty(n)
        pos = 0
        while pos < n:
            start_idx = rng.randint(0, n)
            length = rng.geometric(1.0 / block_size)
            length = min(length, n - pos)

            for j in range(length):
                sampled[pos] = ret_arr[(start_idx + j) % n]
                pos += 1
                if pos >= n:
                    break

        equity = _equity_from_returns(sampled)
        sharpes.append(metrics.sharpe_ratio(pd.Series(sampled)))
        cagrs.append(metrics.cagr(equity))
        max_dds.append(metrics.max_drawdown(equity))

        if equity.iloc[-1] < 1.0:
            n_loss += 1

    return MonteCarloResult(
        sharpes=sharpes,
        cagrs=cagrs,
        max_dds=max_dds,
        p_loss=n_loss / n_simulations,
    )


def skip_trade_jitter(
    returns: pd.Series,
    skip_rate: float = 0.10,
    delay_rate: float = 0.05,
    n_simulations: int = 500,
    seed: int = 42,
) -> MonteCarloResult:
    """Randomly drop or delay signals.

    Answers: "does the edge survive imperfect execution?"
    skip_rate: fraction of days where signal is dropped (return = 0)
    delay_rate: fraction of days where fill is delayed by 1 day
    """
    rng = np.random.RandomState(seed)
    ret_arr = returns.values.copy()
    n = len(ret_arr)

    sharpes, cagrs, max_dds = [], [], []
    n_loss = 0

    for _ in range(n_simulations):
        modified = ret_arr.copy()

        skip_mask = rng.random(n) < skip_rate
        modified[skip_mask] = 0.0

        delay_mask = rng.random(n) < delay_rate
        delayed = np.zeros(n)
        carry = 0.0
        for i in range(n):
            # A delayed day's return moves onto the next day; carrying it
            # forward keeps back-to-back delays from deleting a return
            # outright, which would understate the edge rather than jitter it.
            value = carry
            carry = 0.0
            if delay_mask[i] and i < n - 1:
                carry = modified[i]
            else:
                value += modified[i]
            delayed[i] = value
        modified = delayed

        equity = _equity_from_returns(modified)
        sharpes.append(metrics.sharpe_ratio(pd.Series(modified)))
        cagrs.append(metrics.cagr(equity))
        max_dds.append(metrics.max_drawdown(equity))

        if equity.iloc[-1] < 1.0:
            n_loss += 1

    return MonteCarloResult(
        sharpes=sharpes,
        cagrs=cagrs,
        max_dds=max_dds,
        p_loss=n_loss / n_simulations,
    )

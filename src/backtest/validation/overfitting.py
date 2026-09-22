"""Overfitting detection — Deflated Sharpe and PBO via CSCV."""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from backtest import metrics

logger = logging.getLogger(__name__)


def probability_of_backtest_overfitting(
    return_streams: pd.DataFrame,
    n_splits: int = 16,
) -> float:
    """Probability of Backtest Overfitting via CSCV.

    Bailey, Borwein, Lopez de Prado, Zhu (2016).

    Combinatorially splits the return streams into IS and OOS halves,
    measures how often the IS-best performs below median OOS.
    PBO > 0.5 means selection process has no skill.

    return_streams: each column is one strategy/parameter-set's daily returns.

    Returns NaN when the test cannot run -- fewer than two streams to select
    between, or blocks too short to estimate a Sharpe on. Returning 0.0 there
    would be indistinguishable from the best possible result, so a test that
    never executed would read as a clean pass on the verdict card.
    """
    n_cols = return_streams.shape[1]
    if n_cols < 2:
        return float("nan")

    n_rows = len(return_streams)
    if n_rows < n_splits * 2:
        n_splits = max(2, n_rows // 10)

    block_size = n_rows // n_splits
    if block_size < 5:
        return float("nan")

    blocks = []
    for i in range(n_splits):
        start = i * block_size
        end = start + block_size if i < n_splits - 1 else n_rows
        blocks.append(return_streams.iloc[start:end])

    half = n_splits // 2
    all_combos = list(combinations(range(n_splits), half))

    max_combos = min(len(all_combos), 100)
    rng = np.random.RandomState(42)
    if len(all_combos) > max_combos:
        indices = rng.choice(len(all_combos), max_combos, replace=False)
        selected_combos = [all_combos[i] for i in indices]
    else:
        selected_combos = all_combos

    overfit_count = 0

    for is_indices in selected_combos:
        oos_indices = tuple(i for i in range(n_splits) if i not in is_indices)

        is_data = pd.concat([blocks[i] for i in is_indices])
        oos_data = pd.concat([blocks[i] for i in oos_indices])

        is_sharpes = is_data.apply(metrics.sharpe_ratio)
        oos_sharpes = oos_data.apply(metrics.sharpe_ratio)

        best_is = is_sharpes.idxmax()

        oos_median = oos_sharpes.median()
        if oos_sharpes[best_is] < oos_median:
            overfit_count += 1

    return overfit_count / len(selected_combos)

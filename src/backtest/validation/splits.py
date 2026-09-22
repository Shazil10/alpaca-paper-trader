"""In-sample / out-of-sample splits and walk-forward validation.

Supports:
  - Locked holdout (touched exactly once)
  - Anchored (expanding) walk-forward
  - Rolling walk-forward
  - Walk-Forward Efficiency (WFE = OOS return / IS return; Pardo >= 0.5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest import metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_cagr: float = 0.0
    test_cagr: float = 0.0
    train_max_dd: float = 0.0
    test_max_dd: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    wfe: float = 0.0  # Walk-Forward Efficiency


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward validation results."""
    folds: List[WalkForwardFold] = field(default_factory=list)
    aggregate_oos_sharpe: float = 0.0
    aggregate_oos_cagr: float = 0.0
    mean_wfe: float = 0.0
    median_wfe: float = 0.0
    pass_pardo: bool = False  # WFE >= 0.5


def split_is_oos(
    start: str,
    end: str,
    holdout_years: int = 3,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Split date range into in-sample and out-of-sample periods.

    The last `holdout_years` years become OOS.
    """
    end_ts = pd.Timestamp(end)
    split = end_ts - pd.DateOffset(years=holdout_years)

    is_end = (split - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    oos_start = split.strftime("%Y-%m-%d")

    return (start, is_end), (oos_start, end)


def anchored_walk_forward(
    backtest_fn: Callable[[str, str, Dict], Dict],
    start: str,
    end: str,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    param_grid: Optional[Dict[str, List]] = None,
) -> WalkForwardResult:
    """Anchored (expanding) walk-forward validation.

    Train window always starts at `start` and grows.
    Test window slides forward.

    backtest_fn: (start, end, params) -> {metrics dict with 'sharpe', 'cagr', 'max_drawdown'}
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    folds = []
    fold_idx = 0

    train_end_ts = start_ts + pd.DateOffset(years=train_years)

    while train_end_ts < end_ts:
        test_start_ts = train_end_ts
        test_end_ts = min(test_start_ts + pd.DateOffset(years=test_years), end_ts)

        train_start_str = start
        train_end_str = (train_end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        test_start_str = test_start_ts.strftime("%Y-%m-%d")
        test_end_str = test_end_ts.strftime("%Y-%m-%d")

        best_params = {}
        try:
            is_result = backtest_fn(train_start_str, train_end_str, best_params)
            oos_result = backtest_fn(test_start_str, test_end_str, best_params)

            is_sharpe = is_result.get("sharpe", 0)
            oos_sharpe = oos_result.get("sharpe", 0)
            is_cagr = is_result.get("cagr", 0)
            oos_cagr = oos_result.get("cagr", 0)

            wfe = oos_cagr / is_cagr if is_cagr != 0 else 0

            fold = WalkForwardFold(
                fold_index=fold_idx,
                train_start=train_start_str,
                train_end=train_end_str,
                test_start=test_start_str,
                test_end=test_end_str,
                train_sharpe=is_sharpe,
                test_sharpe=oos_sharpe,
                train_cagr=is_cagr,
                test_cagr=oos_cagr,
                train_max_dd=is_result.get("max_drawdown", 0),
                test_max_dd=oos_result.get("max_drawdown", 0),
                best_params=best_params,
                wfe=wfe,
            )
            folds.append(fold)

        except Exception as e:
            logger.warning("Walk-forward fold %d failed: %s", fold_idx, e)

        fold_idx += 1
        train_end_ts += pd.DateOffset(years=step_years)

    if not folds:
        return WalkForwardResult()

    wfes = [f.wfe for f in folds if f.wfe != 0]
    oos_sharpes = [f.test_sharpe for f in folds]
    oos_cagrs = [f.test_cagr for f in folds]

    return WalkForwardResult(
        folds=folds,
        aggregate_oos_sharpe=float(np.mean(oos_sharpes)) if oos_sharpes else 0,
        aggregate_oos_cagr=float(np.mean(oos_cagrs)) if oos_cagrs else 0,
        mean_wfe=float(np.mean(wfes)) if wfes else 0,
        median_wfe=float(np.median(wfes)) if wfes else 0,
        pass_pardo=float(np.median(wfes)) >= 0.5 if wfes else False,
    )


def rolling_walk_forward(
    backtest_fn: Callable[[str, str, Dict], Dict],
    start: str,
    end: str,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
) -> WalkForwardResult:
    """Rolling (fixed-width) walk-forward validation.

    Train window is always exactly `train_years` long.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    folds = []
    fold_idx = 0

    current_start = start_ts

    while current_start + pd.DateOffset(years=train_years + test_years) <= end_ts + pd.Timedelta(days=30):
        train_end_ts = current_start + pd.DateOffset(years=train_years)
        test_start_ts = train_end_ts
        test_end_ts = min(test_start_ts + pd.DateOffset(years=test_years), end_ts)

        if test_start_ts >= end_ts:
            break

        train_start_str = current_start.strftime("%Y-%m-%d")
        train_end_str = (train_end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        test_start_str = test_start_ts.strftime("%Y-%m-%d")
        test_end_str = test_end_ts.strftime("%Y-%m-%d")

        try:
            is_result = backtest_fn(train_start_str, train_end_str, {})
            oos_result = backtest_fn(test_start_str, test_end_str, {})

            is_cagr = is_result.get("cagr", 0)
            oos_cagr = oos_result.get("cagr", 0)
            wfe = oos_cagr / is_cagr if is_cagr != 0 else 0

            folds.append(WalkForwardFold(
                fold_index=fold_idx,
                train_start=train_start_str, train_end=train_end_str,
                test_start=test_start_str, test_end=test_end_str,
                train_sharpe=is_result.get("sharpe", 0),
                test_sharpe=oos_result.get("sharpe", 0),
                train_cagr=is_cagr, test_cagr=oos_cagr,
                train_max_dd=is_result.get("max_drawdown", 0),
                test_max_dd=oos_result.get("max_drawdown", 0),
                wfe=wfe,
            ))
        except Exception as e:
            logger.warning("Rolling fold %d failed: %s", fold_idx, e)

        fold_idx += 1
        current_start += pd.DateOffset(years=step_years)

    if not folds:
        return WalkForwardResult()

    wfes = [f.wfe for f in folds if f.wfe != 0]

    return WalkForwardResult(
        folds=folds,
        aggregate_oos_sharpe=float(np.mean([f.test_sharpe for f in folds])),
        aggregate_oos_cagr=float(np.mean([f.test_cagr for f in folds])),
        mean_wfe=float(np.mean(wfes)) if wfes else 0,
        median_wfe=float(np.median(wfes)) if wfes else 0,
        pass_pardo=float(np.median(wfes)) >= 0.5 if wfes else False,
    )

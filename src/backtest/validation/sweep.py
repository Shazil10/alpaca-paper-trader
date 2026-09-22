"""Parameter grid sweep execution."""
from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def grid_sweep(
    backtest_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    param_grid: Dict[str, List],
) -> pd.DataFrame:
    """Run backtest_fn over all parameter combinations.

    backtest_fn: (params) -> metrics dict
    param_grid: {param_name: [values]}

    Returns DataFrame with one row per combo, params + metrics columns.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    logger.info("Running grid sweep: %d combinations", len(combos))

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        try:
            result = backtest_fn(params)
            row = dict(params)
            row.update(result)
            row["_status"] = "ok"
        except Exception as e:
            row = dict(params)
            row["_status"] = f"error: {e}"

        results.append(row)

        if (i + 1) % 50 == 0:
            logger.info("Sweep progress: %d/%d", i + 1, len(combos))

    return pd.DataFrame(results)

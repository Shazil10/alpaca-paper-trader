"""Parameter stability analysis — heatmaps and plateau scoring."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def stability_heatmap(
    sweep_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Pivot sweep results into a heatmap (param_x vs param_y).

    Returns a DataFrame suitable for seaborn.heatmap.
    """
    if param_x not in sweep_df.columns or param_y not in sweep_df.columns:
        raise ValueError(f"Parameters {param_x}, {param_y} not in sweep results")
    if metric not in sweep_df.columns:
        raise ValueError(f"Metric {metric} not in sweep results")

    return sweep_df.pivot_table(
        index=param_y, columns=param_x, values=metric, aggfunc="mean"
    ).sort_index(ascending=False)


def plateau_score(
    heatmap: pd.DataFrame,
    neighborhood: int = 1,
    k: float = 1.0,
) -> pd.DataFrame:
    """Score each cell by its neighbourhood stability.

    plateau_score = mean(NxN neighbourhood) - k * std(NxN neighbourhood)

    Higher = more stable. A peak surrounded by low values scores poorly.
    """
    values = heatmap.values.astype(float)
    scores = np.full_like(values, np.nan)

    rows, cols = values.shape

    for i in range(rows):
        for j in range(cols):
            r_start = max(0, i - neighborhood)
            r_end = min(rows, i + neighborhood + 1)
            c_start = max(0, j - neighborhood)
            c_end = min(cols, j + neighborhood + 1)

            nbr = values[r_start:r_end, c_start:c_end]
            valid = nbr[~np.isnan(nbr)]

            if len(valid) >= 2:
                scores[i, j] = float(np.mean(valid) - k * np.std(valid))
            elif len(valid) == 1:
                scores[i, j] = float(valid[0])

    return pd.DataFrame(scores, index=heatmap.index, columns=heatmap.columns)


def worst_step_decay(
    heatmap: pd.DataFrame,
    row_idx: int,
    col_idx: int,
) -> float:
    """How much the metric drops moving one grid step in the worst direction."""
    values = heatmap.values.astype(float)
    center = values[row_idx, col_idx]

    if np.isnan(center):
        return float("nan")

    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = row_idx + di, col_idx + dj
            if 0 <= ni < values.shape[0] and 0 <= nj < values.shape[1]:
                if not np.isnan(values[ni, nj]):
                    neighbors.append(values[ni, nj])

    if not neighbors:
        return 0.0

    return center - min(neighbors)

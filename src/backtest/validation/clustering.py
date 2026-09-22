"""Return-stream correlation clustering and SPP analysis."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


def spp_percentile(
    sweep_metrics: pd.DataFrame,
    metric: str = "sharpe",
    percentile: float = 25,
) -> float:
    """Strategy Parameter Permutation expected OOS performance.

    Instead of reporting the best cell, use a lower percentile of the
    distribution across all parameter sets as the expected OOS performance.
    """
    values = sweep_metrics[metric].dropna()
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def return_stream_clustering(
    return_streams: pd.DataFrame,
    n_clusters: int = 5,
    method: str = "ward",
) -> Dict[str, Any]:
    """Cluster parameter sets by the correlation of their return streams.

    return_streams: DataFrame where each column is a parameter set's daily returns.

    Returns:
        {
            'labels': array of cluster labels,
            'linkage_matrix': for dendrogram plotting,
            'correlation_matrix': pairwise correlations,
            'cluster_sizes': {cluster_id: count},
        }
    """
    if return_streams.shape[1] < 3:
        return {"labels": np.array([0] * return_streams.shape[1])}

    corr = return_streams.corr()

    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 0)

    dist = (dist + dist.T) / 2

    try:
        condensed = squareform(dist)
        Z = linkage(condensed, method=method)
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    except Exception as e:
        logger.warning("Clustering failed: %s", e)
        return {"labels": np.zeros(return_streams.shape[1])}

    cluster_sizes = {}
    for label in np.unique(labels):
        cluster_sizes[int(label)] = int((labels == label).sum())

    return {
        "labels": labels,
        "linkage_matrix": Z,
        "correlation_matrix": corr,
        "cluster_sizes": cluster_sizes,
        "column_names": list(return_streams.columns),
    }


def best_in_cluster(
    sweep_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    metric: str = "sharpe",
) -> Dict[str, Any]:
    """Identify which cluster the best parameter set belongs to."""
    labels = np.asarray(cluster_labels)
    if len(sweep_df) != len(labels):
        return {}

    values = sweep_df[metric].to_numpy(dtype=float)
    if np.all(np.isnan(values)):
        return {}

    # Positional, not label-based: cluster_labels is aligned to row order, so
    # idxmax() on a non-RangeIndex sweep would index the wrong cluster (or
    # raise). Row order is the only thing the two share.
    best_pos = int(np.nanargmax(values))
    best_cluster = int(labels[best_pos])
    cluster_size = int((labels == best_cluster).sum())
    total = len(labels)

    return {
        "best_idx": best_pos,
        "best_label": sweep_df.index[best_pos],
        "best_cluster": best_cluster,
        "cluster_size": cluster_size,
        "cluster_pct": cluster_size / total,
        "is_core": cluster_size / total > 0.1,  # >10% of all param sets
    }

"""As-of-date price adjustment.

adj_close embeds every split and dividend up to the download date, so historical
price *levels* leak future corporate actions. Returns are unaffected, but any
comparison to an absolute dollar threshold (e.g. MIN_PRICE=10, share rounding)
will be wrong in a backtest.

This module re-anchors the adjustment to a specific as-of date so that price
levels are historically honest while returns remain correct.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Sequence, Union
from data_pipeline import store, schema

DateLike = Union[str, pd.Timestamp, None]


def adjustment_factor(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Compute per-symbol scaling factor that re-anchors adj_close to as_of.

    prices: long-format lake frame with date, symbol, close, adj_close columns.
    as_of: the date to anchor adjustments to (inclusive).

    Returns a DataFrame with columns [date, symbol, factor] where
    factor = (adj_close_t / close_t) / (adj_close_T / close_T)
    and T is the latest date <= as_of for each symbol.
    """
    df = prices.copy()
    df = df[df[schema.DATE] <= as_of]

    close = df[schema.CLOSE].astype(float)
    adj = df[schema.ADJ_CLOSE].astype(float)
    raw_factor = (adj / close.where(close > 0)).fillna(1.0)
    df["_raw_factor"] = raw_factor

    anchor = (
        df.sort_values(schema.DATE)
        .groupby(schema.SYMBOL)
        .last()["_raw_factor"]
        .rename("_anchor_factor")
    )

    df = df.merge(anchor, left_on=schema.SYMBOL, right_index=True, how="left")
    df["factor"] = df["_raw_factor"] / df["_anchor_factor"]

    return df[[schema.DATE, schema.SYMBOL, "factor"]].reset_index(drop=True)


def load_prices_asof(
    symbols: Optional[Sequence[str]] = None,
    start: DateLike = None,
    end: DateLike = None,
    as_of: Optional[pd.Timestamp] = None,
    *,
    root=None,
) -> pd.DataFrame:
    """Load lake prices with adj_close re-anchored to as_of.

    If as_of is None, returns prices as-is (today's adjustment, standard behaviour).
    If as_of is set, adj_close and all OHLC are scaled so that price levels
    reflect what was observable on as_of, not today.

    The returned frame has the same schema as store.load_prices().
    """
    effective_end = as_of if as_of is not None else end
    prices = store.load_prices(symbols, start, effective_end, root=root)

    if as_of is None or len(prices) == 0:
        if end is not None and as_of is not None:
            end_ts = pd.Timestamp(end)
            prices = prices[prices[schema.DATE] <= end_ts]
        return prices

    factors = adjustment_factor(prices, pd.Timestamp(as_of))
    prices = prices.merge(
        factors, on=[schema.DATE, schema.SYMBOL], how="left"
    )
    prices["factor"] = prices["factor"].fillna(1.0)

    for col in [schema.OPEN, schema.HIGH, schema.LOW, schema.CLOSE]:
        prices[col] = prices[col] * prices["factor"]
    prices[schema.ADJ_CLOSE] = prices[schema.CLOSE]

    prices = prices.drop(columns=["factor"])

    if end is not None:
        end_ts = pd.Timestamp(end)
        prices = prices[prices[schema.DATE] <= end_ts]

    return prices


def load_close_matrix_asof(
    symbols: Optional[Sequence[str]] = None,
    start: DateLike = None,
    end: DateLike = None,
    as_of: Optional[pd.Timestamp] = None,
    *,
    root=None,
) -> pd.DataFrame:
    """Like store.load_close_matrix but with as-of-date adjustment.

    Returns date x symbol matrix of adjusted close prices, re-anchored to as_of.
    """
    prices = load_prices_asof(symbols, start, end, as_of, root=root)
    if len(prices) == 0:
        return pd.DataFrame(dtype="float64")

    matrix = prices.pivot_table(
        index=schema.DATE, columns=schema.SYMBOL,
        values=schema.ADJ_CLOSE, aggfunc="last"
    )
    matrix.index.name = schema.DATE
    matrix.columns.name = None

    wanted = store._normalize_symbols(symbols)
    if wanted is not None:
        ordered = [s for s in wanted if s in matrix.columns]
        matrix = matrix[ordered]

    matrix = matrix.dropna(how="all")
    return matrix.astype("float64").sort_index()

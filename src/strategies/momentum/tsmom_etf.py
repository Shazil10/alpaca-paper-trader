"""Time-series momentum across a cross-asset ETF book.

Moskowitz-Ooi-Pedersen time-series momentum: hold each asset long or short on the
sign of its own trailing 12-month return, and size every position so that each
contributes the same volatility. Nothing here is cross-sectional -- an asset
competes only against its own past, so the book can be long everything or short
everything.

This module was extracted from ``analysis/strategies/momentum/tsmom_etf.ipynb``,
which held the only copy of the logic. The notebook now imports from here, so the
vectorized research path and the event-driven backtest run the same arithmetic
instead of two implementations that agree until one is edited.

Two shapes are exposed deliberately:

* :func:`run_tsmom` is the notebook's vectorized backtest -- weights as a matrix,
  returns as a dot product. Fast, frictionless about cash and share counts, and
  the right tool for a thousand-run parameter sweep.
* :func:`target_weights` is the engine interface. Slower, but it pays spread,
  rounds to whole shares, respects a cash balance and cannot see past ``as_of``.

``tests/backtest/test_tsmom_parity.py`` asserts the two agree once the frictions
are switched off. Divergence means one of them is wrong.

Leverage is not incidental. Sizing to a 40% volatility target means an asset
running at 10% realized vol asks for 4x notional, and gross exposure across the
book routinely lands between 2x and 3x. That is the strategy as researched, so
configs must set ``max_leverage`` to match or the risk layer will quietly
backtest something else. There is no financing model, so levered stretches are
overstated by roughly the call rate on the borrowed portion.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The research universe: equity regions, duration, credit, commodities, real
#: estate, sectors and the dollar. Breadth across *asset classes* is the point --
#: time-series momentum earns its diversification from assets that trend at
#: different times, not from many names inside one market.
TICKERS: Tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ",
    "TLT", "IEF", "LQD", "HYG", "TIP",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBB",
    "VNQ", "XLE", "XLF", "XLK", "XLU", "UUP",
)

TRADING_DAYS = 252

#: Trailing months of return whose sign decides long or short.
LOOKBACK_MONTHS = 12

#: EWMA halflife in days for the volatility estimate. Long enough to be stable,
#: short enough to react to a regime change within a quarter.
VOL_HALFLIFE = 60

#: Annualized volatility each position is scaled to contribute.
SIGMA_TARGET = 0.40

#: One-way cost in basis points applied to turnover in the vectorized path. The
#: engine models spread, slippage and commission separately and ignores this.
COST_BPS = 10.0

#: Sessions of history before the first signal can be trusted: the lookback in
#: months plus the vol halflife, with slack.
MIN_SESSIONS = LOOKBACK_MONTHS * 21 + VOL_HALFLIFE + 10


# ---------------------------------------------------------------------------
# Decision functions -- shared by the vectorized and event-driven paths
# ---------------------------------------------------------------------------

def month_end_sessions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last *trading* session of each month present in ``index``.

    Not calendar month-ends. The 31st is frequently a weekend or a holiday, and
    reindexing onto a date the market never opened silently forward-fills a stale
    price into the signal.
    """
    if len(index) == 0:
        return pd.DatetimeIndex([])
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.groupby(index.to_period("M")).max().values)


def ewma_annualized_vol(
    returns: pd.DataFrame,
    halflife: int = VOL_HALFLIFE,
) -> pd.DataFrame:
    """Annualized EWMA volatility per asset.

    ``min_periods=halflife`` is what keeps a newly listed ETF out of the book: a
    vol estimate from four observations is small and noisy, and dividing the
    target by it would hand that asset an enormous weight.
    """
    return returns.ewm(halflife=halflife, min_periods=int(halflife)).std() * np.sqrt(
        TRADING_DAYS
    )


def momentum_sign(
    monthly_closes: pd.DataFrame,
    lookback: int = LOOKBACK_MONTHS,
) -> pd.DataFrame:
    """Sign of each asset's trailing ``lookback``-month return. NaN where unknown.

    ``np.sign`` of exactly zero is zero, which reads as "flat" rather than as a
    direction -- correct, and rare enough not to matter.
    """
    return np.sign(monthly_closes / monthly_closes.shift(lookback) - 1.0)


def vol_target_weights(
    sign: pd.DataFrame,
    vol: pd.DataFrame,
    sigma_target: float = SIGMA_TARGET,
) -> pd.DataFrame:
    """Scale each signal to the volatility target and split across live assets.

    ``sigma_target / vol`` is the notional that makes an asset contribute
    ``sigma_target`` of annualized volatility. Dividing by the number of assets
    with a live signal keeps the book comparable as coverage grows: without it,
    adding ETFs would mechanically lever the strategy up.

    A missing signal or a missing vol becomes a zero weight, not a dropped
    column, so the weight matrix stays rectangular for the dot product.
    """
    raw = (sign * sigma_target / vol).where(sign.notna() & vol.notna(), 0.0)
    live = sign.notna().sum(axis=1).clip(lower=1)
    return raw.div(live, axis=0)


def month_end_weights(
    monthly_closes: pd.DataFrame,
    vol: pd.DataFrame,
    lookback: int = LOOKBACK_MONTHS,
    sigma_target: float = SIGMA_TARGET,
    *,
    long_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(weights, sign)`` indexed on month-end sessions.

    ``vol`` is expected on the same month-end index as ``monthly_closes``.
    """
    sign = momentum_sign(monthly_closes, lookback)
    weights = vol_target_weights(sign, vol, sigma_target)
    if long_only:
        weights = weights.clip(lower=0.0)
    return weights, sign


def daily_streams(
    month_end_w: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cost_bps: float = COST_BPS,
    start: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Month-end weights to a daily net return and turnover series.

    ``.shift(1)`` is the entire execution assumption of the vectorized path: a
    weight decided on the month-end close is earned from the *next* session
    onward. Dropping the shift books the rebalancing day's move as profit the
    signal could not have captured, which is the most common way a vectorized
    backtest flatters itself.
    """
    held = month_end_w.reindex(returns.index).ffill().shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - (cost_bps / 1e4) * turnover

    if start is not None:
        return net.loc[start:], turnover.loc[start:]
    return net, turnover


def run_tsmom(
    closes: pd.DataFrame,
    lookback: int = LOOKBACK_MONTHS,
    halflife: int = VOL_HALFLIFE,
    *,
    sigma_target: float = SIGMA_TARGET,
    cost_bps: float = COST_BPS,
    start: Optional[str] = None,
    vol: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """The vectorized backtest. Returns long-short and long-only streams.

    ``vol`` accepts a precomputed *daily* volatility frame. A parameter sweep over
    (lookback, halflife) pairs recomputes the same EWMA once per lookback
    otherwise, and the EWMA is the expensive part; the sweep in the notebook
    memoizes it per halflife and passes it in.
    """
    closes = closes.sort_index()
    returns = closes.pct_change()

    me = month_end_sessions(closes.index)
    monthly = closes.loc[me]
    daily_vol = ewma_annualized_vol(returns, halflife) if vol is None else vol
    vol_me = daily_vol.loc[me]

    weights, sign = month_end_weights(monthly, vol_me, lookback, sigma_target)

    ls_net, ls_turn = daily_streams(
        weights, returns, cost_bps=cost_bps, start=start
    )
    lo_net, lo_turn = daily_streams(
        weights.clip(lower=0.0), returns, cost_bps=cost_bps, start=start
    )

    return {
        "ls": ls_net, "lo": lo_net,
        "ls_turn": ls_turn, "lo_turn": lo_turn,
        "weights": weights, "sign": sign,
    }


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------

def is_rebalance_day(ctx) -> bool:
    """True on the last trading session of a month.

    The vectorized path applies month-end weights on the following session via
    ``.shift(1)``. The engine fills a decision made on day D at D+1's open, so
    deciding on the month-end session reproduces that shift exactly rather than
    approximating it.
    """
    sessions = ctx.sessions()
    if sessions is None or len(sessions) == 0:
        return True

    as_of = pd.Timestamp(ctx.as_of)
    month = sessions[(sessions.year == as_of.year) & (sessions.month == as_of.month)]
    if len(month) == 0:
        return True

    # as_of is the newest session the context will return, so it is the month's
    # last session so far. Whether a later one exists is not knowable here --
    # that is the point of the firewall -- so "latest in this month" is the only
    # available reading, and on the true month-end it is the correct one.
    return pd.Timestamp(month[-1]) == as_of and _is_last_session_of_month(ctx, as_of)


def _is_last_session_of_month(ctx, as_of: pd.Timestamp) -> bool:
    """Whether ``as_of`` is the final session of its month.

    Decided from the calendar rather than from future prices: the next session
    exists in the trading calendar the engine was handed, and asking whether it
    falls in a later month is not look-ahead into *prices*. Without this, every
    session would look like a month-end and the sleeve would rebalance daily.
    """
    sessions = ctx._trading_sessions
    if sessions is None or len(sessions) == 0:
        return True

    later = sessions[sessions > as_of]
    if len(later) == 0:
        return True
    return pd.Timestamp(later[0]).month != as_of.month


def tradable(ctx, tickers: Sequence[str] = TICKERS) -> List[str]:
    """Universe members the context can actually price with enough history.

    The lake does not carry every ETF in ``TICKERS``, and a symbol it cannot
    price is absent rather than an error. Reporting which ones survived is how a
    run says out loud that it traded a smaller book than the research did.
    """
    matrix = ctx.prices(list(tickers))
    if matrix is None or matrix.empty:
        return []
    return [
        str(c) for c in matrix.columns
        if int(matrix[c].notna().sum()) >= MIN_SESSIONS
    ]


def decide(ctx, **params) -> Optional[dict]:
    """Compute this month-end's target weights and the inputs behind them."""
    lookback = int(params.get("lookback_months", LOOKBACK_MONTHS))
    halflife = int(params.get("vol_halflife", VOL_HALFLIFE))
    sigma_target = float(params.get("sigma_target", SIGMA_TARGET))
    long_only = bool(params.get("long_only", False))

    symbols = tradable(ctx)
    if not symbols:
        logger.warning("TSMOM/ctx %s: nothing tradable", ctx.as_of.date())
        return None

    closes = ctx.prices(symbols)
    if closes is None or closes.empty:
        return None

    returns = closes.pct_change()
    me = month_end_sessions(closes.index)
    if len(me) <= lookback:
        logger.info(
            "TSMOM/ctx %s: %d month-end(s), need more than %d",
            ctx.as_of.date(), len(me), lookback,
        )
        return None

    monthly = closes.loc[me]
    vol_me = ewma_annualized_vol(returns, halflife).loc[me]

    weights, sign = month_end_weights(
        monthly, vol_me, lookback, sigma_target, long_only=long_only
    )

    latest = weights.iloc[-1]
    latest_sign = sign.iloc[-1]

    return {
        "as_of": str(pd.Timestamp(ctx.as_of).date()),
        "tradable": symbols,
        "n_month_ends": int(len(me)),
        "long_only": long_only,
        "signs": {
            str(k): float(v) for k, v in latest_sign.items() if pd.notna(v)
        },
        "vol": {
            str(k): float(v) for k, v in vol_me.iloc[-1].items() if pd.notna(v)
        },
        "weights": {
            str(k): float(v) for k, v in latest.items() if abs(float(v)) > 1e-9
        },
    }


def target_weights(ctx) -> Dict[str, float]:
    """TSMOM as target weights. {symbol: fraction of sleeve equity}.

    Negative weights are shorts. The engine only honours them once short support
    is enabled; until then set ``long_only: true`` in params, which is what the
    notebook's long-only stream measures.
    """
    params = ctx.params or {}

    if not is_rebalance_day(ctx):
        return {sym: float(w) for sym, w in ctx.portfolio.weights.items() if w != 0}

    decision = decide(ctx, **params)
    if decision is None:
        return {sym: float(w) for sym, w in ctx.portfolio.weights.items() if w != 0}

    weights = decision["weights"]
    logger.info(
        "TSMOM/ctx %s: %d tradable, %d long, %d short, gross %.2f",
        decision["as_of"], len(decision["tradable"]),
        sum(1 for w in weights.values() if w > 0),
        sum(1 for w in weights.values() if w < 0),
        sum(abs(w) for w in weights.values()),
    )
    return weights

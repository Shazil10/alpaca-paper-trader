"""Triple-trigger momentum: short-term return, volume confirmation, range position.

A cross-sectional screen that requires three independent conditions at once and
ranks on their product:

1. **Momentum** -- a positive 14-day return.
2. **Volume confirmation** -- today's volume above its own 20-day average.
3. **Range position** -- price in the upper half of its 325-day high-low range.

The product is the score, and all three gates must pass, so a name with huge
momentum on thin volume scores nothing at all rather than scoring slightly less.
That conjunction is the strategy: the volume term asks whether anyone else
noticed, and the range term asks whether the move is a recovery from a low or a
genuine advance.

Extracted from ``analysis/strategies/momentum/triple_trigger_momentum.ipynb``,
which held the only copy. The notebook now imports from here.

Volume is the reason this sleeve could not simply reuse an existing adapter: it is
the only strategy in the repository that reads a field other than price.
``ctx.price_panel`` carries it, and ``data/prices/_schema.md`` Contract 4 matters
here -- volume is nullable and is never zero-filled, because zero volume asserts
"no shares traded", which is a different and false claim. A missing volume
therefore produces a missing ratio and the name is screened out, which is the
correct outcome: without volume the second trigger cannot be evaluated.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Trading days of return for trigger 1.
MOM_LOOKBACK = 14

#: Trading days in the volume average for trigger 2.
VOL_LOOKBACK = 20

#: Trading days in the high-low range for trigger 3. Long by design: a 325-day
#: range spans more than a year, so "upper half" means upper half of a full cycle
#: rather than of a recent swing.
RANGE_LOOKBACK = 325

#: Names held at a time.
TOP_N = 15

#: Trading days between rebalances. Weekly, so the 14-day momentum signal has
#: time to mean something between decisions.
REBALANCE_DAYS = 5

#: Trigger thresholds, named rather than inlined so a sweep can reach them.
MIN_MOMENTUM = 0.0
MIN_VOLUME_RATIO = 1.0
MIN_RANGE_POSITION = 0.5

#: Sessions needed before any score is valid.
MIN_SESSIONS = RANGE_LOOKBACK + VOL_LOOKBACK


# ---------------------------------------------------------------------------
# Decision functions
# ---------------------------------------------------------------------------

def compute_scores(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    mom_lookback: int = MOM_LOOKBACK,
    vol_lookback: int = VOL_LOOKBACK,
    range_lookback: int = RANGE_LOOKBACK,
    min_momentum: float = MIN_MOMENTUM,
    min_volume_ratio: float = MIN_VOLUME_RATIO,
    min_range_position: float = MIN_RANGE_POSITION,
) -> pd.DataFrame:
    """Daily score matrix. NaN wherever any of the three triggers fails.

    ``min_periods`` equals the window on both rolling statistics, so a partial
    window yields NaN rather than an average of whatever happens to be there --
    a 20-day volume average computed from three days would make almost any
    volume look above average.
    """
    momentum = closes.pct_change(mom_lookback)

    volume_average = volumes.rolling(vol_lookback, min_periods=vol_lookback).mean()
    volume_ratio = volumes / volume_average.replace(0, np.nan)

    rolling_high = closes.rolling(range_lookback, min_periods=range_lookback).max()
    rolling_low = closes.rolling(range_lookback, min_periods=range_lookback).min()
    span = (rolling_high - rolling_low).replace(0, np.nan)
    range_position = (closes - rolling_low) / span

    score = momentum * volume_ratio * range_position

    passes = (
        (momentum > min_momentum)
        & (volume_ratio > min_volume_ratio)
        & (range_position > min_range_position)
    )
    return score.where(passes)


def rank_picks(
    scores_on_date: pd.Series,
    top_n: int = TOP_N,
) -> List[str]:
    """The top ``top_n`` symbols by score on one date."""
    live = scores_on_date.dropna().sort_values(ascending=False)
    return [str(s) for s in live.head(top_n).index]


def equal_weights(symbols: Sequence[str]) -> Dict[str, float]:
    """Equal weight across the picks. Empty means all cash.

    Equal weight, not score weight: the score is a product of three terms on
    different scales, so its *magnitude* is not a risk-adjusted conviction and
    sizing by it would lever whichever term happened to spike.
    """
    if not symbols:
        return {}
    weight = 1.0 / len(symbols)
    return {str(s): weight for s in symbols}


def run_backtest_vectorized(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    top_n: int = TOP_N,
    rebalance_days: int = REBALANCE_DAYS,
    cost_bps: float = 10.0,
    **score_params,
) -> pd.Series:
    """The notebook's screen as a daily return stream.

    Deliberately frictionless about cash and share counts -- this is the fast
    path for sweeps, and ``target_weights`` through the engine is the honest one.
    Weights are applied from the session *after* the score, matching the engine's
    D+1 fill.
    """
    scores = compute_scores(closes, volumes, **score_params)
    returns = closes.pct_change()

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    current: Dict[str, float] = {}

    for i, date in enumerate(scores.index):
        if i < MIN_SESSIONS:
            continue
        if (i - MIN_SESSIONS) % rebalance_days == 0:
            current = equal_weights(rank_picks(scores.loc[date], top_n))
        for symbol, weight in current.items():
            if symbol in weights.columns:
                weights.loc[date, symbol] = weight

    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    return gross - (cost_bps / 1e4) * turnover


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------

def volumes_from_ctx(ctx, symbols: Sequence[str]) -> pd.DataFrame:
    """Date x symbol volume matrix, truncated at as_of.

    Pivoted out of ``ctx.price_panel`` because ``ctx.prices`` only serves
    adjusted closes. Volume is scaled by the inverse of the price adjustment
    factor for the same reason prices are scaled by it: after a 2-for-1 split the
    vendor restates pre-split volume, and comparing today's raw volume against an
    unrestated average would read as a doubling in participation.
    """
    panel = ctx.price_panel(symbols)
    if panel is None or len(panel) == 0:
        return pd.DataFrame(dtype="float64")

    frame = panel.copy()
    close = frame["close"].astype("float64")
    adj = frame["adj_close"].astype("float64")
    factor = (adj / close.where(close > 0)).fillna(1.0)
    frame["scaled_volume"] = frame["volume"].astype("float64") / factor.where(
        factor > 0, 1.0
    )

    matrix = frame.pivot_table(
        index="date", columns="symbol", values="scaled_volume", aggfunc="last"
    )
    matrix.columns.name = None
    return matrix.sort_index()


def is_rebalance_day(ctx, rebalance_days: int = REBALANCE_DAYS) -> bool:
    """Every ``rebalance_days``-th session, counted from the first tradable one.

    Anchored on the session index rather than on the weekday: a holiday week
    would otherwise shift the cadence and, worse, make it depend on which
    holidays fall inside the backtest window.
    """
    sessions = ctx.sessions()
    if sessions is None or len(sessions) == 0:
        return True
    if not ctx.held_symbols:
        return True
    return (len(sessions) - 1) % max(rebalance_days, 1) == 0


def decide(ctx, **params) -> Optional[dict]:
    """Score the universe and return the picks with the triggers behind them."""
    top_n = int(params.get("top_n", TOP_N))
    score_params = {
        key: params[key]
        for key in (
            "mom_lookback", "vol_lookback", "range_lookback",
            "min_momentum", "min_volume_ratio", "min_range_position",
        )
        if key in params
    }

    universe = sorted(ctx.universe() or set())
    if not universe:
        return None

    closes = ctx.prices(universe)
    if closes is None or closes.empty or len(closes) < MIN_SESSIONS:
        logger.info(
            "TripleTrigger/ctx %s: %d session(s), need %d",
            ctx.as_of.date(), 0 if closes is None else len(closes), MIN_SESSIONS,
        )
        return None

    volumes = volumes_from_ctx(ctx, list(closes.columns))
    if volumes.empty:
        logger.warning("TripleTrigger/ctx %s: no volume data", ctx.as_of.date())
        return None

    # Align, because a symbol can carry a close and no volume for a session.
    volumes = volumes.reindex(index=closes.index, columns=closes.columns)

    scores = compute_scores(closes, volumes, **score_params)
    latest = scores.iloc[-1]
    picks = rank_picks(latest, top_n)

    return {
        "as_of": str(pd.Timestamp(ctx.as_of).date()),
        "universe": len(universe),
        "passing": int(latest.notna().sum()),
        "picks": picks,
        "scores": {
            str(s): float(latest[s]) for s in picks if pd.notna(latest.get(s))
        },
        "weights": equal_weights(picks),
    }


def target_weights(ctx) -> Dict[str, float]:
    """Triple-trigger momentum as target weights. {symbol: fraction of equity}."""
    params = ctx.params or {}
    rebalance_days = int(params.get("rebalance_days", REBALANCE_DAYS))

    if not is_rebalance_day(ctx, rebalance_days):
        return {sym: float(w) for sym, w in ctx.portfolio.weights.items() if w > 0}

    decision = decide(ctx, **params)
    if decision is None:
        return {sym: float(w) for sym, w in ctx.portfolio.weights.items() if w > 0}

    logger.info(
        "TripleTrigger/ctx %s: %d of %d passed all three triggers, holding %d",
        decision["as_of"], decision["passing"], decision["universe"],
        len(decision["picks"]),
    )
    return decision["weights"]

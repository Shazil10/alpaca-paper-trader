"""Ranked Sector Allocation (V10 DAF 2x) on the backtest ``target_weights`` interface.

This is the *same* strategy as :mod:`strategies.ranks.ranked_asset_alloc`. Every
decision function is imported from there rather than copied -- ``_v4_today``,
``_v8aw_today``, ``_blend``, ``_daf_leverage``, ``_regime``. A copy would drift,
and the whole point of a parity gate is that there is one implementation of the
logic and two ways of feeding it.

The only thing that changes is where prices come from. The live sleeve calls
``_load_data()``, which reads the lake (or yfinance) directly and implicitly
means "now". Here the panel is rebuilt from :class:`backtest.context.StrategyContext`,
which is truncated at ``ctx.as_of`` and raises ``LookaheadError`` on any request
past it. The panel is assembled to be byte-for-byte the shape ``_lake_data()``
returns: the same 730-day window, the same ``ALL_TICKERS`` column order, the
same ``dropna(how="all")`` row filter, the same adjusted OHLC dict.

Two conversions are worth stating out loud:

**Dollars to weights.** The live sleeve emits ``notional = BASE_BUDGET * lev * w``.
The sleeve's nominal capital is ``BASE_BUDGET`` at 1x, so the equivalent fraction
of sleeve equity is ``w * lev``, not ``w``. Gross exposure therefore sums to
``lev`` and **exceeds 1.0 when DAF levers up** -- that is the strategy, not a bug.
Dividing by ``BASE_BUDGET * lev`` instead would normalise the leverage away and
silently backtest a different, unleveraged strategy.

**Rebalance cadence.** The live sleeve emits signals only on a rebalance day and
an empty signal list otherwise, which means "hold". ``target_weights`` has no
empty-means-hold escape hatch -- an empty dict means "sell everything". So on a
non-rebalance day this returns the *current* portfolio weights, which the engine
diffs to zero orders. Same outcome, different encoding.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from strategies.ranks import ranked_asset_alloc as ra

logger = logging.getLogger(__name__)

#: Re-exported so callers (configs, tests, the fund simulator) can reach the
#: universe without importing both modules.
ALL_TICKERS = ra.ALL_TICKERS
SECTOR_ETFS = ra.SECTOR_ETFS
CASH_ETF = ra.CASH_ETF
BASE_BUDGET = ra.BASE_BUDGET


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def panel_from_ctx(ctx) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Rebuild ``_lake_data()``'s ``(closes, ohlc_dict)`` out of a context.

    Three details carry the parity, and each one silently changes decisions if
    dropped:

    * The window is ``ra._window(as_of)`` -- 730 calendar days. ``_daf_leverage``
      uses ``expanding()``, so a longer window moves the 35th-percentile vol
      threshold and flips the 2x leverage decision on different days.
    * ``dropna(how="all")`` reproduces ``store.load_close_matrix``. Without it a
      backtest-wide panel contributes sessions on which none of these 16 ETFs
      traded, and every positional lookback (``iloc[idx - 84]``) shifts by a row.
    * Column order follows ``ALL_TICKERS``, matching both data paths.
    """
    start, end = ra._window(ctx.as_of)

    closes = ctx.prices(ra.ALL_TICKERS, start=start, end=end)
    closes = closes[[t for t in ra.ALL_TICKERS if t in closes.columns]]
    closes = closes.dropna(how="all")

    ohlc_dict: Dict[str, pd.DataFrame] = {}
    for ticker in ra.SECTOR_ETFS:
        frame = ctx.ohlc(ticker)
        if frame is None or len(frame) == 0:
            continue
        frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
        if len(frame) > 0:
            ohlc_dict[ticker] = frame

    return closes, ohlc_dict


# ---------------------------------------------------------------------------
# Rebalance cadence
# ---------------------------------------------------------------------------

def is_rebalance_day(ctx) -> bool:
    """True on the first trading session of a month, or when holding nothing.

    The live sleeve approximates this with ``today.day <= 3``, because it is
    woken by a daily cron and cannot see the calendar. A backtest can, so it
    uses the exact first session -- the behaviour the docstring always claimed.
    """
    if not ctx.held_symbols:
        return True

    as_of = pd.Timestamp(ctx.as_of)
    sessions = ctx.sessions()
    if sessions is None or len(sessions) == 0:
        return True

    month = sessions[(sessions.year == as_of.year) & (sessions.month == as_of.month)]
    return len(month) > 0 and pd.Timestamp(month[0]) == as_of


def _hold(ctx) -> Dict[str, float]:
    """Current weights: the engine diffs these to zero orders."""
    return {
        sym: float(w)
        for sym, w in ctx.portfolio.weights.items()
        if w > 0
    }


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def decide(ctx) -> Optional[dict]:
    """Run the sleeve's decision functions against the context panel.

    Returns the same decision record the canary fixture records, or None when
    the sleeve declines to act. Exposed separately from :func:`target_weights`
    so parity tests can compare regimes and the DAF factor, not just the final
    weights.
    """
    closes, ohlc_dict = panel_from_ctx(ctx)

    if closes.empty:
        logger.warning("Ranked/ctx %s: empty panel", ctx.as_of.date())
        return None

    # Below this, _regime degrades to neutral and _daf_leverage to 1x without
    # raising. Acting on that would rotate the book on a technicality.
    if len(closes) < ra.MIN_SESSIONS:
        logger.info(
            "Ranked/ctx %s: %d session(s), need %d -- skipping",
            ctx.as_of.date(), len(closes), ra.MIN_SESSIONS,
        )
        return None

    # Presence of a column is not presence of data. Once the lake reaches back
    # past an ETF's inception -- XLC listed 2018-06-19, XLRE 2015-10-08 -- the
    # column exists on every earlier session as all-NaN, so a membership test
    # alone passes and the ranking then sorts NaNs. Require real history.
    missing = [
        t for t in ra.SECTOR_ETFS
        if t not in closes.columns or int(closes[t].notna().sum()) < ra.MIN_SESSIONS
    ]
    if missing:
        logger.info(
            "Ranked/ctx %s: sector ETF(s) without %d sessions of history: %s "
            "-- skipping",
            ctx.as_of.date(), ra.MIN_SESSIONS, ", ".join(missing),
        )
        return None

    sector_closes = closes[ra.SECTOR_ETFS]
    idx = len(closes) - 1

    alloc_v4 = ra._v4_today(sector_closes, ohlc_dict, closes, idx)
    alloc_v8 = ra._v8aw_today(sector_closes, ohlc_dict, closes, idx)
    target = ra._blend(alloc_v4, alloc_v8)
    lev = float(ra._daf_leverage(closes))

    return {
        "sessions": int(len(closes)),
        "last_session": str(pd.Timestamp(closes.index[-1]).date()),
        "regime_v4": ra._regime(closes, idx, ra.V4_BREADTH),
        "regime_v8": ra._regime(closes, idx, ra.V8_BREADTH),
        "alloc_v4": alloc_v4,
        "alloc_v8": alloc_v8,
        "target": target,
        "daf_leverage": lev,
        "target_symbols": sorted(
            s for s, w in target.items() if s != ra.CASH_ETF and w > 1e-6
        ),
    }


def target_weights(ctx) -> Dict[str, float]:
    """Ranked sector allocation as target weights. {symbol: fraction of sleeve equity}."""
    if not is_rebalance_day(ctx):
        return _hold(ctx)

    decision = decide(ctx)
    if decision is None:
        return _hold(ctx)

    lev = decision["daf_leverage"]
    weights = {
        sym: w * lev
        for sym, w in decision["target"].items()
        if sym != ra.CASH_ETF and w > 1e-6
    }

    logger.info(
        "Ranked/ctx %s: V4=%s V8=%s lev=%.1fx gross=%.2f -> %s",
        ctx.as_of.date(), decision["regime_v4"], decision["regime_v8"],
        lev, sum(weights.values()),
        {k: f"{v:.0%}" for k, v in sorted(weights.items())},
    )
    return weights

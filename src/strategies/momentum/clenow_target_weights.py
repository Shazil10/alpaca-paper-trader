"""Clenow momentum on the backtest ``target_weights`` interface.

Same strategy as :mod:`strategies.momentum.clenow_trend`. Every decision comes
from there -- ``score_symbol``, ``rank_records``, ``regime_from_closes``,
``_inverse_vol_weights``, ``_generate_exit_signals``. Only the data source
changes: the live sleeve calls ``yf.download(..., period="1y")`` twice, which
means "now" and, at a 09:30 ET cron, means a partial session. Here every price
arrives through :class:`backtest.context.StrategyContext`, truncated at
``ctx.as_of``, which removes that look-ahead as a side effect of the migration.

Signals to weights
------------------
The live sleeve is incremental: it emits SELLs for rule violations and BUYs only
into free slots, sized as ``budget * w``. ``target_weights`` has to return the
*whole* intended book, so the translation is:

* a held name that trips no exit rule keeps its current portfolio weight,
* free slots are filled from the top of the ranking with inverse-vol weights,
* an exited name is simply absent, which the engine reads as "close it".

Two live quirks are preserved rather than quietly corrected, because a parity
gate that "fixes" things proves nothing. Both are reported in
:func:`decide`'s record so a run can see them.

**Slot accounting counts names that are leaving.** ``slots_available`` is
``top_n - len(held)`` measured *before* exits are applied, so a sleeve exiting
three of seven positions still buys nothing that day. That is what the live
sleeve does.

One deviation is unavoidable, and it is in sizing a partial fill.
``_inverse_vol_weights`` normalizes to 1.0 whatever it is handed, which in the
live sleeve means "spend the whole remaining budget on these names". As a
*fraction of equity* that is only correct when the whole book is being filled: a
single free slot would ask for 100% of equity in one position, the risk layer
would clip it, and gross exposure would oscillate with the number of free slots.
So new weights are scaled by ``slots_filled / top_n``. A seven-slot rebalance
still sums to 1.0; a one-slot top-up asks for 1/7.

**Sizing is equal-weight, not inverse-volatility, at the live ``top_n``.**
``_inverse_vol_weights`` clamps to [2%, 10%] and then renormalizes to sum to 1,
and the renormalization cancels the clamp. Seven names clamp to 10% each, sum to
70%, and scale straight back to 14.3% each -- and because *every* name hits the
same ceiling, the inverse-vol tilt is erased with it. An 11x spread in volatility
produces a 0x spread in weight. The tilt only reappears above ten positions,
where ``1/n`` falls under the 10% cap.

So feature 4 of ``clenow_trend``'s docstring does not operate in production. It
is preserved here because correcting it changes live position sizes, which is a
strategy decision rather than part of a migration. ``tests/backtest/
test_clenow_parity.py`` pins the behaviour so a future fix is a deliberate,
visible change.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from strategies.momentum import clenow_trend as ct

logger = logging.getLogger(__name__)

#: Re-exported so configs and tests reach the universe without importing both.
REGIME_ETFS = ct.REGIME_ETFS
DEFAULT_TOP_N = ct.DEFAULT_TOP_N
EXIT_RANK_CUTOFF = ct.EXIT_RANK_CUTOFF

#: Longest window any gate needs: the 200-day entry SMA, plus slack for the
#: 52-week high. Requesting less silently drops every symbol on the
#: ``len(close) < max(lookback, entry_sma)`` guard inside ``score_symbol``.
MIN_SESSIONS = 252

#: Calendar days of history to pull. ``period="1y"`` in the live sleeve, and the
#: gates need 252 sessions, so a calendar year is the matching request.
LOOKBACK_CALENDAR_DAYS = 365


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def closes_from_ctx(ctx, symbols: Optional[List[str]] = None) -> Dict[str, pd.Series]:
    """``{symbol: close series}`` over the sleeve's window, truncated at as_of.

    Mirrors what ``yf.download(period="1y")`` hands the live sleeve: one adjusted
    close series per symbol, most recent bar last, NaNs dropped per symbol rather
    than across the panel -- a name that halted for a day must not shorten
    everyone else's window.
    """
    start = pd.Timestamp(ctx.as_of) - pd.Timedelta(days=LOOKBACK_CALENDAR_DAYS)

    wanted = list(symbols) if symbols is not None else None
    matrix = ctx.prices(wanted, start=start)
    if matrix is None or matrix.empty:
        return {}

    out: Dict[str, pd.Series] = {}
    for symbol in matrix.columns:
        series = matrix[symbol].dropna()
        if len(series) > 0:
            out[str(symbol)] = series
    return out


def _universe(ctx) -> List[str]:
    """Tradable symbols on ``as_of``, regime ETFs excluded.

    The regime ETFs are an input to the gate, not candidates for it. The live
    sleeve gets this for free by reading universe.csv, which holds stocks only;
    a point-in-time universe unions the fixed ETFs in, so they are removed here.
    """
    members = ctx.universe() or set()
    return sorted(s for s in members if s not in set(ct.REGIME_ETFS))


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def decide(ctx, top_n: int = DEFAULT_TOP_N) -> Optional[dict]:
    """Score, rank, gate and size -- returning the full decision record.

    Exposed separately from :func:`target_weights` so a parity test can compare
    the ranking, the regime and the exit reasons, not just the final weights.
    Returns None when the context cannot support a decision at all.
    """
    candidates = _universe(ctx)
    held: Set[str] = set(ctx.held_symbols)

    # Regime ETFs are fetched alongside the candidates in one slice.
    closes = closes_from_ctx(ctx, candidates + list(ct.REGIME_ETFS))
    if not closes:
        logger.warning("Clenow/ctx %s: empty panel", ctx.as_of.date())
        return None

    records = []
    for symbol in candidates:
        series = closes.get(symbol)
        if series is None:
            continue
        record = ct.score_symbol(symbol, series)
        if record is not None:
            records.append(record)

    scores = ct.rank_records(records)
    risk_on = ct.regime_from_closes(closes)

    exit_signals = ct._generate_exit_signals(
        scores, held, "strategies.momentum.clenow_target_weights"
    )
    exiting = {s.symbol for s in exit_signals}

    # Slots are counted against everything held, including names leaving today.
    # See the module docstring: this is the live sleeve's behaviour.
    slots_available = max(top_n - len(held), 0)

    new_weights: Dict[str, float] = {}
    if risk_on and slots_available > 0 and not scores.empty:
        available = (
            scores[~scores["Symbol"].isin(sorted(held))]
            .head(slots_available)
            .copy()
        )
        if not available.empty:
            sized = ct._inverse_vol_weights(available.set_index("Symbol")["Vol20"])
            # _inverse_vol_weights always normalizes to 1.0, which means "deploy
            # the whole budget across these names". That is right for a full
            # rebalance and wrong for a fill-in: one free slot out of seven would
            # ask for 100% of equity in a single name. Scale by the share of the
            # book actually being filled, so a full seven-slot rebalance still
            # sums to 1.0 and a one-slot top-up asks for 1/7. See the module
            # docstring -- this is the one place the adapter must deviate,
            # because the live encoding is dollars against a running budget and
            # has no fraction-of-equity equivalent for a partial fill.
            book_share = min(slots_available, len(available)) / max(top_n, 1)
            new_weights = {
                str(k): float(v) * book_share for k, v in sized.items()
            }

    return {
        "as_of": str(pd.Timestamp(ctx.as_of).date()),
        "scored": int(len(scores)),
        "risk_on": bool(risk_on),
        "held": sorted(held),
        "exiting": sorted(exiting),
        "exit_reasons": {s.symbol: s.reason for s in exit_signals},
        "slots_available": int(slots_available),
        "ranking": (
            scores.head(max(top_n, EXIT_RANK_CUTOFF))["Symbol"].tolist()
            if not scores.empty else []
        ),
        "new_weights": new_weights,
    }


def target_weights(ctx, top_n: int = DEFAULT_TOP_N) -> Dict[str, float]:
    """Clenow momentum as target weights. {symbol: fraction of sleeve equity}."""
    params = ctx.params or {}
    top_n = int(params.get("top_n", top_n))

    decision = decide(ctx, top_n=top_n)
    if decision is None:
        # No panel is not a reason to liquidate: hold what is held.
        return {
            sym: float(w) for sym, w in ctx.portfolio.weights.items() if w > 0
        }

    exiting = set(decision["exiting"])
    weights: Dict[str, float] = {
        sym: float(w)
        for sym, w in ctx.portfolio.weights.items()
        if w > 0 and sym not in exiting
    }
    weights.update(decision["new_weights"])

    logger.info(
        "Clenow/ctx %s: %d scored, regime=%s, %d held, %d exiting, %d new -> gross %.2f",
        decision["as_of"], decision["scored"],
        "ON" if decision["risk_on"] else "OFF",
        len(decision["held"]), len(exiting), len(decision["new_weights"]),
        sum(weights.values()),
    )
    return weights

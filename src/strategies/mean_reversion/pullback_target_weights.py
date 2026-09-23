"""52-week-high pullback mean reversion on the ``target_weights`` interface.

Same strategy as :mod:`strategies.mean_reversion.52W_mean_reversion_strat`, whose
module name is not importable, so it is loaded by path below. Every decision comes
from there -- ``_score_series`` for the entry gates, ``regime_from_panel`` for the
R2 trend stack, ``exit_reason`` for target/stop/timeout. Only the data source
changes.

The live sleeve reads two things the market cannot tell it: what we paid, and
when. It gets them from the Alpaca position's ``avg_entry_price`` and by
reconstructing entry dates from order history. In a backtest both come straight
off ``ctx.portfolio`` -- ``Position.avg_entry_price`` and ``Position.entry_date``
-- which is the same arithmetic against a different source, and is why
``exit_reason`` takes them as arguments rather than fetching them.

The cooldown question
---------------------
The live module declares ``COOLDOWN_DAYS = 20`` (after any exit) and
``STOP_ENTRY_COOLDOWN = 30`` (after a stop) and **enforces neither**. They are
constants nothing reads. So "backtest the documented strategy" and "backtest the
running strategy" are two different strategies, and picking one silently would
misrepresent whichever was not chosen.

This module models the *running* strategy by default, and makes the documented
one measurable: set ``params: {cooldown_days: 20}`` to enforce a cooldown and see
what it would have cost or saved. Default is 0, which is the live behaviour.

The stop-specific 30-day variant is **not** modelled, and cannot be without more
plumbing: the engine records that a position closed, not why. The strategy's exit
reason is consumed by the adapter and never reaches the fill, so a backtest can
tell that a name was exited 12 days ago but not whether it was stopped out.
Threading reasons through would be a larger change than this migration, and
guessing would be worse than declining.

Sizing
------
Pullback-weighted: deeper pullback, larger allocation. The live sleeve normalizes
those weights to sum to 1 over however many slots are free, which as a *fraction
of equity* means one free slot asks for the entire book. As in the Clenow
adapter, new weights are scaled by ``slots_filled / top_n`` so a full rebalance
sums to 1.0 and a one-slot top-up asks for 1/5.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def _load_live_module():
    """Import ``52W_mean_reversion_strat`` by path.

    The filename starts with a digit, so it is not a legal module name and
    ``import`` cannot reach it. Renaming the file would break the live
    ``trade.py`` import and the canary fixtures, so the path load stays.
    """
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "live_pullback_strat", here / "52W_mean_reversion_strat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pb = _load_live_module()

#: Re-exported so configs and tests reach these without the path dance.
REGIME_ETFS = pb.REGIME_ETFS
TOP_N = pb.TOP_N
ENTRY_DEPTH = pb.ENTRY_DEPTH
EXIT_PULLBACK = pb.EXIT_PULLBACK
COOLDOWN_DAYS = pb.COOLDOWN_DAYS
STOP_ENTRY_COOLDOWN = pb.STOP_ENTRY_COOLDOWN
LOOKBACK_DAYS = pb.LOOKBACK_DAYS
REGIME_LOOKBACK_DAYS = pb.REGIME_LOOKBACK_DAYS

#: The entry gates need a 252-day high and a 200-day average, so a symbol with
#: less than this is skipped by ``_score_series`` regardless.
MIN_SESSIONS = max(pb.HIGH_LOOKBACK, pb.MA_LOOKBACK)


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def panel_from_ctx(ctx, symbols: List[str], days: int) -> pd.DataFrame:
    """Date x symbol close panel over a trailing window, truncated at as_of.

    Matches ``_close_panel``: the same trailing calendar window, adjusted closes,
    and no cross-symbol dropna -- every gate in this sleeve reads a trailing
    slice of one symbol, so one halted name must not shorten anyone else's
    history.
    """
    start = pd.Timestamp(ctx.as_of) - pd.Timedelta(days=days)
    matrix = ctx.prices(symbols, start=start)
    if matrix is None or matrix.empty:
        return pd.DataFrame(dtype="float64")
    return matrix


def _candidates(ctx) -> List[str]:
    """Tradable stocks on as_of. Regime ETFs are inputs, not candidates."""
    members = ctx.universe() or set()
    return sorted(s for s in members if s not in set(pb.REGIME_ETFS))


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def decide(
    ctx,
    top_n: int = TOP_N,
    cooldown_days: int = 0,
) -> Optional[dict]:
    """Score, gate, exit and size. Returns the full decision record."""
    candidates = _candidates(ctx)
    held: Set[str] = set(ctx.held_symbols)

    panel = panel_from_ctx(ctx, candidates + list(pb.REGIME_ETFS), pb.LOOKBACK_DAYS)
    if panel.empty:
        logger.warning("Pullback/ctx %s: empty panel", ctx.as_of.date())
        return None

    records = []
    for symbol in panel.columns:
        if symbol in set(pb.REGIME_ETFS):
            continue
        record = pb._score_series(str(symbol), panel[symbol])
        if record is not None:
            records.append(record)

    scores = pd.DataFrame(records)
    if not scores.empty:
        scores = scores.sort_values("pullback", ascending=False).reset_index(drop=True)

    regime_panel = panel_from_ctx(
        ctx, list(pb.REGIME_ETFS), pb.REGIME_LOOKBACK_DAYS
    )
    risk_on = pb.regime_from_panel(regime_panel)

    # ---- Exits -----------------------------------------------------------
    # Every rule needs a current price and a 52-week high, which come from the
    # held name's own series rather than from the scored set: a holding that
    # failed the *entry* gates is not therefore an exit, and treating it as one
    # would sell every position the moment it started recovering.
    exit_reasons: Dict[str, str] = {}
    missing_prices: List[str] = []

    for symbol in sorted(held):
        series = panel[symbol].dropna() if symbol in panel.columns else pd.Series(dtype=float)
        if len(series) < 10:
            # Absent data is an infrastructure condition, not a market signal.
            # The live sleeve holds and defers; so does this.
            missing_prices.append(symbol)
            continue

        price = float(series.iloc[-1])
        high52 = pb._high_52w(series)
        if price <= 0 or pd.isna(high52) or high52 <= 0:
            missing_prices.append(symbol)
            continue

        position = ctx.portfolio.positions.get(symbol)
        reason = pb.exit_reason(
            pullback=(high52 - price) / high52,
            price=price,
            entry_price=float(position.avg_entry_price) if position else None,
            entry_date=(
                pd.Timestamp(position.entry_date).to_pydatetime()
                if position is not None else None
            ),
            now=pd.Timestamp(ctx.as_of).to_pydatetime(),
        )
        if reason:
            exit_reasons[symbol] = reason

    exiting = set(exit_reasons)

    # ---- Entries ---------------------------------------------------------
    slots_available = max(top_n - len(held), 0)
    blocked: List[str] = []
    new_weights: Dict[str, float] = {}

    if risk_on and slots_available > 0 and not scores.empty:
        eligible = scores[~scores["symbol"].isin(sorted(held))]

        if cooldown_days > 0:
            keep = []
            for symbol in eligible["symbol"]:
                since = ctx.days_since_exit(str(symbol))
                if since is not None and since < cooldown_days:
                    blocked.append(str(symbol))
                else:
                    keep.append(symbol)
            eligible = eligible[eligible["symbol"].isin(keep)]

        picks = eligible.head(slots_available)
        if not picks.empty:
            depths = picks.set_index("symbol")["pullback"].clip(lower=1e-6)
            shares = depths / depths.sum()
            book_share = len(picks) / max(top_n, 1)
            new_weights = {
                str(k): float(v) * book_share for k, v in shares.items()
            }

    return {
        "as_of": str(pd.Timestamp(ctx.as_of).date()),
        "scored": int(len(scores)),
        "risk_on": bool(risk_on),
        "held": sorted(held),
        "exiting": sorted(exiting),
        "exit_reasons": exit_reasons,
        "missing_prices": sorted(missing_prices),
        "slots_available": int(slots_available),
        "cooldown_blocked": sorted(blocked),
        "candidates": [] if scores.empty else scores["symbol"].tolist()[: top_n * 3],
        "new_weights": new_weights,
    }


def target_weights(ctx, top_n: int = TOP_N) -> Dict[str, float]:
    """Pullback mean reversion as target weights. {symbol: fraction of equity}."""
    params = ctx.params or {}
    top_n = int(params.get("top_n", top_n))
    cooldown_days = int(params.get("cooldown_days", 0))

    decision = decide(ctx, top_n=top_n, cooldown_days=cooldown_days)
    if decision is None:
        return {sym: float(w) for sym, w in ctx.portfolio.weights.items() if w > 0}

    exiting = set(decision["exiting"])
    weights: Dict[str, float] = {
        sym: float(w)
        for sym, w in ctx.portfolio.weights.items()
        if w > 0 and sym not in exiting
    }
    weights.update(decision["new_weights"])

    logger.info(
        "Pullback/ctx %s: %d candidate(s), regime=%s, %d held, %d exiting, "
        "%d new%s -> gross %.2f",
        decision["as_of"], decision["scored"],
        "ON" if decision["risk_on"] else "OFF",
        len(decision["held"]), len(exiting), len(decision["new_weights"]),
        f", {len(decision['cooldown_blocked'])} on cooldown"
        if decision["cooldown_blocked"] else "",
        sum(weights.values()),
    )
    return weights

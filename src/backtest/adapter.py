"""Adapter bridging target_weights and generate_signals in both directions.

Two directions:
1. weights_to_signals: A migrated (target_weights) strategy drives live trade.py
2. signals_to_weights: An un-migrated (generate_signals) strategy is backtested

This allows gradual migration — strategies move one at a time, each verified
for decision parity before the switch.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set
from uuid import uuid4

from backtest.context import StrategyContext

logger = logging.getLogger(__name__)


def weights_to_signals(
    weights: Dict[str, float],
    budget: float,
    strategy_id: str,
    held_symbols: Set[str],
) -> list:
    """Convert target weights to Signal objects for the live system.

    This lets a migrated strategy (returning target_weights) drive the
    existing trade.py without changes.

    Args:
        weights: {symbol: fraction} from target_weights()
        budget: remaining budget from committed_dollars_from_orders
        strategy_id: the strategy module path
        held_symbols: currently held symbols

    Returns:
        List[Signal] compatible with trade.py
    """
    from trade_models import Signal, Side

    signals = []
    target_symbols = set(weights.keys())

    # SELL: held but not in target (or target weight is 0/negative)
    for sym in held_symbols:
        if sym not in target_symbols or weights.get(sym, 0) <= 0:
            signals.append(Signal(
                symbol=sym,
                side=Side.SELL,
                reason="exit:not_in_target",
                strategy_id=strategy_id,
                client_order_id=f"{strategy_id}:{uuid4().hex[:16]}",
            ))

    # BUY: in target, not already held
    for sym, w in weights.items():
        if w <= 0:
            continue
        if sym in held_symbols:
            continue  # Already held, skip (rebalancing handled separately)

        notional = budget * w
        if notional < 1.0:
            continue

        signals.append(Signal(
            symbol=sym,
            side=Side.BUY,
            reason=f"target_weight={w:.3f}",
            notional=notional,
            strategy_id=strategy_id,
            client_order_id=f"{strategy_id}:{uuid4().hex[:16]}",
        ))

    return signals


def signals_to_weights(
    signals: list,
    equity: float,
) -> Dict[str, float]:
    """Convert Signal objects to target weights for the backtester.

    This lets an un-migrated strategy (returning generate_signals) be
    backtested through the new engine without porting.

    Args:
        signals: List[Signal] from generate_signals()
        equity: current portfolio equity

    Returns:
        {symbol: weight} compatible with the backtest engine
    """
    if equity <= 0:
        return {}

    weights: Dict[str, float] = {}

    for sig in signals:
        side = str(getattr(sig, 'side', '')).upper()
        if 'SELL' in side:
            weights[sig.symbol] = 0.0  # exit signal
        elif 'BUY' in side:
            notional = getattr(sig, 'notional', None) or 0
            if notional > 0:
                weights[sig.symbol] = notional / equity
            else:
                weights[sig.symbol] = 0.05
    return weights


def wrap_generate_signals(
    module_path: str,
    default_budget: float = 100_000.0,
) -> Callable[[StrategyContext], Dict[str, float]]:
    """Wrap an existing generate_signals module as a target_weights function.

    The wrapped function:
    1. Extracts budget, strategy_id, held_symbols from ctx
    2. Calls module.generate_signals(budget=..., strategy_id=..., held_symbols=...)
    3. Converts the returned signals to target weights

    This is the migration bridge — use it to backtest existing strategies
    without rewriting them.

    IMPORTANT: This wrapper does NOT enforce the PIT firewall on the
    strategy's internal data fetches. The strategy may still call
    yfinance or store directly. For a proper backtest, the strategy must
    be migrated to use ctx.prices() instead.

    For strategies that already support as_of (ranked_asset_alloc,
    52W_mean_reversion_strat), a better wrapper would patch as_of
    into the call. This generic wrapper is the fallback.
    """
    import importlib

    module = importlib.import_module(module_path)
    generate_fn = getattr(module, 'generate_signals')

    def wrapped(ctx: StrategyContext) -> Dict[str, float]:
        budget = ctx.equity * (1 - 0.10)  # 10% cash reserve
        if default_budget > 0:
            budget = min(budget, default_budget)

        strategy_id = module_path
        held = set(ctx.held_symbols)

        signals = list(generate_fn(
            budget=budget,
            strategy_id=strategy_id,
            held_symbols=held,
        ))

        return signals_to_weights(signals, ctx.equity)

    wrapped.__name__ = f"wrapped_{module_path.replace('.', '_')}"
    wrapped.__doc__ = f"Wrapped generate_signals from {module_path}"
    return wrapped


def wrap_with_asof(
    module_path: str,
    default_budget: float = 100_000.0,
) -> Callable[[StrategyContext], Dict[str, float]]:
    """Wrap a strategy that supports as_of parameter.

    For ranked_asset_alloc and 52W_mean_reversion_strat which accept
    as_of in their internal data loading. This is a better wrapper than
    wrap_generate_signals because it patches the date, limiting (but not
    eliminating) look-ahead bias.

    Full PIT enforcement still requires migrating to ctx.prices().
    """
    import importlib
    import os

    module = importlib.import_module(module_path)
    generate_fn = getattr(module, 'generate_signals')

    def wrapped(ctx: StrategyContext) -> Dict[str, float]:
        budget = ctx.equity * (1 - 0.10)
        if default_budget > 0:
            budget = min(budget, default_budget)

        strategy_id = module_path
        held = set(ctx.held_symbols)

        import inspect
        sig = inspect.signature(generate_fn)
        kwargs: Dict[str, Any] = {
            'budget': budget,
            'strategy_id': strategy_id,
            'held_symbols': held,
        }
        if 'as_of' in sig.parameters:
            kwargs['as_of'] = ctx.as_of

        old_source = os.environ.get('PRICE_SOURCE', '')
        os.environ['PRICE_SOURCE'] = 'lake'

        try:
            signals = list(generate_fn(**kwargs))
        finally:
            if old_source:
                os.environ['PRICE_SOURCE'] = old_source
            else:
                os.environ.pop('PRICE_SOURCE', None)

        return signals_to_weights(signals, ctx.equity)

    wrapped.__name__ = f"wrapped_asof_{module_path.replace('.', '_')}"
    return wrapped

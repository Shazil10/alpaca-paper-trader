"""trade.py

Runs all enabled strategies and places orders.

Contract:
- ``config.STRATEGY_ALLOCATIONS`` maps strategy module path -> max dollars to deploy.
- Each strategy exposes ``generate_signals(budget=..., strategy_id=..., held_symbols=...)``
  and returns a list of ``Signal`` objects (BUY **and** SELL).
- Strategies own sizing by setting ``Signal.notional``.  This runner executes signals safely.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional, Set
from uuid import uuid4

import config
import orders
from trade_models import Side, Signal, committed_dollars_from_orders

try:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
except ImportError:
    GetOrdersRequest = None  # type: ignore[assignment,misc]
    QueryOrderStatus = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


def _get_orders_page(client, *, status: str = "all", limit: int = 500) -> list:
    """Fetch a page of orders, handling SDK differences.

    Some alpaca-py versions accept keyword args directly; others require
    a ``GetOrdersRequest`` object.  We try both so it works everywhere.
    """
    # Try the request-object approach first (newer SDK)
    if GetOrdersRequest is not None:
        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus(status),
                limit=limit,
                nested=True,
            )
            return list(client.get_orders(req))
        except Exception:
            pass

    # Fallback: raw kwargs (older SDK)
    try:
        return list(client.get_orders(status=status, limit=limit, direction="desc", nested=True))
    except TypeError:
        pass

    # Minimal fallback
    try:
        return list(client.get_orders(limit=limit))
    except TypeError:
        return list(client.get_orders())


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def _positions_by_symbol(client) -> Dict[str, object]:
    """Return a dict mapping SYMBOL -> Alpaca position object."""
    positions: Dict[str, object] = {}
    try:
        for p in client.get_all_positions():
            sym = str(getattr(p, "symbol", "")).strip().upper()
            if sym:
                positions[sym] = p
    except Exception:
        logger.exception("Failed to fetch positions")
    return positions


def _held_symbols_for_strategy(client, strategy_id: str) -> Set[str]:
    """Return the set of symbols *this* strategy currently holds.

    We attribute ownership via ``client_order_id`` prefix:
        ``f"{strategy_id}:..."``

    A symbol is considered held if:
      1. There is at least one *filled* BUY order with this strategy's prefix, AND
      2. We currently have a position in that symbol.

    This is deliberately conservative – if we cannot determine attribution, we
    skip the symbol (the strategy will not generate a spurious SELL).
    """
    # All symbols we currently hold as positions.
    all_positions = _positions_by_symbol(client)

    # Scan order history for BUY fills tagged with this strategy.
    prefix = f"{strategy_id}:"
    bought_symbols: Set[str] = set()

    try:
        batch = _get_orders_page(client, status="all", limit=500)
    except Exception:
        logger.exception("Failed to fetch orders for held-symbol scan (%s)", strategy_id)
        batch = []

    if batch:
        for o in batch:
            cid = str(getattr(o, "client_order_id", "") or "")
            if not cid.startswith(prefix):
                continue
            side = str(getattr(o, "side", "")).upper()
            status = str(getattr(o, "status", "")).lower()
            sym = str(getattr(o, "symbol", "")).strip().upper()
            if side == "BUY" and status == "filled" and sym:
                bought_symbols.add(sym)

    # Intersection: only include symbols we *still* hold.
    return bought_symbols & set(all_positions.keys())


def _existing_symbols(client) -> Set[str]:
    """Return symbols we already hold or have pending buy orders for."""
    existing: Set[str] = set()

    # 1) Current positions
    try:
        for p in client.get_all_positions():
            existing.add(str(p.symbol).strip().upper())
    except Exception:
        logger.exception("Failed to fetch positions")

    # 2) Open orders (prevents double-buys when a previous run already submitted orders)
    try:
        for o in client.get_orders():
            side = str(getattr(o, "side", "")).upper()
            status = str(getattr(o, "status", "")).lower()
            sym = str(getattr(o, "symbol", "")).strip().upper()
            if not sym:
                continue
            if side == "BUY" and status in {"new", "accepted", "pending_new", "held", "partially_filled"}:
                existing.add(sym)
    except Exception:
        logger.exception("Failed to fetch open orders")

    return existing


def _strategy_committed_dollars(client, *, strategy_id: str) -> float:
    """Return lifetime dollars committed for a strategy.

    "Committed" means BUY orders that are either filled or still open/pending.
    We attribute orders to strategies by ``client_order_id`` prefix.
    """
    try:
        batch = _get_orders_page(client, status="all", limit=500)
    except Exception:
        logger.exception("Failed to fetch orders for strategy accounting (%s)", strategy_id)
        batch = []

    return committed_dollars_from_orders(batch, strategy_id=strategy_id)


# ---------------------------------------------------------------------------
# SELL execution
# ---------------------------------------------------------------------------

def _execute_sell_signals(
    client,
    sell_signals: List[Signal],
    positions: Dict[str, object],
    strategy_path: str,
) -> None:
    """Close positions for each SELL signal (full liquidation)."""
    for s in sell_signals:
        symbol = s.normalized_symbol()
        pos = positions.get(symbol)
        if pos is None:
            logger.info("SELL signal for %s but no position found (skipping)", symbol)
            continue

        qty = _safe_float(getattr(pos, "qty", 0.0))
        if qty <= 0:
            logger.info("SELL signal for %s but qty=%.4f (skipping)", symbol, qty)
            continue

        try:
            client_order_id = f"{strategy_path}:{uuid4().hex[:16]}"
            orders.sell_market_qty(client, symbol, qty)
            logger.info(
                "Submitted SELL %s qty=%.4f reason=%s (strategy=%s)",
                symbol,
                qty,
                s.reason or "exit",
                strategy_path,
            )
        except Exception:
            logger.exception("SELL order failed for %s (strategy=%s)", symbol, strategy_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def execute_daily_trades() -> None:
    """Run all configured strategies and place the resulting trades."""
    client = config.get_client()

    # Risk guard: keep some cash untouched.
    try:
        account = client.get_account()
        cash = _safe_float(getattr(account, "cash", 0.0))
    except Exception:
        logger.exception("Failed to fetch account")
        return

    cash_reserve = cash * 0.10  # keep 10 % as a safety buffer

    existing = _existing_symbols(client)
    positions = _positions_by_symbol(client)
    submitted_this_run: Set[str] = set()

    for strategy_path, budget in config.STRATEGY_ALLOCATIONS.items():
        if budget <= 0:
            logger.info("Skipping %s (budget=%s)", strategy_path, budget)
            continue

        committed = _strategy_committed_dollars(client, strategy_id=strategy_path)
        remaining_budget = float(budget) - committed

        logger.info(
            "Strategy=%s budget=%.2f committed=%.2f remaining=%.2f",
            strategy_path,
            float(budget),
            committed,
            remaining_budget,
        )

        # Even if the lifetime cap is reached, we still need to run the strategy
        # so it can generate EXIT signals for positions it already holds.
        try:
            module = importlib.import_module(strategy_path)
        except ModuleNotFoundError:
            logger.exception("Strategy module not found: %s", strategy_path)
            continue

        generate_signals = getattr(module, "generate_signals", None)
        if not callable(generate_signals):
            logger.error("Strategy %s has no callable generate_signals()", strategy_path)
            continue

        # Tell the strategy which symbols it currently owns so it can decide exits.
        held = _held_symbols_for_strategy(client, strategy_path)

        try:
            signals = list(
                generate_signals(
                    budget=max(remaining_budget, 0),
                    strategy_id=strategy_path,
                    held_symbols=held,
                )
            )
        except Exception:
            logger.exception("Strategy %s failed while generating signals", strategy_path)
            continue

        # ── Process SELL signals first ──
        sell_signals: List[Signal] = [
            s for s in signals if isinstance(s, Signal) and s.side == Side.SELL and s.normalized_symbol()
        ]
        if sell_signals:
            logger.info("Strategy %s returned %d SELL signals", strategy_path, len(sell_signals))
            _execute_sell_signals(client, sell_signals, positions, strategy_path)

        # ── Process BUY signals ──
        if remaining_budget <= 0:
            logger.info(
                "Skipping BUY signals for %s (lifetime cap reached: budget=%.2f committed=%.2f)",
                strategy_path,
                float(budget),
                committed,
            )
            continue

        buy_signals: List[Signal] = [
            s for s in signals if isinstance(s, Signal) and s.side == Side.BUY and s.normalized_symbol()
        ]
        if not buy_signals:
            logger.info("Strategy %s returned no BUY signals", strategy_path)
            continue

        for s in buy_signals:
            symbol = s.normalized_symbol()
            try:
                if symbol in existing or symbol in submitted_this_run:
                    logger.info("Already have/pending %s (skipping)", symbol)
                    continue

                dollars = _safe_float(s.notional)
                if dollars <= 0:
                    logger.info("No sizing for %s (strategy=%s), skipping", symbol, strategy_path)
                    continue

                # Don't spend into the cash reserve.
                if cash - cash_reserve < dollars:
                    logger.warning(
                        "Not enough cash left for %s (need %.2f, cash %.2f, reserve %.2f)",
                        symbol,
                        dollars,
                        cash,
                        cash_reserve,
                    )
                    continue

                client_order_id = f"{strategy_path}:{uuid4().hex[:16]}"

                orders.buy_market_notional(client, symbol, dollars, client_order_id=client_order_id)
                submitted_this_run.add(symbol)
                cash -= dollars
                logger.info(
                    "Submitted BUY %s notional=%.2f (strategy=%s, client_order_id=%s)",
                    symbol,
                    dollars,
                    strategy_path,
                    client_order_id,
                )
            except Exception:
                logger.exception(
                    "Order failed for %s (strategy=%s, notional=%.2f)",
                    symbol,
                    strategy_path,
                    _safe_float(getattr(s, "notional", 0.0)),
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    execute_daily_trades()
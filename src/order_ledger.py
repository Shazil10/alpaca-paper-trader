"""order_ledger.py

One implementation of "turn Alpaca orders into an attributed trade ledger".

Both ``report.py`` (human-readable order report) and ``export_portfolio.py``
(microsite data feed) read from here so the two can never disagree.

Two things this module owns:

1. **Attribution.** Orders are tagged ``client_order_id = f"{strategy_id}:{uuid}"``
   by ``trade.py``.  ``strategy_id_of()`` recovers the strategy from that tag.

2. **Realized PnL via per-strategy FIFO.** Cost basis is keyed by
   ``(strategy_id, symbol)`` rather than ``symbol`` alone.  This matters when
   two sleeves hold the same ticker — GEO was held by both the Clenow and
   pullback sleeves, and a symbol-only FIFO queue silently books one sleeve's
   sell against the other sleeve's entry price.

   Lots are quantity-aware: a sell consumes buy lots proportionally instead of
   assuming one order equals one lot.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from trade_models import _enum_str, safe_float

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def strategy_id_of(client_order_id: str) -> str:
    """Return the strategy id encoded in a ``client_order_id``, or "".

    ``"strategies.momentum.clenow_trend:9f2a..."`` -> ``"strategies.momentum.clenow_trend"``

    Untagged orders (hand-placed from the Alpaca dashboard) have no colon
    prefix we control, so they return "" and are treated as discretionary.
    """
    if not client_order_id:
        return ""
    left, sep, _ = client_order_id.partition(":")
    if not sep:
        return ""
    # Only accept our own namespace; a raw broker UUID is not a strategy.
    if not left.startswith("strategies."):
        return ""
    return left


def parse_strategy_id(client_order_id: str) -> Tuple[str, str]:
    """Split a ``client_order_id`` into human ``(type, name)`` labels.

    ``"strategies.momentum.clenow_trend:..."`` -> ``("Momentum", "Clenow Trend")``
    """
    if not client_order_id:
        return ("", "")
    left, _, _ = client_order_id.partition(":")

    parts = left.split(".")
    if len(parts) >= 3 and parts[0] == "strategies":
        return (parts[1].replace("_", " ").title(), parts[2].replace("_", " ").title())

    return (left.replace("_", " ").title(), "")


# ---------------------------------------------------------------------------
# Ledger rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LedgerRow:
    """One order, normalized, with realized PnL attached to closing sells."""

    submitted_at: str
    symbol: str
    side: str            # "BUY" / "SELL"
    status: str          # "FILLED", "CANCELED", ...
    notional: float
    filled_qty: float
    filled_avg_price: float
    filled_value: float
    client_order_id: str
    strategy_id: str     # "" for untagged / discretionary
    pnl: Optional[float] = None

    @property
    def is_filled(self) -> bool:
        return self.status == "FILLED" and self.filled_qty > 0

    @property
    def is_discretionary(self) -> bool:
        return not self.strategy_id


def _normalize(order: object) -> LedgerRow:
    """Read one Alpaca order object into a LedgerRow (no PnL yet)."""
    symbol = str(getattr(order, "symbol", "") or "").strip().upper()
    side = _enum_str(getattr(order, "side", "")).upper()
    status = _enum_str(getattr(order, "status", "")).upper()

    notional = safe_float(getattr(order, "notional", 0.0))
    filled_qty = safe_float(getattr(order, "filled_qty", 0.0))
    filled_avg_price = safe_float(getattr(order, "filled_avg_price", 0.0))
    filled_value = (
        filled_qty * filled_avg_price
        if filled_qty > 0 and filled_avg_price > 0
        else 0.0
    )

    client_order_id = str(getattr(order, "client_order_id", "") or "")

    return LedgerRow(
        submitted_at=str(getattr(order, "submitted_at", "") or ""),
        symbol=symbol,
        side=side,
        status=status,
        notional=notional,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        filled_value=filled_value,
        client_order_id=client_order_id,
        strategy_id=strategy_id_of(client_order_id),
    )


def build_ledger(orders: Iterable[object]) -> List[LedgerRow]:
    """Normalize orders and attach realized PnL to filled SELL rows.

    Returned newest-first for display.  FIFO matching runs oldest-first
    internally regardless of input order.
    """
    rows = [_normalize(o) for o in orders]

    # Oldest-first so cost basis accumulates before sells consume it.
    rows.sort(key=lambda r: r.submitted_at)

    # (strategy_id, symbol) -> deque of [remaining_qty, entry_price]
    lots: Dict[Tuple[str, str], Deque[List[float]]] = {}

    priced: List[LedgerRow] = []
    for row in rows:
        if not row.is_filled or row.filled_avg_price <= 0:
            priced.append(row)
            continue

        key = (row.strategy_id, row.symbol)

        if row.side == "BUY":
            lots.setdefault(key, deque()).append([row.filled_qty, row.filled_avg_price])
            priced.append(row)
            continue

        if row.side != "SELL":
            priced.append(row)
            continue

        # Consume FIFO lots for this (strategy, symbol) pair.
        queue = lots.get(key)
        remaining = row.filled_qty
        realized = 0.0
        matched_any = False

        while remaining > 1e-9 and queue:
            lot = queue[0]
            take = min(lot[0], remaining)
            realized += (row.filled_avg_price - lot[1]) * take
            lot[0] -= take
            remaining -= take
            matched_any = True
            if lot[0] <= 1e-9:
                queue.popleft()

        if remaining > 1e-9:
            # Sold more than we have a recorded entry for. Alpaca caps order
            # history (200 orders), so entries can predate the window.
            logger.debug(
                "Unmatched sell qty %.4f for %s (%s) — entry likely outside history window",
                remaining, row.symbol, row.strategy_id or "discretionary",
            )

        priced.append(
            LedgerRow(
                submitted_at=row.submitted_at,
                symbol=row.symbol,
                side=row.side,
                status=row.status,
                notional=row.notional,
                filled_qty=row.filled_qty,
                filled_avg_price=row.filled_avg_price,
                filled_value=row.filled_value,
                client_order_id=row.client_order_id,
                strategy_id=row.strategy_id,
                pnl=round(realized, 2) if matched_any else None,
            )
        )

    priced.sort(key=lambda r: r.submitted_at, reverse=True)
    return priced


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def realized_pnl_by_strategy(rows: Iterable[LedgerRow]) -> Dict[str, float]:
    """Sum realized (closed-trade) PnL per strategy id.

    Only closing sells carry PnL, so this is a closed-trade figure: open
    positions contribute nothing until they are sold.
    """
    totals: Dict[str, float] = {}
    for row in rows:
        if row.pnl is None or not row.strategy_id:
            continue
        totals[row.strategy_id] = round(totals.get(row.strategy_id, 0.0) + row.pnl, 2)
    return totals


def held_symbols_by_strategy(rows: Iterable[LedgerRow]) -> Dict[str, set]:
    """Symbols each strategy has ever bought and filled.

    Intersect with live Alpaca positions to get current holdings — this is the
    same attribution rule ``trade.py._held_symbols_for_strategy`` applies.
    """
    bought: Dict[str, set] = {}
    for row in rows:
        if not row.strategy_id or row.side != "BUY" or not row.is_filled:
            continue
        bought.setdefault(row.strategy_id, set()).add(row.symbol)
    return bought

"""Portfolio accounting with FIFO lot tracking.

Tracks cash, positions (as FIFO lots), equity, realized and unrealized PnL.
Every mutation returns a new state — no in-place modification.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd

from backtest.types import (
    Fill, Lot, Order, OrderSide, Position, PortfolioSnapshot,
)

logger = logging.getLogger(__name__)


class Portfolio:
    """Mutable portfolio state. The engine owns one instance per simulation."""

    def __init__(self, initial_cash: float, strategy_id: str = ""):
        self.cash: float = initial_cash
        self.lots: List[Lot] = []
        self.realized_pnl: float = 0.0
        self.strategy_id = strategy_id

    def apply_fill(self, fill: Fill) -> float:
        """Process a fill. Returns realized PnL (0 for buys).
        
        BUY: deduct cash, add lot.
        SELL: remove lots FIFO, add proceeds to cash, compute realized PnL.
        """
        cost = fill.fill_price * fill.fill_shares + fill.commission
        realized = 0.0

        if fill.order.side == OrderSide.BUY:
            self.cash -= cost
            new_lot = Lot(
                symbol=fill.order.symbol,
                shares=fill.fill_shares,
                entry_price=fill.fill_price,
                entry_date=fill.fill_date,
                strategy_id=fill.order.strategy_id,
                lot_id=uuid4().hex[:12],
            )
            self.lots.append(new_lot)

        elif fill.order.side == OrderSide.SELL:
            proceeds = fill.fill_price * fill.fill_shares - fill.commission
            self.cash += proceeds
            realized = self._sell_fifo(
                fill.order.symbol, fill.fill_shares, fill.fill_price,
                strategy_id=fill.order.strategy_id or None,
            )
            self.realized_pnl += realized

        return realized

    def _sell_fifo(
        self,
        symbol: str,
        shares_to_sell: float,
        sell_price: float,
        strategy_id: Optional[str] = None,
    ) -> float:
        """Remove shares FIFO. Returns realized PnL.

        ``strategy_id`` restricts the sale to one sleeve's lots. Without it, a
        fund sleeve selling its own position could consume another sleeve's lots
        in the same symbol -- both would then disagree with the ledger about what
        they hold, and the attribution would be silently wrong.
        """
        remaining = shares_to_sell
        realized = 0.0
        new_lots = []

        for lot in self.lots:
            same_owner = strategy_id is None or lot.strategy_id == strategy_id
            if lot.symbol != symbol or remaining <= 0 or not same_owner:
                new_lots.append(lot)
                continue

            if lot.shares <= remaining:
                realized += (sell_price - lot.entry_price) * lot.shares
                remaining -= lot.shares
            else:
                realized += (sell_price - lot.entry_price) * remaining
                reduced = Lot(
                    symbol=lot.symbol,
                    shares=lot.shares - remaining,
                    entry_price=lot.entry_price,
                    entry_date=lot.entry_date,
                    strategy_id=lot.strategy_id,
                    lot_id=lot.lot_id,
                )
                new_lots.append(reduced)
                remaining = 0.0

        self.lots = new_lots
        return realized

    def snapshot(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float],
        *,
        strategy_id: Optional[str] = None,
        cash_override: Optional[float] = None,
    ) -> PortfolioSnapshot:
        """Create an immutable snapshot at current market prices.

        prices: {symbol: current_adj_close}

        ``strategy_id`` and ``cash_override`` exist for the fund engine, where one
        ledger holds every sleeve's lots over a single cash balance. A sleeve has
        to see *its own* book to decide -- Clenow exits the names it holds, the
        pullback sleeve stops out against the price it paid -- so the fund asks
        for a snapshot filtered to one sleeve's lots, paired with that sleeve's
        share of cash. Passing neither gives the whole portfolio, which is what
        the single-strategy engine wants.
        """
        positions: Dict[str, Position] = {}

        lots = self.lots
        if strategy_id is not None:
            lots = [l for l in lots if l.strategy_id == strategy_id]

        lots_by_symbol: Dict[str, List[Lot]] = {}
        for lot in lots:
            lots_by_symbol.setdefault(lot.symbol, []).append(lot)

        total_unrealized = 0.0

        for symbol, symbol_lots in lots_by_symbol.items():
            total_shares = sum(l.shares for l in symbol_lots)
            if total_shares <= 1e-9:
                continue

            total_cost = sum(l.shares * l.entry_price for l in symbol_lots)
            avg_entry = total_cost / total_shares
            market_price = prices.get(symbol, avg_entry)
            market_value = total_shares * market_price
            unrealized = market_value - total_cost
            total_unrealized += unrealized
            earliest_entry = min(l.entry_date for l in symbol_lots)

            positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                avg_entry_price=avg_entry,
                market_price=market_price,
                market_value=market_value,
                unrealized_pnl=unrealized,
                entry_date=earliest_entry,
                lots=tuple(symbol_lots),
            )

        cash = self.cash if cash_override is None else float(cash_override)
        equity = cash + sum(p.market_value for p in positions.values())

        return PortfolioSnapshot(
            date=date,
            cash=cash,
            equity=equity,
            positions=positions,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=total_unrealized,
            strategy_id=strategy_id or self.strategy_id,
        )

    def position_shares(self, symbol: str) -> float:
        """Total shares held in a symbol."""
        return sum(l.shares for l in self.lots if l.symbol == symbol)

    def drop_dust(self, tolerance: float = 1e-9) -> List[str]:
        """Discard lots too small to be tradable. Returns the symbols dropped.

        ``snapshot`` already hides a sub-tolerance holding, so it is invisible to
        a strategy while still sitting in ``self.lots`` -- which makes
        ``held_symbols`` and the snapshot disagree. Anything this small cannot be
        sold and is worth fractions of a cent; carrying it only creates that
        inconsistency.
        """
        by_symbol: Dict[str, float] = {}
        for lot in self.lots:
            by_symbol[lot.symbol] = by_symbol.get(lot.symbol, 0.0) + lot.shares

        dust = {s for s, shares in by_symbol.items() if 0 < shares <= tolerance}
        if not dust:
            return []

        self.lots = [l for l in self.lots if l.symbol not in dust]
        return sorted(dust)

    def held_symbols(self) -> set:
        """Symbols currently held."""
        return {l.symbol for l in self.lots if l.shares > 1e-9}

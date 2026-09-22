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
                fill.order.symbol, fill.fill_shares, fill.fill_price
            )
            self.realized_pnl += realized

        return realized

    def _sell_fifo(self, symbol: str, shares_to_sell: float, sell_price: float) -> float:
        """Remove shares FIFO. Returns realized PnL."""
        remaining = shares_to_sell
        realized = 0.0
        new_lots = []

        for lot in self.lots:
            if lot.symbol != symbol or remaining <= 0:
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
    ) -> PortfolioSnapshot:
        """Create an immutable snapshot at current market prices.
        
        prices: {symbol: current_adj_close}
        """
        positions: Dict[str, Position] = {}

        lots_by_symbol: Dict[str, List[Lot]] = {}
        for lot in self.lots:
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

        equity = self.cash + sum(p.market_value for p in positions.values())

        return PortfolioSnapshot(
            date=date,
            cash=self.cash,
            equity=equity,
            positions=positions,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=total_unrealized,
            strategy_id=self.strategy_id,
        )

    def position_shares(self, symbol: str) -> float:
        """Total shares held in a symbol."""
        return sum(l.shares for l in self.lots if l.symbol == symbol)

    def held_symbols(self) -> set:
        """Symbols currently held."""
        return {l.symbol for l in self.lots if l.shares > 1e-9}

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
    Fill, Lot, Order, OrderSide, Position, PortfolioSnapshot, ShortConfig,
)

logger = logging.getLogger(__name__)

#: Share quantities at or below this are treated as zero. A lot this small cannot
#: be traded and is worth fractions of a cent; carrying it only makes
#: ``held_symbols`` disagree with ``snapshot``.
DUST = 1e-9


class Portfolio:
    """Mutable portfolio state. The engine owns one instance per simulation."""

    def __init__(self, initial_cash: float, strategy_id: str = ""):
        self.cash: float = initial_cash
        self.lots: List[Lot] = []
        self.realized_pnl: float = 0.0
        self.borrow_paid: float = 0.0
        self.strategy_id = strategy_id
        self._last_borrow_accrual: Optional[pd.Timestamp] = None

    def apply_fill(self, fill: Fill) -> float:
        """Process a fill. Returns realized PnL.

        Cash moves the same way regardless of direction: a buy debits
        ``price * shares + commission``, a sell credits ``price * shares -
        commission``. That is already correct for the short side -- opening a short
        credits the proceeds, covering debits them back -- so no special case is
        needed for cash.

        The position side does need care, because one fill can do two things. A
        buy of 150 against a short of 100 covers the short *and* opens a 50-share
        long, realizing PnL on the first 100 and none on the rest. Treating it as
        either one or the other loses money that was made or invents money that
        was not.
        """
        signed = (
            fill.fill_shares
            if fill.order.side == OrderSide.BUY
            else -fill.fill_shares
        )

        if fill.order.side == OrderSide.BUY:
            self.cash -= fill.fill_price * fill.fill_shares + fill.commission
        else:
            self.cash += fill.fill_price * fill.fill_shares - fill.commission

        realized = self._apply_signed(
            symbol=fill.order.symbol,
            signed_shares=signed,
            price=fill.fill_price,
            fill_date=fill.fill_date,
            strategy_id=fill.order.strategy_id or None,
        )
        self.realized_pnl += realized
        return realized

    def _apply_signed(
        self,
        symbol: str,
        signed_shares: float,
        price: float,
        fill_date: pd.Timestamp,
        strategy_id: Optional[str] = None,
    ) -> float:
        """Close opposing lots FIFO, then open a lot with whatever is left.

        ``strategy_id`` restricts the operation to one sleeve's lots. Without it a
        fund sleeve could consume another sleeve's lots in the same symbol, and
        both would then disagree with the ledger about what they hold.

        Realized PnL is ``(price - entry) * signed_shares_closed``, which is
        correct in both directions without a branch: a short lot carries negative
        shares, so covering at a lower price than entry multiplies two negatives
        into a gain.
        """
        remaining = float(signed_shares)
        realized = 0.0
        kept: List[Lot] = []

        for lot in self.lots:
            owned = strategy_id is None or lot.strategy_id == strategy_id
            opposing = lot.shares * remaining < 0

            if lot.symbol != symbol or not owned or not opposing or abs(remaining) <= DUST:
                kept.append(lot)
                continue

            lot_sign = 1.0 if lot.shares > 0 else -1.0
            closed = min(abs(remaining), abs(lot.shares))

            realized += (price - lot.entry_price) * (closed * lot_sign)
            remaining += closed * lot_sign

            residual = lot.shares - closed * lot_sign
            if abs(residual) > DUST:
                kept.append(Lot(
                    symbol=lot.symbol,
                    shares=residual,
                    entry_price=lot.entry_price,
                    entry_date=lot.entry_date,
                    strategy_id=lot.strategy_id,
                    lot_id=lot.lot_id,
                ))

        self.lots = kept

        if abs(remaining) > DUST:
            self.lots.append(Lot(
                symbol=symbol,
                shares=remaining,
                entry_price=price,
                entry_date=fill_date,
                strategy_id=strategy_id or "",
                lot_id=uuid4().hex[:12],
            ))

        return realized

    # -----------------------------------------------------------------------
    # Short-side carry
    # -----------------------------------------------------------------------

    def accrue_borrow_fees(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float],
        config: ShortConfig,
    ) -> float:
        """Charge the borrow on every short position. Returns the amount debited.

        Accrued on *calendar* days elapsed, not sessions, because a borrow is
        charged over a weekend. Using sessions would understate the annual carry
        by roughly 30% -- on a 20% hard-to-borrow name held all year that is six
        percentage points of return conjured out of the calendar.

        The first call only sets the clock; there is nothing to charge before a
        position has been held for a day.
        """
        previous, self._last_borrow_accrual = self._last_borrow_accrual, date
        if previous is None:
            return 0.0

        days = (pd.Timestamp(date) - pd.Timestamp(previous)).days
        if days <= 0:
            return 0.0

        charge = 0.0
        rebate = 0.0
        for symbol, shares in self._net_by_symbol().items():
            if shares >= -DUST:
                continue
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            value = abs(shares) * price
            charge += value * config.borrow_rate(symbol) * days / 365.0
            if config.short_proceeds_earn_interest:
                rebate += value * config.short_rebate_annual * days / 365.0

        net = charge - rebate
        if abs(net) > 0:
            self.cash -= net
            self.borrow_paid += net
        return net

    def _net_by_symbol(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for lot in self.lots:
            totals[lot.symbol] = totals.get(lot.symbol, 0.0) + lot.shares
        return totals

    def short_symbols(self, strategy_id: Optional[str] = None) -> set:
        """Symbols currently held short."""
        totals: Dict[str, float] = {}
        for lot in self.lots:
            if strategy_id is not None and lot.strategy_id != strategy_id:
                continue
            totals[lot.symbol] = totals.get(lot.symbol, 0.0) + lot.shares
        return {s for s, shares in totals.items() if shares < -DUST}

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
            # abs, not >0: a short nets negative and the long-only form silently
            # dropped every short position from the snapshot, so a strategy saw
            # itself as flat while the ledger carried the risk.
            if abs(total_shares) <= DUST:
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
        """Net shares held in a symbol. Negative means short."""
        return sum(l.shares for l in self.lots if l.symbol == symbol)

    def drop_dust(self, tolerance: float = DUST) -> List[str]:
        """Discard positions too small to be tradable. Returns symbols dropped.

        ``snapshot`` already hides a sub-tolerance holding, so it is invisible to
        a strategy while still sitting in ``self.lots`` -- which makes
        ``held_symbols`` and the snapshot disagree. Anything this small cannot be
        traded and is worth fractions of a cent.

        Measured on the *net* magnitude, so a long and a short of the same size in
        one symbol collapse to nothing rather than leaving two live lots that
        offset. Offsetting lots would still accrue borrow on the short leg.
        """
        dust = {
            symbol for symbol, shares in self._net_by_symbol().items()
            if abs(shares) <= tolerance
        }
        if not dust:
            return []

        self.lots = [l for l in self.lots if l.symbol not in dust]
        return sorted(dust)

    def held_symbols(self) -> set:
        """Symbols with an open position, long or short."""
        return {
            symbol for symbol, shares in self._net_by_symbol().items()
            if abs(shares) > DUST
        }

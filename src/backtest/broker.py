"""Simulated broker — converts orders into fills.

Default execution convention:
  - Signals computed from data through close of day D
  - Orders fill at open of day D+1
  - Market orders with configurable slippage
  - Participation cap: no order above X% of daily volume
  - Whole shares by default
  - Delisting: liquidate at last close with configurable haircut
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import numpy as np

from backtest.types import (
    CostConfig, ExecutionConfig, Fill, FillType, Order, OrderSide,
)
from backtest import costs

logger = logging.getLogger(__name__)


def adjusted_bars(
    panel: pd.DataFrame, date: pd.Timestamp
) -> Dict[str, Dict[str, float]]:
    """Extract {symbol: {open, high, low, close, adj_close, volume}} for a date.

    OHLC is rescaled onto the adjusted scale. The lake stores the unadjusted
    print plus a back-adjusted ``adj_close``, while mark-to-market runs off
    ``close_matrix`` (= ``adj_close``). Filling at the raw open and marking at
    ``adj_close`` books the whole corporate-action factor as instant PnL: a bar
    taken before a 2-for-1 split fills 100 shares at $100 and marks them at $50.

    yfinance scales every OHLC field by the same factor, so the adjusted series
    is ``raw * adj_close / close`` (see
    ``data_pipeline.store.load_ohlc_adjusted``). Volume is scaled the other way,
    because the share counts compared against it are adjusted shares.

    Each bar also carries the untouched print (``raw_open``, ``raw_close``) and
    the ``factor`` used. ``adj_close`` is back-adjusted using every corporate
    action up to the *download* date, so adjusted levels on a historical date
    know about splits that had not happened yet. Returns are unaffected, but
    anything compared against an absolute dollar amount -- a minimum-price
    screen, whole-share rounding -- must use the raw print instead. The raw
    fields are what make that possible without a second data load.
    """
    day = panel[panel["date"] == date]
    bars: Dict[str, Dict[str, float]] = {}

    for _, row in day.iterrows():
        raw_open = float(row.get("open", 0) or 0)
        raw_close = float(row.get("close", 0) or 0)
        adj_close = float(row.get("adj_close", 0) or 0)
        factor = adj_close / raw_close if raw_close > 0 and adj_close > 0 else 1.0
        volume = float(row.get("volume", 0) or 0)

        bars[row["symbol"]] = {
            "open": raw_open * factor,
            "high": float(row.get("high", 0)) * factor,
            "low": float(row.get("low", 0)) * factor,
            "close": raw_close * factor,
            "adj_close": adj_close,
            "volume": volume / factor if factor > 0 else volume,
            "raw_open": raw_open,
            "raw_close": raw_close,
            "factor": factor,
        }

    return bars


#: Tolerance when converting an adjusted share count back to raw shares.
#: A position is always built from whole raw shares, so ``shares * factor``
#: should land on an integer -- but it arrives there through a divide and a
#: multiply, so it lands on 117.99999999998 instead. Flooring that gives 117 and
#: strands a share. See ``_floor_whole_shares``.
SHARE_EPSILON = 1e-6


def _floor_whole_shares(adjusted_shares: float, factor: float) -> float:
    """Round down to a whole *raw* share, expressed back in adjusted shares.

    The engine trades in adjusted shares so that corporate actions need no
    explicit handling, but the exchange only ever filled whole raw shares.
    ``raw = adjusted * factor``, so the integrality constraint lives in raw
    space. Flooring the adjusted count instead would quietly permit fractions
    of a real share wherever ``factor < 1`` -- 2010 AAPL has a factor near
    0.03, so one "adjusted share" is ~0.03 of a share actually buyable.

    The epsilon is not cosmetic. A fill stores ``raw / factor`` adjusted shares,
    so a later full exit passes that number back and ``adjusted * factor`` misses
    the integer by a float residue. Flooring it sold one share fewer than the
    position held, every time, and the leftover fraction meant the position never
    closed: the engine re-issued the same exit order the next session, and the
    next, while the strategy logged the same exit reason for months. A Clenow run
    over 641 sessions produced 33 fills and sat in near-cash at beta 0.04 because
    of it.
    """
    if factor <= 0 or not np.isfinite(factor):
        return float(math.floor(adjusted_shares + SHARE_EPSILON))
    raw_shares = math.floor(adjusted_shares * factor + SHARE_EPSILON)
    if raw_shares <= 0:
        return 0.0
    return raw_shares / factor


class SimulatedBroker:
    """Simulates order fills against historical OHLCV data."""

    def __init__(
        self,
        cost_config: CostConfig,
        execution_config: ExecutionConfig,
    ):
        self.cost = cost_config
        self.execution = execution_config

    def fill_orders(
        self,
        orders: List[Order],
        fill_date: pd.Timestamp,
        bars: Dict[str, Dict[str, float]],
    ) -> List[Fill]:
        """Simulate fills for a batch of orders.
        
        Args:
            orders: orders to fill.
            fill_date: the date fills occur (D+1).
            bars: {symbol: {open, high, low, close, adj_close, volume}} for fill_date.
        
        Returns:
            List of Fill objects. Orders for missing symbols are dropped with warning.
        """
        fills = []

        for order in orders:
            bar = bars.get(order.symbol)
            if bar is None:
                logger.warning(
                    "No bar for %s on %s — order dropped",
                    order.symbol, fill_date,
                )
                continue

            fill = self._fill_single(order, fill_date, bar)
            if fill is not None:
                fills.append(fill)

        return fills

    def _fill_single(
        self,
        order: Order,
        fill_date: pd.Timestamp,
        bar: Dict[str, float],
    ) -> Optional[Fill]:
        """Fill a single order against a bar."""
        raw_price = self._execution_price(bar)
        if raw_price is None or raw_price <= 0:
            logger.warning("Invalid price for %s on %s", order.symbol, fill_date)
            return None

        fill_price = costs.compute_fill_price(raw_price, order.side, self.cost)

        if order.shares > 0:
            target_shares = order.shares
        elif order.notional > 0:
            target_shares = order.notional / fill_price
        else:
            return None

        # A full exit is exempt from integrality: whatever the position holds is
        # sellable by definition, and rounding it down leaves dust that keeps the
        # position open forever.
        if not self.execution.fractional_shares and not order.close_position:
            target_shares = _floor_whole_shares(target_shares, bar.get("factor", 1.0))

        if target_shares <= 0:
            return None

        volume = bar.get("volume", 0) or 0
        max_shares = costs.max_shares_for_volume(float(volume), self.cost)
        if target_shares > max_shares and max_shares > 0:
            logger.debug(
                "Participation cap: %s capped from %.0f to %.0f shares (vol=%.0f)",
                order.symbol, target_shares, max_shares, volume,
            )
            target_shares = (
                max_shares
                if self.execution.fractional_shares
                else _floor_whole_shares(max_shares, bar.get("factor", 1.0))
            )

        if target_shares <= 0:
            return None

        commission = costs.compute_commission(target_shares, self.cost)

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_shares=target_shares,
            fill_date=fill_date,
            commission=commission,
            slippage_bps=self.cost.total_one_way_bps,
            fill_type=self.execution.fill_type,
            fill_id=uuid4().hex[:12],
        )

    def _execution_price(self, bar: Dict[str, float]) -> Optional[float]:
        """Extract the execution price from a bar based on fill_type."""
        if self.execution.fill_type == FillType.MARKET_OPEN:
            return bar.get("open")
        elif self.execution.fill_type == FillType.MARKET_CLOSE:
            return bar.get("close")
        elif self.execution.fill_type == FillType.VWAP:
            o = bar.get("open", 0)
            c = bar.get("close", 0)
            if o > 0 and c > 0:
                return (o + c) / 2
            return bar.get("close")
        return bar.get("open")

    def handle_delisting(
        self,
        symbol: str,
        last_close: float,
        shares: float,
        date: pd.Timestamp,
        strategy_id: str = "",
    ) -> Optional[Fill]:
        """Simulate forced liquidation on delisting.
        
        Applies the delisting_return haircut to the last known close.
        """
        adjusted_price = last_close * (1 + self.execution.delisting_return)
        adjusted_price = max(adjusted_price, 0.01)

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            notional=0.0,
            shares=shares,
            strategy_id=strategy_id,
            reason=f"delisting_haircut={self.execution.delisting_return:.0%}",
            created_date=date,
            order_id=f"delist_{uuid4().hex[:8]}",
        )

        return Fill(
            order=order,
            fill_price=adjusted_price,
            fill_shares=shares,
            fill_date=date,
            commission=0.0,
            slippage_bps=0.0,
            fill_type=FillType.MARKET_CLOSE,
            fill_id=f"delist_{uuid4().hex[:8]}",
        )

"""Hand-verifiable FIFO accounting tests for Portfolio.

Every assertion is a number computed on paper; the arithmetic is shown in a
comment next to it. Prices and share counts are round numbers so that float
results land exactly on the expected double.
"""
import pytest
import pandas as pd

from backtest.portfolio import Portfolio
from backtest.types import Fill, FillType, Order, OrderSide

DAY1 = pd.Timestamp("2024-01-02")
DAY2 = pd.Timestamp("2024-01-03")
DAY3 = pd.Timestamp("2024-01-04")


def _fill(side, shares, price, date=DAY1, symbol="AAPL", commission=0.0):
    """Build a Fill directly, bypassing the broker."""
    order = Order(
        symbol=symbol,
        side=side,
        notional=shares * price,
        shares=shares,
        strategy_id="test",
        created_date=date,
    )
    return Fill(
        order=order,
        fill_price=price,
        fill_shares=shares,
        fill_date=date,
        commission=commission,
        fill_type=FillType.MARKET_OPEN,
    )


class TestSingleBuy:
    def test_cash_shares_and_avg_entry(self):
        pf = Portfolio(initial_cash=10_000.0, strategy_id="test")
        pf.apply_fill(_fill(OrderSide.BUY, 100, 50.0))

        # 10000 - 100 * 50 = 5000
        assert pf.cash == 5000.0
        assert pf.position_shares("AAPL") == 100.0
        assert pf.realized_pnl == 0.0
        assert pf.held_symbols() == {"AAPL"}

        snap = pf.snapshot(DAY1, {"AAPL": 50.0})
        assert snap.positions["AAPL"].avg_entry_price == 50.0
        # cash 5000 + 100 * 50 = 10000, no PnL yet
        assert snap.equity == 10_000.0
        assert snap.unrealized_pnl == 0.0


class TestFullExit:
    def test_realized_pnl_and_cash_after_profitable_sell(self):
        pf = Portfolio(initial_cash=10_000.0, strategy_id="test")
        pf.apply_fill(_fill(OrderSide.BUY, 100, 50.0, date=DAY1))
        realized = pf.apply_fill(_fill(OrderSide.SELL, 100, 60.0, date=DAY2))

        # (60 - 50) * 100 = 1000
        assert realized == 1000.0
        assert pf.realized_pnl == 1000.0
        # 10000 - 5000 + 6000 = 11000
        assert pf.cash == 11_000.0
        assert pf.position_shares("AAPL") == 0.0
        assert pf.held_symbols() == set()


class TestFifoAcrossLots:
    """Two lots, one partial sell. FIFO must consume the older lot first."""

    def _two_lots_then_sell(self):
        pf = Portfolio(initial_cash=20_000.0, strategy_id="test")
        pf.apply_fill(_fill(OrderSide.BUY, 100, 50.0, date=DAY1))
        pf.apply_fill(_fill(OrderSide.BUY, 100, 60.0, date=DAY2))
        realized = pf.apply_fill(_fill(OrderSide.SELL, 150, 70.0, date=DAY3))
        return pf, realized

    def test_realized_pnl_follows_fifo_not_average_cost(self):
        pf, realized = self._two_lots_then_sell()

        # FIFO: all 100 of lot 1 at (70 - 50) * 100 = 2000
        #       50 of lot 2      at (70 - 60) * 50  = 500
        # total = 2500. Average cost would have given (70 - 55) * 150 = 2250.
        assert realized == 2500.0
        assert pf.realized_pnl == 2500.0
        # 20000 - 5000 - 6000 + 150 * 70 = 20000 - 11000 + 10500 = 19500
        assert pf.cash == 19_500.0
        # 200 bought - 150 sold = 50
        assert pf.position_shares("AAPL") == 50.0

    def test_partial_lot_keeps_original_entry_price(self):
        pf, _ = self._two_lots_then_sell()

        remaining = [lot for lot in pf.lots if lot.symbol == "AAPL"]
        assert len(remaining) == 1
        # The survivor is the remnant of lot 2: entry price stays 60, not
        # re-averaged to 55 or marked up to the 70 sell price.
        assert remaining[0].shares == 50.0
        assert remaining[0].entry_price == 60.0
        assert remaining[0].entry_date == DAY2

        snap = pf.snapshot(DAY3, {"AAPL": 60.0})
        assert snap.positions["AAPL"].avg_entry_price == 60.0
        # 50 * 60 - 50 * 60 = 0
        assert snap.unrealized_pnl == 0.0


class TestCommission:
    def test_commission_deducted_on_both_sides(self):
        pf = Portfolio(initial_cash=10_000.0, strategy_id="test")
        # BUY: cash -= price * shares + commission
        pf.apply_fill(_fill(OrderSide.BUY, 100, 50.0, date=DAY1, commission=1.0))
        # 10000 - (100 * 50 + 1.00) = 4999.00
        assert pf.cash == 4999.0

        # SELL: cash += price * shares - commission
        realized = pf.apply_fill(
            _fill(OrderSide.SELL, 100, 60.0, date=DAY2, commission=2.5)
        )
        # 4999 + (100 * 60 - 2.50) = 4999 + 5997.50 = 10996.50
        assert pf.cash == 10_996.50

        # Round trip cost 1.00 + 2.50 = 3.50, so cash gain is 1000 - 3.50.
        assert pf.cash - 10_000.0 == 996.50
        # realized_pnl is gross of commission by design: price delta only.
        assert realized == 1000.0


class TestUnrealizedPnl:
    def test_snapshot_marks_to_market(self):
        pf = Portfolio(initial_cash=10_000.0, strategy_id="test")
        pf.apply_fill(_fill(OrderSide.BUY, 100, 50.0))

        snap = pf.snapshot(DAY2, {"AAPL": 55.0})

        # 100 * 55 - 100 * 50 = 500
        assert snap.unrealized_pnl == 500.0
        assert snap.positions["AAPL"].market_value == 5500.0
        # cash 5000 + 5500 = 10500
        assert snap.cash == 5000.0
        assert snap.equity == 10_500.0
        assert snap.equity == snap.cash + 5500.0
        # 5500 / 10500
        assert snap.weights["AAPL"] == pytest.approx(5500.0 / 10_500.0)

    def test_missing_price_falls_back_to_entry(self):
        pf = Portfolio(initial_cash=10_000.0, strategy_id="test")
        pf.apply_fill(_fill(OrderSide.BUY, 100, 50.0))

        snap = pf.snapshot(DAY2, {})

        # No quote: marked at cost, so unrealized is exactly 0 and equity
        # is unchanged at 5000 + 100 * 50 = 10000.
        assert snap.positions["AAPL"].market_price == 50.0
        assert snap.unrealized_pnl == 0.0
        assert snap.equity == 10_000.0

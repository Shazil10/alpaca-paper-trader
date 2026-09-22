"""Hand-verifiable fill tests for SimulatedBroker.

Covers the four things that separate a simulated fill from a free one:
slippage, whole-share rounding, the participation cap, and missing data.
Plus the delisting haircut. Every expected number is shown as arithmetic.
"""
import pytest
import pandas as pd

from backtest.broker import SimulatedBroker, adjusted_bars
from backtest.types import (
    CostConfig, ExecutionConfig, FillType, Order, OrderSide,
)

FILL_DATE = pd.Timestamp("2024-01-03")

FREE = CostConfig(
    spread_bps=0.0,
    slippage_bps=0.0,
    commission_per_share=0.0,
    min_commission=0.0,
    participation_rate=1.0,
)


def _bar(price=100.0, volume=1_000_000):
    """A flat bar: open == high == low == close, so fill_type does not matter."""
    return {
        "open": price, "high": price, "low": price,
        "close": price, "adj_close": price, "volume": float(volume),
    }


def _order(side=OrderSide.BUY, symbol="AAPL", notional=0.0, shares=0.0):
    return Order(
        symbol=symbol,
        side=side,
        notional=notional,
        shares=shares,
        strategy_id="test",
        created_date=pd.Timestamp("2024-01-02"),
    )


class TestParticipationCap:
    def test_order_above_cap_is_partially_filled(self):
        cost = CostConfig(
            spread_bps=0.0, slippage_bps=0.0, commission_per_share=0.0,
            participation_rate=0.05,
        )
        broker = SimulatedBroker(cost, ExecutionConfig(fractional_shares=False))

        fills = broker.fill_orders(
            [_order(shares=1000)], FILL_DATE, {"AAPL": _bar(100.0, volume=10_000)}
        )

        assert len(fills) == 1
        # 10000 shares * 0.05 = 500 max, so a 1000-share order fills 500.
        assert fills[0].fill_shares == 500.0

    def test_order_below_cap_fills_in_full(self):
        cost = CostConfig(
            spread_bps=0.0, slippage_bps=0.0, commission_per_share=0.0,
            participation_rate=0.05,
        )
        broker = SimulatedBroker(cost, ExecutionConfig(fractional_shares=False))

        fills = broker.fill_orders(
            [_order(shares=400)], FILL_DATE, {"AAPL": _bar(100.0, volume=10_000)}
        )

        # 400 < 10000 * 0.05 = 500, untouched.
        assert fills[0].fill_shares == 400.0


class TestShareRounding:
    def test_whole_shares_floor_the_target(self):
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=False))

        fills = broker.fill_orders(
            [_order(notional=1070.0)], FILL_DATE, {"AAPL": _bar(100.0)}
        )

        # 1070 / 100 = 10.7 -> floor -> 10 shares, 70 dollars left undeployed.
        assert fills[0].fill_shares == 10.0

    def test_fractional_shares_fill_the_exact_amount(self):
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=True))

        fills = broker.fill_orders(
            [_order(notional=1070.0)], FILL_DATE, {"AAPL": _bar(100.0)}
        )

        # 1070 / 100 = 10.7 exactly
        assert fills[0].fill_shares == 10.7

    def test_notional_below_one_share_produces_no_fill(self):
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=False))

        fills = broker.fill_orders(
            [_order(notional=50.0)], FILL_DATE, {"AAPL": _bar(100.0)}
        )

        # 50 / 100 = 0.5 -> floor -> 0 shares -> dropped.
        assert fills == []


class TestSlippageDirection:
    """Cost is spread_bps / 2 + slippage_bps, one way, against the trader."""

    COST = CostConfig(
        spread_bps=4.0, slippage_bps=3.0, commission_per_share=0.0,
        participation_rate=1.0,
    )

    def test_buy_fills_above_raw_price(self):
        broker = SimulatedBroker(self.COST, ExecutionConfig(fractional_shares=True))

        fills = broker.fill_orders(
            [_order(OrderSide.BUY, shares=10)], FILL_DATE, {"AAPL": _bar(100.0)}
        )

        # 4/2 + 3 = 5 bps = 0.0005; 100 * 1.0005 = 100.05
        assert fills[0].fill_price == 100.05
        assert fills[0].slippage_bps == 5.0

    def test_sell_fills_below_raw_price(self):
        broker = SimulatedBroker(self.COST, ExecutionConfig(fractional_shares=True))

        fills = broker.fill_orders(
            [_order(OrderSide.SELL, shares=10)], FILL_DATE, {"AAPL": _bar(100.0)}
        )

        # 100 * (1 - 0.0005) = 99.95
        assert fills[0].fill_price == 99.95

    def test_round_trip_loses_exactly_twice_the_one_way_cost(self):
        broker = SimulatedBroker(self.COST, ExecutionConfig(fractional_shares=True))
        bars = {"AAPL": _bar(100.0)}

        buy = broker.fill_orders([_order(OrderSide.BUY, shares=10)], FILL_DATE, bars)[0]
        sell = broker.fill_orders([_order(OrderSide.SELL, shares=10)], FILL_DATE, bars)[0]

        # (100.05 - 99.95) * 10 = 1.00 on 1000.50 deployed
        assert (buy.fill_price - sell.fill_price) * 10 == pytest.approx(1.0, abs=1e-9)


class TestCommissionOnFill:
    def test_per_share_commission(self):
        cost = CostConfig(
            spread_bps=0.0, slippage_bps=0.0,
            commission_per_share=0.005, participation_rate=1.0,
        )
        broker = SimulatedBroker(cost, ExecutionConfig(fractional_shares=False))

        fills = broker.fill_orders(
            [_order(shares=200)], FILL_DATE, {"AAPL": _bar(100.0)}
        )

        # 200 * 0.005 = 1.00
        assert fills[0].commission == pytest.approx(1.0, abs=1e-12)


class TestMissingData:
    def test_missing_bar_drops_the_order_without_raising(self):
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=True))

        fills = broker.fill_orders(
            [_order(symbol="MSFT", shares=10)], FILL_DATE, {"AAPL": _bar(100.0)}
        )

        assert fills == []

    def test_one_missing_symbol_does_not_block_the_others(self):
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=True))

        fills = broker.fill_orders(
            [_order(symbol="MSFT", shares=10), _order(symbol="AAPL", shares=10)],
            FILL_DATE,
            {"AAPL": _bar(100.0)},
        )

        assert len(fills) == 1
        assert fills[0].order.symbol == "AAPL"

    def test_zero_price_bar_produces_no_fill(self):
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=True))

        fills = broker.fill_orders(
            [_order(shares=10)], FILL_DATE, {"AAPL": _bar(0.0)}
        )

        assert fills == []


class TestAdjustedBars:
    """Bars handed to the broker must be on the same scale as the mark."""

    def _panel(self, close, adj_close, open_=None, volume=1_000_000.0):
        return pd.DataFrame([{
            "date": FILL_DATE, "symbol": "AAPL",
            "open": open_ if open_ is not None else close,
            "high": close, "low": close,
            "close": close, "adj_close": adj_close, "volume": volume,
        }])

    def test_unadjusted_bar_passes_through(self):
        bars = adjusted_bars(self._panel(close=100.0, adj_close=100.0), FILL_DATE)

        # factor = 100 / 100 = 1
        assert bars["AAPL"]["open"] == 100.0
        assert bars["AAPL"]["close"] == 100.0
        assert bars["AAPL"]["volume"] == 1_000_000.0

    def test_pre_split_bar_is_halved(self):
        # Raw print $100 with adj_close $50: a 2-for-1 split happened later.
        bars = adjusted_bars(self._panel(close=100.0, adj_close=50.0), FILL_DATE)

        # factor = 50 / 100 = 0.5
        assert bars["AAPL"]["open"] == 50.0
        assert bars["AAPL"]["close"] == 50.0
        assert bars["AAPL"]["adj_close"] == 50.0
        # Adjusted share counts are doubled, so the tradeable volume is too:
        # 1_000_000 / 0.5 = 2_000_000
        assert bars["AAPL"]["volume"] == 2_000_000.0

    def test_open_keeps_its_gap_to_close_after_rescaling(self):
        bars = adjusted_bars(
            self._panel(close=100.0, adj_close=50.0, open_=110.0), FILL_DATE
        )

        # 110 * 0.5 = 55, i.e. the +10% gap survives the rescale.
        assert bars["AAPL"]["open"] == 55.0
        assert bars["AAPL"]["open"] / bars["AAPL"]["close"] == pytest.approx(1.1)

    def test_zero_close_falls_back_to_factor_one(self):
        bars = adjusted_bars(self._panel(close=0.0, adj_close=50.0), FILL_DATE)

        # No usable factor: pass the raw values through rather than divide by 0.
        assert bars["AAPL"]["volume"] == 1_000_000.0

    def test_missing_date_returns_empty(self):
        panel = self._panel(close=100.0, adj_close=100.0)
        assert adjusted_bars(panel, pd.Timestamp("2024-06-03")) == {}


class TestDelisting:
    def test_haircut_applied_to_last_close(self):
        broker = SimulatedBroker(
            FREE, ExecutionConfig(fractional_shares=True, delisting_return=-0.30)
        )

        fill = broker.handle_delisting(
            "DEAD", last_close=100.0, shares=50.0, date=FILL_DATE, strategy_id="test"
        )

        # 100 * (1 - 0.30) = 70.00
        assert fill.fill_price == 70.0
        assert fill.fill_shares == 50.0
        # 50 * 70 = 3500 recovered instead of 50 * 100 = 5000
        assert fill.fill_price * fill.fill_shares == 3500.0
        assert fill.order.side == OrderSide.SELL
        assert fill.commission == 0.0
        assert fill.fill_type == FillType.MARKET_CLOSE

    def test_haircut_floors_at_a_penny(self):
        broker = SimulatedBroker(
            FREE, ExecutionConfig(fractional_shares=True, delisting_return=-1.0)
        )

        fill = broker.handle_delisting(
            "DEAD", last_close=100.0, shares=50.0, date=FILL_DATE
        )

        # 100 * 0 = 0, floored at 0.01 so the position never marks negative.
        assert fill.fill_price == 0.01

"""Short-side accounting, with every expected number shown as arithmetic.

Shorts are where signs go wrong quietly. A long-only bug shows up as an absurd
return; a sign error on the short leg produces a plausible-looking equity curve
that is exactly backwards, and no summary statistic reveals it. So these tests are
mostly hand-computed rather than property-based.

Five things have to hold:

* **Equity is unchanged at the moment of sale.** Shorting $5,000 credits $5,000 of
  cash and books -$5,000 of market value. If it moves, the proceeds are being
  double-counted.
* **PnL has the right sign.** A short makes money when the price falls.
* **One fill can cover and flip.** Buying 150 against a short of 100 realizes PnL
  on 100 and opens a 50-share long, and loses money if treated as either alone.
* **Borrow is charged on calendar days**, not sessions, or a year of carry comes
  out ~30% light.
* **Nothing changes for long-only.** ``allow_shorts`` defaults off and a negative
  target is dropped, not reinterpreted.

Run with: ./venv/bin/python -m pytest tests/backtest/test_shorts.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import generate_orders
from backtest.portfolio import Portfolio
from backtest.types import (
    CostConfig, Fill, FillType, Order, OrderSide, PortfolioSnapshot, Position,
    RiskConfig, ShortConfig,
)
from backtest import risk

DAY = pd.Timestamp("2024-01-03")
LATER = pd.Timestamp("2024-01-10")

SHORTS_ON = ShortConfig(allow_shorts=True, borrow_rate_annual=0.0)


def _fill(symbol, side, shares, price, date=DAY, commission=0.0, strategy_id="s"):
    return Fill(
        order=Order(
            symbol=symbol, side=side, notional=0.0, shares=shares,
            strategy_id=strategy_id, created_date=date,
        ),
        fill_price=price,
        fill_shares=shares,
        fill_date=date,
        commission=commission,
        fill_type=FillType.MARKET_OPEN,
    )


# ---------------------------------------------------------------------------
# Opening a short
# ---------------------------------------------------------------------------

class TestOpeningAShort:
    def test_equity_is_unchanged_at_the_moment_of_sale(self):
        """Short 100 at $50: cash 100,000 -> 105,000, market value -5,000.

        Equity must still be 100,000. If it rose, the proceeds are being counted
        as profit -- the single most common short-accounting error.
        """
        pf = Portfolio(100_000.0)

        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))
        snap = pf.snapshot(DAY, {"AAA": 50.0})

        assert pf.cash == pytest.approx(105_000.0)
        assert snap.positions["AAA"].shares == pytest.approx(-100.0)
        assert snap.positions["AAA"].market_value == pytest.approx(-5_000.0)
        assert snap.equity == pytest.approx(100_000.0)

    def test_the_position_is_visible_in_the_snapshot(self):
        """The long-only form dropped shorts on a `shares <= 0` guard, so a
        strategy saw itself as flat while the ledger carried the risk."""
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        snap = pf.snapshot(DAY, {"AAA": 50.0})

        assert "AAA" in snap.positions
        assert "AAA" in pf.held_symbols()
        assert pf.short_symbols() == {"AAA"}
        assert snap.positions["AAA"].is_short

    def test_entry_price_is_the_sale_price(self):
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        position = pf.snapshot(DAY, {"AAA": 50.0}).positions["AAA"]

        assert position.avg_entry_price == pytest.approx(50.0)

    def test_gross_and_net_exposure_differ(self):
        """A market-neutral book nets to zero at 2x gross. Anything checking
        leverage on the net sum would call it flat."""
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("LONG", OrderSide.BUY, 100, 50.0))
        pf.apply_fill(_fill("SHRT", OrderSide.SELL, 100, 50.0))

        snap = pf.snapshot(DAY, {"LONG": 50.0, "SHRT": 50.0})

        assert snap.market_value == pytest.approx(0.0)
        assert snap.gross_value == pytest.approx(10_000.0)
        assert snap.long_value == pytest.approx(5_000.0)
        assert snap.short_value == pytest.approx(5_000.0)


# ---------------------------------------------------------------------------
# Profit and loss
# ---------------------------------------------------------------------------

class TestShortPnL:
    def test_a_falling_price_is_an_unrealized_gain(self):
        """Short 100 at $50, price falls to $40: market value -4,000, so equity
        is 105,000 - 4,000 = 101,000. A $1,000 gain on a $10 fall."""
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        snap = pf.snapshot(LATER, {"AAA": 40.0})

        assert snap.positions["AAA"].unrealized_pnl == pytest.approx(1_000.0)
        assert snap.equity == pytest.approx(101_000.0)

    def test_a_rising_price_is_an_unrealized_loss(self):
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        snap = pf.snapshot(LATER, {"AAA": 60.0})

        assert snap.positions["AAA"].unrealized_pnl == pytest.approx(-1_000.0)
        assert snap.equity == pytest.approx(99_000.0)

    def test_covering_lower_realizes_a_gain(self):
        """Short 100 at $50, cover at $40: realized (50-40)*100 = +1,000."""
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        realized = pf.apply_fill(_fill("AAA", OrderSide.BUY, 100, 40.0, LATER))

        assert realized == pytest.approx(1_000.0)
        assert pf.cash == pytest.approx(101_000.0)
        assert pf.position_shares("AAA") == pytest.approx(0.0)

    def test_covering_higher_realizes_a_loss(self):
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        realized = pf.apply_fill(_fill("AAA", OrderSide.BUY, 100, 60.0, LATER))

        assert realized == pytest.approx(-1_000.0)
        assert pf.cash == pytest.approx(99_000.0)

    def test_a_partial_cover_leaves_the_rest_short(self):
        """Short 100 at $50, buy 40 at $45: realized (50-45)*40 = +200, 60 left."""
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        realized = pf.apply_fill(_fill("AAA", OrderSide.BUY, 40, 45.0, LATER))

        assert realized == pytest.approx(200.0)
        assert pf.position_shares("AAA") == pytest.approx(-60.0)

    def test_commission_is_charged_on_both_legs(self):
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0, commission=5.0))
        pf.apply_fill(_fill("AAA", OrderSide.BUY, 100, 50.0, LATER, commission=5.0))

        # Flat price, so the only change is two commissions.
        assert pf.cash == pytest.approx(99_990.0)


# ---------------------------------------------------------------------------
# One fill doing two things
# ---------------------------------------------------------------------------

class TestCoverAndFlip:
    def test_a_buy_can_cover_a_short_and_open_a_long(self):
        """Short 100 at $50, then buy 150 at $40.

        Realized is (50-40)*100 = +1,000 on the covered part and nothing on the
        rest, leaving a 50-share long at $40. Treating the fill as purely a cover
        would strand 50 shares; treating it as purely a buy would lose the $1,000.
        """
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        realized = pf.apply_fill(_fill("AAA", OrderSide.BUY, 150, 40.0, LATER))

        assert realized == pytest.approx(1_000.0)
        assert pf.position_shares("AAA") == pytest.approx(50.0)
        position = pf.snapshot(LATER, {"AAA": 40.0}).positions["AAA"]
        assert position.avg_entry_price == pytest.approx(40.0)
        assert not position.is_short

    def test_a_sell_can_close_a_long_and_open_a_short(self):
        """Long 100 at $40, sell 150 at $50: realized +1,000, 50 short at $50."""
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.BUY, 100, 40.0))

        realized = pf.apply_fill(_fill("AAA", OrderSide.SELL, 150, 50.0, LATER))

        assert realized == pytest.approx(1_000.0)
        assert pf.position_shares("AAA") == pytest.approx(-50.0)
        assert pf.snapshot(LATER, {"AAA": 50.0}).positions["AAA"].is_short

    def test_fifo_order_is_respected_across_short_lots(self):
        """Two short lots at $50 and $60; buying 100 must close the $50 lot first.

        Realized = (50-45)*100 = +500, not (60-45)*100 = +1,500.
        """
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 60.0, LATER))

        realized = pf.apply_fill(_fill("AAA", OrderSide.BUY, 100, 45.0, LATER))

        assert realized == pytest.approx(500.0)
        assert pf.position_shares("AAA") == pytest.approx(-100.0)

    def test_offsetting_long_and_short_collapse_to_nothing(self):
        """Otherwise two live lots offset while the short leg still accrues borrow."""
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.BUY, 100, 50.0))
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 50.0))

        assert pf.position_shares("AAA") == pytest.approx(0.0)
        assert pf.held_symbols() == set()


# ---------------------------------------------------------------------------
# Borrow
# ---------------------------------------------------------------------------

class TestBorrow:
    def test_borrow_accrues_on_calendar_days_not_sessions(self):
        """Friday to Monday is three days of carry, not one.

        Charging per session understates a year of borrow by roughly 30% -- on a
        20% hard-to-borrow name that is six points of return invented by the
        calendar.
        """
        config = ShortConfig(allow_shorts=True, borrow_rate_annual=0.365)
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 100.0, pd.Timestamp("2024-01-05")))

        pf.accrue_borrow_fees(pd.Timestamp("2024-01-05"), {"AAA": 100.0}, config)
        charged = pf.accrue_borrow_fees(
            pd.Timestamp("2024-01-08"), {"AAA": 100.0}, config
        )

        # $10,000 short at 36.5%/yr = $10/day, over 3 calendar days = $30.
        assert charged == pytest.approx(30.0)

    def test_the_first_call_only_starts_the_clock(self):
        config = ShortConfig(allow_shorts=True, borrow_rate_annual=0.365)
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 100.0))

        assert pf.accrue_borrow_fees(DAY, {"AAA": 100.0}, config) == 0.0

    def test_hard_to_borrow_names_cost_more(self):
        config = ShortConfig(
            allow_shorts=True,
            borrow_rate_annual=0.01,
            hard_to_borrow_rate_annual=0.365,
            hard_to_borrow=frozenset({"HTB"}),
        )

        assert config.borrow_rate("HTB") == pytest.approx(0.365)
        assert config.borrow_rate("AAA") == pytest.approx(0.01)

    def test_longs_are_never_charged_borrow(self):
        config = ShortConfig(allow_shorts=True, borrow_rate_annual=0.365)
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.BUY, 100, 100.0))

        pf.accrue_borrow_fees(DAY, {"AAA": 100.0}, config)
        charged = pf.accrue_borrow_fees(LATER, {"AAA": 100.0}, config)

        assert charged == pytest.approx(0.0)

    def test_borrow_reduces_cash_and_is_tracked(self):
        config = ShortConfig(allow_shorts=True, borrow_rate_annual=0.365)
        pf = Portfolio(100_000.0)
        pf.apply_fill(_fill("AAA", OrderSide.SELL, 100, 100.0))
        before = pf.cash

        pf.accrue_borrow_fees(DAY, {"AAA": 100.0}, config)
        pf.accrue_borrow_fees(DAY + pd.Timedelta(days=1), {"AAA": 100.0}, config)

        assert pf.cash == pytest.approx(before - 10.0)
        assert pf.borrow_paid == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Margin
# ---------------------------------------------------------------------------

class TestMargin:
    def _snapshot(self, long_value, short_value, equity):
        positions = {}
        if long_value:
            positions["L"] = Position(
                symbol="L", shares=1.0, avg_entry_price=long_value,
                market_price=long_value, market_value=long_value,
                unrealized_pnl=0.0, entry_date=DAY,
            )
        if short_value:
            positions["S"] = Position(
                symbol="S", shares=-1.0, avg_entry_price=short_value,
                market_price=short_value, market_value=-short_value,
                unrealized_pnl=0.0, entry_date=DAY,
            )
        return PortfolioSnapshot(
            date=DAY, cash=equity - long_value + short_value,
            equity=equity, positions=positions,
        )

    def test_requirement_is_stricter_on_the_short_side(self):
        """25% on longs, 30% on shorts: losses on a short have no floor."""
        config = ShortConfig(allow_shorts=True)

        long_only = self._snapshot(10_000.0, 0.0, 10_000.0)
        short_only = self._snapshot(0.0, 10_000.0, 10_000.0)

        assert long_only.margin_requirement(config) == pytest.approx(2_500.0)
        assert short_only.margin_requirement(config) == pytest.approx(3_000.0)

    def test_gross_book_requires_both_legs(self):
        config = ShortConfig(allow_shorts=True)
        snapshot = self._snapshot(10_000.0, 10_000.0, 10_000.0)

        # 10,000 * 0.25 + 10,000 * 0.30
        assert snapshot.margin_requirement(config) == pytest.approx(5_500.0)


# ---------------------------------------------------------------------------
# Order generation
# ---------------------------------------------------------------------------

class TestOrderGeneration:
    def _snapshot(self, positions=None, cash=100_000.0, equity=100_000.0):
        return PortfolioSnapshot(
            date=DAY, cash=cash, equity=equity, positions=positions or {}
        )

    def _position(self, symbol, shares, price):
        return Position(
            symbol=symbol, shares=shares, avg_entry_price=price,
            market_price=price, market_value=shares * price,
            unrealized_pnl=0.0, entry_date=DAY,
        )

    def test_a_negative_target_is_dropped_when_shorts_are_off(self):
        """The default. Every long-only strategy behaves as it always did."""
        orders = generate_orders(
            {"AAA": -0.5}, self._snapshot(), {"AAA": 50.0}, "s",
            short_config=None,
        )

        assert orders == []

    def test_a_negative_target_becomes_a_sell_when_shorts_are_on(self):
        orders = generate_orders(
            {"AAA": -0.5}, self._snapshot(), {"AAA": 50.0}, "s",
            short_config=SHORTS_ON,
        )

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].notional == pytest.approx(50_000.0)

    def test_an_existing_short_is_closed_by_buying(self):
        snapshot = self._snapshot({"AAA": self._position("AAA", -100.0, 50.0)})

        orders = generate_orders(
            {}, snapshot, {"AAA": 50.0}, "s", short_config=SHORTS_ON
        )

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].shares == pytest.approx(100.0)
        assert orders[0].close_position

    def test_growing_a_short_sells_more(self):
        # Short 100 at 50 = -5,000 of 100,000 equity = -5%. Target -20%.
        snapshot = self._snapshot({"AAA": self._position("AAA", -100.0, 50.0)})

        orders = generate_orders(
            {"AAA": -0.20}, snapshot, {"AAA": 50.0}, "s", short_config=SHORTS_ON
        )

        assert orders[0].side == OrderSide.SELL
        assert orders[0].notional == pytest.approx(15_000.0)

    def test_shrinking_a_short_buys_back_shares_not_notional(self):
        """Sized in shares so a gap cannot push the position through zero."""
        snapshot = self._snapshot({"AAA": self._position("AAA", -100.0, 50.0)})

        orders = generate_orders(
            {"AAA": -0.025}, snapshot, {"AAA": 50.0}, "s", short_config=SHORTS_ON
        )

        assert orders[0].side == OrderSide.BUY
        assert orders[0].shares == pytest.approx(50.0)
        assert orders[0].notional == 0.0

    def test_flipping_side_closes_first_rather_than_crossing_zero(self):
        """One order carrying a position through zero overshoots on a gap."""
        snapshot = self._snapshot({"AAA": self._position("AAA", 100.0, 50.0)})

        orders = generate_orders(
            {"AAA": -0.30}, snapshot, {"AAA": 50.0}, "s", short_config=SHORTS_ON
        )

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].shares == pytest.approx(100.0)
        assert orders[0].close_position
        assert "flip_side" in orders[0].reason

    def test_long_only_behaviour_is_unchanged(self):
        """The regression guard for every existing strategy."""
        snapshot = self._snapshot({"AAA": self._position("AAA", 100.0, 50.0)})

        without = generate_orders(
            {"AAA": 0.20, "BBB": 0.10}, snapshot,
            {"AAA": 50.0, "BBB": 25.0}, "s", short_config=None,
        )
        with_off = generate_orders(
            {"AAA": 0.20, "BBB": 0.10}, snapshot,
            {"AAA": 50.0, "BBB": 25.0}, "s", short_config=ShortConfig(),
        )

        assert len(without) == len(with_off) == 2
        assert {o.side for o in without} == {OrderSide.BUY}


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------

class TestRiskLimits:
    def test_the_short_book_can_be_capped_separately_from_gross(self):
        config = RiskConfig(
            max_position_pct=1.0, max_sector_pct=1.0, cash_reserve_pct=0.0,
            max_leverage=2.0, max_short_pct=0.30,
        )
        snapshot = PortfolioSnapshot(date=DAY, cash=0.0, equity=100_000.0)

        adjusted = risk.apply_risk_limits(
            {"L": 1.0, "S1": -0.40, "S2": -0.20}, snapshot, config
        )

        assert sum(-w for w in adjusted.values() if w < 0) == pytest.approx(0.30)
        # The long leg is untouched, and the shorts keep their 2:1 ratio.
        assert adjusted["L"] == pytest.approx(1.0)
        assert adjusted["S1"] / adjusted["S2"] == pytest.approx(2.0)

    def test_gross_leverage_counts_both_sides(self):
        config = RiskConfig(
            max_position_pct=1.0, max_sector_pct=1.0, cash_reserve_pct=0.0,
            max_leverage=1.0,
        )
        snapshot = PortfolioSnapshot(date=DAY, cash=0.0, equity=100_000.0)

        adjusted = risk.apply_risk_limits({"L": 0.8, "S": -0.8}, snapshot, config)

        assert sum(abs(w) for w in adjusted.values()) == pytest.approx(1.0)

    def test_cash_reserve_is_measured_on_gross_not_net(self):
        """A market-neutral book nets to zero while consuming all the capital.

        On the net form the reserve would never bind.
        """
        config = RiskConfig(
            max_position_pct=1.0, max_sector_pct=1.0, cash_reserve_pct=0.10,
            max_leverage=1.0,
        )
        snapshot = PortfolioSnapshot(date=DAY, cash=0.0, equity=100_000.0)

        adjusted = risk.apply_risk_limits({"L": 0.5, "S": -0.5}, snapshot, config)

        assert sum(abs(w) for w in adjusted.values()) == pytest.approx(0.90)

    def test_position_cap_applies_to_shorts_by_magnitude(self):
        config = RiskConfig(
            max_position_pct=0.10, max_sector_pct=1.0, cash_reserve_pct=0.0,
            max_leverage=2.0,
        )
        snapshot = PortfolioSnapshot(date=DAY, cash=0.0, equity=100_000.0)

        adjusted = risk.apply_risk_limits({"S": -0.50}, snapshot, config)

        assert adjusted["S"] == pytest.approx(-0.10)

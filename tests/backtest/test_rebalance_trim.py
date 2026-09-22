"""The engine must sell down, not only buy up.

Target weights are a two-sided instruction. An engine that only buys toward
target and exits positions dropped from the target does not implement them: it
implements "buy toward target, never sell down". Gross exposure then ratchets,
because nothing ever reduces a position whose target *fell* -- whether because
a winner drifted above its weight, or because the strategy deliberately
de-levered.

This was a real bug. The DAF sleeve backtest reached 2.59x realized gross
against a 2.0x cap and sat at 2.22x through a month whose target gross was
1.00x, which makes every return number from that run describe a book more
levered than the strategy ever asked for.

Run with: ./venv/bin/python -m pytest tests/backtest/test_rebalance_trim.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import REBALANCE_BAND, BacktestEngine, generate_orders
from backtest.portfolio import Portfolio
from backtest.types import (
    BacktestConfig,
    CostConfig,
    ExecutionConfig,
    Fill,
    FillType,
    Order,
    OrderSide,
    RiskConfig,
)


def _portfolio_holding(symbol: str, shares: float, price: float, cash: float):
    """A portfolio holding `shares` of `symbol`, with `cash` left over.

    `cash` is the balance *after* the purchase, so the portfolio is seeded with
    enough to cover the fill. Passing the post-trade figure straight to
    Portfolio() would have it deducted a second time.
    """
    pf = Portfolio(cash + shares * price, "test")
    order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        notional=shares * price,
        shares=shares,
        strategy_id="test",
        created_date=pd.Timestamp("2024-01-02"),
        order_id="seed",
    )
    pf.apply_fill(Fill(
        order=order,
        fill_price=price,
        fill_shares=shares,
        fill_date=pd.Timestamp("2024-01-02"),
        fill_type=FillType.MARKET_OPEN,
        fill_id="seed",
    ))
    return pf


class TestTrimOrderGeneration:
    def test_overweight_position_is_trimmed(self):
        """Holding 100% against a 50% target must produce a SELL."""
        # 100 shares @ $100 = $10,000 held, $0 cash -> weight 1.0
        pf = _portfolio_holding("AAPL", 100, 100.0, cash=0.0)
        snap = pf.snapshot(pd.Timestamp("2024-01-03"), {"AAPL": 100.0})
        assert snap.equity == pytest.approx(10_000.0)
        assert snap.weights["AAPL"] == pytest.approx(1.0)

        orders = generate_orders({"AAPL": 0.5}, snap, {"AAPL": 100.0}, "test")

        sells = [o for o in orders if o.side == OrderSide.SELL]
        assert len(sells) == 1, "an overweight position must generate a trim"

        # Halving a 100-share position sells exactly 50 shares.
        assert sells[0].shares == pytest.approx(50.0)
        assert "trim" in sells[0].reason

    def test_trim_never_exceeds_the_position(self):
        """A trim must not sell more shares than are held (accidental short)."""
        pf = _portfolio_holding("AAPL", 100, 100.0, cash=0.0)
        snap = pf.snapshot(pd.Timestamp("2024-01-03"), {"AAPL": 100.0})

        # Target far below current, the case most likely to overshoot.
        orders = generate_orders({"AAPL": 0.01}, snap, {"AAPL": 100.0}, "test")

        for o in orders:
            if o.side == OrderSide.SELL:
                assert o.shares <= snap.positions["AAPL"].shares

    def test_underweight_still_buys(self):
        """The trim path must not break the ordinary buy path."""
        pf = _portfolio_holding("AAPL", 10, 100.0, cash=9_000.0)
        snap = pf.snapshot(pd.Timestamp("2024-01-03"), {"AAPL": 100.0})
        # $1,000 of AAPL against $10,000 equity -> weight 0.1

        orders = generate_orders({"AAPL": 0.5}, snap, {"AAPL": 100.0}, "test")

        buys = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buys) == 1
        # 0.5 - 0.1 = 0.4 of $10,000
        assert buys[0].notional == pytest.approx(4_000.0)

    def test_dead_band_applies_symmetrically(self):
        """Drift smaller than the band trades in neither direction."""
        pf = _portfolio_holding("AAPL", 100, 100.0, cash=0.0)
        snap = pf.snapshot(pd.Timestamp("2024-01-03"), {"AAPL": 100.0})

        # Just inside the band on the overweight side.
        target = 1.0 - (REBALANCE_BAND / 2)
        orders = generate_orders({"AAPL": target}, snap, {"AAPL": 100.0}, "test")
        assert orders == [], "drift inside the dead band must not trade"

    def test_exit_still_sells_whole_position(self):
        """Dropping a symbol entirely remains a full exit, not a trim."""
        pf = _portfolio_holding("AAPL", 100, 100.0, cash=0.0)
        snap = pf.snapshot(pd.Timestamp("2024-01-03"), {"AAPL": 100.0})

        orders = generate_orders({}, snap, {"AAPL": 100.0}, "test")

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].shares == pytest.approx(100.0)
        assert "exit" in orders[0].reason


class TestGrossExposureTracksTarget:
    """End-to-end: a strategy that de-levers must actually de-lever."""

    def test_de_levering_reduces_gross_exposure(self):
        dates = pd.bdate_range("2024-01-02", periods=30)
        rows = []
        for d in dates:
            rows.append({
                "date": d, "symbol": "AAPL",
                "open": 100.0, "high": 100.0, "low": 100.0,
                "close": 100.0, "adj_close": 100.0, "volume": 10_000_000,
            })
        panel = pd.DataFrame(rows)
        matrix = panel.pivot_table(index="date", columns="symbol", values="adj_close")

        config = BacktestConfig(
            strategy_id="test",
            strategy_module="test",
            start_date="2024-01-02",
            end_date=dates[-1].strftime("%Y-%m-%d"),
            initial_capital=100_000.0,
            benchmark="AAPL",
            cost=CostConfig(spread_bps=0, slippage_bps=0, commission_per_share=0,
                            participation_rate=1.0),
            risk=RiskConfig(cash_reserve_pct=0.0, max_position_pct=1.0),
            execution=ExecutionConfig(fractional_shares=True),
        )

        # Hold 80% for the first half, then cut the target to 20%.
        calls = {"n": 0}

        def strategy(ctx):
            calls["n"] += 1
            return {"AAPL": 0.8} if calls["n"] <= 15 else {"AAPL": 0.2}

        result = BacktestEngine(config).run(strategy, panel, matrix)

        final = result.snapshots[-1]
        final_weight = final.weights.get("AAPL", 0.0)

        # Prices are flat, so the book can track the target exactly. Without a
        # trim path this stays pinned near 0.8 forever.
        assert final_weight == pytest.approx(0.2, abs=0.02), (
            f"target dropped to 20% but book still holds {final_weight:.1%} -- "
            "the engine is not selling down"
        )

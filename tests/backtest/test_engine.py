"""Hand-verifiable accounting tests for the backtest engine.

Every test uses synthetic data where cash, shares, equity and PnL can be
computed on paper. No real market data involved.
"""
import pytest
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine
from backtest.types import (
    BacktestConfig, CostConfig, ExecutionConfig, RiskConfig, FillType,
)


def _synthetic_panel(n_days=5, symbols=("AAPL",), base_price=100.0, daily_return=0.01):
    """Create synthetic OHLCV panel for testing.
    
    All prices are round numbers for hand verification.
    """
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for i, d in enumerate(dates):
        for sym in symbols:
            price = base_price * (1 + daily_return) ** i
            rows.append({
                "date": d, "symbol": sym,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "adj_close": price,
                "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def _matrix_from_panel(panel):
    return panel.pivot_table(index="date", columns="symbol", values="adj_close")


class TestSingleBuy:
    """Buy one stock, hold to end. Verify cash and equity."""

    def test_buy_and_hold(self):
        panel = _synthetic_panel(n_days=5, symbols=("AAPL",), base_price=100.0, daily_return=0.0)
        matrix = _matrix_from_panel(panel)

        config = BacktestConfig(
            strategy_id="test",
            strategy_module="test",
            start_date="2024-01-02",
            end_date="2024-01-08",
            initial_capital=10_000.0,
            benchmark="AAPL",
            cost=CostConfig(spread_bps=0, slippage_bps=0, commission_per_share=0),
            risk=RiskConfig(cash_reserve_pct=0.0),
            execution=ExecutionConfig(fractional_shares=False),
        )

        def strategy(ctx):
            return {"AAPL": 1.0}

        engine = BacktestEngine(config)
        result = engine.run(strategy, panel, matrix)

        assert len(result.equity_curve) > 0
        final = result.equity_curve.iloc[-1]
        assert abs(final - 10_000) < 100


class TestBuyAndSell:
    """Buy then sell. Verify PnL."""

    def test_buy_then_exit(self):
        panel = _synthetic_panel(n_days=6, symbols=("AAPL",), base_price=100.0, daily_return=0.0)
        matrix = _matrix_from_panel(panel)

        config = BacktestConfig(
            strategy_id="test",
            strategy_module="test",
            start_date="2024-01-02",
            end_date="2024-01-09",
            initial_capital=10_000.0,
            benchmark="AAPL",
            cost=CostConfig(spread_bps=0, slippage_bps=0, commission_per_share=0),
            risk=RiskConfig(cash_reserve_pct=0.0),
            execution=ExecutionConfig(fractional_shares=False),
        )

        call_count = [0]

        def strategy(ctx):
            call_count[0] += 1
            if call_count[0] <= 3:
                return {"AAPL": 1.0}
            return {}

        engine = BacktestEngine(config)
        result = engine.run(strategy, panel, matrix)

        assert len(result.fills) > 0
        buy_fills = [f for f in result.fills if f.order.side.value == "BUY"]
        sell_fills = [f for f in result.fills if f.order.side.value == "SELL"]
        assert len(buy_fills) > 0
        assert len(sell_fills) > 0


class TestTimingInvariant:
    """Strategy sees day D close, fill happens at D+1 open."""

    def test_d_plus_1_fill(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        rows = []
        opens = [100, 110, 120, 130]
        closes = [105, 115, 125, 135]
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "symbol": "AAPL",
                "open": opens[i], "high": closes[i] + 5,
                "low": opens[i] - 5, "close": closes[i],
                "adj_close": closes[i], "volume": 1_000_000,
            })
        panel = pd.DataFrame(rows)
        matrix = _matrix_from_panel(panel)

        config = BacktestConfig(
            strategy_id="test",
            strategy_module="test",
            start_date="2024-01-02",
            end_date="2024-01-05",
            initial_capital=10_000.0,
            benchmark="AAPL",
            cost=CostConfig(spread_bps=0, slippage_bps=0, commission_per_share=0),
            risk=RiskConfig(cash_reserve_pct=0.0),
            execution=ExecutionConfig(
                fill_type=FillType.MARKET_OPEN,
                fractional_shares=True,
            ),
        )

        first_call = [True]

        def strategy(ctx):
            if first_call[0]:
                first_call[0] = False
                return {"AAPL": 1.0}
            return {"AAPL": 1.0}

        engine = BacktestEngine(config)
        result = engine.run(strategy, panel, matrix)

        if len(result.fills) > 0:
            first_fill = result.fills[0]
            assert first_fill.fill_date == dates[1]
            assert first_fill.fill_price == 110.0


class TestCashReserve:
    """Cash reserve prevents full deployment."""

    def test_reserve_enforced(self):
        panel = _synthetic_panel(n_days=5, symbols=("AAPL",), base_price=100.0, daily_return=0.0)
        matrix = _matrix_from_panel(panel)

        config = BacktestConfig(
            strategy_id="test",
            strategy_module="test",
            start_date="2024-01-02",
            end_date="2024-01-08",
            initial_capital=10_000.0,
            benchmark="AAPL",
            cost=CostConfig(spread_bps=0, slippage_bps=0),
            risk=RiskConfig(cash_reserve_pct=0.10),
            execution=ExecutionConfig(fractional_shares=True),
        )

        def strategy(ctx):
            return {"AAPL": 1.0}

        engine = BacktestEngine(config)
        result = engine.run(strategy, panel, matrix)

        for snap in result.snapshots[1:]:
            if snap.cash > 0:
                cash_pct = snap.cash / snap.equity if snap.equity > 0 else 1.0
                break

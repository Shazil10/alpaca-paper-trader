"""Tests for the PIT firewall (StrategyContext).

The single most important property: requesting data past as_of raises LookaheadError.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import pandas as pd
import numpy as np

from backtest.context import StrategyContext, LookaheadError
from backtest.types import PortfolioSnapshot


def _make_panel():
    """Create a tiny price panel for testing."""
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    symbols = ["AAPL", "SPY"]
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({
                "date": d, "symbol": s,
                "open": 100.0, "high": 105.0, "low": 95.0,
                "close": 102.0, "adj_close": 102.0, "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def _make_close_matrix(panel):
    return panel.pivot_table(index="date", columns="symbol", values="adj_close")


class TestLookaheadPrevention:
    def test_prices_end_after_as_of_raises(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        with pytest.raises(LookaheadError):
            ctx.prices(end=pd.Timestamp("2024-01-15"))

    def test_prices_truncated_at_as_of(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        result = ctx.prices()
        assert result.index.max() <= as_of

    def test_prices_lookback(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-15")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        result = ctx.prices(lookback=3)
        assert len(result) == 3

    def test_sessions_after_as_of_raises(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        with pytest.raises(LookaheadError):
            ctx.sessions(end=pd.Timestamp("2024-01-15"))

    def test_sessions_truncated(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        sessions = ctx.sessions()
        assert all(s <= as_of for s in sessions)


class TestContextProperties:
    def test_universe_from_price_panel(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        universe = ctx.universe()
        assert "AAPL" in universe
        assert "SPY" in universe

    def test_universe_from_pit_function(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        def pit_universe(date):
            return {"AAPL"}  # SPY not in the PIT universe

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
            universe_fn=pit_universe,
        )

        universe = ctx.universe()
        assert universe == {"AAPL"}

    def test_portfolio_defaults(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        assert ctx.equity == 0.0
        assert ctx.cash == 0.0
        assert len(ctx.positions) == 0

    def test_latest_close(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        close = ctx.latest_close("AAPL")
        assert close is not None
        assert close == 102.0

    def test_has_lookback(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-15")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
        )

        assert ctx.has_lookback(["AAPL", "SPY"], 3)
        assert not ctx.has_lookback(["AAPL"], 100)

    def test_params(self):
        panel = _make_panel()
        matrix = _make_close_matrix(panel)
        as_of = pd.Timestamp("2024-01-08")

        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
            params={"lookback": 60, "top_n": 7},
        )

        assert ctx.params["lookback"] == 60
        assert ctx.params["top_n"] == 7

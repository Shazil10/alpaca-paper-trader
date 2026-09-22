"""Multi-strategy fund simulation.

Runs multiple strategies sharing one cash balance, one risk manager,
and one equity curve. Per-sleeve attribution preserved via strategy_id.

This is the "hedge fund" mode: allocate capital across strategies,
track per-sleeve performance, and measure the combined portfolio.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import pandas as pd
import numpy as np

from backtest.types import (
    BacktestConfig, BacktestResult, CostConfig, ExecutionConfig,
    Fill, Order, OrderSide, PortfolioSnapshot, RiskConfig,
)
from backtest.context import StrategyContext
from backtest.portfolio import Portfolio
from backtest.broker import SimulatedBroker, adjusted_bars
from backtest.engine import generate_orders
from backtest import costs, risk, metrics

logger = logging.getLogger(__name__)


class SleeveConfig:
    """Configuration for a single strategy sleeve."""

    def __init__(
        self,
        strategy_id: str,
        strategy_fn: Callable[[StrategyContext], Dict[str, float]],
        allocation_pct: float,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.strategy_id = strategy_id
        self.strategy_fn = strategy_fn
        self.allocation_pct = allocation_pct
        self.params = params or {}


class FundResult:
    """Results from a multi-strategy fund simulation."""

    def __init__(self):
        self.fund_equity: pd.Series = pd.Series(dtype=float)
        self.fund_returns: pd.Series = pd.Series(dtype=float)
        self.benchmark_equity: pd.Series = pd.Series(dtype=float)
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        self.fund_metrics: Dict[str, Any] = {}

        # Per-sleeve attribution
        self.sleeve_equity: Dict[str, pd.Series] = {}
        self.sleeve_metrics: Dict[str, Dict[str, Any]] = {}
        self.sleeve_fills: Dict[str, List[Fill]] = {}
        self.sleeve_orders: Dict[str, List[Order]] = {}

        # Combined
        self.all_fills: List[Fill] = []
        self.all_orders: List[Order] = []
        self.snapshots: List[PortfolioSnapshot] = []

        self.warnings: List[str] = []
        self.elapsed_seconds: float = 0.0


class FundEngine:
    """Multi-strategy fund simulator."""

    def __init__(
        self,
        sleeves: List[SleeveConfig],
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000.0,
        benchmark: str = "SPY",
        cost_config: Optional[CostConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        self.sleeves = sleeves
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        self.cost = cost_config or CostConfig()
        self.risk = risk_config or RiskConfig()
        self.execution = execution_config or ExecutionConfig()

        total_alloc = sum(s.allocation_pct for s in sleeves)
        if total_alloc > 1.0 + 1e-6:
            raise ValueError(f"Sleeve allocations sum to {total_alloc:.1%}, must be <= 100%")

    def run(
        self,
        price_panel: pd.DataFrame,
        close_matrix: pd.DataFrame,
        ohlc_adjusted: Optional[Dict] = None,
        universe_fn: Optional[Callable] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> FundResult:
        """Run the full multi-strategy simulation."""
        started_at = time.time()
        result = FundResult()

        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)

        sessions = close_matrix.index[
            (close_matrix.index >= start) & (close_matrix.index <= end)
        ]
        if len(sessions) == 0:
            result.warnings.append("No trading sessions in range")
            return result

        all_sessions = pd.DatetimeIndex(sorted(close_matrix.index))

        sleeve_portfolios: Dict[str, Portfolio] = {}
        sleeve_pending: Dict[str, List[Order]] = {}

        for sleeve in self.sleeves:
            sleeve_capital = self.initial_capital * sleeve.allocation_pct
            sleeve_portfolios[sleeve.strategy_id] = Portfolio(sleeve_capital, sleeve.strategy_id)
            sleeve_pending[sleeve.strategy_id] = []
            result.sleeve_fills[sleeve.strategy_id] = []
            result.sleeve_orders[sleeve.strategy_id] = []

        broker = SimulatedBroker(self.cost, self.execution)

        equity_dates = []
        equity_values = []
        sleeve_equity_values: Dict[str, List[float]] = {s.strategy_id: [] for s in self.sleeves}

        for session in sessions:
            bars = self._get_bars(price_panel, session)

            for sleeve in self.sleeves:
                sid = sleeve.strategy_id
                pending = sleeve_pending[sid]
                if pending:
                    fills = broker.fill_orders(pending, session, bars)
                    for fill in fills:
                        sleeve_portfolios[sid].apply_fill(fill)
                        result.sleeve_fills[sid].append(fill)
                        result.all_fills.append(fill)
                    result.sleeve_orders[sid].extend(pending)
                    result.all_orders.extend(pending)
                    sleeve_pending[sid] = []

            current_prices = self._get_close_prices(close_matrix, session)

            fund_equity = 0.0
            for sleeve in self.sleeves:
                sid = sleeve.strategy_id
                snap = sleeve_portfolios[sid].snapshot(session, current_prices)
                sleeve_equity_values[sid].append(snap.equity)
                fund_equity += snap.equity

            equity_dates.append(session)
            equity_values.append(fund_equity)

            for sleeve in self.sleeves:
                sid = sleeve.strategy_id
                portfolio = sleeve_portfolios[sid]
                snap = portfolio.snapshot(session, current_prices)

                ctx = StrategyContext(
                    as_of=session,
                    price_panel=price_panel,
                    close_matrix=close_matrix,
                    ohlc_adjusted=ohlc_adjusted,
                    universe_fn=universe_fn,
                    portfolio=snap,
                    params=sleeve.params,
                    trading_sessions=all_sessions,
                    sector_map=sector_map,
                )

                try:
                    target_weights = sleeve.strategy_fn(ctx)
                except Exception as e:
                    logger.error("Strategy %s error on %s: %s", sid, session, e)
                    result.warnings.append(f"{session}: {sid} error: {e}")
                    continue

                adjusted_weights = risk.apply_risk_limits(
                    target_weights, snap, self.risk, sector_map
                )

                orders = self._generate_orders(adjusted_weights, snap, current_prices, sid)
                sleeve_pending[sid] = orders

        result.fund_equity = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates))
        result.fund_returns = result.fund_equity.pct_change().fillna(0.0)

        for sleeve in self.sleeves:
            sid = sleeve.strategy_id
            result.sleeve_equity[sid] = pd.Series(
                sleeve_equity_values[sid],
                index=pd.DatetimeIndex(equity_dates),
            )

        if self.benchmark in close_matrix.columns:
            bench = close_matrix[self.benchmark].loc[sessions].dropna()
            if len(bench) > 0:
                result.benchmark_equity = (bench / bench.iloc[0]) * self.initial_capital
                result.benchmark_returns = bench.pct_change().fillna(0.0)

        result.fund_metrics = metrics.compute_full_metrics(
            result.fund_equity,
            result.fund_returns,
            result.benchmark_equity,
            result.benchmark_returns,
        )

        for sleeve in self.sleeves:
            sid = sleeve.strategy_id
            eq = result.sleeve_equity[sid]
            ret = eq.pct_change().fillna(0.0)
            result.sleeve_metrics[sid] = metrics.compute_full_metrics(
                eq, ret,
                result.benchmark_equity,
                result.benchmark_returns,
            )

        result.elapsed_seconds = time.time() - started_at

        logger.info(
            "Fund simulation complete: %.1fs, %d sessions, %d total fills, "
            "final equity $%.2f (%d sleeves)",
            result.elapsed_seconds, len(sessions), len(result.all_fills),
            result.fund_equity.iloc[-1] if len(result.fund_equity) > 0 else 0,
            len(self.sleeves),
        )

        return result

    def _get_bars(self, panel, date):
        """Bars for `date` on the adjusted scale, matching the mark-to-market."""
        return adjusted_bars(panel, date)

    def _get_close_prices(self, matrix, date):
        if date not in matrix.index:
            return {}
        row = matrix.loc[date]
        return {sym: float(p) for sym, p in row.items() if pd.notna(p)}

    def _generate_orders(self, target_weights, snapshot, current_prices, strategy_id):
        """Delegate to the engine's shared implementation.

        This used to be a second copy of that logic, and the copy was missing
        the trim case -- so fund runs silently ratcheted gross exposure. One
        implementation, one place to fix it.
        """
        return generate_orders(target_weights, snapshot, current_prices, strategy_id)

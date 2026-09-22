"""Backtest engine — the daily simulation clock.

This is the central orchestrator. It:
1. Loads all data once at start
2. Iterates through trading sessions
3. On each day: build context -> call strategy -> generate orders -> simulate fills -> update portfolio
4. Collects results
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import pandas as pd
import numpy as np

from backtest.types import (
    BacktestConfig, BacktestResult, CostConfig, ExecutionConfig,
    Fill, FillType, Order, OrderSide, PortfolioSnapshot, RiskConfig,
)
from backtest.context import StrategyContext
from backtest.portfolio import Portfolio
from backtest.broker import SimulatedBroker, adjusted_bars
from backtest import costs, risk
from backtest.calendar import TradingCalendar

logger = logging.getLogger(__name__)


#: Minimum |target - current| weight gap before the engine trades, applied
#: symmetrically to buys and trims.
#:
#: Rebalancing on every basis point of drift would churn the book daily and
#: hand the cost model a turnover bill with no economic content behind it. Too
#: wide, though, and the portfolio stops tracking its target: the band is the
#: permanent error you accept on every position, so at 50bps a 20-name book
#: tolerates up to 10% of gross drifting away from what the strategy asked for.
REBALANCE_BAND = 0.005


class BacktestEngine:
    """Event-driven daily backtest engine."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        strategy_fn: Callable[[StrategyContext], Dict[str, float]],
        price_panel: pd.DataFrame,
        close_matrix: pd.DataFrame,
        ohlc_adjusted: Optional[Dict] = None,
        universe_fn: Optional[Callable] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> BacktestResult:
        """Execute the full backtest.
        
        Args:
            strategy_fn: target_weights(ctx) -> {symbol: weight}
            price_panel: long-format lake data (date, symbol, open, high, low, close, adj_close, volume)
            close_matrix: date x symbol matrix of adj_close
            ohlc_adjusted: {symbol: DataFrame[Open,High,Low,Close]} (optional)
            universe_fn: callable(date) -> Set[str] for PIT membership (optional)
            sector_map: {symbol: sector} for risk limits (optional)
        
        Returns:
            BacktestResult with equity curve, trades, metrics, etc.
        """
        started_at = time.time()

        cfg = self.config
        start = pd.Timestamp(cfg.start_date)
        end = pd.Timestamp(cfg.end_date)

        sessions_in_range = close_matrix.index[
            (close_matrix.index >= start) & (close_matrix.index <= end)
        ]
        if len(sessions_in_range) == 0:
            logger.error("No trading sessions in [%s, %s]", start, end)
            return BacktestResult(config=cfg, warnings=["No trading sessions in range"])

        all_sessions = pd.DatetimeIndex(sorted(close_matrix.index))

        portfolio = Portfolio(cfg.initial_capital, cfg.strategy_id)
        broker = SimulatedBroker(cfg.cost, cfg.execution)

        all_orders: List[Order] = []
        all_fills: List[Fill] = []
        snapshots: List[PortfolioSnapshot] = []
        equity_dates: List[pd.Timestamp] = []
        equity_values: List[float] = []
        warnings: List[str] = []

        pending_orders: List[Order] = []

        logger.info(
            "Starting backtest: %s from %s to %s ($%.0f)",
            cfg.strategy_id, start, end, cfg.initial_capital,
        )

        for i, session in enumerate(sessions_in_range):
            # Step 1: Execute pending orders from previous day (D+1 fills)
            if pending_orders:
                bars = self._get_bars(price_panel, session)
                fills = broker.fill_orders(pending_orders, session, bars)

                for fill in fills:
                    portfolio.apply_fill(fill)
                    all_fills.append(fill)

                all_orders.extend(pending_orders)
                pending_orders = []

            # Step 2: Mark-to-market at close
            current_prices = self._get_close_prices(close_matrix, session)
            snap = portfolio.snapshot(session, current_prices)
            snapshots.append(snap)
            equity_dates.append(session)
            equity_values.append(snap.equity)

            # Step 3: Build strategy context (PIT firewall)
            ctx = StrategyContext(
                as_of=session,
                price_panel=price_panel,
                close_matrix=close_matrix,
                ohlc_adjusted=ohlc_adjusted,
                universe_fn=universe_fn,
                portfolio=snap,
                params=cfg.params,
                trading_sessions=all_sessions,
                sector_map=sector_map,
            )

            # Step 4: Call strategy -> target weights
            try:
                target_weights = strategy_fn(ctx)
            except Exception as e:
                logger.error("Strategy error on %s: %s", session, e)
                warnings.append(f"{session:%Y-%m-%d}: strategy error: {e}")
                continue

            # Step 5: Apply risk limits
            adjusted_weights = risk.apply_risk_limits(
                target_weights, snap, cfg.risk, sector_map
            )

            # Step 6: Generate orders (diff current vs target)
            pending_orders = self._generate_orders(
                adjusted_weights, snap, current_prices, cfg
            )

        # Execute any remaining pending orders on the last+1 day
        if pending_orders:
            next_session_idx = close_matrix.index.get_indexer([sessions_in_range[-1]], method="pad")[0] + 1
            if next_session_idx < len(close_matrix.index):
                next_session = close_matrix.index[next_session_idx]
                bars = self._get_bars(price_panel, next_session)
                fills = broker.fill_orders(pending_orders, next_session, bars)
                for fill in fills:
                    portfolio.apply_fill(fill)
                    all_fills.append(fill)
                all_orders.extend(pending_orders)

        equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), name="equity")
        returns = equity_curve.pct_change().fillna(0.0)

        bench_equity, bench_returns = self._compute_benchmark(
            close_matrix, cfg.benchmark, sessions_in_range, cfg.initial_capital
        )

        elapsed = time.time() - started_at

        limitations = []
        if universe_fn is None:
            limitations.append("No PIT universe — results carry survivorship bias")

        result = BacktestResult(
            config=cfg,
            equity_curve=equity_curve,
            returns=returns,
            benchmark_equity=bench_equity,
            benchmark_returns=bench_returns,
            orders=all_orders,
            fills=all_fills,
            snapshots=snapshots,
            run_id=cfg.run_id(),
            started_at=pd.Timestamp.now().isoformat(),
            elapsed_seconds=elapsed,
            warnings=warnings,
            limitations=limitations,
        )

        logger.info(
            "Backtest complete: %.1fs, %d sessions, %d fills, final equity $%.2f",
            elapsed, len(sessions_in_range), len(all_fills), equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
        )

        return result

    def _get_bars(
        self, panel: pd.DataFrame, date: pd.Timestamp
    ) -> Dict[str, Dict[str, float]]:
        """Bars for `date` on the adjusted scale, matching the mark-to-market."""
        return adjusted_bars(panel, date)

    def _get_close_prices(
        self, matrix: pd.DataFrame, date: pd.Timestamp
    ) -> Dict[str, float]:
        """Get {symbol: adj_close} for mark-to-market."""
        if date not in matrix.index:
            return {}
        row = matrix.loc[date]
        return {sym: float(p) for sym, p in row.items() if pd.notna(p)}

    def _generate_orders(
        self,
        target_weights: Dict[str, float],
        snapshot: PortfolioSnapshot,
        current_prices: Dict[str, float],
        config: BacktestConfig,
    ) -> List[Order]:
        """Convert target weights into orders for this run's strategy."""
        return generate_orders(
            target_weights, snapshot, current_prices, config.strategy_id
        )

    def _compute_benchmark(
        self,
        close_matrix: pd.DataFrame,
        benchmark: str,
        sessions: pd.DatetimeIndex,
        initial_capital: float,
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute benchmark equity curve (buy-and-hold)."""
        if benchmark not in close_matrix.columns:
            return (
                pd.Series(dtype=float),
                pd.Series(dtype=float),
            )

        bench = close_matrix[benchmark].loc[sessions].dropna()
        if len(bench) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        bench_equity = (bench / bench.iloc[0]) * initial_capital
        bench_returns = bench.pct_change().fillna(0.0)

        return bench_equity, bench_returns


def generate_orders(
    target_weights: Dict[str, float],
    snapshot: PortfolioSnapshot,
    current_prices: Dict[str, float],
    strategy_id: str,
) -> List[Order]:
    """Convert target weights into buy/sell orders.

    Shared by the single-strategy engine and the multi-strategy fund engine.
    It lives in one place deliberately: this logic was duplicated once, and the
    copy silently missed the trim case below for every run made against it.

    Three cases, and the third is the one that gets forgotten:

    1. Held but absent from the target -- full exit.
    2. Under target by more than the band -- buy the difference.
    3. **Over** target by more than the band -- sell the difference.

    Without case 3 this does not implement target weights at all; it implements
    "buy toward target, never sell down", so gross exposure ratchets. A winner
    drifting above its target is never trimmed, and a strategy that *lowers* a
    target (de-levering, or rotating to a smaller position) keeps the old size.
    Observed before case 3 existed: the DAF sleeve reached 2.59x realized gross
    against a 2.0x cap, and sat at 2.22x through a month targeting 1.00x.
    """
    orders: List[Order] = []
    equity = snapshot.equity

    if equity <= 0:
        return orders

    current_weights = snapshot.weights
    target_symbols = set(target_weights.keys())
    held_symbols = set(current_weights.keys())

    # Full exit: held but not wanted.
    for sym in held_symbols:
        if sym not in target_symbols or target_weights.get(sym, 0) <= 0:
            pos = snapshot.positions.get(sym)
            if pos and pos.shares > 0:
                orders.append(Order(
                    symbol=sym,
                    side=OrderSide.SELL,
                    notional=0.0,
                    shares=pos.shares,
                    strategy_id=strategy_id,
                    reason="exit:not_in_target",
                    created_date=snapshot.date,
                    order_id=f"{strategy_id}:{uuid4().hex[:12]}",
                    close_position=True,
                ))

    # Buy up / trim down toward target.
    for sym, target_w in target_weights.items():
        if target_w <= 0:
            continue

        current_w = current_weights.get(sym, 0.0)
        delta_w = target_w - current_w

        # Dead band, applied symmetrically so trims and buys share one rule.
        if abs(delta_w) < REBALANCE_BAND:
            continue

        price = current_prices.get(sym)
        if price is None or price <= 0:
            continue

        if delta_w > 0:
            orders.append(Order(
                symbol=sym,
                side=OrderSide.BUY,
                notional=equity * delta_w,
                shares=0.0,
                strategy_id=strategy_id,
                reason=f"target_weight={target_w:.3f}",
                created_date=snapshot.date,
                order_id=f"{strategy_id}:{uuid4().hex[:12]}",
            ))
            continue

        pos = snapshot.positions.get(sym)
        if pos is None or pos.shares <= 0 or current_w <= 0:
            continue

        # Size the trim in shares, not notional. Handing the broker a notional
        # makes it re-derive shares from its own fill price, which on a gap can
        # exceed the position and drive it negative -- an accidental short.
        excess_shares = min(pos.shares * (-delta_w) / current_w, pos.shares)
        if excess_shares <= 0:
            continue

        orders.append(Order(
            symbol=sym,
            side=OrderSide.SELL,
            notional=0.0,
            shares=excess_shares,
            strategy_id=strategy_id,
            reason=f"trim_to_target={target_w:.3f}",
            created_date=snapshot.date,
            order_id=f"{strategy_id}:{uuid4().hex[:12]}",
        ))

    return orders

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
    ShortConfig,
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
        exit_dates: Dict[str, pd.Timestamp] = {}

        logger.info(
            "Starting backtest: %s from %s to %s ($%.0f)",
            cfg.strategy_id, start, end, cfg.initial_capital,
        )

        for i, session in enumerate(sessions_in_range):
            # Step 1: Execute pending orders from previous day (D+1 fills)
            if pending_orders:
                bars = self._get_bars(price_panel, session)
                fills = broker.fill_orders(pending_orders, session, bars)

                held_before = portfolio.held_symbols()
                for fill in fills:
                    portfolio.apply_fill(fill)
                    all_fills.append(fill)

                # Record closures so re-entry rules have an exit history. A trim
                # is not a closure, hence the before/after comparison rather than
                # simply logging every sell.
                portfolio.drop_dust()
                for symbol in held_before - portfolio.held_symbols():
                    exit_dates[symbol] = session

                all_orders.extend(pending_orders)
                pending_orders = []

            # Step 1b: Short-side carry and constraints, before marking.
            current_prices = self._get_close_prices(close_matrix, session)
            if cfg.short.allow_shorts:
                portfolio.accrue_borrow_fees(session, current_prices, cfg.short)
                forced = self._force_buy_ins(
                    portfolio, session, current_prices, cfg, broker, all_fills,
                    all_orders,
                )
                for symbol in forced:
                    warnings.append(
                        f"{session:%Y-%m-%d}: bought in {symbol} (unborrowable)"
                    )

            # Step 2: Mark-to-market at close
            snap = portfolio.snapshot(session, current_prices)

            if cfg.short.allow_shorts:
                breach = self._margin_call(portfolio, snap, cfg, session)
                if breach:
                    warnings.append(breach)
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
                exit_dates=exit_dates,
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

        if cfg.short.allow_shorts:
            # Stated on every short run, because the borrow model is the weakest
            # assumption in the whole engine and it is invisible in the returns.
            limitations.append(
                f"Borrow modelled as a flat "
                f"{cfg.short.borrow_rate_annual:.1%} annual rate "
                f"({cfg.short.hard_to_borrow_rate_annual:.0%} for "
                f"{len(cfg.short.hard_to_borrow)} named hard-to-borrow symbol(s)). "
                "Real borrow is a daily per-name broker quote with no free "
                "history, so this is a policy with the right shape, not a "
                "reconstruction."
            )
            limitations.append(
                "Short availability is a static list. In reality a crowded short "
                "becomes unborrowable exactly when it is squeezing, so forced "
                "buy-ins here are rarer and cheaper than they would have been."
            )
            if portfolio.borrow_paid:
                limitations.append(
                    f"Borrow cost charged over the run: "
                    f"${portfolio.borrow_paid:,.0f}"
                )

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

    def _force_buy_ins(
        self,
        portfolio: Portfolio,
        session: pd.Timestamp,
        prices: Dict[str, float],
        cfg: BacktestConfig,
        broker: SimulatedBroker,
        all_fills: List[Fill],
        all_orders: List[Order],
    ) -> List[str]:
        """Cover any short in a name that is no longer borrowable.

        A real buy-in is not optional and not scheduled: the lender recalls, and
        the position is closed at whatever the market is, immediately. So this
        fills at the *current session's* price rather than waiting for the next
        open, which is the one place the D+1 convention is deliberately broken --
        pretending a forced cover gets tomorrow's open would be the optimistic
        error, since buy-ins cluster exactly when a crowded short is squeezing.
        """
        recalled = [
            symbol for symbol in portfolio.short_symbols()
            if not cfg.short.borrowable(symbol)
        ]
        if not recalled:
            return []

        covered: List[str] = []
        for symbol in recalled:
            shares = abs(portfolio.position_shares(symbol))
            price = prices.get(symbol)
            if shares <= 0 or price is None or price <= 0:
                continue

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                notional=0.0,
                shares=shares,
                strategy_id=cfg.strategy_id,
                reason="forced_buy_in:unborrowable",
                created_date=session,
                order_id=f"{cfg.strategy_id}:{uuid4().hex[:12]}",
                close_position=True,
            )
            fill_price = costs.compute_fill_price(price, OrderSide.BUY, cfg.cost)
            fill = Fill(
                order=order,
                fill_price=fill_price,
                fill_shares=shares,
                fill_date=session,
                commission=costs.compute_commission(shares, cfg.cost),
                slippage_bps=cfg.cost.total_one_way_bps,
                fill_type=FillType.MARKET_CLOSE,
                fill_id=uuid4().hex[:12],
            )
            portfolio.apply_fill(fill)
            all_fills.append(fill)
            all_orders.append(order)
            covered.append(symbol)
            logger.warning(
                "Forced buy-in: %s %.0f shares at %.2f on %s",
                symbol, shares, fill_price, session.date(),
            )

        return covered

    def _margin_call(
        self,
        portfolio: Portfolio,
        snapshot: PortfolioSnapshot,
        cfg: BacktestConfig,
        session: pd.Timestamp,
    ) -> Optional[str]:
        """Liquidate proportionally when equity falls under maintenance margin.

        Scaled down uniformly rather than closing the largest position, for the
        same reason the risk layer scales rather than clips: a forced liquidation
        that re-ranks the book changes the strategy being measured. A real broker
        would not be so considerate, but modelling its arbitrary choice would add
        noise rather than realism.

        Only a *breach* triggers this. Sitting one dollar above maintenance is
        legal and uncomfortable, which is exactly the state a levered short book
        lives in.
        """
        required = snapshot.margin_requirement(cfg.short)
        equity = snapshot.equity
        if required <= 0 or equity >= required:
            return None

        if equity <= 0:
            # Past insolvency: nothing to scale, close everything.
            scale = 0.0
        else:
            scale = max(min(equity / required, 1.0), 0.0)

        reduction = 1.0 - scale
        closed_value = 0.0

        for symbol, position in list(snapshot.positions.items()):
            price = position.market_price
            if price <= 0:
                continue
            shares = abs(position.shares) * reduction
            if shares <= 0:
                continue

            side = OrderSide.SELL if position.shares > 0 else OrderSide.BUY
            order = Order(
                symbol=symbol,
                side=side,
                notional=0.0,
                shares=shares,
                strategy_id=cfg.strategy_id,
                reason="margin_call",
                created_date=session,
                order_id=f"{cfg.strategy_id}:{uuid4().hex[:12]}",
                close_position=reduction >= 1.0,
            )
            fill_price = costs.compute_fill_price(price, side, cfg.cost)
            portfolio.apply_fill(Fill(
                order=order,
                fill_price=fill_price,
                fill_shares=shares,
                fill_date=session,
                commission=costs.compute_commission(shares, cfg.cost),
                slippage_bps=cfg.cost.total_one_way_bps,
                fill_type=FillType.MARKET_CLOSE,
                fill_id=uuid4().hex[:12],
            ))
            closed_value += shares * fill_price

        message = (
            f"{session:%Y-%m-%d}: margin call — equity ${equity:,.0f} below "
            f"${required:,.0f} required; liquidated {reduction:.0%} "
            f"(${closed_value:,.0f})"
        )
        logger.warning(message)
        return message

    def _generate_orders(
        self,
        target_weights: Dict[str, float],
        snapshot: PortfolioSnapshot,
        current_prices: Dict[str, float],
        config: BacktestConfig,
    ) -> List[Order]:
        """Convert target weights into orders for this run's strategy."""
        return generate_orders(
            target_weights, snapshot, current_prices, config.strategy_id,
            short_config=config.short,
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
    short_config: Optional[ShortConfig] = None,
) -> List[Order]:
    """Convert target weights into buy/sell orders.

    Shared by the single-strategy engine and the multi-strategy fund engine.
    It lives in one place deliberately: this logic was duplicated once, and the
    copy silently missed the trim case below for every run made against it.

    Weights are signed. A negative target is a short, and is honoured only when
    ``short_config.allow_shorts`` is set -- otherwise it is dropped, so every
    long-only strategy behaves exactly as it did before the short side existed.

    Five cases, and the ones that get forgotten are the third and the fifth:

    1. Held but absent from the target -- full exit, in whichever direction closes
       it. A short is closed by buying.
    2. Under target in the target's own direction -- trade the difference.
    3. **Over** target -- trade back the difference.
    4. Target on the opposite side of the current position -- close it outright and
       let the next session open the other side from flat.
    5. Shorts, which every one of the above has to handle with a sign rather than
       an assumption.

    Without case 3 this does not implement target weights at all; it implements
    "buy toward target, never sell down", so gross exposure ratchets. A winner
    drifting above its target is never trimmed, and a strategy that *lowers* a
    target (de-levering, or rotating to a smaller position) keeps the old size.
    Observed before case 3 existed: the DAF sleeve reached 2.59x realized gross
    against a 2.0x cap, and sat at 2.22x through a month targeting 1.00x.

    Case 4 is split across two sessions on purpose. Sizing a flip as one order
    means handing the broker a notional that has to carry the position through
    zero, and a gap between decision and fill then overshoots into an unintended
    position on the far side.
    """
    orders: List[Order] = []
    equity = snapshot.equity

    if equity <= 0:
        return orders

    allow_shorts = short_config is not None and short_config.allow_shorts
    current_weights = snapshot.weights
    held_symbols = set(current_weights.keys())

    def wanted(symbol: str) -> float:
        """The target weight, with shorts suppressed unless enabled.

        A negative target is dropped rather than reinterpreted when shorts are
        off. Clipping it to zero would be the same thing here -- absent from the
        target means exit -- but being explicit keeps the intent visible.
        """
        weight = float(target_weights.get(symbol, 0.0))
        if weight < 0 and not allow_shorts:
            return 0.0
        return weight

    target_symbols = {s for s in target_weights if abs(wanted(s)) > 0}

    # Full exit: held but not wanted. Signed, so a short is closed by buying.
    for sym in held_symbols:
        if sym in target_symbols:
            continue
        pos = snapshot.positions.get(sym)
        if pos is None or abs(pos.shares) <= 0:
            continue
        orders.append(Order(
            symbol=sym,
            side=OrderSide.SELL if pos.shares > 0 else OrderSide.BUY,
            notional=0.0,
            shares=abs(pos.shares),
            strategy_id=strategy_id,
            reason="exit:not_in_target",
            created_date=snapshot.date,
            order_id=f"{strategy_id}:{uuid4().hex[:12]}",
            close_position=True,
        ))

    # Move each wanted symbol toward its target.
    for sym in target_symbols:
        target_w = wanted(sym)
        current_w = current_weights.get(sym, 0.0)
        delta_w = target_w - current_w

        # Dead band, applied symmetrically so every direction shares one rule.
        if abs(delta_w) < REBALANCE_BAND:
            continue

        price = current_prices.get(sym)
        if price is None or price <= 0:
            continue

        pos = snapshot.positions.get(sym)
        crossing_zero = pos is not None and target_w * pos.shares < 0

        if crossing_zero:
            # Flipping side is two decisions, not one, and sizing it as a single
            # notional would leave the broker to re-derive shares across the flip.
            # Close the existing position outright; the next session opens the
            # other side from flat, which is also how a real desk would do it.
            orders.append(Order(
                symbol=sym,
                side=OrderSide.SELL if pos.shares > 0 else OrderSide.BUY,
                notional=0.0,
                shares=abs(pos.shares),
                strategy_id=strategy_id,
                reason=f"flip_side:target={target_w:.3f}",
                created_date=snapshot.date,
                order_id=f"{strategy_id}:{uuid4().hex[:12]}",
                close_position=True,
            ))
            continue

        # Growing the position -- further long, or further short.
        growing = (
            (target_w > 0 and delta_w > 0) or (target_w < 0 and delta_w < 0)
        )
        if growing:
            orders.append(Order(
                symbol=sym,
                side=OrderSide.BUY if delta_w > 0 else OrderSide.SELL,
                notional=equity * abs(delta_w),
                shares=0.0,
                strategy_id=strategy_id,
                reason=f"target_weight={target_w:.3f}",
                created_date=snapshot.date,
                order_id=f"{strategy_id}:{uuid4().hex[:12]}",
            ))
            continue

        # Shrinking toward the target without crossing it.
        if pos is None or abs(pos.shares) <= 0 or abs(current_w) <= 0:
            continue

        # Size the trim in shares, not notional. Handing the broker a notional
        # makes it re-derive shares from its own fill price, which on a gap can
        # exceed the position and drive it through zero -- an accidental short.
        excess_shares = min(
            abs(pos.shares) * abs(delta_w) / abs(current_w), abs(pos.shares)
        )
        if excess_shares <= 0:
            continue

        orders.append(Order(
            symbol=sym,
            side=OrderSide.SELL if pos.shares > 0 else OrderSide.BUY,
            notional=0.0,
            shares=excess_shares,
            strategy_id=strategy_id,
            reason=f"trim_to_target={target_w:.3f}",
            created_date=snapshot.date,
            order_id=f"{strategy_id}:{uuid4().hex[:12]}",
        ))

    return orders

"""Multi-strategy fund simulation on one balance sheet.

The distinction that matters: this is not several backtests added together. It is
one account. One cash balance, one position per ticker, one set of exposure
limits, and orders netted across sleeves before anything reaches the market. If
the momentum sleeve wants to buy 100 SPY on the same session the defensive sleeve
wants to sell 60, the fund trades 40 and pays spread on 40 -- not on 160.

Running sleeves as separate sub-accounts, which is what this module used to do,
gets three things wrong and all three flatter the result:

* **Cost.** Offsetting trades both pay spread, so the fund is charged for
  churn that never left the building.
* **Risk.** A leverage or sector cap applied per sleeve does not bind on the
  combined book. Two sleeves each at 90% gross and 35% in technology are a fund
  at 180% gross and 70% in one sector, and neither sleeve's check fires.
* **Capital.** A sleeve that has drawn down keeps trading against its original
  allocation, and a sleeve that is doing well cannot be given more.

What each sleeve still sees
---------------------------
Every sleeve decides against *its own* book, because its rules depend on it --
Clenow exits the names it holds, the pullback sleeve stops out against the price
it paid. So lots are tagged with ``strategy_id`` and each sleeve gets a snapshot
filtered to its own lots, paired with its own cash sub-ledger. The sub-ledgers
sum to the fund's single real cash balance, and a test asserts that invariant
every session; if it ever drifts, the attribution is lying.

How netting prices out
----------------------
For one symbol on one session, with signed share requests per sleeve:

    crossed = min(total buys, total sells)
    net     = total buys - total sells

The ``net`` residual goes to the broker and pays spread, slippage and commission.
The ``crossed`` quantity transfers between sleeves at the untouched execution
price -- the buyer pays what the seller receives, so it nets to zero in fund cash,
which is exactly what an internal cross is. Sleeves on the net side are allocated
their pro-rata slice of the residual at the broker's fill price and the remainder
at the crossing price; sleeves on the other side fill entirely at the crossing
price.

That construction makes the cash identity hold exactly: the sleeve cash deltas
sum to the fund's, because the crossed halves cancel pairwise.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from backtest import costs, metrics, risk
from backtest.broker import SimulatedBroker, adjusted_bars
from backtest.context import StrategyContext
from backtest.engine import generate_orders
from backtest.portfolio import Portfolio
from backtest.types import (
    BacktestConfig, CostConfig, ExecutionConfig, Fill, FillType, Order,
    OrderSide, PortfolioSnapshot, RiskConfig, ShortConfig,
)

logger = logging.getLogger(__name__)

#: Share quantities below this are treated as zero when netting. Netting sums
#: signed floats, so two exactly offsetting requests leave a float residue rather
#: than a clean zero, and sending that to the broker books a commission on
#: nothing.
SHARE_TOLERANCE = 1e-9


@dataclass
class SleeveConfig:
    """One strategy inside the fund."""
    strategy_id: str
    strategy_fn: Callable[[StrategyContext], Dict[str, float]]
    allocation_pct: float
    params: Dict[str, Any] = field(default_factory=dict)

    #: Per-sleeve risk limits, applied to that sleeve's own targets before the
    #: fund-level pass. None means the fund's limits are the only ones.
    risk: Optional[RiskConfig] = None


@dataclass
class NettingRecord:
    """What netting saved on one symbol on one session. Kept for the report."""
    date: pd.Timestamp
    symbol: str
    gross_shares: float
    net_shares: float
    crossed_shares: float
    crossing_price: float

    @property
    def saved_notional(self) -> float:
        return self.crossed_shares * self.crossing_price


@dataclass
class FundResult:
    """Results from a multi-strategy fund simulation."""
    fund_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fund_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fund_metrics: Dict[str, Any] = field(default_factory=dict)

    # Per-sleeve attribution, retained internally even though the book is shared.
    sleeve_equity: Dict[str, pd.Series] = field(default_factory=dict)
    sleeve_cash: Dict[str, pd.Series] = field(default_factory=dict)
    sleeve_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sleeve_fills: Dict[str, List[Fill]] = field(default_factory=dict)
    sleeve_orders: Dict[str, List[Order]] = field(default_factory=dict)

    all_fills: List[Fill] = field(default_factory=list)
    all_orders: List[Order] = field(default_factory=list)
    snapshots: List[PortfolioSnapshot] = field(default_factory=list)

    netting: List[NettingRecord] = field(default_factory=list)
    gross_exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    reallocations: List[Dict[str, Any]] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def netting_saved_notional(self) -> float:
        """Notional that never reached the market because sleeves crossed."""
        return float(sum(r.saved_notional for r in self.netting))

    @property
    def shared_symbols(self) -> Dict[str, List[str]]:
        """{symbol: sleeves that traded it}, for symbols more than one touched.

        This is what decides whether netting can save anything at all. Sleeves
        with disjoint universes -- an ETF trend follower beside a single-stock
        screen -- never hold the same instrument, so there is nothing to cross and
        a zero saving is the correct answer rather than a broken one. Reporting
        the overlap alongside the saving distinguishes those two cases, which
        otherwise look identical.
        """
        owners: Dict[str, set] = {}
        for sid, fills in self.sleeve_fills.items():
            for fill in fills:
                owners.setdefault(fill.order.symbol, set()).add(sid)
        return {
            symbol: sorted(sleeves)
            for symbol, sleeves in sorted(owners.items())
            if len(sleeves) > 1
        }


class FundEngine:
    """Multi-strategy fund simulator over a single balance sheet."""

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
        short_config: Optional[ShortConfig] = None,
        *,
        reallocate: str = "none",
        reallocate_every_months: int = 12,
    ):
        """``reallocate`` controls capital movement between sleeves.

        ``"none"`` leaves each sleeve with whatever its own trading produced --
        winners compound, losers shrink. ``"fixed"`` returns each sleeve to its
        target ``allocation_pct`` of fund equity on the schedule, which is what a
        fund with a mandate actually does and is the only mode that keeps a
        drawn-down sleeve trading at a meaningful size.
        """
        self.sleeves = sleeves
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        self.cost = cost_config or CostConfig()
        self.risk = risk_config or RiskConfig()
        self.execution = execution_config or ExecutionConfig()
        self.short = short_config or ShortConfig()
        self.reallocate = reallocate
        self.reallocate_every_months = max(int(reallocate_every_months), 1)

        if not sleeves:
            raise ValueError("a fund needs at least one sleeve")

        total = sum(s.allocation_pct for s in sleeves)
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"Sleeve allocations sum to {total:.1%}, must be <= 100%"
            )

        ids = [s.strategy_id for s in sleeves]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate sleeve ids: {ids}")

    # -----------------------------------------------------------------------
    # Netting
    # -----------------------------------------------------------------------

    def _requested_shares(
        self,
        order: Order,
        bar: Dict[str, float],
        broker: SimulatedBroker,
    ) -> float:
        """Signed share quantity an order is asking for, at this bar's price.

        Netting has to happen in shares, but a buy arrives as a notional. The
        conversion uses the same execution price the broker will use, so the two
        cannot disagree about what "100 dollars of SPY" means.
        """
        price = broker.execution_price(bar)
        if price is None or price <= 0:
            return 0.0

        if order.shares > 0:
            shares = float(order.shares)
        elif order.notional > 0:
            shares = float(order.notional) / price
        else:
            return 0.0

        return shares if order.side == OrderSide.BUY else -shares

    def _net_and_fill(
        self,
        pending: Dict[str, List[Order]],
        session: pd.Timestamp,
        bars: Dict[str, Dict[str, float]],
        broker: SimulatedBroker,
        ledger: Portfolio,
        sleeve_cash: Dict[str, float],
        result: FundResult,
    ) -> None:
        """Net every sleeve's orders per symbol, execute the residual, allocate back."""
        # symbol -> [(strategy_id, signed_shares, order)]
        by_symbol: Dict[str, List[Tuple[str, float, Order]]] = {}
        for sid, orders in pending.items():
            for order in orders:
                bar = bars.get(order.symbol)
                if bar is None:
                    logger.warning(
                        "No bar for %s on %s — %s order dropped",
                        order.symbol, session.date(), sid,
                    )
                    continue
                signed = self._requested_shares(order, bar, broker)
                if abs(signed) > SHARE_TOLERANCE:
                    by_symbol.setdefault(order.symbol, []).append((sid, signed, order))

        for symbol, requests in by_symbol.items():
            bar = bars[symbol]
            crossing_price = broker.execution_price(bar)
            if crossing_price is None or crossing_price <= 0:
                continue

            buys = sum(s for _, s, _ in requests if s > 0)
            sells = -sum(s for _, s, _ in requests if s < 0)
            crossed = min(buys, sells)
            net = buys - sells

            net_price = crossing_price
            net_commission = 0.0
            filled_net = 0.0

            if abs(net) > SHARE_TOLERANCE:
                side = OrderSide.BUY if net > 0 else OrderSide.SELL
                # A netted exit that closes every sleeve's position must not be
                # rounded down, or the fund inherits the dust problem the
                # single-strategy engine already fixed.
                closing = all(
                    o.close_position for _, s, o in requests if (s < 0) == (net < 0)
                )
                netted = Order(
                    symbol=symbol,
                    side=side,
                    notional=0.0,
                    shares=abs(net),
                    strategy_id="fund",
                    reason=f"netted:{len(requests)}_sleeve_order(s)",
                    created_date=session,
                    order_id=f"fund:{uuid4().hex[:12]}",
                    close_position=closing,
                )
                fills = broker.fill_orders([netted], session, {symbol: bar})
                if not fills:
                    continue
                fill = fills[0]
                filled_net = float(fill.fill_shares)
                net_price = float(fill.fill_price)
                net_commission = float(fill.commission)
                result.all_orders.append(netted)

                if filled_net + SHARE_TOLERANCE < abs(net):
                    # The participation cap bit. Scale every sleeve's residual
                    # slice pro-rata rather than filling the first sleeve in full.
                    logger.debug(
                        "Netted %s capped: %.0f of %.0f shares",
                        symbol, filled_net, abs(net),
                    )

            result.netting.append(NettingRecord(
                date=session, symbol=symbol,
                gross_shares=buys + sells,
                net_shares=net,
                crossed_shares=crossed,
                crossing_price=float(crossing_price),
            ))

            self._allocate(
                symbol, requests, session, net, filled_net, net_price,
                net_commission, float(crossing_price), ledger, sleeve_cash, result,
            )

    def _allocate(
        self,
        symbol: str,
        requests: List[Tuple[str, float, Order]],
        session: pd.Timestamp,
        net: float,
        filled_net: float,
        net_price: float,
        net_commission: float,
        crossing_price: float,
        ledger: Portfolio,
        sleeve_cash: Dict[str, float],
        result: FundResult,
    ) -> None:
        """Split the executed residual and the internal cross back to sleeves."""
        net_side_sign = 1.0 if net > 0 else -1.0
        net_side_total = sum(
            abs(s) for _, s, _ in requests if np.sign(s) == net_side_sign
        )
        fill_ratio = (
            filled_net / abs(net) if abs(net) > SHARE_TOLERANCE else 0.0
        )

        for sid, signed, order in requests:
            on_net_side = np.sign(signed) == net_side_sign and net_side_total > 0

            if on_net_side:
                share_of_residual = abs(signed) / net_side_total
                at_net = abs(net) * share_of_residual * fill_ratio
                # Whatever is not part of the residual crossed internally, capped
                # so a partial residual fill cannot allocate more than requested.
                at_cross = max(min(abs(signed) - at_net, abs(signed)), 0.0)
                commission = net_commission * share_of_residual
            else:
                at_net = 0.0
                at_cross = abs(signed)
                commission = 0.0

            total_shares = at_net + at_cross
            if total_shares <= SHARE_TOLERANCE:
                continue

            blended = (
                (at_net * net_price + at_cross * crossing_price) / total_shares
            )

            allocated = Order(
                symbol=symbol,
                side=order.side,
                notional=0.0,
                shares=total_shares,
                strategy_id=sid,
                reason=order.reason,
                created_date=order.created_date,
                order_id=order.order_id,
                close_position=order.close_position,
            )
            fill = Fill(
                order=allocated,
                fill_price=blended,
                fill_shares=total_shares,
                fill_date=session,
                commission=commission,
                slippage_bps=self.cost.total_one_way_bps * (at_net / total_shares),
                fill_type=self.execution.fill_type,
                fill_id=uuid4().hex[:12],
            )

            ledger.apply_fill(fill)

            cash_delta = -(total_shares * blended + commission)
            if order.side == OrderSide.SELL:
                cash_delta = total_shares * blended - commission
            sleeve_cash[sid] = sleeve_cash.get(sid, 0.0) + cash_delta

            result.all_fills.append(fill)
            result.sleeve_fills.setdefault(sid, []).append(fill)
            result.sleeve_orders.setdefault(sid, []).append(allocated)

    # -----------------------------------------------------------------------
    # Fund-level risk
    # -----------------------------------------------------------------------

    def _apply_fund_limits(
        self,
        sleeve_targets: Dict[str, Dict[str, float]],
        sleeve_equity: Dict[str, float],
        fund_snapshot: PortfolioSnapshot,
        sector_map: Optional[Dict[str, str]],
    ) -> Dict[str, Dict[str, float]]:
        """Scale sleeve targets so the *combined* book respects the fund limits.

        Per-sleeve checks cannot see the fund. Two sleeves each at 90% gross are a
        fund at 180%, and each one's own leverage check passes. So the sleeve
        targets are folded into one fund-level weight vector, the fund's limits
        are applied to that, and the resulting scale factor is pushed back down.

        Scaling rather than clipping is deliberate: clipping the largest position
        would silently re-rank a sleeve's conviction, while a uniform scale
        preserves the shape of what every sleeve asked for and only changes the
        size.
        """
        fund_equity = fund_snapshot.equity
        if fund_equity <= 0:
            return sleeve_targets

        # Sleeve weights are fractions of sleeve equity; convert to fractions of
        # fund equity before they can be added together.
        combined: Dict[str, float] = {}
        for sid, targets in sleeve_targets.items():
            scale = sleeve_equity.get(sid, 0.0) / fund_equity
            for symbol, weight in targets.items():
                combined[symbol] = combined.get(symbol, 0.0) + weight * scale

        if not combined:
            return sleeve_targets

        limited = risk.apply_risk_limits(
            combined, fund_snapshot, self.risk, sector_map
        )

        gross_before = sum(abs(w) for w in combined.values())
        gross_after = sum(abs(w) for w in limited.values())
        if gross_before <= 0 or gross_after >= gross_before - 1e-12:
            return sleeve_targets

        factor = gross_after / gross_before
        logger.debug(
            "Fund risk: scaling every sleeve by %.3f (gross %.2f -> %.2f)",
            factor, gross_before, gross_after,
        )
        return {
            sid: {symbol: weight * factor for symbol, weight in targets.items()}
            for sid, targets in sleeve_targets.items()
        }

    # -----------------------------------------------------------------------
    # Capital reallocation
    # -----------------------------------------------------------------------

    def _reallocate(
        self,
        session: pd.Timestamp,
        ledger: Portfolio,
        sleeve_cash: Dict[str, float],
        sleeve_equity: Dict[str, float],
        result: FundResult,
    ) -> None:
        """Move cash between sleeve sub-ledgers toward target allocations.

        Only cash moves. Positions stay with the sleeve that opened them, because
        transferring a position would hand one sleeve another's entry price and
        break every exit rule that reads it.

        A sleeve can therefore be owed capital it cannot immediately be given --
        if it is fully invested there is nothing to take from it. The transfer is
        capped at available cash rather than forced, and the shortfall closes on
        its own as positions are exited.
        """
        fund_equity = sum(sleeve_equity.values())
        if fund_equity <= 0:
            return

        targets = {
            s.strategy_id: fund_equity * s.allocation_pct for s in self.sleeves
        }
        surplus = {
            sid: sleeve_equity.get(sid, 0.0) - target
            for sid, target in targets.items()
        }

        donors = {sid: amt for sid, amt in surplus.items() if amt > 0}
        receivers = {sid: -amt for sid, amt in surplus.items() if amt < 0}
        if not donors or not receivers:
            return

        # A donor can only give what it holds in cash.
        available = {
            sid: min(amt, max(sleeve_cash.get(sid, 0.0), 0.0))
            for sid, amt in donors.items()
        }
        pot = sum(available.values())
        wanted = sum(receivers.values())
        moved = min(pot, wanted)
        if moved <= 0:
            return

        for sid, amount in available.items():
            if pot > 0:
                sleeve_cash[sid] -= moved * (amount / pot)
        for sid, amount in receivers.items():
            sleeve_cash[sid] += moved * (amount / wanted)

        result.reallocations.append({
            "date": str(session.date()),
            "moved": float(moved),
            "requested": float(wanted),
            "constrained_by_cash": bool(moved < wanted - 1e-9),
            "from": {sid: float(a) for sid, a in available.items() if a > 0},
            "to": {sid: float(a) for sid, a in receivers.items()},
        })
        logger.info(
            "Reallocated $%.0f of $%.0f requested on %s",
            moved, wanted, session.date(),
        )

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(
        self,
        price_panel: pd.DataFrame,
        close_matrix: pd.DataFrame,
        ohlc_adjusted: Optional[Dict] = None,
        universe_fn: Optional[Callable] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> FundResult:
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
        broker = SimulatedBroker(self.cost, self.execution)

        # One ledger, one cash balance. Sleeve sub-ledgers must sum to it.
        ledger = Portfolio(self.initial_capital, "fund")
        sleeve_cash: Dict[str, float] = {
            s.strategy_id: self.initial_capital * s.allocation_pct
            for s in self.sleeves
        }
        unallocated = self.initial_capital - sum(sleeve_cash.values())
        if unallocated > 1e-6:
            # Allocations summing under 100% leave a cash buffer that belongs to
            # the fund and is never traded. Recorded so fund equity reconciles.
            logger.info(
                "Fund holds $%.0f (%.1f%%) unallocated to any sleeve",
                unallocated, 100 * unallocated / self.initial_capital,
            )

        pending: Dict[str, List[Order]] = {s.strategy_id: [] for s in self.sleeves}
        exit_dates: Dict[str, Dict[str, pd.Timestamp]] = {
            s.strategy_id: {} for s in self.sleeves
        }
        for s in self.sleeves:
            result.sleeve_fills[s.strategy_id] = []
            result.sleeve_orders[s.strategy_id] = []

        equity_dates: List[pd.Timestamp] = []
        equity_values: List[float] = []
        gross_values: List[float] = []
        sleeve_equity_history: Dict[str, List[float]] = {
            s.strategy_id: [] for s in self.sleeves
        }
        sleeve_cash_history: Dict[str, List[float]] = {
            s.strategy_id: [] for s in self.sleeves
        }
        last_reallocation: Optional[pd.Timestamp] = None

        for session in sessions:
            bars = self._get_bars(price_panel, session)

            # 1. Execute yesterday's orders, netted across sleeves.
            if any(pending.values()):
                held_before = {
                    sid: self._held(ledger, sid) for sid in pending
                }
                self._net_and_fill(
                    pending, session, bars, broker, ledger, sleeve_cash, result
                )
                ledger.drop_dust()
                for sid, before in held_before.items():
                    for symbol in before - self._held(ledger, sid):
                        exit_dates[sid][symbol] = session
                pending = {s.strategy_id: [] for s in self.sleeves}

            # 2. Mark to market. Cash lives in the sub-ledgers plus the buffer.
            prices = self._get_close_prices(close_matrix, session)

            # Borrow is charged on the fund's shorts, then split across the
            # sleeves that hold them -- a sleeve must carry the cost of its own
            # short, or the cheapest way to look good is to be the short sleeve.
            if self.short.allow_shorts:
                self._charge_borrow(
                    ledger, session, prices, sleeve_cash
                )
            sleeve_snapshots: Dict[str, PortfolioSnapshot] = {}
            sleeve_equity: Dict[str, float] = {}

            for sleeve in self.sleeves:
                sid = sleeve.strategy_id
                snapshot = ledger.snapshot(
                    session, prices,
                    strategy_id=sid, cash_override=sleeve_cash[sid],
                )
                sleeve_snapshots[sid] = snapshot
                sleeve_equity[sid] = snapshot.equity
                sleeve_equity_history[sid].append(snapshot.equity)
                sleeve_cash_history[sid].append(sleeve_cash[sid])

            fund_snapshot = ledger.snapshot(session, prices)
            result.snapshots.append(fund_snapshot)
            equity_dates.append(session)
            equity_values.append(fund_snapshot.equity)
            gross_values.append(
                fund_snapshot.market_value / fund_snapshot.equity
                if fund_snapshot.equity > 0 else 0.0
            )

            # 3. Optional capital reallocation.
            if self.reallocate == "fixed":
                due = last_reallocation is None or (
                    (session - last_reallocation).days
                    >= 30 * self.reallocate_every_months
                )
                if due:
                    self._reallocate(
                        session, ledger, sleeve_cash, sleeve_equity, result
                    )
                    last_reallocation = session

            # 4. Ask every sleeve what it wants, against its own book.
            sleeve_targets: Dict[str, Dict[str, float]] = {}
            for sleeve in self.sleeves:
                sid = sleeve.strategy_id
                ctx = StrategyContext(
                    as_of=session,
                    price_panel=price_panel,
                    close_matrix=close_matrix,
                    ohlc_adjusted=ohlc_adjusted,
                    universe_fn=universe_fn,
                    portfolio=sleeve_snapshots[sid],
                    params=sleeve.params,
                    trading_sessions=all_sessions,
                    sector_map=sector_map,
                    exit_dates=exit_dates[sid],
                )
                try:
                    targets = sleeve.strategy_fn(ctx)
                except Exception as exc:
                    logger.error("Sleeve %s failed on %s: %s", sid, session, exc)
                    result.warnings.append(f"{session:%Y-%m-%d}: {sid}: {exc}")
                    continue

                sleeve_targets[sid] = risk.apply_risk_limits(
                    targets, sleeve_snapshots[sid],
                    sleeve.risk or self.risk, sector_map,
                )

            # 5. Fund-level limits on the combined book, then orders per sleeve.
            sleeve_targets = self._apply_fund_limits(
                sleeve_targets, sleeve_equity, fund_snapshot, sector_map
            )
            for sid, targets in sleeve_targets.items():
                pending[sid] = generate_orders(
                    targets, sleeve_snapshots[sid], prices, sid,
                    short_config=self.short,
                )

        return self._finalize(
            result, equity_dates, equity_values, gross_values,
            sleeve_equity_history, sleeve_cash_history,
            close_matrix, sessions, started_at,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _charge_borrow(
        self,
        ledger: Portfolio,
        session: pd.Timestamp,
        prices: Dict[str, float],
        sleeve_cash: Dict[str, float],
    ) -> float:
        """Charge the fund's borrow, then bill each sleeve for its own shorts.

        The fee is levied once on the shared ledger -- the fund holds one position
        per ticker, so it borrows once -- and then attributed in proportion to each
        sleeve's short market value. Leaving it at the fund level would let the
        sleeve running the shorts look costless while the others paid for it, which
        makes the per-sleeve attribution useless for deciding what to keep.
        """
        charged = ledger.accrue_borrow_fees(session, prices, self.short)
        if abs(charged) <= 0:
            return 0.0

        exposures: Dict[str, float] = {}
        for lot in ledger.lots:
            if lot.shares >= 0:
                continue
            price = prices.get(lot.symbol)
            if price is None or price <= 0:
                continue
            exposures[lot.strategy_id] = (
                exposures.get(lot.strategy_id, 0.0) + abs(lot.shares) * price
            )

        total = sum(exposures.values())
        if total <= 0:
            return charged

        for sleeve_id, exposure in exposures.items():
            if sleeve_id in sleeve_cash:
                sleeve_cash[sleeve_id] -= charged * (exposure / total)

        return charged

    @staticmethod
    def _held(ledger: Portfolio, strategy_id: str) -> set:
        return {
            lot.symbol for lot in ledger.lots
            if lot.strategy_id == strategy_id and lot.shares > 1e-9
        }

    def _get_bars(self, panel, date):
        return adjusted_bars(panel, date)

    def _get_close_prices(self, matrix, date):
        if date not in matrix.index:
            return {}
        row = matrix.loc[date]
        return {sym: float(p) for sym, p in row.items() if pd.notna(p)}

    def _finalize(
        self, result, equity_dates, equity_values, gross_values,
        sleeve_equity_history, sleeve_cash_history,
        close_matrix, sessions, started_at,
    ) -> FundResult:
        index = pd.DatetimeIndex(equity_dates)
        result.fund_equity = pd.Series(equity_values, index=index, name="equity")
        result.fund_returns = result.fund_equity.pct_change().fillna(0.0)
        result.gross_exposure = pd.Series(gross_values, index=index, name="gross")

        for sleeve in self.sleeves:
            sid = sleeve.strategy_id
            result.sleeve_equity[sid] = pd.Series(
                sleeve_equity_history[sid], index=index
            )
            result.sleeve_cash[sid] = pd.Series(
                sleeve_cash_history[sid], index=index
            )

        if self.benchmark in close_matrix.columns:
            bench = close_matrix[self.benchmark].loc[sessions].dropna()
            if len(bench) > 0:
                result.benchmark_equity = (
                    bench / bench.iloc[0]
                ) * self.initial_capital
                result.benchmark_returns = bench.pct_change().fillna(0.0)

        result.fund_metrics = metrics.compute_full_metrics(
            result.fund_equity, result.fund_returns,
            result.benchmark_equity, result.benchmark_returns,
        )
        for sleeve in self.sleeves:
            sid = sleeve.strategy_id
            equity = result.sleeve_equity[sid]
            result.sleeve_metrics[sid] = metrics.compute_full_metrics(
                equity, equity.pct_change().fillna(0.0),
                result.benchmark_equity, result.benchmark_returns,
            )

        result.elapsed_seconds = time.time() - started_at
        logger.info(
            "Fund complete: %.1fs, %d sessions, %d fills across %d sleeve(s), "
            "final equity $%.2f, peak gross %.2fx, netting saved $%.0f of flow",
            result.elapsed_seconds, len(sessions), len(result.all_fills),
            len(self.sleeves),
            result.fund_equity.iloc[-1] if len(result.fund_equity) else 0.0,
            result.gross_exposure.max() if len(result.gross_exposure) else 0.0,
            result.netting_saved_notional,
        )
        return result

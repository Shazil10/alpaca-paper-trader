"""Phase 5: the fast screen and the full engine must agree when friction is off.

`fast.fast_backtest` exists to reject bad ideas cheaply, so it is only useful if
it is the *same* backtest as `engine.BacktestEngine` minus the frictions. This
test pins that down on a frictionless synthetic series.

Why the two legitimately diverge, and how each source is neutralised here:

* **Costs.** The fast engine charges `turnover * cost_bps` at rebalance; the full
  engine charges spread, slippage and commission per fill. Neutralised with
  `cost_bps=0` and a zeroed `CostConfig`.
* **Integer shares.** The full engine floors share counts, leaving stub cash that
  the fast engine (which only ever compounds weights) never has. Neutralised
  with `fractional_shares=True`.
* **Cash drag and risk limits.** The fast engine has no cash account.
  Neutralised with `cash_reserve_pct=0`, `max_position_pct=1.0`,
  `max_leverage=1.0`, so target weights survive `apply_risk_limits` unchanged.
* **Participation caps.** The full engine truncates orders above
  `participation_rate * volume`. Neutralised with `participation_rate=1.0` and
  large volume.
* **Fill timing.** This one is structural and cannot be configured away: the
  full engine fills at the D+1 **open** while the fast engine compounds
  close-to-close. The fixture removes it by setting `open == close` on every bar
  and making the day-0 to day-1 return zero, so the full engine's entry fill
  happens at the day-0 mark and both curves are invested over the same window.
* **Rebalance drift.** `generate_orders` trims as well as buys, but it applies a
  symmetric `REBALANCE_BAND` dead band, so drift under that threshold is left
  alone while the fast engine rebalances exactly. The fixture gives every symbol
  an identical return path, so equal weights never drift at all and rebalancing
  is a no-op for both engines regardless of the band.

With all of that controlled the two curves are the same arithmetic in a
different order, so the only remaining gap is float association: the observed
relative disagreement on final equity is ~4e-16, and on Sharpe ~2e-14. The
`rel=1e-6` and `abs=0.01` tolerances are therefore loose by ten orders of
magnitude on purpose — they are set where a *behavioural* regression (a
reintroduced cost, a share-rounding change, an off-by-one in the fill clock)
trips them, and well clear of float noise on any platform.
"""
import numpy as np
import pandas as pd
import pytest

from backtest import fast, metrics
from backtest.engine import BacktestEngine
from backtest.types import (
    BacktestConfig, CostConfig, ExecutionConfig, FillType, RiskConfig,
)

INITIAL_CAPITAL = 100_000.0
SYMBOLS = ("AAA", "BBB", "CCC")
BASE_PRICES = (100.0, 50.0, 250.0)
N_DAYS = 60
START = "2024-01-02"


def _returns_path(n=N_DAYS, seed=42):
    """Deterministic return path shared by every symbol.

    Index 0 is the base level and index 1 is flat: with a flat day 1 the full
    engine's D+1 entry fill costs it nothing relative to the fast engine, which
    is invested from day 0.
    """
    rng = np.random.default_rng(seed)
    path = rng.normal(loc=0.0005, scale=0.01, size=n)
    path[0] = 0.0
    path[1] = 0.0
    return path


def _panel():
    """Lake-shaped panel. open == close == adj_close, identical returns."""
    rets = _returns_path()
    levels = np.cumprod(1.0 + rets)
    dates = pd.bdate_range(START, periods=N_DAYS)

    rows = []
    for i, d in enumerate(dates):
        for sym, base in zip(SYMBOLS, BASE_PRICES):
            price = base * levels[i]
            rows.append({
                "date": d, "symbol": sym,
                "open": price, "high": price, "low": price,
                "close": price, "adj_close": price,
                "volume": 50_000_000.0,
            })
    return pd.DataFrame(rows), dates


def _matrix(panel):
    return panel.pivot_table(index="date", columns="symbol", values="adj_close")


EQUAL_WEIGHTS = {sym: 1.0 / len(SYMBOLS) for sym in SYMBOLS}


def _run_full(panel, dates):
    config = BacktestConfig(
        strategy_id="parity",
        strategy_module="test",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        initial_capital=INITIAL_CAPITAL,
        benchmark="__none__",
        cost=CostConfig(
            spread_bps=0.0, slippage_bps=0.0,
            commission_per_share=0.0, min_commission=0.0,
            participation_rate=1.0,
        ),
        risk=RiskConfig(
            cash_reserve_pct=0.0, max_position_pct=1.0, max_leverage=1.0,
        ),
        execution=ExecutionConfig(
            fill_type=FillType.MARKET_OPEN, fractional_shares=True,
        ),
    )
    engine = BacktestEngine(config)
    return engine.run(lambda ctx: dict(EQUAL_WEIGHTS), panel, _matrix(panel))


def _run_fast(panel, dates):
    return fast.fast_backtest(
        lambda history, date, params: dict(EQUAL_WEIGHTS),
        _matrix(panel),
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        rebalance_freq="D",
        cost_bps=0.0,
        benchmark="__none__",
    )


@pytest.fixture(scope="module")
def parity():
    panel, dates = _panel()
    return _run_full(panel, dates), _run_fast(panel, dates)


class TestFixtureAssumptions:
    """If the fixture stops being frictionless, the parity claim is vacuous."""

    def test_open_equals_close_and_day_one_is_flat(self):
        panel, _ = _panel()
        assert (panel["open"] == panel["close"]).all()
        assert (panel["adj_close"] == panel["close"]).all()

        day0 = panel[panel["date"] == panel["date"].min()]
        day1 = panel[panel["date"] == sorted(panel["date"].unique())[1]]
        for sym in SYMBOLS:
            p0 = float(day0.loc[day0["symbol"] == sym, "close"].iloc[0])
            p1 = float(day1.loc[day1["symbol"] == sym, "close"].iloc[0])
            assert p1 == pytest.approx(p0, rel=1e-15)

    def test_path_is_non_degenerate(self, parity):
        full, _ = parity
        # A flat or constant curve would satisfy every parity assertion
        # trivially, so require real dispersion and a real move.
        assert full.returns.iloc[2:].std() > 0.005
        assert abs(full.equity_curve.iloc[-1] / INITIAL_CAPITAL - 1.0) > 0.02

    def test_full_engine_deploys_all_cash_and_never_rebalances(self, parity):
        full, _ = parity
        # One buy per symbol on the first fill date and nothing after: with
        # identical return paths, equal weights never drift past the 0.5% band.
        assert len(full.fills) == len(SYMBOLS)
        assert {f.order.symbol for f in full.fills} == set(SYMBOLS)
        # 100000 fully deployed leaves no cash to drag on the curve.
        assert full.snapshots[-1].cash == pytest.approx(0.0, abs=1e-6)


class TestEquityParity:
    def test_same_number_of_sessions(self, parity):
        full, fastr = parity
        assert len(full.equity_curve) == N_DAYS
        assert len(fastr["equity"]) == N_DAYS
        assert (full.equity_curve.index == fastr["equity"].index).all()

    def test_final_equity_agrees(self, parity):
        full, fastr = parity
        full_growth = full.equity_curve.iloc[-1] / INITIAL_CAPITAL
        fast_growth = fastr["equity"].iloc[-1]  # fast engine starts at 1.0
        assert full_growth == pytest.approx(fast_growth, rel=1e-6)

    def test_whole_curve_agrees_not_just_the_endpoint(self, parity):
        full, fastr = parity
        normalized = full.equity_curve / INITIAL_CAPITAL
        max_rel_gap = ((normalized - fastr["equity"]).abs() / fastr["equity"]).max()
        assert max_rel_gap < 1e-6

    def test_daily_returns_agree(self, parity):
        full, fastr = parity
        gap = (full.returns - fastr["returns"]).abs().max()
        assert gap < 1e-9


class TestMetricParity:
    def test_sharpe_agrees(self, parity):
        full, fastr = parity
        full_sharpe = metrics.sharpe_ratio(full.returns)
        fast_sharpe = fastr["metrics"]["sharpe"]
        assert full_sharpe == pytest.approx(fast_sharpe, abs=0.01)

    def test_max_drawdown_agrees(self, parity):
        full, fastr = parity
        full_dd = metrics.max_drawdown(full.equity_curve)
        assert full_dd == pytest.approx(fastr["metrics"]["max_drawdown"], abs=1e-6)


class TestFrictionBreaksParity:
    """A guard on the guard: with friction on, the curves must separate.

    Without this, a bug that made both engines return the same constant would
    satisfy every assertion above.
    """

    def test_costs_make_the_full_engine_underperform(self, parity):
        _, fastr = parity
        panel, dates = _panel()

        config = BacktestConfig(
            strategy_id="parity_costly",
            strategy_module="test",
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            initial_capital=INITIAL_CAPITAL,
            benchmark="__none__",
            cost=CostConfig(
                spread_bps=20.0, slippage_bps=20.0,
                commission_per_share=0.0, participation_rate=1.0,
            ),
            risk=RiskConfig(
                cash_reserve_pct=0.0, max_position_pct=1.0, max_leverage=1.0,
            ),
            execution=ExecutionConfig(
                fill_type=FillType.MARKET_OPEN, fractional_shares=True,
            ),
        )
        costly = BacktestEngine(config).run(
            lambda ctx: dict(EQUAL_WEIGHTS), panel, _matrix(panel)
        )

        costly_growth = costly.equity_curve.iloc[-1] / INITIAL_CAPITAL
        # 20/2 + 20 = 30 bps paid once on entry, so the costly run must trail
        # the frictionless fast curve by roughly 0.30%.
        drag = 1.0 - costly_growth / fastr["equity"].iloc[-1]
        assert drag == pytest.approx(0.0030, abs=1e-4)

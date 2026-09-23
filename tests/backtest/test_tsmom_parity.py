"""Parity gate for the TSMOM ETF migration.

TSMOM had no live sleeve -- the logic existed only inside
``analysis/strategies/momentum/tsmom_etf.ipynb`` as a vectorized computation. So
"the old backtest" here is that vectorized path, now extracted into
``strategies.momentum.tsmom_etf`` and imported by the notebook. These tests
compare it against the event-driven engine over the same data.

The two cannot agree exactly, and the reason is worth stating because it makes
the vectorized path optimistic rather than merely different.

``daily_streams`` holds a *constant weight vector* between month-ends and takes a
dot product against daily returns. A constant-weight portfolio is one that
rebalances to those weights every single day -- as a winner runs, it is trimmed
back overnight for free. But turnover is measured only across month-end changes,
so the cost of that daily rebalancing is never charged. The engine instead buys
shares once and lets the position drift, which is what an actual account does.

So the assertions here are: identical decisions (signs, weights, rebalance dates),
and a *correlated* return stream with the engine's drift accounted for -- not
equality. Anything that breaks the decision-level assertions is a real bug;
tracking error between the two paths is expected and bounded.

Run with: ./venv/bin/python -m pytest tests/backtest/test_tsmom_parity.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.context import StrategyContext
from backtest.types import PortfolioSnapshot
from data_pipeline import store
from strategies.momentum import tsmom_etf as ts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def etf_closes():
    """Adjusted closes for the TSMOM universe, straight from the lake."""
    matrix = store.load_close_matrix(sorted(ts.TICKERS))
    if matrix.empty:
        pytest.skip("no ETF data in the lake")

    deep = [c for c in matrix.columns if int(matrix[c].notna().sum()) >= ts.MIN_SESSIONS]
    if len(deep) < 5:
        pytest.skip(f"only {len(deep)} ETF(s) carry enough history")

    return matrix[deep].sort_index()


@pytest.fixture(scope="module")
def lake_panel():
    panel = store.load_prices(sorted(ts.TICKERS))
    if len(panel) == 0:
        pytest.skip("no ETF bars in the lake")
    return panel


def _ctx(panel, closes, as_of, params=None, held=None):
    sessions = pd.DatetimeIndex(closes.index)
    portfolio = PortfolioSnapshot(date=as_of, cash=0.0, equity=0.0)
    if held:
        portfolio = PortfolioSnapshot(
            date=as_of, cash=0.0, equity=float(sum(abs(v) for v in held.values()))
        )

    return StrategyContext(
        as_of=as_of,
        price_panel=panel,
        close_matrix=closes,
        universe_fn=lambda _d: set(closes.columns),
        portfolio=portfolio,
        params=params or {},
        trading_sessions=sessions,
    )


# ---------------------------------------------------------------------------
# Decision-level parity
# ---------------------------------------------------------------------------

class TestSignalParity:
    def test_ctx_weights_match_the_vectorized_month_end(self, lake_panel, etf_closes):
        """The engine path must reproduce the matrix path's weight vector."""
        vector = ts.run_tsmom(etf_closes)
        month_ends = ts.month_end_sessions(etf_closes.index)
        as_of = pd.Timestamp(month_ends[-1])

        expected = vector["weights"].loc[as_of]
        expected = {str(k): float(v) for k, v in expected.items() if abs(float(v)) > 1e-9}

        decision = ts.decide(_ctx(lake_panel, etf_closes, as_of))

        assert decision is not None
        assert set(decision["weights"]) == set(expected)
        for symbol, weight in expected.items():
            assert decision["weights"][symbol] == pytest.approx(weight, rel=1e-9)

    def test_signs_match(self, lake_panel, etf_closes):
        vector = ts.run_tsmom(etf_closes)
        as_of = pd.Timestamp(ts.month_end_sessions(etf_closes.index)[-1])

        expected = {
            str(k): float(v)
            for k, v in vector["sign"].loc[as_of].items() if pd.notna(v)
        }
        got = ts.decide(_ctx(lake_panel, etf_closes, as_of))["signs"]

        assert got == pytest.approx(expected)

    def test_long_only_clips_shorts_without_resizing_longs(
        self, lake_panel, etf_closes
    ):
        """Long-only is a clip, not a renormalization. Gross falls; longs do not."""
        as_of = pd.Timestamp(ts.month_end_sessions(etf_closes.index)[-1])
        ctx = _ctx(lake_panel, etf_closes, as_of)

        both = ts.decide(ctx)["weights"]
        longs = ts.decide(ctx, long_only=True)["weights"]

        assert all(w > 0 for w in longs.values())
        for symbol, weight in longs.items():
            assert both[symbol] == pytest.approx(weight)
        assert set(longs) == {s for s, w in both.items() if w > 0}


# ---------------------------------------------------------------------------
# The vol target
# ---------------------------------------------------------------------------

class TestVolTargeting:
    def test_low_vol_asset_gets_more_notional(self):
        sign = pd.DataFrame({"CALM": [1.0], "WILD": [1.0]})
        vol = pd.DataFrame({"CALM": [0.10], "WILD": [0.40]})

        weights = ts.vol_target_weights(sign, vol, sigma_target=0.40)

        # 0.40/0.10 = 4.0 and 0.40/0.40 = 1.0, each split across 2 live signals.
        assert weights["CALM"].iloc[0] == pytest.approx(2.0)
        assert weights["WILD"].iloc[0] == pytest.approx(0.5)

    def test_short_signal_flips_the_sign_not_the_size(self):
        sign = pd.DataFrame({"A": [-1.0]})
        vol = pd.DataFrame({"A": [0.20]})

        weights = ts.vol_target_weights(sign, vol, sigma_target=0.40)

        assert weights["A"].iloc[0] == pytest.approx(-2.0)

    def test_missing_signal_becomes_zero_not_a_dropped_column(self):
        """The weight matrix must stay rectangular for the dot product."""
        sign = pd.DataFrame({"A": [1.0], "B": [np.nan]})
        vol = pd.DataFrame({"A": [0.20], "B": [0.20]})

        weights = ts.vol_target_weights(sign, vol)

        assert list(weights.columns) == ["A", "B"]
        assert weights["B"].iloc[0] == 0.0
        # Only A is live, so it is not halved.
        assert weights["A"].iloc[0] == pytest.approx(2.0)

    def test_vol_needs_a_full_halflife_before_it_reports(self):
        """A four-observation vol estimate would hand that asset a huge weight."""
        returns = pd.DataFrame(
            {"A": np.random.RandomState(0).normal(0, 0.01, 100)},
            index=pd.bdate_range("2024-01-01", periods=100),
        )

        vol = ts.ewma_annualized_vol(returns, halflife=60)

        assert vol["A"].iloc[:59].isna().all()
        assert pd.notna(vol["A"].iloc[-1])

    def test_gross_exposure_is_levered_by_design(self, etf_closes):
        """Sizing to 40% vol on assets running at 15% asks for more than 1x."""
        vector = ts.run_tsmom(etf_closes)
        gross = vector["weights"].abs().sum(axis=1).dropna()

        if gross.empty:
            pytest.skip("no weights computed")
        assert gross.median() > 1.0, "a 40% vol target on ETFs must lever up"


# ---------------------------------------------------------------------------
# Execution convention
# ---------------------------------------------------------------------------

class TestExecutionTiming:
    def test_month_end_weights_are_earned_from_the_next_session(self):
        """.shift(1) is the whole execution assumption of the matrix path."""
        idx = pd.bdate_range("2024-01-01", periods=45)
        returns = pd.DataFrame({"A": [0.0] * 45}, index=idx)
        # A 10% move on the session right after the month-end.
        me = ts.month_end_sessions(idx)[0]
        day_after = idx[idx.get_loc(me) + 1]
        returns.loc[day_after, "A"] = 0.10

        weights = pd.DataFrame({"A": [1.0]}, index=[me])
        net, _ = ts.daily_streams(weights, returns, cost_bps=0.0)

        assert net.loc[me] == pytest.approx(0.0), "the signal day must earn nothing"
        assert net.loc[day_after] == pytest.approx(0.10)

    def test_turnover_is_charged_on_weight_changes(self):
        idx = pd.bdate_range("2024-01-01", periods=45)
        returns = pd.DataFrame({"A": [0.0] * 45}, index=idx)
        me = ts.month_end_sessions(idx)
        weights = pd.DataFrame({"A": [1.0]}, index=[me[0]])

        net, turn = ts.daily_streams(weights, returns, cost_bps=100.0)

        # One unit of turnover when the position goes on, charged at 100 bps.
        assert turn.sum() == pytest.approx(1.0)
        assert net.sum() == pytest.approx(-0.01)

    def test_month_end_is_a_trading_session_not_a_calendar_date(self):
        """2024-03-31 was a Sunday. The signal must land on the 28th."""
        idx = pd.bdate_range("2024-03-01", "2024-04-10")
        # Good Friday 2024-03-29 removed, so the month's last session is the 28th.
        idx = idx[idx != pd.Timestamp("2024-03-29")]

        ends = ts.month_end_sessions(idx)

        assert pd.Timestamp("2024-03-28") in ends
        assert pd.Timestamp("2024-03-31") not in ends

    def test_rebalance_day_is_the_last_session_of_the_month(
        self, lake_panel, etf_closes
    ):
        ends = set(ts.month_end_sessions(etf_closes.index))
        sessions = pd.DatetimeIndex(etf_closes.index)

        # Take a month-end and the session before it.
        as_of = pd.Timestamp(sorted(ends)[-2])
        prior = sessions[sessions.get_loc(as_of) - 1]

        assert ts.is_rebalance_day(_ctx(lake_panel, etf_closes, as_of))
        assert not ts.is_rebalance_day(_ctx(lake_panel, etf_closes, prior))


# ---------------------------------------------------------------------------
# Engine agreement
# ---------------------------------------------------------------------------

class TestEngineAgreement:
    def test_engine_tracks_the_vectorized_long_only_stream(
        self, lake_panel, etf_closes
    ):
        """Correlated, not equal -- and the gap is the matrix path's optimism.

        ``daily_streams`` holds a constant weight vector between month-ends,
        which is a portfolio rebalanced to target every day, yet charges turnover
        only on month-end changes. The engine buys shares once and lets them
        drift. So the engine is the honest one and the two must move together
        without matching.
        """
        from backtest.engine import BacktestEngine
        from backtest.types import (
            BacktestConfig, CostConfig, ExecutionConfig, RiskConfig,
        )

        start, end = "2015-01-02", "2020-12-31"
        window = etf_closes.loc[start:end]
        if len(window) < 500:
            pytest.skip("not enough history for a multi-year comparison")

        config = BacktestConfig(
            strategy_id="tsmom_parity",
            strategy_module="strategies.momentum.tsmom_etf",
            start_date=start,
            end_date=end,
            initial_capital=1_000_000.0,
            benchmark="SPY",
            cost=CostConfig(
                spread_bps=0.0, slippage_bps=0.0,
                commission_per_share=0.0, participation_rate=1.0,
            ),
            risk=RiskConfig(
                max_position_pct=1.0, max_sector_pct=1.0, cash_reserve_pct=0.0,
                max_leverage=10.0, max_positions=50, fractional_shares=True,
            ),
            execution=ExecutionConfig(fractional_shares=True),
            params={"long_only": True},
        )

        result = BacktestEngine(config).run(
            strategy_fn=ts.target_weights,
            price_panel=lake_panel,
            close_matrix=etf_closes,
            universe_fn=lambda _d: set(etf_closes.columns),
        )

        vector = ts.run_tsmom(etf_closes, cost_bps=0.0)["lo"].loc[start:end]
        engine = result.returns

        common = vector.index.intersection(engine.index)
        assert len(common) > 400, "the two paths must cover the same sessions"

        correlation = float(np.corrcoef(vector.loc[common], engine.loc[common])[0, 1])
        assert correlation > 0.90, f"paths diverged: correlation {correlation:.3f}"

        # Same sign of edge, same rough magnitude.
        assert np.sign(vector.loc[common].mean()) == np.sign(engine.loc[common].mean())
        ratio = engine.loc[common].std() / vector.loc[common].std()
        assert 0.5 < ratio < 2.0, f"volatility ratio {ratio:.2f} is not comparable"

    def test_constant_weights_imply_daily_rebalancing(self):
        """Pin the reason the two paths differ, so it is not mistaken for a bug.

        Hold w=1.0 in one asset that doubles, then halves. A constant-weight
        stream returns to its starting value; a buy-and-hold share count does
        too, but only because the path is symmetric -- the point is that the
        matrix path never has to trade to stay at w=1.0.
        """
        idx = pd.bdate_range("2024-01-01", periods=45)
        me = ts.month_end_sessions(idx)[0]
        returns = pd.DataFrame({"A": [0.0] * 45}, index=idx)
        after = idx[idx.get_loc(me) + 1:]
        returns.loc[after[0], "A"] = 1.0    # +100%
        returns.loc[after[1], "A"] = -0.5   # -50%

        net, turn = ts.daily_streams(
            pd.DataFrame({"A": [1.0]}, index=[me]), returns, cost_bps=0.0
        )

        # Compounded back to flat.
        assert float((1 + net).prod()) == pytest.approx(1.0)

        # Turnover is charged once, when the position goes on...
        assert turn.loc[after[0]] == pytest.approx(1.0)
        # ...and never again, although holding w=1.0 through a double and a halve
        # would have required real trades to stay at that weight. That unbilled
        # rebalancing is exactly what makes the matrix path optimistic.
        assert turn.loc[after[1]] == pytest.approx(0.0)
        assert turn.loc[after[2]] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Universe handling
# ---------------------------------------------------------------------------

class TestUniverse:
    def test_tradable_reports_only_symbols_with_enough_history(
        self, lake_panel, etf_closes
    ):
        as_of = pd.Timestamp(etf_closes.index[-1])
        ctx = _ctx(lake_panel, etf_closes, as_of)

        symbols = ts.tradable(ctx)

        assert symbols
        assert set(symbols) <= set(ts.TICKERS)

    def test_early_session_declines_rather_than_guessing(self, lake_panel, etf_closes):
        """Before the lookback is covered there is no signal, so hold."""
        early = pd.Timestamp(etf_closes.index[30])
        ctx = _ctx(lake_panel, etf_closes, early)

        assert ts.decide(ctx) is None
        assert ts.target_weights(ctx) == {}

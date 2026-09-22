"""Corporate action invariants: a split or a dividend must not move equity.

How the engine consumes prices matters here. The lake stores the *unadjusted*
print in `close`/`open` plus a back-adjusted `adj_close`
(`data/prices/_schema.md`). The engine marks to market off `close_matrix`, which
is `adj_close`, while the broker fills off the OHLC in `price_panel`. Those two
are on different scales whenever a corporate action sits between the bar and the
end of the series, and the ratio is exactly `adj_close / close` — the same factor
`store.load_ohlc_adjusted()` uses.

The economic invariants under test:

* A 2-for-1 split shows up as `close` halving while `adj_close` stays
  continuous. Holding through it must not change equity: you own twice as many
  shares at half the price.
* A dividend back-adjusts every pre-ex-date price down by `1 - d / P`.
  `adj_close` is the total-return series, so a holder's equity must move
  continuously through the ex-date and must capture the dividend.

Both are asserted as "the equity curve is proportional to the adjusted price
series", which is the strongest statement available and is independent of which
scale the engine chooses internally — as long as it picks one and sticks to it.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from backtest.types import (
    BacktestConfig, CostConfig, ExecutionConfig, FillType, RiskConfig,
)

INITIAL_CAPITAL = 10_000.0
N_DAYS = 9
SPLIT_IDX = 4  # index into the session list where the action takes effect


def _panel_with_action(adjusted: np.ndarray, raw_factor: np.ndarray):
    """Build a lake-shaped panel from an adjusted series plus a scale factor.

    `adjusted` is the continuous (total-return) series that ends up in
    `adj_close`. `raw_factor` is `close / adj_close` per bar, i.e. what the
    unadjusted print was before back-adjustment. open == close so that the
    D+1 open fill and the day-D close mark sit at the same price.
    """
    dates = pd.bdate_range("2024-01-02", periods=len(adjusted))
    rows = []
    for i, d in enumerate(dates):
        raw = adjusted[i] * raw_factor[i]
        rows.append({
            "date": d, "symbol": "AAPL",
            "open": raw, "high": raw * 1.001, "low": raw * 0.999,
            "close": raw, "adj_close": adjusted[i],
            # Raw share volume: a split doubles it, matching the price halving.
            "volume": 1_000_000.0 * raw_factor[i],
        })
    return pd.DataFrame(rows), dates


def _matrix(panel):
    return panel.pivot_table(index="date", columns="symbol", values="adj_close")


def _buy_and_hold_config(dates):
    return BacktestConfig(
        strategy_id="test",
        strategy_module="test",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        initial_capital=INITIAL_CAPITAL,
        benchmark="AAPL",
        cost=CostConfig(
            spread_bps=0.0, slippage_bps=0.0,
            commission_per_share=0.0, participation_rate=1.0,
        ),
        risk=RiskConfig(cash_reserve_pct=0.0, max_position_pct=1.0, max_leverage=1.0),
        execution=ExecutionConfig(
            fill_type=FillType.MARKET_OPEN, fractional_shares=True,
        ),
    )


def _run_buy_and_hold(panel, dates):
    engine = BacktestEngine(_buy_and_hold_config(dates))
    return engine.run(lambda ctx: {"AAPL": 1.0}, panel, _matrix(panel))


def _drift(n=N_DAYS, base=50.0, daily=0.005):
    """Continuous adjusted series. Flat on day 1 so the D+1 entry fill is clean."""
    series = [base, base]
    for _ in range(n - 2):
        series.append(series[-1] * (1 + daily))
    return np.array(series[:n])


class TestTwoForOneSplit:
    """close halves on the split date; adj_close does not."""

    def _panel(self):
        adjusted = _drift()
        # Back-adjusted convention: pre-split prints were twice today's scale.
        factor = np.where(np.arange(N_DAYS) < SPLIT_IDX, 2.0, 1.0)
        return _panel_with_action(adjusted, factor)

    def test_fixture_really_contains_a_split(self):
        panel, _ = self._panel()
        closes = panel["close"].to_numpy()
        adj = panel["adj_close"].to_numpy()

        # close drops ~50% across the split boundary while adj_close drifts +0.5%.
        assert closes[SPLIT_IDX] / closes[SPLIT_IDX - 1] == pytest.approx(0.5025)
        assert adj[SPLIT_IDX] / adj[SPLIT_IDX - 1] == pytest.approx(1.005)

    def test_equity_is_continuous_across_the_split(self):
        panel, dates = self._panel()
        result = _run_buy_and_hold(panel, dates)

        equity = result.equity_curve
        assert len(equity) == N_DAYS

        # The only price move on the split date is the underlying +0.5% drift.
        # A scale mismatch between the fill price and the mark would show up
        # here as roughly -50% or +100%.
        split_return = equity.iloc[SPLIT_IDX] / equity.iloc[SPLIT_IDX - 1] - 1
        assert split_return == pytest.approx(0.005, abs=1e-9)

        # No day in the whole run may move more than the 0.5% drift.
        assert result.returns.iloc[1:].abs().max() == pytest.approx(0.005, abs=1e-9)

    def test_equity_tracks_the_adjusted_series_exactly(self):
        panel, dates = self._panel()
        result = _run_buy_and_hold(panel, dates)

        adjusted = _drift()
        equity = result.equity_curve

        # Fully invested from the D+1 fill on day index 1 at adj price 50.
        # 10000 / 50 = 200 split-adjusted shares, held to the end.
        for i in range(1, N_DAYS):
            expected = INITIAL_CAPITAL * adjusted[i] / adjusted[1]
            assert equity.iloc[i] == pytest.approx(expected, rel=1e-12)

    def test_entry_fill_is_on_the_same_scale_as_the_mark(self):
        panel, dates = self._panel()
        result = _run_buy_and_hold(panel, dates)

        buys = [f for f in result.fills if f.order.side.value == "BUY"]
        assert len(buys) == 1
        # Day 1 adj_close is 50 and the raw print is 100 (pre-split). The fill
        # must use 50, otherwise 10000 buys 100 shares and marks at 5000.
        assert buys[0].fill_price == pytest.approx(50.0, rel=1e-12)
        assert buys[0].fill_shares == pytest.approx(200.0, rel=1e-12)

        first_snap = result.snapshots[1]
        assert first_snap.equity == pytest.approx(INITIAL_CAPITAL, rel=1e-12)
        assert first_snap.cash == pytest.approx(0.0, abs=1e-6)


class TestDividend:
    """An ex-dividend date back-adjusts every earlier print downward."""

    DIVIDEND_FACTOR = 1.0 / (1.0 - 0.02)  # a 2% dividend

    def _panel(self):
        adjusted = _drift()
        factor = np.where(
            np.arange(N_DAYS) < SPLIT_IDX, self.DIVIDEND_FACTOR, 1.0
        )
        return _panel_with_action(adjusted, factor)

    def test_fixture_really_contains_a_dividend(self):
        panel, _ = self._panel()
        closes = panel["close"].to_numpy()

        # Raw close drops through the ex-date by the dividend net of drift:
        # 1.005 * 0.98 = 0.9849
        assert closes[SPLIT_IDX] / closes[SPLIT_IDX - 1] == pytest.approx(0.9849)

    def test_equity_is_continuous_across_the_ex_date(self):
        panel, dates = self._panel()
        result = _run_buy_and_hold(panel, dates)

        equity = result.equity_curve
        ex_return = equity.iloc[SPLIT_IDX] / equity.iloc[SPLIT_IDX - 1] - 1
        # A holder earns the drift and keeps the dividend: +0.5%, not -1.5%.
        assert ex_return == pytest.approx(0.005, abs=1e-9)

    def test_entry_does_not_lose_the_dividend_to_a_scale_mismatch(self):
        panel, dates = self._panel()
        result = _run_buy_and_hold(panel, dates)

        # Continuity alone is weak here: shares are fixed after entry, so the
        # curve is smooth even if the entry level is wrong. The level is what
        # the back-adjustment factor corrupts, so assert it directly.
        # Day 1 raw print is 50 / 0.98 = 51.02; the fill must use adj 50.
        buys = [f for f in result.fills if f.order.side.value == "BUY"]
        assert buys[0].fill_price == pytest.approx(50.0, rel=1e-12)
        # 10000 / 50 = 200 shares marked at 50 = 10000, not 196 * 50 = 9800.
        assert result.equity_curve.iloc[1] == pytest.approx(INITIAL_CAPITAL, rel=1e-12)

    def test_total_return_equals_the_adjusted_series_return(self):
        panel, dates = self._panel()
        result = _run_buy_and_hold(panel, dates)

        adjusted = _drift()
        equity = result.equity_curve

        # Held from day index 1 to the end: (1.005 ** 7) - 1.
        expected = adjusted[-1] / adjusted[1]
        assert equity.iloc[-1] / equity.iloc[1] == pytest.approx(expected, rel=1e-12)
        assert expected == pytest.approx(1.005 ** (N_DAYS - 2), rel=1e-12)


class TestNoCorporateAction:
    """Control: with adj_close == close the fixture must behave identically."""

    def test_equity_tracks_prices(self):
        adjusted = _drift()
        panel, dates = _panel_with_action(adjusted, np.ones(N_DAYS))
        result = _run_buy_and_hold(panel, dates)

        equity = result.equity_curve
        for i in range(1, N_DAYS):
            expected = INITIAL_CAPITAL * adjusted[i] / adjusted[1]
            assert equity.iloc[i] == pytest.approx(expected, rel=1e-12)

"""Parity gate for the triple-trigger momentum migration.

Like TSMOM, this strategy had no live sleeve -- the logic lived only in
``analysis/strategies/momentum/triple_trigger_momentum.ipynb``. So the comparison
is between the extracted vectorized screen and the engine path, both reading the
same lake.

The triggers get hand-built fixtures rather than only lake data, because the
strategy is a *conjunction* and the interesting failures are in the corners: a
name that passes two triggers and fails the third must score nothing, not score
less. Lake data cannot be relied on to contain each of those corners.

This is also the only sleeve that reads volume, so the volume path gets its own
section: the adjustment scaling, and what happens when volume is missing rather
than zero.

Run with: ./venv/bin/python -m pytest tests/backtest/test_triple_trigger_parity.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.context import StrategyContext
from backtest.types import PortfolioSnapshot
from data_pipeline import schema, store
from strategies.momentum import triple_trigger as tt

N = tt.MIN_SESSIONS + 40


# ---------------------------------------------------------------------------
# Hand-built fixtures
# ---------------------------------------------------------------------------

def _frames(price_tail, volume_tail, base_price=100.0, base_volume=1_000_000.0):
    """A long flat history followed by a controlled tail.

    The flat run is what lets the rolling windows warm up; the tail is the only
    part any assertion looks at.
    """
    idx = pd.bdate_range("2020-01-01", periods=N)
    n_tail = len(price_tail)

    prices = [base_price] * (N - n_tail) + list(price_tail)
    volumes = [base_volume] * (N - n_tail) + list(volume_tail)

    closes = pd.DataFrame({"AAA": prices}, index=idx, dtype="float64")
    vols = pd.DataFrame({"AAA": volumes}, index=idx, dtype="float64")
    return closes, vols


def _rising_tail(n=30, step=1.0, start=100.0):
    return [start + step * i for i in range(n)]


def _volume_spike(multiple=3.0, base=1_000_000.0, n=30, spike_len=5):
    """Base volume for most of the tail, elevated only at the end.

    The spike has to be short relative to ``VOL_LOOKBACK`` or the ratio collapses
    to exactly 1.0: a volume that has been elevated for longer than its own
    averaging window *is* the average, and ``ratio > 1`` then fails. Easy to get
    wrong, and it fails as "the trigger is broken" rather than "the fixture is".
    """
    return [base] * (n - spike_len) + [base * multiple] * spike_len


# ---------------------------------------------------------------------------
# The three triggers
# ---------------------------------------------------------------------------

class TestTriggers:
    def test_all_three_passing_produces_a_score(self):
        """Rising price, heavy volume, near the top of the range."""
        closes, volumes = _frames(_rising_tail(), _volume_spike())

        score = tt.compute_scores(closes, volumes)["AAA"].iloc[-1]

        assert pd.notna(score)
        assert score > 0

    def test_flat_momentum_fails_even_with_volume_and_range(self):
        """Trigger 1 alone is enough to disqualify: the gates are a conjunction."""
        closes, volumes = _frames([100.0] * 30, [5_000_000.0] * 30)

        assert pd.isna(tt.compute_scores(closes, volumes)["AAA"].iloc[-1])

    def test_thin_volume_fails_even_with_strong_momentum(self):
        """A move nobody else participated in scores nothing, not less."""
        closes, volumes = _frames(_rising_tail(step=3.0), [400_000.0] * 30)

        assert pd.isna(tt.compute_scores(closes, volumes)["AAA"].iloc[-1])

    def test_lower_half_of_the_range_fails(self):
        """A bounce off the lows is not an advance.

        Price spends the history at 200, collapses to 100, then ticks up. The
        14-day return is positive and volume is heavy, but the 325-day range
        still spans 100-200 so the name sits at the bottom of it.
        """
        idx = pd.bdate_range("2020-01-01", periods=N)
        prices = [200.0] * (N - 30) + [100.0] * 15 + _rising_tail(15, 0.5, 100.0)
        closes = pd.DataFrame({"AAA": prices}, index=idx, dtype="float64")
        volumes = pd.DataFrame(
            {"AAA": [1_000_000.0] * (N - 30) + [4_000_000.0] * 30},
            index=idx, dtype="float64",
        )

        scores = tt.compute_scores(closes, volumes)
        position = (
            (closes - closes.rolling(tt.RANGE_LOOKBACK).min())
            / (closes.rolling(tt.RANGE_LOOKBACK).max()
               - closes.rolling(tt.RANGE_LOOKBACK).min())
        )["AAA"].iloc[-1]

        assert position < tt.MIN_RANGE_POSITION
        assert pd.isna(scores["AAA"].iloc[-1])

    def test_partial_windows_yield_nan_not_a_short_average(self):
        """A 20-day volume average from three days makes anything look heavy."""
        idx = pd.bdate_range("2020-01-01", periods=10)
        closes = pd.DataFrame({"AAA": _rising_tail(10)}, index=idx)
        volumes = pd.DataFrame({"AAA": [1e6] * 10}, index=idx)

        assert tt.compute_scores(closes, volumes)["AAA"].isna().all()

    def test_score_is_the_product_of_the_three_terms(self):
        closes, volumes = _frames(_rising_tail(), _volume_spike())

        scores = tt.compute_scores(closes, volumes)

        momentum = closes.pct_change(tt.MOM_LOOKBACK)["AAA"].iloc[-1]
        ratio = (
            volumes["AAA"]
            / volumes["AAA"].rolling(tt.VOL_LOOKBACK).mean()
        ).iloc[-1]
        low = closes["AAA"].rolling(tt.RANGE_LOOKBACK).min().iloc[-1]
        high = closes["AAA"].rolling(tt.RANGE_LOOKBACK).max().iloc[-1]
        position = (closes["AAA"].iloc[-1] - low) / (high - low)

        assert scores["AAA"].iloc[-1] == pytest.approx(momentum * ratio * position)

    def test_thresholds_are_reachable_as_parameters(self):
        """A sweep must be able to move the gates."""
        closes, volumes = _frames(_rising_tail(step=0.2), _volume_spike(multiple=1.4))

        strict = tt.compute_scores(closes, volumes, min_volume_ratio=2.0)
        loose = tt.compute_scores(closes, volumes, min_volume_ratio=1.0)

        assert pd.isna(strict["AAA"].iloc[-1])
        assert pd.notna(loose["AAA"].iloc[-1])


# ---------------------------------------------------------------------------
# Ranking and sizing
# ---------------------------------------------------------------------------

class TestRankingAndSizing:
    def test_picks_are_the_highest_scores_in_order(self):
        scores = pd.Series({"A": 0.1, "B": 0.5, "C": np.nan, "D": 0.3})

        assert tt.rank_picks(scores, top_n=2) == ["B", "D"]

    def test_failing_names_are_not_picked_even_to_fill_the_book(self):
        """A NaN score is disqualified, not merely last."""
        scores = pd.Series({"A": 0.1, "B": np.nan, "C": np.nan})

        assert tt.rank_picks(scores, top_n=3) == ["A"]

    def test_weights_are_equal_not_score_weighted(self):
        """The score is a product of three scales, so its size is not conviction."""
        weights = tt.equal_weights(["A", "B", "C", "D"])

        assert set(weights.values()) == {0.25}
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_no_picks_means_all_cash(self):
        assert tt.equal_weights([]) == {}


# ---------------------------------------------------------------------------
# Volume handling -- unique to this sleeve
# ---------------------------------------------------------------------------

class TestVolume:
    def _ctx(self, panel, closes, as_of, params=None):
        return StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=closes,
            universe_fn=lambda _d: set(closes.columns),
            params=params or {},
            trading_sessions=pd.DatetimeIndex(closes.index),
        )

    def test_volume_is_rescaled_by_the_adjustment_factor(self):
        """Restated post-split volume must be compared on one scale.

        The vendor halves pre-split prices and doubles pre-split volume. Feeding
        raw volume into a rolling average that spans the split would read the
        restatement as a surge in participation.
        """
        rows = [
            # A 2-for-1 split: adj_close is half the print, so factor = 0.5 and
            # volume scales the other way.
            ("2024-01-02", "AAA", 200.0, 200.0, 200.0, 200.0, 100.0, 1_000_000),
            ("2024-01-03", "AAA", 100.0, 100.0, 100.0, 100.0, 100.0, 2_000_000),
        ]
        panel = pd.DataFrame(rows, columns=schema.COLUMNS)
        panel[schema.DATE] = pd.to_datetime(panel[schema.DATE])
        closes = panel.pivot_table(
            index=schema.DATE, columns=schema.SYMBOL,
            values=schema.ADJ_CLOSE, aggfunc="last",
        )
        closes.columns.name = None

        ctx = self._ctx(panel, closes, pd.Timestamp("2024-01-03"))
        volumes = tt.volumes_from_ctx(ctx, ["AAA"])

        # factor 0.5 on the first bar: 1,000,000 / 0.5 = 2,000,000.
        assert volumes["AAA"].iloc[0] == pytest.approx(2_000_000.0)
        # factor 1.0 on the second: unchanged.
        assert volumes["AAA"].iloc[1] == pytest.approx(2_000_000.0)

    def test_missing_volume_screens_the_name_out(self):
        """Volume is nullable and never zero-filled. No volume, no trigger 2."""
        closes, volumes = _frames(_rising_tail(), _volume_spike())
        volumes.iloc[-1, 0] = np.nan

        assert pd.isna(tt.compute_scores(closes, volumes)["AAA"].iloc[-1])

    def test_zero_volume_average_does_not_divide_by_zero(self):
        closes, volumes = _frames(_rising_tail(), _volume_spike())
        volumes.iloc[:, 0] = 0.0

        scores = tt.compute_scores(closes, volumes)

        assert scores["AAA"].isna().all()
        assert not np.isinf(scores["AAA"].fillna(0)).any()


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------

class TestCadence:
    def _ctx(self, closes, as_of, held=None):
        positions = {}
        if held:
            from backtest.types import Position
            positions = {
                s: Position(
                    symbol=s, shares=1.0, avg_entry_price=100.0, market_price=100.0,
                    market_value=100.0, unrealized_pnl=0.0, entry_date=as_of,
                )
                for s in held
            }
        portfolio = PortfolioSnapshot(
            date=as_of, cash=0.0,
            equity=100.0 * max(len(positions), 1), positions=positions,
        )
        panel = pd.DataFrame(columns=schema.COLUMNS)
        return StrategyContext(
            as_of=as_of, price_panel=panel, close_matrix=closes,
            portfolio=portfolio, trading_sessions=pd.DatetimeIndex(closes.index),
        )

    def test_an_empty_book_always_rebalances(self):
        """Otherwise a run starting off-cadence sits in cash for four sessions."""
        idx = pd.bdate_range("2024-01-01", periods=7)
        closes = pd.DataFrame({"AAA": [1.0] * 7}, index=idx)

        assert tt.is_rebalance_day(self._ctx(closes, idx[3]))

    def test_cadence_counts_sessions_not_weekdays(self):
        idx = pd.bdate_range("2024-01-01", periods=11)
        closes = pd.DataFrame({"AAA": [1.0] * 11}, index=idx)

        held = ["AAA"]
        # Session index 0, 5, 10 rebalance; the rest hold.
        assert tt.is_rebalance_day(self._ctx(closes.iloc[:1], idx[0], held))
        assert tt.is_rebalance_day(self._ctx(closes.iloc[:6], idx[5], held))
        assert tt.is_rebalance_day(self._ctx(closes.iloc[:11], idx[10], held))
        assert not tt.is_rebalance_day(self._ctx(closes.iloc[:4], idx[3], held))


# ---------------------------------------------------------------------------
# Engine path against the lake
# ---------------------------------------------------------------------------

class TestAgainstTheLake:
    @pytest.fixture(scope="class")
    def lake(self):
        calendar = store.trading_calendar()
        if len(calendar) == 0:
            pytest.skip("price lake is empty")

        as_of = pd.Timestamp(calendar[-1])
        panel = store.load_prices(start=as_of - pd.Timedelta(days=900), end=as_of)
        if len(panel) == 0:
            pytest.skip("no bars in the window")

        closes = store.load_close_matrix(
            start=as_of - pd.Timedelta(days=900), end=as_of
        )
        deep = [
            c for c in closes.columns
            if int(closes[c].notna().sum()) >= tt.MIN_SESSIONS
        ]
        if len(deep) < tt.TOP_N:
            pytest.skip(f"only {len(deep)} symbol(s) with enough history")

        return as_of, panel, closes[sorted(deep)[:150]]

    def test_decide_returns_a_full_book_of_picks(self, lake):
        as_of, panel, closes = lake
        ctx = StrategyContext(
            as_of=as_of, price_panel=panel, close_matrix=closes,
            universe_fn=lambda _d: set(closes.columns),
            trading_sessions=pd.DatetimeIndex(closes.index),
        )

        decision = tt.decide(ctx)

        assert decision is not None
        assert decision["passing"] <= decision["universe"]
        assert len(decision["picks"]) <= tt.TOP_N
        assert sum(decision["weights"].values()) == pytest.approx(
            1.0 if decision["picks"] else 0.0
        )

    def test_engine_path_and_vectorized_path_pick_the_same_names(self, lake):
        """One screen implementation, reached two ways."""
        as_of, panel, closes = lake
        ctx = StrategyContext(
            as_of=as_of, price_panel=panel, close_matrix=closes,
            universe_fn=lambda _d: set(closes.columns),
            trading_sessions=pd.DatetimeIndex(closes.index),
        )

        volumes = tt.volumes_from_ctx(ctx, list(closes.columns)).reindex(
            index=closes.index, columns=closes.columns
        )
        expected = tt.rank_picks(
            tt.compute_scores(closes, volumes).iloc[-1], tt.TOP_N
        )

        assert tt.decide(ctx)["picks"] == expected

    def test_weights_never_exceed_the_book(self, lake):
        as_of, panel, closes = lake
        ctx = StrategyContext(
            as_of=as_of, price_panel=panel, close_matrix=closes,
            universe_fn=lambda _d: set(closes.columns),
            trading_sessions=pd.DatetimeIndex(closes.index),
        )

        weights = tt.target_weights(ctx)

        assert sum(weights.values()) <= 1.0 + 1e-9
        assert all(w > 0 for w in weights.values())

"""Price *levels* must be honest, not just returns.

``adj_close`` is back-adjusted with every corporate action up to the download
date. Returns survive that; levels do not. A 2010 AAPL bar reads about $8 on the
adjusted scale against a $251 print, so a ``price >= 10`` screen run on adjusted
levels drops a name that was never cheap, and whole-share rounding rounds to a
fraction of a share that could not have been bought.

These tests pin the three places the fix has to hold:

* the bar handed to the broker still carries the untouched print,
* whole-share rounding is enforced in raw-share space, and
* ``StrategyContext.raw_close`` returns the print, not the restated level.

Plus the two wiring details that are easy to get silently wrong: the runner's
as-of anchoring must not overwrite the raw print, and the PIT universe must
still contain the fixed ETFs, which are never index constituents.

Run with: ./venv/bin/python -m pytest tests/backtest/test_asof_levels.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from backtest import runner
from backtest.broker import SimulatedBroker, adjusted_bars, _floor_whole_shares
from backtest.context import StrategyContext
from backtest.types import CostConfig, ExecutionConfig, Order, OrderSide
from data_pipeline import schema
from data_pipeline.schema import COLUMNS

FILL_DATE = pd.Timestamp("2024-01-03")

FREE = CostConfig(
    spread_bps=0.0,
    slippage_bps=0.0,
    commission_per_share=0.0,
    min_commission=0.0,
    participation_rate=1.0,
)


def _row(date, symbol, close, adj, volume=1_000_000):
    """One bar with OHLC all equal to `close` and adj_close stated separately."""
    return (date, symbol, close, close, close, close, adj, volume)


def _panel(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df[schema.DATE] = pd.to_datetime(df[schema.DATE])
    return df


# A 4-for-1 split lands after the window: the $200 print is restated to $50,
# so the stored factor on every earlier bar is 0.25.
SPLIT_PANEL = _panel([
    _row("2024-01-02", "ACME", 200.0, 50.0),
    _row("2024-01-03", "ACME", 204.0, 51.0),
    _row("2024-01-02", "PLAIN", 20.0, 20.0),
    _row("2024-01-03", "PLAIN", 21.0, 21.0),
])


class TestBarCarriesTheRawPrint:
    def test_raw_fields_survive_adjustment(self):
        bars = adjusted_bars(SPLIT_PANEL, FILL_DATE)

        assert bars["ACME"]["raw_close"] == pytest.approx(204.0)
        assert bars["ACME"]["raw_open"] == pytest.approx(204.0)
        # 51 / 204 = 0.25
        assert bars["ACME"]["factor"] == pytest.approx(0.25)
        # The adjusted scale is still what marks run on.
        assert bars["ACME"]["close"] == pytest.approx(51.0)

    def test_factor_is_one_without_corporate_actions(self):
        bars = adjusted_bars(SPLIT_PANEL, FILL_DATE)

        assert bars["PLAIN"]["factor"] == pytest.approx(1.0)
        assert bars["PLAIN"]["raw_close"] == pytest.approx(21.0)
        assert bars["PLAIN"]["close"] == pytest.approx(21.0)


class TestWholeShareRoundingIsRawSpace:
    def test_floor_lands_on_a_whole_raw_share(self):
        # 10 adjusted shares at factor 0.25 is 2.5 real shares, which never
        # traded. Floor to 2 real shares = 8 adjusted shares.
        assert _floor_whole_shares(10.0, 0.25) == pytest.approx(8.0)

    def test_sub_one_raw_share_is_rejected(self):
        # 3 adjusted shares * 0.25 = 0.75 of a real share: not buyable.
        assert _floor_whole_shares(3.0, 0.25) == 0.0

    def test_unit_factor_is_a_plain_floor(self):
        assert _floor_whole_shares(7.9, 1.0) == pytest.approx(7.0)

    def test_degenerate_factor_falls_back_to_plain_floor(self):
        assert _floor_whole_shares(7.9, 0.0) == pytest.approx(7.0)
        assert _floor_whole_shares(7.9, float("nan")) == pytest.approx(7.0)

    def test_fill_shares_map_to_whole_real_shares(self):
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=False))
        bars = adjusted_bars(SPLIT_PANEL, FILL_DATE)

        # $1,000 at an adjusted $51 is 19.6 adjusted shares = 4.9 real shares.
        order = Order(
            symbol="ACME", side=OrderSide.BUY, notional=1000.0,
            strategy_id="test", created_date=pd.Timestamp("2024-01-02"),
        )
        fills = broker.fill_orders([order], FILL_DATE, bars)

        assert len(fills) == 1
        real_shares = fills[0].fill_shares * bars["ACME"]["factor"]
        assert real_shares == pytest.approx(4.0)
        # 4 real shares at the $204 print = $816 = 16 adjusted shares at $51.
        assert fills[0].fill_shares == pytest.approx(16.0)

    def test_epsilon_recovers_a_share_lost_to_float_residue(self):
        """`shares * factor` should be whole but arrives a hair under.

        A fill stores ``raw / factor`` adjusted shares. Passing that back through
        ``adjusted * factor`` lands on 117.99999999998, and a bare floor called
        that 117 -- selling one share fewer than the position held.
        """
        factor = 0.25
        held_adjusted = 118.0 / factor  # exactly 118 raw shares

        assert _floor_whole_shares(held_adjusted, factor) == pytest.approx(held_adjusted)
        # And the residue case explicitly.
        assert _floor_whole_shares(
            held_adjusted - 1e-12, factor
        ) == pytest.approx(held_adjusted)

    def test_full_exit_is_exempt_from_rounding(self):
        """A position is sellable in full by definition.

        Without the exemption the unsold fraction keeps the position open, the
        engine re-issues the same exit next session, and the strategy logs the
        same exit reason for months. A 641-session Clenow run produced 33 fills
        and sat at beta 0.04 on this alone.
        """
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=False))
        bars = adjusted_bars(SPLIT_PANEL, FILL_DATE)

        # A holding that sits between two whole raw shares, as it would after a
        # dividend adjustment lands mid-position.
        dusty = 17.3
        closing = Order(
            symbol="ACME", side=OrderSide.SELL, notional=0.0, shares=dusty,
            strategy_id="test", created_date=pd.Timestamp("2024-01-02"),
            close_position=True,
        )
        partial = Order(
            symbol="ACME", side=OrderSide.SELL, notional=0.0, shares=dusty,
            strategy_id="test", created_date=pd.Timestamp("2024-01-02"),
        )

        closed = broker.fill_orders([closing], FILL_DATE, bars)
        trimmed = broker.fill_orders([partial], FILL_DATE, bars)

        assert closed[0].fill_shares == pytest.approx(dusty), "must sell it all"
        assert trimmed[0].fill_shares < dusty, "a trim still rounds"

    def test_fractional_mode_skips_raw_rounding(self):
        broker = SimulatedBroker(FREE, ExecutionConfig(fractional_shares=True))
        bars = adjusted_bars(SPLIT_PANEL, FILL_DATE)

        order = Order(
            symbol="ACME", side=OrderSide.BUY, notional=1000.0,
            strategy_id="test", created_date=pd.Timestamp("2024-01-02"),
        )
        fills = broker.fill_orders([order], FILL_DATE, bars)

        assert fills[0].fill_shares == pytest.approx(1000.0 / 51.0)


class TestContextRawClose:
    def _ctx(self, as_of):
        close_matrix = SPLIT_PANEL.pivot_table(
            index=schema.DATE, columns=schema.SYMBOL,
            values=schema.ADJ_CLOSE, aggfunc="last",
        )
        close_matrix.columns.name = None
        return StrategyContext(
            as_of=as_of,
            price_panel=SPLIT_PANEL,
            close_matrix=close_matrix,
        )

    def test_raw_close_is_the_print(self):
        ctx = self._ctx(FILL_DATE)

        assert ctx.raw_close("ACME") == pytest.approx(204.0)
        # The adjusted level is four times lower and is what a naive screen sees.
        assert ctx.latest_close("ACME") == pytest.approx(51.0)

    def test_raw_close_respects_as_of(self):
        ctx = self._ctx(pd.Timestamp("2024-01-02"))

        assert ctx.raw_close("ACME") == pytest.approx(200.0)

    def test_raw_close_is_none_for_unknown_symbol(self):
        assert self._ctx(FILL_DATE).raw_close("NOPE") is None

    def test_raw_closes_batch_matches_singles(self):
        ctx = self._ctx(FILL_DATE)

        assert ctx.raw_closes(["ACME", "PLAIN"]) == {
            "ACME": pytest.approx(204.0),
            "PLAIN": pytest.approx(21.0),
        }

    def test_min_price_screen_disagrees_between_scales(self):
        """The whole point: a $100 screen keeps ACME on prints, drops it adjusted."""
        ctx = self._ctx(FILL_DATE)

        assert ctx.raw_close("ACME") >= 100.0
        assert ctx.latest_close("ACME") < 100.0


class TestLoadPanels:
    def _lake(self, root: Path):
        """A lake where a 2-for-1 split lands after the 2024-01-03 window."""
        frame = pd.DataFrame(
            [
                _row("2024-01-02", "ACME", 200.0, 100.0),
                _row("2024-01-03", "ACME", 204.0, 102.0),
                _row("2024-01-04", "ACME", 104.0, 104.0),
            ],
            columns=COLUMNS,
        )
        schema.write_year(frame, 2024, root, hot=True)

    def test_asof_anchoring_returns_the_print_on_the_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._lake(root)

            panel, closes, ohlc = runner.load_panels(
                "2024-01-01", "2024-01-03", "asof", root=root
            )

            by_date = panel.set_index(schema.DATE)
            # Standing on 01-03 the split has not happened: adjusted == print.
            assert float(by_date.loc["2024-01-03", schema.ADJ_CLOSE]) == pytest.approx(204.0)
            assert float(by_date.loc["2024-01-02", schema.ADJ_CLOSE]) == pytest.approx(200.0)
            assert float(closes.loc["2024-01-03", "ACME"]) == pytest.approx(204.0)
            assert float(ohlc["ACME"].loc["2024-01-03", "Close"]) == pytest.approx(204.0)

    def test_asof_anchoring_leaves_the_raw_print_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._lake(root)

            panel, _, _ = runner.load_panels(
                "2024-01-01", "2024-01-03", "asof", root=root
            )

            by_date = panel.set_index(schema.DATE)
            for col in (schema.OPEN, schema.HIGH, schema.LOW, schema.CLOSE):
                assert float(by_date.loc["2024-01-02", col]) == pytest.approx(200.0)

    def test_today_mode_keeps_the_stored_adjustment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._lake(root)

            panel, closes, _ = runner.load_panels(
                "2024-01-01", "2024-01-03", "today", root=root
            )

            by_date = panel.set_index(schema.DATE)
            assert float(by_date.loc["2024-01-03", schema.ADJ_CLOSE]) == pytest.approx(102.0)
            assert float(closes.loc["2024-01-03", "ACME"]) == pytest.approx(102.0)

    def test_returns_are_identical_across_modes(self):
        """Anchoring moves levels only. Returns must not budge."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._lake(root)

            _, asof_closes, _ = runner.load_panels(
                "2024-01-01", "2024-01-03", "asof", root=root
            )
            _, today_closes, _ = runner.load_panels(
                "2024-01-01", "2024-01-03", "today", root=root
            )

            a = asof_closes["ACME"].pct_change().dropna()
            b = today_closes["ACME"].pct_change().dropna()
            pd.testing.assert_series_equal(a, b)

    def test_unknown_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._lake(root)

            with pytest.raises(ValueError, match="price_adjustment"):
                runner.load_panels("2024-01-01", "2024-01-03", "yesterday", root=root)


class TestDefaultUniverseFn:
    def test_etfs_are_unioned_into_the_pit_universe(self, monkeypatch):
        from data_pipeline import membership

        monkeypatch.setattr(membership, "has_pit_membership", lambda *a, **k: True)
        monkeypatch.setattr(
            membership, "members_asof", lambda *a, **k: {"AAPL", "MSFT"}
        )

        universe_fn = runner.default_universe_fn("pit_sp500")
        universe = universe_fn(pd.Timestamp("2024-01-03"))

        assert {"AAPL", "MSFT"}.issubset(universe)
        # An ETF sleeve trades instruments that are never index constituents.
        assert "XLK" in universe
        assert "SPY" in universe

    def test_no_membership_data_means_no_universe_fn(self, monkeypatch):
        from data_pipeline import membership

        monkeypatch.setattr(membership, "has_pit_membership", lambda *a, **k: False)

        assert runner.default_universe_fn("pit_sp500") is None

    def test_etf_rotation_source_is_etfs_only(self, monkeypatch):
        from data_pipeline import membership

        monkeypatch.setattr(membership, "has_pit_membership", lambda *a, **k: True)
        monkeypatch.setattr(
            membership, "members_asof", lambda *a, **k: {"AAPL", "MSFT"}
        )

        universe = runner.default_universe_fn("etf_rotation")(pd.Timestamp("2024-01-03"))

        assert "XLK" in universe
        assert "AAPL" not in universe

    def test_non_pit_source_falls_back(self):
        assert runner.default_universe_fn("current") is None
        assert runner.default_universe_fn("") is None

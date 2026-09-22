"""Decision parity: the ctx-fed ranked sleeve must decide what the live one decides.

The migration gate for Phase 7. ``strategies.ranks.ranked_target_weights`` is
only allowed to exist if, given the same day and the same lake, it picks the
same symbols in the same proportions as
``strategies.ranks.ranked_asset_alloc.generate_signals``. Anything else means
the migration changed the strategy while claiming to move it.

``as_of`` is pinned to 2026-08-19 to line up with
``tests/fixtures/ranked_decisions.json``, the pre-lake yfinance baseline. The
lake covers 2023-01-03 to 2026-08-21, so the fixture date and its full 730-day
lookback are both inside coverage; no substitute date is needed.

``PRICE_SOURCE=lake`` is forced for the comparison. Left at its default the old
path downloads from yfinance while the ctx path reads the lake, and the two
sides are then compared over different data -- a test that passes or fails for
reasons unrelated to the code under test.

Assertions are on decisions, never on price levels: ``adj_close`` agrees between
sources to float32 precision, not bit-exactly.

Requires the local lake. Skipped where it is absent, because it asserts a
property of the data as much as of the code.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ranked_decisions.json"

#: The fixture date. Inside lake coverage (2023-01-03 .. 2026-08-21).
AS_OF = pd.Timestamp("2026-08-19")

#: Warmup handed to the context. Wider than the sleeve's own 730-day window on
#: purpose -- the strategy must do its own windowing, and a context that only
#: just covers the window would hide a strategy that forgot to.
WARMUP_DAYS = 900


def _lake_covers() -> bool:
    try:
        from data_pipeline import store
        from strategies.ranks import ranked_asset_alloc as ra

        closes = store.load_close_matrix(
            ra.ALL_TICKERS,
            start=AS_OF - pd.Timedelta(days=ra.LOOKBACK_DAYS),
            end=AS_OF,
        )
        return len(closes) >= ra.MIN_SESSIONS and pd.Timestamp(closes.index[-1]) == AS_OF
    except Exception:
        return False


def _build_ctx(as_of: pd.Timestamp):
    """A StrategyContext over the real lake, restricted to the sleeve's tickers."""
    from backtest.context import StrategyContext
    from backtest.types import PortfolioSnapshot
    from data_pipeline import store
    from strategies.ranks import ranked_asset_alloc as ra

    warmup_start = as_of - pd.Timedelta(days=WARMUP_DAYS)

    price_panel = store.load_prices(ra.ALL_TICKERS, start=warmup_start, end=as_of)
    close_matrix = store.load_close_matrix(ra.ALL_TICKERS, start=warmup_start, end=as_of)
    ohlc = store.load_ohlc_adjusted(ra.SECTOR_ETFS, start=warmup_start, end=as_of)

    return StrategyContext(
        as_of=as_of,
        price_panel=price_panel,
        close_matrix=close_matrix,
        ohlc_adjusted=ohlc,
        portfolio=PortfolioSnapshot(date=as_of, cash=100_000.0, equity=100_000.0),
        trading_sessions=pd.DatetimeIndex(close_matrix.index),
    )


def _normalised(weights: dict) -> dict:
    """Weights rescaled to sum to 1, so leverage does not mask a shape mismatch."""
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in weights.items()}


@unittest.skipUnless(_lake_covers(), "local price lake does not cover 2026-08-19")
class RankedTargetWeightsParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from strategies.ranks import ranked_asset_alloc as ra
        from strategies.ranks import ranked_target_weights as rtw

        cls.ra = ra
        cls.rtw = rtw

        cls._saved_source = os.environ.get("PRICE_SOURCE")
        os.environ["PRICE_SOURCE"] = "lake"

        cls.ctx = _build_ctx(AS_OF)
        cls.weights = rtw.target_weights(cls.ctx)
        cls.decision = rtw.decide(cls.ctx)
        cls.signals = ra.generate_signals(
            budget=None,
            strategy_id="parity",
            held_symbols=set(),
            as_of=AS_OF,
        )

    @classmethod
    def tearDownClass(cls):
        if cls._saved_source is None:
            os.environ.pop("PRICE_SOURCE", None)
        else:
            os.environ["PRICE_SOURCE"] = cls._saved_source

    # -- the gate ---------------------------------------------------------

    def test_same_symbols_are_selected(self):
        live = {s.symbol for s in self.signals if "BUY" in str(s.side).upper()}
        self.assertEqual(
            sorted(self.weights), sorted(live),
            f"ctx={sorted(self.weights)} live={sorted(live)}",
        )

    def test_relative_weights_agree(self):
        """Shapes must match once leverage is divided out."""
        live_notional = {
            s.symbol: float(s.notional)
            for s in self.signals
            if "BUY" in str(s.side).upper()
        }
        want = _normalised(live_notional)
        got = _normalised(self.weights)

        self.assertEqual(sorted(want), sorted(got))
        for sym in want:
            with self.subTest(symbol=sym):
                self.assertAlmostEqual(got[sym], want[sym], places=6)

    def test_gross_exposure_carries_the_daf_leverage(self):
        """Weights sum to the DAF factor. Normalising it away would backtest an
        unlevered strategy while reporting the levered one's name."""
        lev = self.decision["daf_leverage"]
        self.assertGreater(lev, 0)
        self.assertAlmostEqual(sum(self.weights.values()), lev, places=6)

        # And the dollar reconstruction round-trips to the live notionals.
        live_notional = {
            s.symbol: float(s.notional)
            for s in self.signals
            if "BUY" in str(s.side).upper()
        }
        for sym, w in self.weights.items():
            with self.subTest(symbol=sym):
                self.assertAlmostEqual(
                    w * self.ra.BASE_BUDGET, live_notional[sym], places=2
                )

    # -- against the recorded pre-lake baseline ---------------------------

    def test_ctx_path_reproduces_the_recorded_fixture(self):
        """Same baseline the lake canary asserts, now reached through ctx."""
        want = json.loads(FIXTURE.read_text())
        self.assertEqual(want["as_of"], str(AS_OF.date()))

        def rounded(d):
            return {k: round(float(v), 6) for k, v in sorted(d.items())}

        got = dict(self.decision)
        got["alloc_v4"] = rounded(got["alloc_v4"])
        got["alloc_v8"] = rounded(got["alloc_v8"])
        got["target"] = rounded(got["target"])
        got["daf_leverage"] = round(got["daf_leverage"], 6)

        for key in (
            "sessions", "last_session", "regime_v4", "regime_v8",
            "alloc_v4", "alloc_v8", "target", "daf_leverage", "target_symbols",
        ):
            with self.subTest(key=key):
                self.assertEqual(got[key], want[key])

    # -- the PIT firewall actually holds ----------------------------------

    def test_panel_stops_at_as_of_and_spans_the_pinned_window(self):
        closes, ohlc = self.rtw.panel_from_ctx(self.ctx)
        start, end = self.ra._window(AS_OF)

        self.assertEqual(pd.Timestamp(closes.index[-1]), AS_OF)
        self.assertGreaterEqual(pd.Timestamp(closes.index[0]), start)
        self.assertEqual(list(closes.columns), self.ra.ALL_TICKERS)

        for ticker in self.ra.SECTOR_ETFS:
            with self.subTest(ticker=ticker):
                self.assertIn(ticker, ohlc)
                self.assertLessEqual(pd.Timestamp(ohlc[ticker].index[-1]), end)

    def test_ctx_panel_matches_the_direct_lake_read(self):
        """The context slice and ``_lake_data`` must produce the same frame.

        Not a tautology: the context is fed a 900-day panel and has to window it
        back to 730 itself, and it must drop the all-NaN rows that
        ``load_close_matrix`` drops. Either slip shifts every positional
        lookback by a row.
        """
        ctx_closes, _ = self.rtw.panel_from_ctx(self.ctx)
        lake_closes, _ = self.ra._lake_data(AS_OF)
        pd.testing.assert_frame_equal(ctx_closes, lake_closes)

    def test_requesting_tomorrow_is_refused(self):
        from backtest.context import LookaheadError

        with self.assertRaises(LookaheadError):
            self.ctx.prices(self.ra.ALL_TICKERS, end=AS_OF + pd.Timedelta(days=1))


@unittest.skipUnless(_lake_covers(), "local price lake does not cover 2026-08-19")
class RebalanceCadenceTests(unittest.TestCase):
    """Holding flat between rebalances, expressed in the weights dialect."""

    @classmethod
    def setUpClass(cls):
        from strategies.ranks import ranked_target_weights as rtw

        cls.rtw = rtw
        cls.ctx = _build_ctx(AS_OF)

    def test_empty_book_always_rebalances(self):
        self.assertTrue(self.rtw.is_rebalance_day(self.ctx))

    def test_mid_month_with_holdings_holds_current_weights(self):
        from backtest.types import PortfolioSnapshot, Position
        from backtest.context import StrategyContext

        pos = Position(
            symbol="XLK", shares=10.0, avg_entry_price=100.0,
            market_price=120.0, market_value=1_200.0,
            unrealized_pnl=200.0, entry_date=AS_OF,
        )
        held = PortfolioSnapshot(
            date=AS_OF, cash=8_800.0, equity=10_000.0, positions={"XLK": pos}
        )
        ctx = StrategyContext(
            as_of=AS_OF,
            price_panel=self.ctx._price_panel,
            close_matrix=self.ctx._close_matrix,
            ohlc_adjusted=self.ctx._ohlc_adjusted,
            portfolio=held,
            trading_sessions=self.ctx._trading_sessions,
        )

        self.assertFalse(self.rtw.is_rebalance_day(ctx))
        self.assertEqual(self.rtw.target_weights(ctx), {"XLK": 0.12})

    def test_first_session_of_the_month_rebalances(self):
        from backtest.types import PortfolioSnapshot, Position
        from backtest.context import StrategyContext

        sessions = self.ctx.sessions()
        august = sessions[(sessions.year == 2026) & (sessions.month == 8)]
        first = pd.Timestamp(august[0])

        pos = Position(
            symbol="XLK", shares=10.0, avg_entry_price=100.0,
            market_price=120.0, market_value=1_200.0,
            unrealized_pnl=200.0, entry_date=first,
        )
        ctx = StrategyContext(
            as_of=first,
            price_panel=self.ctx._price_panel,
            close_matrix=self.ctx._close_matrix,
            ohlc_adjusted=self.ctx._ohlc_adjusted,
            portfolio=PortfolioSnapshot(
                date=first, cash=8_800.0, equity=10_000.0, positions={"XLK": pos}
            ),
            trading_sessions=self.ctx._trading_sessions,
        )
        self.assertTrue(self.rtw.is_rebalance_day(ctx))


class LiveSleeveUnchangedTests(unittest.TestCase):
    """``as_of`` is additive: omitting it must leave the live path alone."""

    def test_as_of_defaults_to_none(self):
        import inspect
        from strategies.ranks import ranked_asset_alloc as ra

        params = inspect.signature(ra.generate_signals).parameters
        self.assertIn("as_of", params)
        self.assertIsNone(params["as_of"].default)
        self.assertEqual(params["as_of"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_decision_logic_is_imported_not_copied(self):
        """A second copy of the ranking logic would drift out of parity."""
        import ast

        src = (
            REPO_ROOT / "src" / "strategies" / "ranks" / "ranked_target_weights.py"
        ).read_text()
        defined = {
            n.name for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef)
        }
        for owned_by_the_live_sleeve in (
            "_v4_today", "_v8aw_today", "_blend", "_daf_leverage",
            "_regime", "_sector_composite", "_momentum", "_trend_atr",
        ):
            with self.subTest(fn=owned_by_the_live_sleeve):
                self.assertNotIn(owned_by_the_live_sleeve, defined)


if __name__ == "__main__":
    unittest.main()

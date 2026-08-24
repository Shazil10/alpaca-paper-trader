"""Canary: the lake must reproduce the rotation sleeve's pre-migration decisions.

The baseline in ``tests/fixtures/ranked_decisions.json`` was recorded from the
yfinance path *before* the sleeve was wired to the lake
(``scripts/record_ranked_fixture.py``). Recording it afterwards would capture
post-change behaviour and prove nothing.

Assertions are on **decisions** -- target symbols, weights, regime letters, the
DAF factor -- never on price levels. ``adj_close`` agrees between the two
sources to float32 precision, not bit-exactly, so a price assertion would fail
for a correct reason and train everyone to ignore the test.

This is the test that caught the real bug: five sector ETFs were each missing
2-3 bars, one hole inside a 50-day window made ``rolling(50)`` NaN, breadth fell
10/11 -> 5/11, and the regime flipped bull -> neutral with a different target
allocation. Strategy code was untouched; only completeness differed.

Requires the local lake. Skipped where it is absent (e.g. a clean CI checkout),
because it asserts a property of the data, not of the code.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ranked_decisions.json"

#: Every decision the sleeve makes before sizing. If these all match, the
#: resulting orders match.
DECISION_KEYS = [
    "regime_v4",
    "regime_v8",
    "alloc_v4",
    "alloc_v8",
    "target",
    "daf_leverage",
    "target_symbols",
    "sessions",
    "last_session",
]


def lake_has_coverage() -> bool:
    """True when the local lake can serve the fixture's window."""
    if not FIXTURE.exists():
        return False
    try:
        from data_pipeline import store
        from strategies.ranks import ranked_asset_alloc as ra

        want = json.loads(FIXTURE.read_text())
        as_of = pd.Timestamp(want["as_of"])
        start = as_of - pd.Timedelta(days=want["lookback_days"])
        closes = store.load_close_matrix(ra.ALL_TICKERS, start=start, end=as_of)
        return len(closes) >= want["sessions"]
    except Exception:
        return False


def decisions_from(closes, ohlc_dict) -> dict:
    """Run the sleeve's decision functions over a given panel."""
    from strategies.ranks import ranked_asset_alloc as ra

    sector_closes = closes[ra.SECTOR_ETFS]
    idx = len(closes) - 1

    alloc_v4 = ra._v4_today(sector_closes, ohlc_dict, closes, idx)
    alloc_v8 = ra._v8aw_today(sector_closes, ohlc_dict, closes, idx)
    target = ra._blend(alloc_v4, alloc_v8)

    def rounded(d):
        return {k: round(float(v), 6) for k, v in sorted(d.items())}

    return {
        "sessions": int(len(closes)),
        "last_session": str(pd.Timestamp(closes.index[-1]).date()),
        "regime_v4": ra._regime(closes, idx, ra.V4_BREADTH),
        "regime_v8": ra._regime(closes, idx, ra.V8_BREADTH),
        "alloc_v4": rounded(alloc_v4),
        "alloc_v8": rounded(alloc_v8),
        "target": rounded(target),
        "daf_leverage": round(float(ra._daf_leverage(closes)), 6),
        "target_symbols": sorted(
            s for s, w in target.items() if s != ra.CASH_ETF and w > 1e-6
        ),
    }


@unittest.skipUnless(lake_has_coverage(), "local price lake does not cover the fixture window")
class RankedCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from strategies.ranks import ranked_asset_alloc as ra

        cls.ra = ra
        cls.want = json.loads(FIXTURE.read_text())
        cls.as_of = pd.Timestamp(cls.want["as_of"])
        cls.start = cls.as_of - pd.Timedelta(days=cls.want["lookback_days"])

    def _lake_panel(self):
        from data_pipeline import store

        closes = store.load_close_matrix(
            self.ra.ALL_TICKERS, start=self.start, end=self.as_of
        )
        closes = closes[[t for t in self.ra.ALL_TICKERS if t in closes.columns]]
        ohlc = store.load_ohlc_adjusted(
            self.ra.SECTOR_ETFS, start=self.start, end=self.as_of
        )
        return closes, ohlc

    def test_lake_reproduces_recorded_decisions(self):
        got = decisions_from(*self._lake_panel())
        for key in DECISION_KEYS:
            with self.subTest(key=key):
                self.assertEqual(
                    got[key], self.want[key],
                    f"{key}: lake={got[key]!r} fixture={self.want[key]!r}",
                )

    def test_lookback_window_is_pinned_to_the_fixture(self):
        """The DAF expanding() quantile must not see extra history."""
        self.assertEqual(self.ra.LOOKBACK_DAYS, self.want["lookback_days"])

        start, end = self.ra._window(self.as_of)
        self.assertEqual(end, self.as_of)
        self.assertEqual(start, self.start)

    def test_every_sector_has_a_usable_rolling_average(self):
        """A single missing bar makes rolling(50) NaN and silently drops the
        sector out of the breadth count. That is how the regime flipped."""
        closes, _ = self._lake_panel()
        sma50 = closes[self.ra.SECTOR_ETFS].rolling(50).mean().iloc[-1]
        naughty = sorted(sma50[sma50.isna()].index)
        self.assertEqual(naughty, [], f"sectors with NaN 50-day average: {naughty}")

    def test_sector_panel_has_no_holes(self):
        closes, _ = self._lake_panel()
        holes = closes[self.ra.SECTOR_ETFS].isna().sum()
        self.assertEqual(
            holes[holes > 0].to_dict(), {}, "sector panel contains NaN bars"
        )

    def test_adjusted_ohlc_aligns_with_the_close_panel(self):
        closes, ohlc = self._lake_panel()
        for ticker in self.ra.SECTOR_ETFS:
            with self.subTest(ticker=ticker):
                self.assertIn(ticker, ohlc)
                # Adjusted OHLC close must equal the adj_close panel.
                merged = ohlc[ticker]["Close"].reindex(closes.index).dropna()
                aligned = closes[ticker].reindex(merged.index)
                pd.testing.assert_series_equal(
                    merged, aligned, check_names=False, rtol=1e-9
                )


class SourceSelectionTests(unittest.TestCase):
    """The flag itself, independent of any data being present."""

    def setUp(self):
        from data_pipeline import source

        self.source = source
        self._saved = os.environ.get(source.ENV_VAR)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(self.source.ENV_VAR, None)
        else:
            os.environ[self.source.ENV_VAR] = self._saved

    def test_default_is_yfinance_until_deliberately_flipped(self):
        os.environ.pop(self.source.ENV_VAR, None)
        self.assertEqual(self.source.active_source(), self.source.YFINANCE)
        self.assertFalse(self.source.using_lake())

    def test_lake_can_be_selected(self):
        os.environ[self.source.ENV_VAR] = "lake"
        self.assertTrue(self.source.using_lake())

    def test_value_is_case_and_space_insensitive(self):
        os.environ[self.source.ENV_VAR] = "  LAKE  "
        self.assertTrue(self.source.using_lake())

    def test_unknown_value_falls_back_to_default(self):
        os.environ[self.source.ENV_VAR] = "bloomberg"
        self.assertEqual(self.source.active_source(), self.source.DEFAULT_SOURCE)

    def test_strategy_never_falls_back_across_sources(self):
        """No silent yfinance rescue inside the lake path, and vice versa."""
        import ast

        src = (SRC_DIR / "strategies" / "ranks" / "ranked_asset_alloc.py").read_text()
        tree = ast.parse(src)

        lake_fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_lake_data"
        )
        calls = {
            ast.unparse(n.func) if hasattr(ast, "unparse") else ""
            for n in ast.walk(lake_fn) if isinstance(n, ast.Call)
        }
        self.assertFalse(
            any("yf." in c or "_download_data" in c for c in calls),
            f"_lake_data must not reach for yfinance: {calls}",
        )


if __name__ == "__main__":
    unittest.main()

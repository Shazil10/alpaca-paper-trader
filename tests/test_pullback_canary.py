"""Canary: the lake must reproduce the Pullback sleeve's pre-migration decisions.

Baseline in ``tests/fixtures/pullback_decisions.json``, recorded from the
yfinance path before the sleeve was wired
(``scripts/record_pullback_fixture.py``).

Assertions are on decisions -- which symbols clear the gates, and their rank
order -- not on price levels, since ``adj_close`` agrees across sources only to
float32 precision.

Unlike the rotation sleeve, the 40% entry gate is a hard threshold, so a name
sitting exactly on it can legitimately flip between sources. The fixture records
those boundary names and they are excluded from the strict set comparison rather
than pretended away.

Requires the local lake and universe.csv. Skipped where either is absent.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from importlib import import_module
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "pullback_decisions.json"
UNIVERSE = REPO_ROOT / "universe.csv"

MODULE = "strategies.mean_reversion.52W_mean_reversion_strat"


def lake_has_coverage() -> bool:
    if not FIXTURE.exists() or not UNIVERSE.exists():
        return False
    try:
        from data_pipeline import store

        want = json.loads(FIXTURE.read_text())
        as_of = pd.Timestamp(want["as_of"])
        start = as_of - pd.Timedelta(days=want["lookback_days"])
        # Cheap probe: the regime ETFs must reach back far enough.
        panel = store.load_close_matrix(["SPY", "IJH", "IJR"], start=start, end=as_of)
        return len(panel) >= 252
    except Exception:
        return False


@unittest.skipUnless(lake_has_coverage(), "local price lake does not cover the fixture window")
class PullbackCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PRICE_SOURCE"] = "lake"
        from data_pipeline import source

        cls.source = source
        cls.m = import_module(MODULE)
        cls.want = json.loads(FIXTURE.read_text())
        cls.as_of = pd.Timestamp(cls.want["as_of"])
        cls.scores = cls.m._score_universe(str(UNIVERSE), cls.as_of)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("PRICE_SOURCE", None)

    def test_regime_matches(self):
        self.assertEqual(
            self.m.check_regime(self.as_of), self.want["regime"]["risk_on"]
        )

    def test_candidate_set_matches_excluding_boundary_names(self):
        got = set(self.scores["symbol"]) if not self.scores.empty else set()
        want = set(self.want["ranked_symbols"])
        boundary = set(self.want.get("boundary_symbols") or [])

        self.assertEqual(
            got - boundary, want - boundary,
            f"lake-only={sorted(got - want - boundary)} "
            f"fixture-only={sorted(want - got - boundary)}",
        )

    def test_rank_order_matches(self):
        got = self.scores["symbol"].tolist() if not self.scores.empty else []
        self.assertEqual(got, self.want["ranked_symbols"])

    def test_top_picks_match(self):
        got = (
            self.scores["symbol"].head(self.m.TOP_N).tolist()
            if not self.scores.empty else []
        )
        self.assertEqual(got, self.want["top_picks"])

    def test_pullback_depths_agree_within_float_tolerance(self):
        if self.scores.empty:
            self.skipTest("no candidates")
        want = {c["symbol"]: c["pullback"] for c in self.want["candidates"]}
        for _, row in self.scores.iterrows():
            sym = row["symbol"]
            if sym not in want:
                continue
            with self.subTest(symbol=sym):
                self.assertAlmostEqual(float(row["pullback"]), want[sym], places=3)

    def test_every_candidate_clears_the_documented_gates(self):
        """Guards against a gate being silently dropped during the rewrite."""
        if self.scores.empty:
            self.skipTest("no candidates")
        for _, r in self.scores.iterrows():
            with self.subTest(symbol=r["symbol"]):
                self.assertGreaterEqual(r["pullback"], self.m.ENTRY_DEPTH)
                self.assertGreaterEqual(r["price"], self.m.MA_FREEFALL_RATIO * r["high52"])
                self.assertGreaterEqual(r["price"], self.m.MA_BROKEN_RATIO * r["ma200"])

    def test_lookback_windows_are_pinned_to_the_fixture(self):
        self.assertEqual(self.m.LOOKBACK_DAYS, self.want["lookback_days"])
        self.assertEqual(self.m.REGIME_LOOKBACK_DAYS, self.want["regime_lookback_days"])


class ExitSafetyTests(unittest.TestCase):
    """Missing data must never liquidate a position."""

    def setUp(self):
        os.environ["PRICE_SOURCE"] = "lake"
        from data_pipeline import source

        self.source = source
        self.m = import_module(MODULE)

    def tearDown(self):
        os.environ.pop("PRICE_SOURCE", None)

    def test_absent_price_data_does_not_produce_a_sell(self):
        """A plumbing fault must not realise a loss.

        The pre-lake code emitted ``exit:data_unavailable`` and sold. Once the
        source is a local lake, absent bars mean a failed sync far more often
        than a dead instrument, so selling on it is wrong. Failing to exit is
        recoverable; selling by mistake is not.
        """
        original = self.m._close_panel
        self.m._close_panel = lambda *a, **k: pd.DataFrame(dtype="float64")
        try:
            signals = self.m._generate_exit_signals(
                pd.DataFrame(), {"NOSUCHSYM"}, "strategies.test", str(UNIVERSE)
            )
        finally:
            self.m._close_panel = original

        self.assertEqual(
            [s.symbol for s in signals], [],
            "missing price data must not generate a SELL",
        )

    def test_no_held_symbols_yields_no_signals(self):
        self.assertEqual(
            self.m._generate_exit_signals(pd.DataFrame(), set(), "strategies.test", str(UNIVERSE)),
            [],
        )


class ScoreGateTests(unittest.TestCase):
    """The gate logic itself, on synthetic series -- no data required."""

    def setUp(self):
        self.m = import_module(MODULE)

    def _series(
        self, high: float, last: float, *, drop_bars_ago: int = 230, n: int = 300
    ) -> pd.Series:
        """Step series: ``high`` then ``last`` for the final ``drop_bars_ago`` bars.

        ``drop_bars_ago`` is the interesting dial. The 52-week high looks back
        252 bars while the moving average looks back 200, so a drop placed
        between those two figures leaves the high intact while letting the
        average settle at the new price -- which is exactly the "sold off, then
        stabilised" shape the sleeve is built to buy. Placing the drop recently
        instead leaves the average far above price, which gate F3 rejects.
        """
        values = [high] * (n - drop_bars_ago) + [last] * drop_bars_ago
        return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=n))

    def test_shallow_pullback_is_rejected(self):
        # 20% below the high: fails the 40% entry gate.
        self.assertIsNone(self.m._score_series("X", self._series(100.0, 80.0)))

    def test_deep_but_stabilised_pullback_passes(self):
        record = self.m._score_series("X", self._series(100.0, 55.0))
        self.assertIsNotNone(record)
        self.assertGreaterEqual(record["pullback"], self.m.ENTRY_DEPTH)
        self.assertAlmostEqual(record["pullback"], 0.45, places=2)

    def test_freefall_is_rejected(self):
        # Below 50% of the 52-week high, even though it has stabilised.
        self.assertIsNone(self.m._score_series("X", self._series(100.0, 40.0)))

    def test_still_falling_is_rejected_by_the_moving_average_gate(self):
        """Same depth, recent drop: price sits far under its 200-day average."""
        self.assertIsNone(
            self.m._score_series("X", self._series(100.0, 55.0, drop_bars_ago=20))
        )

    def test_insufficient_history_is_rejected(self):
        short = pd.Series(
            [100.0] * 100, index=pd.bdate_range("2026-01-01", periods=100)
        )
        self.assertIsNone(self.m._score_series("X", short))

    def test_all_nan_series_is_rejected(self):
        nan = pd.Series(
            [float("nan")] * 300, index=pd.bdate_range("2024-01-01", periods=300)
        )
        self.assertIsNone(self.m._score_series("X", nan))


if __name__ == "__main__":
    unittest.main()

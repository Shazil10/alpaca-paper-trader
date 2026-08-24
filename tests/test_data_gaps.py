"""Regression tests for lake completeness detection.

These encode failures found while migrating the rotation sleeve. Each one
silently changed live trading decisions without raising anything, so each gets a
test that fails loudly if the detector regresses.

Run with: PYTHONPATH=src ./venv/bin/python -m pytest tests/test_data_gaps.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_pipeline import schema, store  # noqa: E402
from data_pipeline.schema import COLUMNS, DATE, SYMBOL  # noqa: E402


def panel(dates, symbols, *, drop=None) -> pd.DataFrame:
    """Build a dense panel, optionally omitting (symbol, date) pairs."""
    drop = drop or {}
    rows = []
    for i, sym in enumerate(symbols):
        skip = {pd.Timestamp(d) for d in drop.get(sym, [])}
        for d in dates:
            if pd.Timestamp(d) in skip:
                continue
            base = 100.0 + i
            rows.append((d, sym, base, base + 1, base - 1, base, base, 1000))
    return pd.DataFrame(rows, columns=COLUMNS)


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dates = pd.bdate_range("2026-01-05", periods=20)
        self.symbols = [f"S{i:03d}" for i in range(40)]

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, frame):
        for year in schema.years_in(frame):
            rows = frame[frame[DATE].dt.year == year]
            schema.write_year(rows, year, self.root, hot=True)

    def test_mass_dropout_date_is_still_a_session(self):
        """The bug: a proportional quorum forgets under-populated dates.

        One bad fetch window dropped the same 3 dates for ~75% of the universe.
        A 50%-quorum calendar concluded those were not sessions, so no symbol
        looked incomplete and the hole passed review. An absolute floor keeps
        the date, so every symbol missing it is correctly flagged.
        """
        victim_date = self.dates[10]
        # 15 of 40 symbols keep the date: 37.5%, so a 50% quorum would discard
        # it, while the absolute floor of 10 retains it. That is precisely the
        # distinction being tested, and it mirrors the observed 23-32% survival.
        drop = {s: [victim_date] for s in self.symbols[15:]}
        self._write(panel(self.dates, self.symbols, drop=drop))

        calendar = store.trading_calendar(root=self.root)
        self.assertIn(
            pd.Timestamp(victim_date), calendar,
            "under-populated date was dropped from the calendar",
        )

        gaps = store.find_gaps(root=self.root)
        self.assertEqual(len(gaps), 25)  # symbols 15..39
        self.assertIn(pd.Timestamp(victim_date), gaps[self.symbols[20]])
        self.assertNotIn(self.symbols[0], gaps)  # kept the date

    def test_incomplete_reference_symbol_does_not_hide_gaps(self):
        """The bug: keying the calendar to one instrument.

        SPY itself was short 2 sessions, so those dates were absent from the
        calendar and every symbol missing them looked complete.
        """
        victim_date = self.dates[7]
        drop = {"SPY": [victim_date], "S001": [victim_date]}
        symbols = ["SPY"] + self.symbols
        self._write(panel(self.dates, symbols, drop=drop))

        calendar = store.trading_calendar(root=self.root)
        self.assertIn(pd.Timestamp(victim_date), calendar)

        gaps = store.find_gaps(root=self.root)
        self.assertIn("SPY", gaps)
        self.assertIn("S001", gaps)

    def test_phantom_date_is_rejected(self):
        """A date only a couple of symbols report is not a session."""
        self._write(
            pd.concat(
                [
                    panel(self.dates, self.symbols),
                    panel([pd.Timestamp("2026-02-14")], self.symbols[:2]),
                ],
                ignore_index=True,
            )
        )
        calendar = store.trading_calendar(root=self.root)
        self.assertNotIn(pd.Timestamp("2026-02-14"), calendar)

    def test_complete_lake_reports_no_gaps(self):
        self._write(panel(self.dates, self.symbols))
        self.assertEqual(store.find_gaps(root=self.root), {})

    def test_late_listing_is_not_a_gap(self):
        """Missing history before a symbol's first bar is not incompleteness."""
        late = panel(self.dates[10:], ["LATE"])
        self._write(pd.concat([panel(self.dates, self.symbols), late], ignore_index=True))

        gaps = store.find_gaps(root=self.root)
        self.assertNotIn("LATE", gaps)

    def test_delisting_is_not_a_gap(self):
        early = panel(self.dates[:10], ["GONE"])
        self._write(pd.concat([panel(self.dates, self.symbols), early], ignore_index=True))

        gaps = store.find_gaps(root=self.root)
        self.assertNotIn("GONE", gaps)

    def test_interior_hole_is_a_gap(self):
        drop = {"S005": [self.dates[5], self.dates[6]]}
        self._write(panel(self.dates, self.symbols, drop=drop))

        gaps = store.find_gaps(root=self.root)
        self.assertIn("S005", gaps)
        self.assertEqual(len(gaps["S005"]), 2)

    def test_a_single_hole_breaks_rolling_windows(self):
        """Why gaps are a correctness problem, not untidiness.

        One missing bar inside the window makes rolling().mean() NaN, so the
        symbol silently fails any threshold comparison downstream. This is how a
        breadth count fell from 10/11 to 5/11 and flipped the regime.
        """
        drop = {"S003": [self.dates[15]]}
        self._write(panel(self.dates, self.symbols, drop=drop))

        matrix = store.load_close_matrix(["S003", "S004"], root=self.root)
        sma = matrix.rolling(5).mean().iloc[-1]

        self.assertTrue(pd.isna(sma["S003"]), "hole should poison the rolling mean")
        self.assertFalse(pd.isna(sma["S004"]))
        # The comparison that silently goes False:
        self.assertFalse(bool(matrix["S003"].iloc[-1] > sma["S003"]))


class AdjustedOhlcTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ohlc_is_rescaled_onto_the_adjusted_close(self):
        """The rotation sleeve's ATR trend needs adjusted highs and lows.

        Its fetch omitted auto_adjust (defaults True), so its OHLC was adjusted.
        The lake stores raw OHLC plus adj_close, so the adjusted series must be
        reconstructed as raw * (adj_close / close) -- verified against yfinance
        as exactly equal. Feeding raw highs/lows instead would silently change a
        term carrying most of the composite rank weight.
        """
        rows = [
            # close 100 vs adj_close 90 -> factor 0.9
            ("2026-01-05", "AAA", 100.0, 110.0, 90.0, 100.0, 90.0, 1000),
            # unadjusted day -> factor 1.0
            ("2026-01-06", "AAA", 100.0, 110.0, 90.0, 100.0, 100.0, 1000),
        ]
        schema.write_year(pd.DataFrame(rows, columns=COLUMNS), 2026, self.root, hot=True)

        got = store.load_ohlc_adjusted(["AAA"], root=self.root)["AAA"]

        self.assertAlmostEqual(got["High"].iloc[0], 99.0, places=6)   # 110 * 0.9
        self.assertAlmostEqual(got["Low"].iloc[0], 81.0, places=6)    # 90 * 0.9
        self.assertAlmostEqual(got["Close"].iloc[0], 90.0, places=6)
        self.assertAlmostEqual(got["High"].iloc[1], 110.0, places=6)  # factor 1
        self.assertEqual(list(got.columns), ["Open", "High", "Low", "Close"])

    def test_zero_close_does_not_produce_infinite_factor(self):
        rows = [("2026-01-05", "AAA", 0.0, 0.0, 0.0, 0.0, 50.0, 1000)]
        schema.write_year(pd.DataFrame(rows, columns=COLUMNS), 2026, self.root, hot=True)

        got = store.load_ohlc_adjusted(["AAA"], root=self.root)
        for frame in got.values():
            self.assertTrue(frame.notna().all().all())
            for col in ("Open", "High", "Low", "Close"):
                self.assertFalse(bool((frame[col] == float("inf")).any()))


if __name__ == "__main__":
    unittest.main()

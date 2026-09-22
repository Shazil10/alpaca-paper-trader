"""Tests for the security master (data_pipeline.securities + build script).

Two separate risks are covered here.

The first is ticker recycling. A symbol freed by one delisting gets handed to an
unrelated company later, and concatenating both into one price series splices two
businesses together. The builder must not merge spans separated by years, and
must not *split* a company that merely dropped out of the index for a while --
AMD left the S&P 500 from 2013 to 2017 and is still AMD.

The second is the sector cap. ``RiskConfig.max_sector_pct`` reads a
``{ticker: sector}`` map; supply nothing and the cap silently does not bind, so a
run reports itself as sector-limited when it was not. These tests pin that the
map loads, and that its absence is detectable rather than quiet.

Run with: ./venv/bin/python -m pytest tests/test_securities.py -v
"""

from __future__ import annotations

import importlib.util
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

from data_pipeline import securities  # noqa: E402


def _load_build_script():
    """Import scripts/build_securities.py, which is not an installed package."""
    path = Path(REPO_ROOT) / "scripts" / "build_securities.py"
    spec = importlib.util.spec_from_file_location("build_securities", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _intervals(rows):
    """rows: list of (start, end) where end may be None for 'still a member'."""
    return pd.DataFrame(
        [
            {
                "symbol": "TEST",
                "index": "SP500",
                "start_date": pd.Timestamp(s),
                "end_date": pd.Timestamp(e) if e else pd.NaT,
            }
            for s, e in rows
        ]
    )


class CollapseIntervalTests(unittest.TestCase):
    def setUp(self):
        self.build = _load_build_script()

    def test_short_gap_is_one_listing(self):
        """Out of the index for 6 months is the same company, not a new one."""
        spans = self.build.collapse_intervals(
            _intervals([("2010-01-04", "2012-06-01"), ("2012-12-03", None)])
        )

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["start_date"], pd.Timestamp("2010-01-04"))
        self.assertTrue(pd.isna(spans[0]["end_date"]))

    def test_amd_style_multi_year_absence_still_splits(self):
        """A 3.5-year gap exceeds the threshold, so it is flagged for triage.

        This is a deliberate false positive: AMD is one company. The builder
        cannot tell it apart from a reassignment without a corporate-actions
        feed, so it surfaces the gap instead of guessing.
        """
        spans = self.build.collapse_intervals(
            _intervals([("1996-01-02", "2013-09-23"), ("2017-03-20", None)])
        )

        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[1]["gap_years"], 3.49, places=1)

    def test_first_span_has_no_gap(self):
        spans = self.build.collapse_intervals(_intervals([("1996-01-02", None)]))

        self.assertEqual(len(spans), 1)
        self.assertTrue(pd.isna(spans[0]["gap_years"]))

    def test_open_ended_span_absorbs_later_intervals(self):
        """Nothing may follow a still-open span without extending it."""
        spans = self.build.collapse_intervals(
            _intervals([("1996-01-02", None), ("2020-01-02", "2021-01-04")])
        )

        self.assertEqual(len(spans), 1)

    def test_threshold_boundary(self):
        """Just under the threshold merges; just over it splits."""
        under = self.build.collapse_intervals(
            _intervals([("2000-01-03", "2010-01-04"), ("2011-11-01", None)])
        )
        over = self.build.collapse_intervals(
            _intervals([("2000-01-03", "2010-01-04"), ("2012-06-01", None)])
        )

        self.assertEqual(len(under), 1, "1.8 years should merge")
        self.assertEqual(len(over), 2, "2.4 years should split")


class SecurityMasterMixin:
    """A two-listing master written to a temp parquet."""

    def build_master(self, root: Path) -> Path:
        path = root / "securities.parquet"
        pd.DataFrame([
            {
                "security_id": "AMP_1", "ticker": "AMP", "name": "",
                "sector": "Information Technology", "industry": "",
                "start_date": pd.Timestamp("1996-01-02"),
                "end_date": pd.Timestamp("1999-04-05"),
                "delist_reason": "index_removal", "recycle_candidate": True,
                "gap_years": float("nan"), "source": "test",
            },
            {
                "security_id": "AMP_2", "ticker": "AMP", "name": "",
                "sector": "Financials", "industry": "",
                "start_date": pd.Timestamp("2005-10-03"),
                "end_date": pd.NaT,
                "delist_reason": "", "recycle_candidate": True,
                "gap_years": 6.5, "source": "test",
            },
            {
                "security_id": "PLAIN_1", "ticker": "PLAIN", "name": "",
                "sector": "", "industry": "",
                "start_date": pd.Timestamp("2000-01-03"),
                "end_date": pd.NaT,
                "delist_reason": "", "recycle_candidate": False,
                "gap_years": float("nan"), "source": "test",
            },
        ]).to_parquet(path, index=False)
        securities.clear_cache()
        return path

    def tearDown(self):
        securities.clear_cache()


class ResolveTickerTests(SecurityMasterMixin, unittest.TestCase):
    def test_same_ticker_resolves_to_different_ids_by_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build_master(Path(tmp))

            self.assertEqual(securities.resolve_ticker("AMP", "1997-06-30", path=path), "AMP_1")
            self.assertEqual(securities.resolve_ticker("AMP", "2010-06-30", path=path), "AMP_2")

    def test_date_inside_the_gap_resolves_to_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build_master(Path(tmp))

            self.assertIsNone(securities.resolve_ticker("AMP", "2002-06-28", path=path))

    def test_recycle_flag_follows_span_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build_master(Path(tmp))

            self.assertTrue(securities.is_recycled_ticker("AMP", path=path))
            self.assertFalse(securities.is_recycled_ticker("PLAIN", path=path))

    def test_missing_master_passes_the_ticker_through(self):
        securities.clear_cache()
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "nope.parquet"

            self.assertEqual(securities.resolve_ticker("AAPL", "2010-06-30", path=absent), "AAPL")


class SectorMapTests(SecurityMasterMixin, unittest.TestCase):
    def test_labels_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build_master(Path(tmp))

            mapping = securities.sector_map(path=path)

            # The latest span wins: AMP is Ameriprise today, not AMP Inc.
            self.assertEqual(mapping["AMP"], "Financials")

    def test_blank_sectors_are_omitted_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build_master(Path(tmp))

            self.assertNotIn("PLAIN", securities.sector_map(path=path))

    def test_absent_master_yields_an_empty_map(self):
        securities.clear_cache()
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "nope.parquet"

            self.assertEqual(securities.sector_map(path=absent), {})


class RunnerSectorMapTests(unittest.TestCase):
    def tearDown(self):
        securities.clear_cache()

    def test_empty_map_returns_none_so_the_caller_can_warn(self):
        from backtest import runner

        securities.clear_cache()
        original = securities.sector_map
        securities.sector_map = lambda *a, **k: {}
        try:
            self.assertIsNone(runner.default_sector_map())
        finally:
            securities.sector_map = original

    def test_populated_map_is_passed_through(self):
        from backtest import runner

        original = securities.sector_map
        securities.sector_map = lambda *a, **k: {"AAPL": "Information Technology"}
        try:
            self.assertEqual(
                runner.default_sector_map(), {"AAPL": "Information Technology"}
            )
        finally:
            securities.sector_map = original


if __name__ == "__main__":
    unittest.main()

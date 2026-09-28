"""The pilot's comparison logic, on synthetic Alpaca/Yahoo frames.

The pilot itself needs credentials and the network; what it concludes from the
two frames does not, and that is what is tested here.

Run with: PYTHONPATH=src ./venv/bin/python -m pytest tests/test_pilot_alpaca.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_pipeline import schema  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pilot_alpaca_under_test", os.path.join(REPO_ROOT, "scripts", "pilot_alpaca.py")
)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)


def frame(dates, close, adj, volume):
    return pd.DataFrame({
        schema.DATE: dates, schema.SYMBOL: "NVDA", schema.OPEN: close, schema.HIGH: close,
        schema.LOW: close, schema.CLOSE: close, schema.ADJ_CLOSE: adj, schema.VOLUME: volume,
    })


class CompareTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2024-06-03", "2024-06-14")
        pre = dates < pd.Timestamp("2024-06-10")
        economic = pd.Series(range(len(dates)), index=dates, dtype="float64") + 120.0
        # Alpaca: the print (x10 before the split), raw volume (/10 before).
        a_close = economic.where(~pre, economic * 10)
        a_vol = pd.Series(1_000, index=dates).where(~pre, 100)
        self.alpaca = frame(dates, a_close.values, economic.values, a_vol.values)
        # Yahoo: split-adjusted close and volume.
        self.yahoo = frame(dates, economic.values, economic.values, [1_000] * len(dates))

    def test_split_check_detects_split_adjusted_yahoo_close(self):
        result = pilot.compare_symbol("NVDA", self.alpaca, self.yahoo)
        check = [c for c in result["split_checks"] if c["split"] == "2024-06-10"][0]
        self.assertAlmostEqual(check["alpaca_over_yahoo_close_before"], 10.0, places=3)
        self.assertTrue(check["yahoo_close_is_split_adjusted"])
        self.assertFalse(check["closes_agree"])

    def test_adjusted_returns_agree_when_only_the_print_differs(self):
        result = pilot.compare_symbol("NVDA", self.alpaca, self.yahoo)
        self.assertEqual(result["adj_return_disagreement_rate"], 0.0)
        self.assertEqual(result["adj_anchor_ratio_first"], 1.0)

    def test_missing_sessions_are_listed_per_vendor(self):
        result = pilot.compare_symbol("NVDA", self.alpaca.iloc[1:], self.yahoo)
        self.assertEqual(result["sessions_only_yahoo"], ["2024-06-03"])
        self.assertEqual(result["sessions_only_alpaca_count"], 0)

    def test_one_empty_vendor_is_reported_not_raised(self):
        result = pilot.compare_symbol("NVDA", self.alpaca, schema.empty_frame())
        self.assertEqual(result["verdict"], "one vendor returned nothing")


if __name__ == "__main__":
    unittest.main()

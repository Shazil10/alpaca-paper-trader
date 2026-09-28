"""Gap classes A-F and the Tiingo decision, on a synthetic lake.

Run with: PYTHONPATH=src ./venv/bin/python -m pytest tests/test_classify_gaps.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from unittest import mock

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_pipeline import schema  # noqa: E402


def load_script(name: str):
    path = os.path.join(REPO_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses on 3.9 need the module registered
    spec.loader.exec_module(module)
    return module


classify_gaps = load_script("classify_gaps")
audit_data = load_script("audit_data")

CALENDAR = pd.bdate_range("2016-01-04", "2016-03-31")


def bars(symbol, dates):
    return pd.DataFrame({
        schema.DATE: dates, schema.SYMBOL: symbol, schema.OPEN: 10.0, schema.HIGH: 10.0,
        schema.LOW: 10.0, schema.CLOSE: 10.0, schema.ADJ_CLOSE: 10.0, schema.VOLUME: 1,
    })


def interval(symbol, start, end=None):
    return {"symbol": symbol, "index": "SP500", "start_date": pd.Timestamp(start),
            "end_date": pd.Timestamp(end) if end else pd.NaT, "left_censored": False,
            "source": "test"}


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.concat([
            bars("FULL", CALENDAR),
            bars("NEWNAME", CALENDAR),
            bars("HOLEY", CALENDAR[::2]),
            bars("RECYCLED", pd.bdate_range("2020-01-02", periods=5)),
        ], ignore_index=True)
        self.intervals = pd.DataFrame([
            interval("FULL", "2010-01-04"),
            interval("OLDNAME", "2015-06-01"),          # renamed to NEWNAME
            interval("HOLEY", "2016-01-04"),
            interval("ANCIENT", "2005-01-03", "2012-12-31"),
            interval("GONE", "2014-01-02", "2016-02-15"),
            interval("LEHMQ", "2010-01-04", "2016-02-01"),
            interval("RECYCLED", "2016-01-04", "2016-03-01"),
        ])
        self.aliases = pd.DataFrame([{"old_symbol": "OLDNAME", "new_symbol": "NEWNAME",
                                      "effective_date": "2016-02-01", "source": "t",
                                      "recorded_at": ""}])

    def classify(self):
        with mock.patch.object(classify_gaps.membership, "intervals",
                               side_effect=lambda index: self.intervals if index == "SP500"
                               else self.intervals.iloc[0:0]):
            return classify_gaps.classify(prices=self.prices, calendar=CALENDAR,
                                          alias_table=self.aliases)

    def test_each_interval_gets_its_cause(self):
        table = self.classify().set_index("symbol")
        self.assertEqual(table.loc["FULL", "class"], "A")
        self.assertEqual(table.loc["OLDNAME", "class"], "B")
        self.assertEqual(table.loc["OLDNAME", "lake_symbol"], "NEWNAME")
        self.assertEqual(table.loc["HOLEY", "class"], "C")
        self.assertEqual(table.loc["ANCIENT", "class"], "D")
        self.assertEqual(table.loc["GONE", "class"], "E")
        self.assertEqual(table.loc["LEHMQ", "class"], "F")
        self.assertEqual(table.loc["RECYCLED", "class"], "F")

    def test_tiingo_is_recommended_only_for_material_class_e(self):
        table = self.classify()
        with mock.patch.object(classify_gaps.manifest, "providers_by_year",
                               return_value={2016: "alpaca"}):
            summary = classify_gaps.summarize(table)
        self.assertTrue(summary["tiingo"]["recommend"])
        self.assertEqual(summary["tiingo"]["candidates"], ["GONE"])

        table = table[table["class"] != "E"]
        with mock.patch.object(classify_gaps.manifest, "providers_by_year",
                               return_value={2016: "alpaca"}):
            self.assertFalse(classify_gaps.summarize(table)["tiingo"]["recommend"])

    def test_decision_is_deferred_while_the_window_is_not_canonical(self):
        with mock.patch.object(classify_gaps.manifest, "providers_by_year",
                               return_value={2015: "yahoo", 2016: "yahoo"}):
            summary = classify_gaps.summarize(self.classify())
        self.assertIsNone(summary["tiingo"]["recommend"])
        self.assertIn("deferred", summary["tiingo"]["reason"])


class AuditHelperTests(unittest.TestCase):
    def test_year_ranges_compress_runs(self):
        by_year = {2005: "yahoo", 2006: "yahoo", 2016: "alpaca", 2017: "alpaca", 2019: "alpaca"}
        self.assertEqual(audit_data._year_ranges(by_year),
                         "2005-2006 yahoo; 2016-2017 alpaca; 2019 alpaca")

    def test_survivorship_statement_never_claims_survivorship_free(self):
        with mock.patch.object(audit_data.membership, "coverage_start",
                               side_effect=lambda i: pd.Timestamp("1996-01-02") if i == "SP500"
                               else pd.Timestamp("2026-09-01")):
            finding = audit_data.audit_survivorship()
        text = " ".join(finding.notes)
        self.assertIn("NOT survivorship-free", text)
        self.assertIn("2026-09-01", finding.headline)


if __name__ == "__main__":
    unittest.main()

"""Tests for the data audit.

The audit's job is to size each data gap correctly, so the arithmetic is what gets
tested -- on hand-built inputs rather than on the lake, because the lake's own
shape is the thing the audit is supposed to describe and a test that reads it
would just restate whatever is currently there.

Two measurements are easy to get subtly wrong and both are pinned here:

* completeness is measured over a symbol's *own* first-to-last span, not over the
  whole calendar, or every recently listed name reads as incomplete;
* a delisted name counts as usable only if it can be priced *during* the window it
  was a member -- having 2024 bars for a company removed in 2009 proves nothing.

Run with: ./venv/bin/python -m pytest tests/test_audit_data.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_audit():
    """Import scripts/audit_data.py, which is not an installed package.

    The module must be registered in ``sys.modules`` *before* it executes.
    ``audit_data`` uses postponed annotations, so on Python 3.9 the dataclass
    decorator resolves field types by looking its own module up in
    ``sys.modules`` -- and finds nothing if the registration happens after.
    """
    location = Path(REPO_ROOT) / "scripts" / "audit_data.py"
    spec = importlib.util.spec_from_file_location("audit_data", location)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_data"] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def _coverage(rows):
    """rows: (symbol, first, last, bars)."""
    return pd.DataFrame(
        [
            {"symbol": s, "first_date": pd.Timestamp(f),
             "last_date": pd.Timestamp(l), "bars": b}
            for s, f, l, b in rows
        ]
    )


class HistoryCompletenessTests(unittest.TestCase):
    def setUp(self):
        # 2024 business days: a clean calendar to measure against.
        self.calendar = pd.bdate_range("2024-01-01", "2024-12-31")

    def test_completeness_is_measured_over_the_symbols_own_span(self):
        """A name listed in December is not incomplete for missing January.

        Grading against the whole calendar would mark every recent listing as
        ragged and bury the symbols that genuinely have holes.
        """
        december = self.calendar[self.calendar >= "2024-12-01"]
        rows = _coverage([
            ("NEWLY", "2024-12-02", "2024-12-31", len(december[december >= "2024-12-02"])),
            ("OLD", str(self.calendar[0].date()), str(self.calendar[-1].date()), len(self.calendar)),
        ])

        finding = audit.audit_history(rows, self.calendar)

        assert finding.numbers["complete over own span"] == 2
        assert finding.numbers["under 99% complete"] == 0

    def test_a_genuine_hole_is_flagged(self):
        rows = _coverage([
            ("HOLED", str(self.calendar[0].date()), str(self.calendar[-1].date()),
             len(self.calendar) - 40),
        ])

        finding = audit.audit_history(rows, self.calendar)

        assert finding.numbers["under 99% complete"] == 1
        assert finding.table is not None
        assert "HOLED" in finding.table["symbol"].tolist()

    def test_stubs_are_counted_separately(self):
        """A 1-2 bar symbol is invisible to gap detection, so it gets its own count."""
        rows = _coverage([
            ("STUB", "2024-06-03", "2024-06-04", 2),
            ("FULL", str(self.calendar[0].date()), str(self.calendar[-1].date()), len(self.calendar)),
        ])

        finding = audit.audit_history(rows, self.calendar)

        assert finding.numbers["1-2 bar stubs"] == 1

    def test_deep_history_is_counted(self):
        calendar = pd.bdate_range("2005-01-03", "2026-08-21")
        rows = _coverage([
            ("DEEP", "2005-01-03", "2026-08-21", len(calendar)),
            ("SHALLOW", "2024-01-02", "2026-08-21",
             len(calendar[calendar >= "2024-01-02"])),
        ])

        finding = audit.audit_history(rows, calendar)

        assert finding.numbers["span >= 15 years"] == 1
        assert "only 1 symbol(s) carry 15+ years" in " ".join(finding.notes)

    def test_empty_lake_is_reported_not_crashed(self):
        finding = audit.audit_history(pd.DataFrame(), self.calendar)

        assert "empty" in finding.headline


class MembershipPrecisionTests(unittest.TestCase):
    def tearDown(self):
        audit.SNAPSHOT_PATH = self._original

    def _write(self, dates, tmpdir):
        self._original = audit.SNAPSHOT_PATH
        path = Path(tmpdir) / "snapshots.csv"
        pd.DataFrame({"snapshot_date": dates}).to_csv(path, index=False)
        audit.SNAPSHOT_PATH = path
        return path

    def test_gap_distribution_is_the_error_bar(self):
        """A change is dated to the first snapshot showing it, so the preceding
        gap is how wrong the date can be."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # Weekly for a year, then one 90-day hole.
            dates = list(pd.date_range("2020-01-01", periods=52, freq="7D"))
            dates.append(dates[-1] + pd.Timedelta(days=90))
            self._write([d.strftime("%Y-%m-%d") for d in dates], tmp)

            finding = audit.audit_membership_precision()

            assert finding.numbers["snapshots"] == 53
            assert finding.numbers["median gap (days)"] == 7.0
            assert finding.numbers["worst gap (days)"] == 90

    def test_absent_snapshot_file_says_so(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._original = audit.SNAPSHOT_PATH
            audit.SNAPSHOT_PATH = Path(tmp) / "missing.csv"

            finding = audit.audit_membership_precision()

            assert "no snapshot dates" in finding.headline
            assert any("build_membership" in n for n in finding.notes)


class FindingRenderTests(unittest.TestCase):
    def test_render_includes_numbers_notes_and_table(self):
        finding = audit.Finding(
            name="X",
            headline="a headline",
            numbers={"count": 3, "ratio": 0.5},
            notes=["something to know"],
            table=pd.DataFrame({"symbol": ["AAA"]}),
        )

        rendered = finding.render()

        assert "a headline" in rendered
        assert "count" in rendered and "3" in rendered
        assert "note: something to know" in rendered
        assert "AAA" in rendered

    def test_empty_table_is_skipped(self):
        finding = audit.Finding(name="X", table=pd.DataFrame())

        assert "\n\n  " not in finding.render()

    def test_to_dict_is_json_serializable(self):
        import json

        finding = audit.Finding(
            name="X", headline="h", numbers={"a": 1},
            table=pd.DataFrame({"s": ["AAA"]}),
        )

        assert json.dumps(finding.to_dict(), default=str)


if __name__ == "__main__":
    unittest.main()

"""Unfiltered S&P 500/400/600 capture, events, intervals, and the SP1500 union.

No network: snapshots are built in memory, and the Wikipedia parser is fed a
small HTML page with the same table shape as the real ones.

Run with: PYTHONPATH=src ./venv/bin/python -m pytest tests/test_membership_capture.py -v
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

from data_pipeline import membership, membership_capture as capture, schema, writer  # noqa: E402


def snapshot(counts=None, *, drop=None, add=None, move=None) -> pd.DataFrame:
    """A plausible three-index snapshot. Symbols are S5_000, S4_000, S6_000...

    ``drop``/``add``: {index: [symbols]}; ``move``: [(symbol, from, to)].
    """
    counts = counts or {"SP500": 503, "SP400": 400, "SP600": 600}
    rows = []
    for index, n in counts.items():
        tag = index[2]
        for i in range(n):
            rows.append({"index": index, "symbol": f"S{tag}_{i:03d}", "security": "Co",
                         "sector": "Tech", "industry": "Software"})
    frame = pd.DataFrame(rows)
    for index, symbols in (drop or {}).items():
        frame = frame[~((frame["index"] == index) & frame["symbol"].isin(symbols))]
    extra = [{"index": i, "symbol": s, "security": "Co", "sector": "Tech", "industry": "SW"}
             for i, syms in (add or {}).items() for s in syms]
    for symbol, source, target in move or []:
        frame = frame[~((frame["index"] == source) & (frame["symbol"] == symbol))]
        extra.append({"index": target, "symbol": symbol, "security": "Co", "sector": "Tech",
                      "industry": "SW"})
    if extra:
        frame = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)
    return frame


class CaptureCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.paths = dict(
            baseline_path=base / "membership_baseline.csv",
            events_path=base / "membership_events.csv",
            current_path=base / "current_membership.csv",
        )
        self.captured = (self.paths["baseline_path"], self.paths["events_path"])
        self.recon = base / "membership.parquet"
        membership.clear_cache()

    def tearDown(self):
        membership.clear_cache()
        self._tmp.cleanup()

    def capture(self, frame, day):
        result = capture.capture(frame, pd.Timestamp(day), **self.paths)
        membership.clear_cache()
        return result

    def events(self):
        return capture.load_events(self.paths["events_path"])


class ParseTests(unittest.TestCase):
    HTML = """
    <table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
      <tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Insurance</td></tr>
      <tr><td>aapl</td><td>Apple</td><td>Information Technology</td><td>Hardware</td></tr>
    </table>
    <table><tr><th>Date</th><th>Symbol</th><th>Security</th></tr>
      <tr><td>2024-01-01</td><td>ZZZ</td><td>Changes table, not constituents</td></tr>
    </table>"""

    def test_constituents_table_is_parsed_and_symbols_normalized(self):
        frame = capture.parse_table(self.HTML, "SP500")
        self.assertEqual(list(frame["symbol"]), ["AAPL", "BRK-B"])
        self.assertEqual(set(frame["index"]), {"SP500"})
        self.assertEqual(frame.set_index("symbol").loc["BRK-B", "sector"], "Financials")

    def test_a_page_without_a_constituents_table_is_suspicious(self):
        html = "<table><tr><th>Date</th><th>Symbol</th></tr><tr><td>x</td><td>A</td></tr></table>"
        with self.assertRaises(capture.SuspiciousScrapeError):
            capture.parse_table(html, "SP400")


class SuspiciousScrapeTests(CaptureCase):
    def test_a_truncated_scrape_is_refused_and_writes_nothing(self):
        with self.assertRaises(capture.SuspiciousScrapeError):
            self.capture(snapshot({"SP500": 12, "SP400": 400, "SP600": 600}), "2026-09-01")
        for path in self.paths.values():
            self.assertFalse(path.exists())

    def test_a_failed_scrape_cannot_generate_mass_removals(self):
        self.capture(snapshot(), "2026-09-01")
        broken = snapshot(drop={"SP500": [f"S5_{i:03d}" for i in range(60)]},
                          add={"SP500": [f"NEW{i:02d}" for i in range(60)]})
        with self.assertRaises(capture.SuspiciousScrapeError) as ctx:
            self.capture(broken, "2026-09-02")
        self.assertIn("removals", str(ctx.exception))
        self.assertEqual(len(self.events()), 0)

    def test_missing_sectors_are_suspicious(self):
        frame = snapshot()
        frame.loc[frame["index"] == "SP600", "sector"] = ""
        self.assertTrue(any("sector" in p for p in capture.check(frame)))

    def test_missing_index_is_suspicious(self):
        frame = snapshot()
        frame = frame[frame["index"] != "SP400"]
        problems = capture.check(frame)
        self.assertTrue(any(p.startswith("SP400") for p in problems))


class EventTests(CaptureCase):
    def test_first_capture_is_a_baseline_not_additions(self):
        result = self.capture(snapshot(), "2026-09-01")
        self.assertTrue(result["baseline"])
        self.assertEqual(len(self.events()), 0)
        baseline = capture.load_baseline(self.paths["baseline_path"])
        self.assertEqual(len(baseline), 1503)
        self.assertEqual(len(capture.load_current(self.paths["current_path"])), 1503)

    def test_additions_and_removals_are_recorded_with_dates_and_source(self):
        self.capture(snapshot(), "2026-09-01")
        self.capture(snapshot(drop={"SP500": ["S5_007"]}, add={"SP500": ["NEWCO"]}), "2026-09-02")
        events = self.events()
        self.assertEqual(
            events[["index", "symbol", "event"]].values.tolist(),
            [["SP500", "NEWCO", "add"], ["SP500", "S5_007", "remove"]],
        )
        self.assertEqual(set(events["observed_date"]), {"2026-09-02"})
        self.assertEqual(set(events["effective_date"]), {"2026-09-02"})
        self.assertEqual(set(events["source"]), {"wikipedia"})

    def test_rerunning_the_same_capture_adds_nothing(self):
        self.capture(snapshot(), "2026-09-01")
        later = snapshot(drop={"SP600": ["S6_001"]}, add={"SP600": ["SMALL"]})
        self.capture(later, "2026-09-02")
        self.capture(later, "2026-09-02")
        self.capture(later, "2026-09-03")
        self.assertEqual(len(self.events()), 2)

    def test_sp600_to_sp400_move_is_two_events_and_continuous_sp1500(self):
        self.capture(snapshot(), "2026-09-01")
        self.capture(snapshot(move=[("S6_010", "SP600", "SP400")],
                              drop={"SP400": ["S4_399"]}, add={"SP600": ["S6_NEW"]}),
                     "2026-09-08")
        events = self.events()
        moved = events[events["symbol"] == "S6_010"]
        self.assertEqual(sorted(moved[["index", "event"]].values.tolist()),
                         [["SP400", "add"], ["SP600", "remove"]])

        self.assertIn("S6_010", membership.members_asof("2026-09-07", "SP600", captured=self.captured))
        self.assertNotIn("S6_010", membership.members_asof("2026-09-08", "SP600", captured=self.captured))
        self.assertIn("S6_010", membership.members_asof("2026-09-08", "SP400", captured=self.captured))
        union = membership.intervals("SP1500", captured=self.captured)
        spans = union[union["symbol"] == "S6_010"]
        self.assertEqual(len(spans), 1)
        self.assertTrue(pd.isna(spans.iloc[0]["end_date"]))

    def test_a_short_gap_between_indices_is_bridged_in_the_union(self):
        self.capture(snapshot(), "2026-09-01")
        # Day 1: removed from the 600, not yet on the 400 page.
        self.capture(snapshot(drop={"SP600": ["S6_010"]}), "2026-09-08")
        # Day 2: the 400 page catches up.
        self.capture(snapshot(drop={"SP600": ["S6_010"]}, add={"SP400": ["S6_010"]}), "2026-09-09")
        for day in ("2026-09-07", "2026-09-08", "2026-09-09"):
            self.assertIn("S6_010", membership.members_asof(day, "SP1500", captured=self.captured))
        self.assertNotIn("S6_010", membership.members_asof("2026-09-08", "SP600", captured=self.captured))


class IntervalTests(CaptureCase):
    def reconstruction(self):
        pd.DataFrame([
            {"symbol": "S5_000", "index": "SP500", "start_date": pd.Timestamp("2000-01-03"),
             "end_date": pd.NaT},
            {"symbol": "OLDCO", "index": "SP500", "start_date": pd.Timestamp("2001-01-02"),
             "end_date": pd.Timestamp("2010-06-30")},
        ]).to_parquet(self.recon, index=False)

    def test_sp1500_is_the_union_of_the_three_components(self):
        self.capture(snapshot(), "2026-09-01")
        day = "2026-09-15"
        union = membership.members_asof(day, "SP1500", captured=self.captured)
        parts = set()
        for index in ("SP500", "SP400", "SP600"):
            parts |= membership.members_asof(day, index, captured=self.captured)
        self.assertEqual(union, parts)
        self.assertEqual(len(union), 1503)

    def test_mid_and_small_caps_are_left_censored_at_the_baseline(self):
        self.capture(snapshot(), "2026-09-01")
        self.assertEqual(membership.coverage_start("SP400", captured=self.captured),
                         pd.Timestamp("2026-09-01"))
        self.assertEqual(membership.members_asof("2026-08-31", "SP400", captured=self.captured), set())
        changes = membership.membership_changes("2026-01-01", "2026-12-31", "SP400",
                                                captured=self.captured)
        self.assertEqual(len(changes), 0, "a baseline must not read as additions")

    def test_sp500_joins_the_reconstruction_to_the_capture(self):
        self.reconstruction()
        self.capture(snapshot(), "2026-09-01")
        self.capture(snapshot(drop={"SP500": ["S5_000"]}), "2026-09-10")
        spans = membership.intervals("SP500", path=self.recon, captured=self.captured)
        s0 = spans[spans["symbol"] == "S5_000"]
        self.assertEqual(len(s0), 1, "the join must not split one membership in two")
        self.assertEqual(s0.iloc[0]["start_date"], pd.Timestamp("2000-01-03"))
        self.assertEqual(s0.iloc[0]["end_date"], pd.Timestamp("2026-09-09"))
        self.assertFalse(bool(s0.iloc[0]["left_censored"]))
        self.assertIn("OLDCO", membership.members_asof("2005-01-03", "SP500",
                                                        path=self.recon, captured=self.captured))
        self.assertEqual(membership.coverage_start("SP500", path=self.recon, captured=self.captured),
                         pd.Timestamp("2000-01-03"))
        self.assertEqual(membership.coverage_start("SP1500", path=self.recon, captured=self.captured),
                         pd.Timestamp("2026-09-01"))

    def test_removed_symbols_stay_tracked_for_the_lake(self):
        self.reconstruction()
        self.capture(snapshot(), "2026-09-01")
        self.capture(snapshot(drop={"SP600": ["S6_005"]}), "2026-09-10")
        current, historical = membership.tracked_symbols(
            "2016-01-01", path=self.recon, captured=self.captured,
            current_path=self.paths["current_path"],
        )
        self.assertNotIn("S6_005", current)
        self.assertIn("S6_005", historical)
        self.assertNotIn("OLDCO", historical)  # left before 2016


class UniverseFilterTests(CaptureCase):
    def load_universe_module(self):
        spec = importlib.util.spec_from_file_location("universe_under_test",
                                                      os.path.join(SRC_DIR, "universe.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_filter_reads_the_lake_and_never_drops_membership(self):
        universe = self.load_universe_module()
        self.capture(snapshot(), "2026-09-01")
        lake = Path(self._tmp.name) / "prices" / "daily"
        days = pd.bdate_range("2026-08-24", periods=5)
        rows = []
        for day in days:
            rows.append((day, "S5_000", 50, 51, 49, 50.0, 50.0, 1_000_000))   # liquid
            rows.append((day, "S5_001", 5, 5, 5, 5.0, 5.0, 10_000_000))       # under $10
            rows.append((day, "S4_000", 20, 20, 20, 20.0, 20.0, 1_000))       # illiquid
        writer.persist(pd.DataFrame(rows, columns=schema.COLUMNS), root=lake)

        candidates = universe.get_candidates(self.paths["current_path"])
        self.assertEqual(len(candidates), 1503)
        kept = universe.filter_universe(candidates, lake_root=lake, as_of=pd.Timestamp("2026-08-31"))
        self.assertEqual(list(kept["Symbol"]), ["S5_000"])
        # The screen changed universe.csv's input only; membership is untouched.
        self.assertEqual(len(capture.load_current(self.paths["current_path"])), 1503)


if __name__ == "__main__":
    unittest.main()

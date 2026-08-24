"""Tests for the daily price lake (schema + store).

No network, no Alpaca. Every test builds its own lake in a temp directory.

Run with: ./venv/bin/python -m pytest tests/test_data_pipeline.py -v
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
from data_pipeline.schema import (  # noqa: E402
    ADJ_CLOSE,
    CLOSE,
    COLUMNS,
    DATE,
    SYMBOL,
    VOLUME,
)


def bars(rows) -> pd.DataFrame:
    """Build a lake frame from (date, symbol, o, h, l, c, adj, vol) tuples."""
    return pd.DataFrame(rows, columns=COLUMNS)


def one(date: str, symbol: str, close: float, *, adj: float = None, vol=1000):
    adj = close if adj is None else adj
    return (date, symbol, close, close, close, close, adj, vol)


class SchemaContractTests(unittest.TestCase):
    def test_coerce_enforces_dtypes_and_order(self):
        scrambled = pd.DataFrame(
            {
                SYMBOL: ["aapl "],
                DATE: ["2026-01-05"],
                "open": ["10.5"],
                "high": ["11"],
                "low": ["10"],
                "close": ["10.75"],
                "adj_close": ["10.70"],
                "volume": ["1234.0"],
                "extra_junk": ["drop me"],
            }
        )
        out = schema.coerce(scrambled)

        self.assertEqual(list(out.columns), COLUMNS)
        self.assertEqual(out[SYMBOL].iloc[0], "AAPL")  # trimmed + uppercased
        self.assertEqual(str(out[VOLUME].dtype), "Int64")
        self.assertEqual(str(out[CLOSE].dtype), "float64")
        self.assertEqual(out[VOLUME].iloc[0], 1234)

    def test_dates_are_naive_and_midnight(self):
        tz_aware = bars([one("2026-01-05", "SPY", 100.0)])
        tz_aware[DATE] = pd.to_datetime(tz_aware[DATE]).dt.tz_localize("America/New_York")

        out = schema.coerce(tz_aware)
        ts = out[DATE].iloc[0]

        self.assertIsNone(ts.tzinfo)
        self.assertEqual((ts.hour, ts.minute, ts.second), (0, 0, 0))

    def test_volume_nan_survives_and_is_never_zero_filled(self):
        frame = bars([one("2026-01-05", "SPY", 100.0, vol=None)])
        out = schema.coerce(frame)

        # Missing volume must stay missing. Comparing pd.NA to 0 with != is
        # ambiguous by design, so assert on the null mask and the sum instead:
        # a zero-filled column would be non-null and sum to 0.
        self.assertTrue(pd.isna(out[VOLUME].iloc[0]))
        self.assertEqual(int(out[VOLUME].notna().sum()), 0)
        self.assertEqual(str(out[VOLUME].dtype), "Int64")

    def test_canonical_sort_is_date_then_symbol(self):
        out = schema.coerce(
            bars(
                [
                    one("2026-01-06", "ZZZ", 1.0),
                    one("2026-01-05", "MSFT", 2.0),
                    one("2026-01-05", "AAPL", 3.0),
                ]
            )
        )
        self.assertEqual(list(out[SYMBOL]), ["AAPL", "MSFT", "ZZZ"])
        self.assertEqual(
            [str(d.date()) for d in out[DATE]],
            ["2026-01-05", "2026-01-05", "2026-01-06"],
        )

    def test_missing_column_is_an_error(self):
        with self.assertRaises(ValueError):
            schema.coerce(pd.DataFrame({DATE: ["2026-01-05"], SYMBOL: ["SPY"]}))

    def test_assert_unique_key_raises_on_duplicates(self):
        dupes = schema.coerce(
            bars([one("2026-01-05", "SPY", 100.0), one("2026-01-05", "SPY", 101.0)])
        )
        with self.assertRaises(ValueError):
            schema.assert_unique_key(dupes)


class UpsertTests(unittest.TestCase):
    def test_incoming_wins_on_key_collision(self):
        existing = bars([one("2026-01-05", "SPY", 100.0)])
        incoming = bars([one("2026-01-05", "SPY", 999.0)])

        out = schema.upsert(existing, incoming)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[CLOSE].iloc[0], 999.0)

    def test_upsert_is_idempotent(self):
        existing = bars([one("2026-01-05", "SPY", 100.0)])
        once = schema.upsert(existing, existing)
        twice = schema.upsert(once, existing)

        self.assertEqual(len(once), 1)
        self.assertEqual(len(twice), 1)
        pd.testing.assert_frame_equal(once, twice)

    def test_upsert_merges_disjoint_rows(self):
        out = schema.upsert(
            bars([one("2026-01-05", "SPY", 100.0)]),
            bars([one("2026-01-06", "SPY", 101.0), one("2026-01-06", "QQQ", 50.0)]),
        )
        self.assertEqual(len(out), 3)
        schema.assert_unique_key(out)

    def test_empty_sides_are_safe(self):
        frame = bars([one("2026-01-05", "SPY", 100.0)])
        self.assertEqual(len(schema.upsert(schema.empty_frame(), frame)), 1)
        self.assertEqual(len(schema.upsert(frame, schema.empty_frame())), 1)
        self.assertEqual(len(schema.upsert(schema.empty_frame(), schema.empty_frame())), 0)


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_csv_round_trip_preserves_values_and_types(self):
        frame = schema.coerce(
            bars(
                [
                    one("2026-01-05", "SPY", 100.123456, adj=99.654321, vol=1_234_567),
                    one("2026-01-05", "QQQ", 50.5, adj=50.5, vol=None),
                ]
            )
        )
        path = schema.hot_year_path(2026, self.root)
        schema.write_csv(frame, path)

        back = schema.read_frame(path)

        pd.testing.assert_frame_equal(frame, back)
        self.assertTrue(pd.isna(back.loc[back[SYMBOL] == "QQQ", VOLUME].iloc[0]))

    def test_parquet_round_trip_preserves_values_and_types(self):
        frame = schema.coerce(
            bars(
                [
                    one("2025-03-03", "SPY", 100.123456, adj=99.654321, vol=1_234_567),
                    one("2025-03-03", "QQQ", 50.5, adj=50.5, vol=None),
                ]
            )
        )
        path = schema.cold_year_path(2025, self.root)
        schema.write_parquet(frame, path)

        back = schema.read_frame(path)

        pd.testing.assert_frame_equal(frame, back)

    def test_raw_close_and_adj_close_both_survive(self):
        frame = bars([("2026-01-05", "KO", 60.0, 61.0, 59.0, 60.5, 58.25, 5_000)])
        path = schema.hot_year_path(2026, self.root)
        schema.write_csv(frame, path)

        back = schema.read_frame(path)

        self.assertEqual(back[CLOSE].iloc[0], 60.5)
        self.assertEqual(back[ADJ_CLOSE].iloc[0], 58.25)
        self.assertNotEqual(back[CLOSE].iloc[0], back[ADJ_CLOSE].iloc[0])

    def test_csv_write_is_byte_deterministic(self):
        """The whole git-cost argument for CSV rests on this."""
        frame = bars(
            [
                one("2026-01-05", "SPY", 100.1, vol=1000),
                one("2026-01-06", "SPY", 100.2, vol=None),
                one("2026-01-06", "AAPL", 200.987654321, vol=42),
            ]
        )
        a = self.root / "a.csv"
        b = self.root / "b.csv"

        schema.write_csv(frame, a)
        schema.write_csv(frame, b)

        self.assertEqual(a.read_bytes(), b.read_bytes())

        # Re-writing what we just read back must also be stable.
        c = self.root / "c.csv"
        schema.write_csv(schema.read_frame(a), c)
        self.assertEqual(a.read_bytes(), c.read_bytes())

    def test_appending_a_session_leaves_earlier_bytes_untouched(self):
        """Git can only delta cheaply if the prefix is unchanged."""
        day1 = bars([one("2026-01-05", "AAPL", 100.0), one("2026-01-05", "SPY", 400.0)])
        path = schema.hot_year_path(2026, self.root)
        schema.write_csv(day1, path)
        before = path.read_bytes()

        day2 = bars([one("2026-01-06", "AAPL", 101.0), one("2026-01-06", "SPY", 401.0)])
        schema.write_csv(schema.upsert(schema.read_frame(path), day2), path)
        after = path.read_bytes()

        self.assertTrue(after.startswith(before))

    def test_write_year_rejects_foreign_years(self):
        frame = bars([one("2026-01-05", "SPY", 100.0)])
        with self.assertRaises(ValueError):
            schema.write_year(frame, 2025, self.root, hot=True)


class YearResolutionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_year_boundary_lands_in_different_files(self):
        frame = schema.coerce(
            bars([one("2025-12-31", "SPY", 100.0), one("2026-01-02", "SPY", 101.0)])
        )
        for year in schema.years_in(frame):
            rows = frame[frame[DATE].dt.year == year]
            schema.write_year(rows, year, self.root, hot=(year == 2026))

        self.assertTrue(schema.cold_year_path(2025, self.root).exists())
        self.assertTrue(schema.hot_year_path(2026, self.root).exists())

        y2025 = schema.read_frame(schema.cold_year_path(2025, self.root))
        y2026 = schema.read_frame(schema.hot_year_path(2026, self.root))
        self.assertEqual(len(y2025), 1)
        self.assertEqual(len(y2026), 1)
        self.assertEqual(str(y2025[DATE].iloc[0].date()), "2025-12-31")
        self.assertEqual(str(y2026[DATE].iloc[0].date()), "2026-01-02")

    def test_parquet_preferred_when_both_formats_exist(self):
        """Mid-rollover state must resolve to exactly one file per year."""
        rows = bars([one("2025-06-02", "SPY", 100.0)])
        schema.write_csv(rows, schema.hot_year_path(2025, self.root))
        schema.write_parquet(rows, schema.cold_year_path(2025, self.root))

        resolved = schema.resolve_year_path(2025, self.root)
        self.assertEqual(resolved.suffix, ".parquet")

        discovered = schema.discover_year_files(self.root)
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].suffix, ".parquet")

    def test_discover_ignores_unrelated_files(self):
        (self.root / "notes.txt").write_text("hello")
        (self.root / "_schema.md").write_text("docs")
        schema.write_csv(bars([one("2026-01-05", "SPY", 1.0)]), schema.hot_year_path(2026, self.root))

        found = schema.discover_year_files(self.root)
        self.assertEqual([p.name for p in found], ["2026.csv"])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

        cold = bars(
            [
                one("2025-12-30", "SPY", 400.0, adj=399.0),
                one("2025-12-30", "XLE", 90.0, adj=89.5),
                one("2025-12-31", "SPY", 401.0, adj=400.0),
                one("2025-12-31", "XLE", 91.0, adj=90.5),
            ]
        )
        schema.write_parquet(cold, schema.cold_year_path(2025, self.root))

        hot = bars(
            [
                one("2026-01-02", "SPY", 402.0, adj=401.0),
                one("2026-01-02", "XLE", 92.0, adj=91.5),
                one("2026-01-05", "SPY", 403.0, adj=402.0),
                one("2026-01-05", "XLE", 93.0, adj=92.5),
            ]
        )
        schema.write_csv(hot, schema.hot_year_path(2026, self.root))

    def tearDown(self):
        self._tmp.cleanup()

    def test_union_of_cold_and_hot_has_no_duplicate_keys(self):
        frame = store.load_prices(root=self.root)

        self.assertEqual(len(frame), 8)
        schema.assert_unique_key(frame)
        self.assertEqual(schema.years_in(frame), [2025, 2026])

    def test_symbol_and_date_filtering(self):
        frame = store.load_prices(
            ["SPY"], start="2025-12-31", end="2026-01-02", root=self.root
        )
        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame[SYMBOL]), {"SPY"})

    def test_close_matrix_shape_and_column_order(self):
        matrix = store.load_close_matrix(["XLE", "SPY"], root=self.root)

        self.assertEqual(list(matrix.columns), ["SPY", "XLE"])  # requested order
        self.assertEqual(len(matrix), 4)
        self.assertEqual(matrix.index.name, DATE)

    def test_close_matrix_defaults_to_adj_close(self):
        adj = store.load_close_matrix(["SPY"], root=self.root)
        raw = store.load_close_matrix(["SPY"], column=CLOSE, root=self.root)

        self.assertEqual(adj["SPY"].iloc[-1], 402.0)
        self.assertEqual(raw["SPY"].iloc[-1], 403.0)

    def test_lowercase_and_padded_symbol_requests_are_normalized(self):
        matrix = store.load_close_matrix([" spy "], root=self.root)
        self.assertEqual(list(matrix.columns), ["SPY"])

    def test_missing_symbol_returns_partial_not_error(self):
        matrix = store.load_close_matrix(["SPY", "NOPE"], root=self.root)
        self.assertEqual(list(matrix.columns), ["SPY"])

    def test_last_bar_date(self):
        self.assertEqual(
            store.last_bar_date(root=self.root), pd.Timestamp("2026-01-05")
        )

    def test_last_date_per_symbol(self):
        latest = store.last_date_per_symbol(root=self.root)
        self.assertEqual(latest["SPY"], pd.Timestamp("2026-01-05"))
        self.assertEqual(latest["XLE"], pd.Timestamp("2026-01-05"))

    def test_coverage_reports_bars_per_symbol(self):
        cov = store.coverage(root=self.root).set_index("symbol")
        self.assertEqual(cov.loc["SPY", "bars"], 4)
        self.assertEqual(cov.loc["SPY", "first_date"], pd.Timestamp("2025-12-30"))

    def test_has_lookback_gate(self):
        self.assertTrue(store.has_lookback(["SPY", "XLE"], 4, root=self.root))
        self.assertFalse(store.has_lookback(["SPY"], 5, root=self.root))
        self.assertFalse(store.has_lookback(["NOPE"], 1, root=self.root))

    def test_empty_lake_is_graceful(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(len(store.load_prices(root=Path(empty))), 0)
            self.assertTrue(store.load_close_matrix(["SPY"], root=Path(empty)).empty)
            self.assertIsNone(store.last_bar_date(root=Path(empty)))
            self.assertFalse(store.has_lookback(["SPY"], 1, root=Path(empty)))

    def test_bad_column_name_is_rejected(self):
        with self.assertRaises(ValueError):
            store.load_close_matrix(["SPY"], column="vwap", root=self.root)

    def test_store_does_not_import_yfinance(self):
        """The lake must never silently fetch behind a strategy's back.

        Checked against the parsed import graph rather than raw text, since the
        module docstring legitimately mentions yfinance to explain its absence.
        """
        import ast

        source = Path(SRC_DIR, "data_pipeline", "store.py").read_text()
        tree = ast.parse(source)

        imported: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])

        self.assertNotIn("yfinance", imported)
        self.assertNotIn("requests", imported)
        self.assertNotIn("urllib", imported)


if __name__ == "__main__":
    unittest.main()

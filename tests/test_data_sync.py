"""Tests for the lake write path: registry, fetch parsing, writer, sync.

No network. yfinance is monkeypatched with a fake that serves deterministic
bars, so the whole sync loop (batching, checkpointing, upsert, registry
bookkeeping, failure accounting) is exercised offline.

Run with: PYTHONPATH=src ./venv/bin/python -m pytest tests/test_data_sync.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_pipeline import fetch, registry, schema, store, sync_prices, writer  # noqa: E402
from data_pipeline.schema import ADJ_CLOSE, CLOSE, COLUMNS, DATE, SYMBOL, VOLUME  # noqa: E402


def bars(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def one(date: str, symbol: str, close: float, *, adj=None, vol=1000):
    adj = close if adj is None else adj
    return (date, symbol, close, close, close, close, adj, vol)


def sessions(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=periods)


class FakeYFinance:
    """Stands in for the yfinance module inside fetch.fetch_batch."""

    def __init__(self, *, known=None, fail=(), raise_on_call=False):
        self.known = known
        self.fail = set(fail)
        self.raise_on_call = raise_on_call
        self.calls = []

    def download(self, tickers, start=None, end=None, **kwargs):
        self.calls.append({"tickers": list(tickers), "start": start, "end": end, **kwargs})
        if self.raise_on_call:
            raise RuntimeError("simulated network failure")

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        idx = pd.bdate_range(start=start_ts, end=end_ts - pd.Timedelta(days=1))

        frames = {}
        for i, sym in enumerate(tickers):
            if sym in self.fail:
                continue
            if self.known is not None and sym not in self.known:
                continue
            if len(idx) == 0:
                continue
            base = 100.0 + i
            sub = pd.DataFrame(
                {
                    "Open": [base] * len(idx),
                    "High": [base + 1] * len(idx),
                    "Low": [base - 1] * len(idx),
                    "Close": [base + 0.5] * len(idx),
                    "Adj Close": [base + 0.25] * len(idx),
                    "Volume": [1_000 + i] * len(idx),
                },
                index=idx,
            )
            sub.index.name = "Date"
            frames[sym] = sub

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)


class RegistryTests(unittest.TestCase):
    def test_symbol_normalization_yahoo_form(self):
        self.assertEqual(registry.normalize_symbol("brk.b"), "BRK-B")
        self.assertEqual(registry.normalize_symbol("  aapl "), "AAPL")

    def test_target_symbols_unions_registry_universe_and_etfs(self):
        reg = registry.coerce_registry(
            pd.DataFrame([{"symbol": "OLDNAME", "consecutive_failures": 0}])
        )
        uni = pd.DataFrame([{"symbol": "NEWNAME", "sector": "Tech", "industry": "SW"}])

        targets = sync_prices.registry.target_symbols(reg, uni)

        self.assertIn("OLDNAME", targets)   # kept even if not in today's universe
        self.assertIn("NEWNAME", targets)
        for etf in registry.ALWAYS_TRACKED:
            self.assertIn(etf, targets)

    def test_dead_symbols_are_skipped_but_rows_retained(self):
        reg = registry.coerce_registry(
            pd.DataFrame(
                [
                    {"symbol": "DEADCO", "consecutive_failures": registry.MAX_CONSECUTIVE_FAILURES},
                    {"symbol": "ALIVECO", "consecutive_failures": 3},
                ]
            )
        )
        targets = registry.target_symbols(reg, pd.DataFrame(columns=["symbol", "sector", "industry"]))

        self.assertNotIn("DEADCO", targets)
        self.assertIn("ALIVECO", targets)
        self.assertIn("DEADCO", set(reg["symbol"]))  # row survives

    def test_etf_is_never_retired(self):
        reg = registry.coerce_registry(
            pd.DataFrame([{"symbol": "SPY", "consecutive_failures": 999}])
        )
        targets = registry.target_symbols(reg, pd.DataFrame(columns=["symbol", "sector", "industry"]))
        self.assertIn("SPY", targets)

    def test_symbol_back_in_universe_is_revived(self):
        reg = registry.coerce_registry(
            pd.DataFrame([{"symbol": "PHOENIX", "consecutive_failures": 50}])
        )
        uni = pd.DataFrame([{"symbol": "PHOENIX", "sector": "X", "industry": "Y"}])
        self.assertIn("PHOENIX", registry.target_symbols(reg, uni))

    def test_update_registry_tracks_first_seen_last_seen_and_failures(self):
        uni = pd.DataFrame([{"symbol": "AAA", "sector": "Tech", "industry": "SW"}])
        day1 = pd.Timestamp("2026-01-05")

        reg = registry.update_registry(
            registry.empty_registry(), uni, succeeded=["AAA"], failed=[], as_of=day1
        )
        row = reg.set_index("symbol").loc["AAA"]
        self.assertEqual(row["first_seen"], day1)
        self.assertEqual(row["last_seen"], day1)
        self.assertEqual(row["consecutive_failures"], 0)
        self.assertEqual(row["sector"], "Tech")

        # A later failure must not advance last_seen, and must not reset first_seen.
        day2 = pd.Timestamp("2026-01-06")
        reg = registry.update_registry(reg, uni, succeeded=[], failed=["AAA"], as_of=day2)
        row = reg.set_index("symbol").loc["AAA"]
        self.assertEqual(row["first_seen"], day1)
        self.assertEqual(row["last_seen"], day1)
        self.assertEqual(row["consecutive_failures"], 1)

        # Success resets the counter and advances last_seen.
        day3 = pd.Timestamp("2026-01-07")
        reg = registry.update_registry(reg, uni, succeeded=["AAA"], failed=[], as_of=day3)
        row = reg.set_index("symbol").loc["AAA"]
        self.assertEqual(row["last_seen"], day3)
        self.assertEqual(row["consecutive_failures"], 0)

    def test_registry_never_shrinks(self):
        uni = pd.DataFrame(columns=["symbol", "sector", "industry"])
        reg = registry.update_registry(
            registry.empty_registry(), uni,
            succeeded=["AAA", "BBB"], failed=[], as_of=pd.Timestamp("2026-01-05"),
        )
        shrunk = registry.update_registry(
            reg, uni, succeeded=["AAA"], failed=[], as_of=pd.Timestamp("2026-01-06")
        )
        self.assertEqual(set(shrunk["symbol"]), {"AAA", "BBB"})

    def test_registry_round_trip_and_stable_sort(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "master_tickers.csv"
            reg = registry.update_registry(
                registry.empty_registry(),
                pd.DataFrame(columns=["symbol", "sector", "industry"]),
                succeeded=["ZZZ", "AAA", "MMM"], failed=[], as_of=pd.Timestamp("2026-01-05"),
            )
            registry.write_registry(reg, path)
            back = registry.load_registry(path)

            self.assertEqual(list(back["symbol"]), ["AAA", "MMM", "ZZZ"])
            registry.write_registry(back, path)
            self.assertEqual(list(registry.load_registry(path)["symbol"]), ["AAA", "MMM", "ZZZ"])

    def test_universe_metadata_normalizes_dotted_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.csv"
            pd.DataFrame(
                {"Symbol": ["BRK.B", "AAPL"], "Sector": ["Fin", "Tech"], "Industry": ["I", "SW"]}
            ).to_csv(path, index=False)

            meta = registry.load_universe_metadata(path)
            self.assertEqual(set(meta["symbol"]), {"BRK-B", "AAPL"})

    def test_missing_universe_file_is_not_fatal(self):
        meta = registry.load_universe_metadata(Path("/nonexistent/universe.csv"))
        self.assertEqual(len(meta), 0)


class FetchParsingTests(unittest.TestCase):
    def test_multi_ticker_response_is_parsed(self):
        fake = FakeYFinance()
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            got, failed = fetch.fetch_batch(
                ["AAA", "BBB"], pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-09")
            )

        self.assertEqual(failed, [])
        self.assertEqual(set(got[SYMBOL]), {"AAA", "BBB"})
        self.assertEqual(list(got.columns), COLUMNS)

    def test_raw_and_adjusted_close_are_both_captured(self):
        fake = FakeYFinance()
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            got, _ = fetch.fetch_batch(
                ["AAA"], pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07")
            )
        self.assertEqual(got[CLOSE].iloc[0], 100.5)
        self.assertEqual(got[ADJ_CLOSE].iloc[0], 100.25)

    def test_auto_adjust_is_false_so_adj_close_exists(self):
        fake = FakeYFinance()
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            fetch.fetch_batch(["AAA"], pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07"))
        self.assertFalse(fake.calls[0]["auto_adjust"])

    def test_current_session_is_never_stored(self):
        """Contract 2: the 09:30 run must not persist a partial bar."""
        today = fetch.today_naive()
        fake = FakeYFinance()
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            got, _ = fetch.fetch_batch(
                ["AAA"], today - pd.Timedelta(days=10), today + pd.Timedelta(days=1)
            )
        if len(got):
            self.assertLess(got[DATE].max(), today)

    def test_missing_symbols_are_reported_as_failures(self):
        fake = FakeYFinance(known={"AAA"})
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            got, failed = fetch.fetch_batch(
                ["AAA", "GONE"], pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-09")
            )
        self.assertEqual(failed, ["GONE"])
        self.assertEqual(set(got[SYMBOL]), {"AAA"})

    def test_whole_batch_exception_marks_all_failed(self):
        fake = FakeYFinance(raise_on_call=True)
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            got, failed = fetch.fetch_batch(
                ["AAA", "BBB"], pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-09")
            )
        self.assertEqual(len(got), 0)
        self.assertEqual(set(failed), {"AAA", "BBB"})

    def test_batched_splits_evenly(self):
        self.assertEqual(fetch.batched(["a", "b", "c"], 2), [["a", "b"], ["c"]])
        self.assertEqual(fetch.batched([], 2), [])


class WriterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_current_year_goes_to_csv_older_years_to_parquet(self):
        frame = bars([one("2024-06-03", "SPY", 100.0), one("2026-06-03", "SPY", 200.0)])
        writer.persist(frame, root=self.root, current_year=2026)

        self.assertTrue(schema.cold_year_path(2024, self.root).exists())
        self.assertTrue(schema.hot_year_path(2026, self.root).exists())

    def test_existing_year_keeps_its_format(self):
        """January overlap must not rewrite last year's Parquet as CSV."""
        schema.write_parquet(bars([one("2025-12-30", "SPY", 100.0)]), schema.cold_year_path(2025, self.root))

        writer.persist(bars([one("2025-12-31", "SPY", 101.0)]), root=self.root, current_year=2026)

        self.assertTrue(schema.cold_year_path(2025, self.root).exists())
        self.assertFalse(schema.hot_year_path(2025, self.root).exists())
        self.assertEqual(len(schema.read_frame(schema.cold_year_path(2025, self.root))), 2)

    def test_persist_is_idempotent(self):
        frame = bars([one("2026-06-03", "SPY", 100.0)])
        writer.persist(frame, root=self.root, current_year=2026)
        writer.persist(frame, root=self.root, current_year=2026)

        self.assertEqual(len(store.load_prices(root=self.root)), 1)

    def test_persist_applies_corrections(self):
        writer.persist(bars([one("2026-06-03", "SPY", 100.0)]), root=self.root, current_year=2026)
        writer.persist(bars([one("2026-06-03", "SPY", 111.0)]), root=self.root, current_year=2026)

        got = store.load_prices(root=self.root)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[CLOSE].iloc[0], 111.0)

    def test_finalize_year_converts_csv_to_parquet(self):
        rows = bars([one("2026-01-05", "SPY", 100.0), one("2026-06-03", "SPY", 110.0)])
        schema.write_csv(rows, schema.hot_year_path(2026, self.root))

        path = writer.finalize_year(2026, self.root)

        self.assertEqual(path.suffix, ".parquet")
        self.assertTrue(path.exists())
        self.assertFalse(schema.hot_year_path(2026, self.root).exists())
        self.assertEqual(len(schema.read_frame(path)), 2)

    def test_finalize_year_is_idempotent(self):
        schema.write_csv(bars([one("2026-01-05", "SPY", 100.0)]), schema.hot_year_path(2026, self.root))
        writer.finalize_year(2026, self.root)
        again = writer.finalize_year(2026, self.root)  # no CSV left; must not raise
        self.assertTrue(again.exists())


class SyncTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.lake = base / "daily"
        self.registry_path = base / "master_tickers.csv"
        self.universe_path = base / "universe.csv"
        self.checkpoint = base / "sync_state.json"

        pd.DataFrame(
            {"Symbol": ["AAA", "BBB"], "Sector": ["Tech", "Fin"], "Industry": ["SW", "Bank"]}
        ).to_csv(self.universe_path, index=False)

        self.start = pd.Timestamp("2026-01-05")

    def tearDown(self):
        self._tmp.cleanup()

    def _sync(self, fake, **kwargs):
        params = dict(
            lake_root=self.lake,
            registry_path=self.registry_path,
            universe_path=self.universe_path,
            checkpoint_path=self.checkpoint,
            lookback_start=self.start,
            batch_size=50,
        )
        params.update(kwargs)
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            return sync_prices.sync(**params)

    def test_first_sync_populates_lake_and_registry(self):
        summary = self._sync(FakeYFinance(), symbols_override=["AAA", "BBB"])

        self.assertGreater(summary["rows"], 0)
        frame = store.load_prices(root=self.lake)
        self.assertEqual(set(frame[SYMBOL]), {"AAA", "BBB"})

        reg = registry.load_registry(self.registry_path)
        self.assertEqual(set(reg["symbol"]), {"AAA", "BBB"})
        self.assertTrue((reg["consecutive_failures"] == 0).all())

    def test_second_sync_is_idempotent(self):
        self._sync(FakeYFinance(), symbols_override=["AAA"])
        first = len(store.load_prices(root=self.lake))

        self._sync(FakeYFinance(), symbols_override=["AAA"])
        second = len(store.load_prices(root=self.lake))

        self.assertEqual(first, second)
        schema.assert_unique_key(store.load_prices(root=self.lake))

    def test_incremental_sync_refetches_only_the_overlap(self):
        self._sync(FakeYFinance(), symbols_override=["AAA"])
        last = store.last_bar_date(root=self.lake)

        fake = FakeYFinance()
        self._sync(fake, symbols_override=["AAA"])

        # The follow-up request must start near the stored tail, not at inception.
        requested_start = pd.Timestamp(fake.calls[0]["start"])
        self.assertGreaterEqual(
            requested_start, last - pd.Timedelta(days=sync_prices.OVERLAP_DAYS + 1)
        )
        self.assertGreater(requested_start, self.start)

    def test_failed_symbols_do_not_abort_the_run(self):
        summary = self._sync(FakeYFinance(fail={"BBB"}), symbols_override=["AAA", "BBB"])

        self.assertIn("BBB", summary["failed"])
        self.assertEqual(set(store.load_prices(root=self.lake)[SYMBOL]), {"AAA"})

        reg = registry.load_registry(self.registry_path).set_index("symbol")
        self.assertEqual(reg.loc["BBB", "consecutive_failures"], 1)
        self.assertEqual(reg.loc["AAA", "consecutive_failures"], 0)

    def test_total_network_failure_still_writes_registry_and_exits_clean(self):
        summary = self._sync(FakeYFinance(raise_on_call=True), symbols_override=["AAA", "BBB"])

        self.assertEqual(summary["rows"], 0)
        self.assertEqual(set(summary["failed"]), {"AAA", "BBB"})
        self.assertTrue(self.registry_path.exists())

    def test_checkpoint_is_cleared_on_success(self):
        self._sync(FakeYFinance(), symbols_override=["AAA"])
        self.assertFalse(self.checkpoint.exists())

    def test_checkpoint_allows_resume(self):
        signature = sync_prices._signature(["AAA", "BBB"], self.start, fetch.today_naive())
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_text(json.dumps({"signature": signature, "done": ["AAA"]}))

        fake = FakeYFinance()
        self._sync(fake, symbols_override=["AAA", "BBB"])

        requested = {t for call in fake.calls for t in call["tickers"]}
        self.assertNotIn("AAA", requested)  # already done
        self.assertIn("BBB", requested)

    def test_stale_checkpoint_signature_is_discarded(self):
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_text(json.dumps({"signature": "does-not-match", "done": ["AAA"]}))

        fake = FakeYFinance()
        self._sync(fake, symbols_override=["AAA"])

        requested = {t for call in fake.calls for t in call["tickers"]}
        self.assertIn("AAA", requested)

    def test_year_boundary_split_across_formats(self):
        fake = FakeYFinance()
        self._sync(
            fake,
            symbols_override=["AAA"],
            lookback_start=pd.Timestamp("2025-12-29"),
        )

        frame = store.load_prices(root=self.lake)
        self.assertIn(2025, schema.years_in(frame))
        self.assertTrue(schema.cold_year_path(2025, self.lake).exists())

    def test_new_symbol_gets_full_history_existing_gets_overlap(self):
        self._sync(FakeYFinance(), symbols_override=["AAA"])

        fake = FakeYFinance()
        self._sync(fake, symbols_override=["AAA", "CCC"])

        starts = {
            t: pd.Timestamp(call["start"])
            for call in fake.calls
            for t in call["tickers"]
        }
        self.assertEqual(starts["CCC"], self.start)          # full history
        self.assertGreater(starts["AAA"], self.start)        # overlap only


if __name__ == "__main__":
    unittest.main()

"""Provenance manifest, anchor factors, sidecar isolation, and quarantine.

Run with: PYTHONPATH=src ./venv/bin/python -m pytest tests/test_lake_provenance.py -v
"""

from __future__ import annotations

import json
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

from data_pipeline import anchor, manifest, quarantine, schema, store, writer  # noqa: E402

ALPACA = {"provider": "alpaca", "feed": "sip", "adjustment": "raw+all"}
YAHOO = {"provider": "yahoo", "feed": "yfinance", "adjustment": "split-adjusted"}


def frame(symbol="AAA", start="2024-03-01", periods=30, *, base=100.0, drift=0.001, adj_scale=1.0):
    days = pd.bdate_range(start, periods=periods)
    close = [base * (1 + drift) ** i for i in range(periods)]
    return pd.DataFrame({
        schema.DATE: days, schema.SYMBOL: symbol,
        schema.OPEN: close, schema.HIGH: close, schema.LOW: close, schema.CLOSE: close,
        schema.ADJ_CLOSE: [c * adj_scale for c in close], schema.VOLUME: 1_000,
    })


class LakeCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lake = Path(self._tmp.name) / "prices" / "daily"

    def tearDown(self):
        self._tmp.cleanup()


class SidecarTests(unittest.TestCase):
    def test_canonical_layout_keeps_metadata_beside_daily(self):
        root = Path("/x/data/prices/daily")
        self.assertEqual(manifest.manifest_path(root), Path("/x/data/prices/manifest.json"))
        self.assertEqual(anchor.anchor_path(root), Path("/x/data/prices/anchor_factors.csv"))
        self.assertEqual(quarantine.quarantine_dir(root), Path("/x/data/prices/quarantine"))

    def test_other_lake_roots_never_share_metadata(self):
        a, b = Path("/tmp/lake_a"), Path("/tmp/lake_b")
        self.assertNotEqual(manifest.manifest_path(a), manifest.manifest_path(b))
        self.assertEqual(manifest.manifest_path(a).parent, a / "_meta")

    def test_default_lake_manifest_is_the_repository_file(self):
        self.assertEqual(manifest.manifest_path(),
                         Path(REPO_ROOT) / "data" / "prices" / "manifest.json")


class ManifestTests(LakeCase):
    def test_persist_records_every_manifest_field(self):
        writer.persist(frame(), root=self.lake, provenance=ALPACA)
        entry = manifest.partition(2024, self.lake)
        for field in ("year", "provider", "feed", "adjustment", "downloaded_at", "start_date",
                      "end_date", "row_count", "content_hash", "file"):
            self.assertIn(field, entry)
        self.assertEqual(entry["provider"], "alpaca")
        self.assertEqual(entry["row_count"], 30)
        self.assertEqual(entry["start_date"], "2024-03-01")
        self.assertTrue(entry["content_hash"].startswith("sha256:"))

    def test_hash_is_format_independent_and_content_sensitive(self):
        f = frame()
        self.assertEqual(manifest.content_hash(f), manifest.content_hash(f.iloc[::-1]))
        g = f.copy()
        g.loc[3, schema.CLOSE] += 0.01
        self.assertNotEqual(manifest.content_hash(f), manifest.content_hash(g))

    def test_a_year_holds_one_provider(self):
        writer.persist(frame(), root=self.lake, provenance=ALPACA)
        with self.assertRaises(manifest.MixedProviderError):
            writer.persist(frame("BBB"), root=self.lake, provenance=YAHOO)
        with self.assertRaises(manifest.MixedProviderError):
            writer.persist(frame("BBB"), root=self.lake)  # unspecified is a provider too

    def test_refusal_is_checked_before_any_year_is_written(self):
        writer.persist(frame(start="2024-12-02", periods=5), root=self.lake, provenance=YAHOO)
        writer.persist(frame(start="2025-01-02", periods=5), root=self.lake, provenance=ALPACA)
        before = schema.read_frame(schema.resolve_year_path(2024, self.lake))
        spanning = frame(start="2024-12-20", periods=10)
        with self.assertRaises(manifest.MixedProviderError):
            writer.persist(spanning, root=self.lake, provenance=ALPACA)
        after = schema.read_frame(schema.resolve_year_path(2024, self.lake))
        pd.testing.assert_frame_equal(before, after)

    def test_unknown_partition_must_be_bootstrapped_first(self):
        schema.write_year(frame(), 2024, self.lake, hot=False)
        with self.assertRaises(manifest.MixedProviderError):
            writer.persist(frame(), root=self.lake, provenance=ALPACA)
        manifest.bootstrap(2024, ALPACA, self.lake)
        writer.persist(frame(), root=self.lake, provenance=ALPACA)
        with self.assertRaises(manifest.MixedProviderError):
            manifest.bootstrap(2024, YAHOO, self.lake)

    def test_verify_catches_an_unrecorded_change(self):
        writer.persist(frame(), root=self.lake, provenance=ALPACA)
        self.assertEqual(manifest.verify(self.lake), [])
        path = schema.resolve_year_path(2024, self.lake)
        tampered = schema.read_frame(path)
        tampered.loc[0, schema.CLOSE] = 1.0
        schema.write_year(tampered, 2024, self.lake, hot=path.suffix == ".csv")
        self.assertTrue(any("hash mismatch" in p for p in manifest.verify(self.lake)))


class AnchorTests(LakeCase):
    def test_detect_measures_the_factor_on_the_earliest_overlap(self):
        stored = frame(periods=20)
        fresh = frame(periods=20, adj_scale=0.5).iloc[15:]
        events = anchor.detect(stored, fresh)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0][anchor.BEFORE], stored.iloc[15][schema.DATE])
        self.assertAlmostEqual(events.iloc[0][anchor.FACTOR], 0.5)

    def test_rounding_noise_is_not_an_event(self):
        stored = frame(periods=10)
        fresh = stored.copy()
        fresh[schema.ADJ_CLOSE] = fresh[schema.ADJ_CLOSE] * (1 + 1e-7)
        self.assertEqual(len(anchor.detect(stored, fresh)), 0)

    def test_whole_year_factors_compose_into_one_row(self):
        table = anchor.empty_table()
        ev = anchor.events_frame([{anchor.SYMBOL: "AAA", anchor.BEFORE: pd.Timestamp("2026-03-01"),
                                   anchor.FACTOR: 0.5}])
        table = anchor.add(table, 2024, ev, reason="split")
        ev2 = anchor.events_frame([{anchor.SYMBOL: "AAA", anchor.BEFORE: pd.Timestamp("2026-06-01"),
                                    anchor.FACTOR: 0.98}])
        table = anchor.add(table, 2024, ev2, reason="dividend")
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(float(table.iloc[0][anchor.FACTOR]), 0.49)
        applied = anchor.apply(frame(), 2024, table)
        self.assertAlmostEqual(float(applied[schema.ADJ_CLOSE].iloc[0]), 49.0)
        self.assertEqual(float(applied[schema.CLOSE].iloc[0]), 100.0)  # raw untouched

    def test_events_after_the_year_only_are_ignored_for_it(self):
        ev = anchor.events_frame([{anchor.SYMBOL: "AAA", anchor.BEFORE: pd.Timestamp("2024-01-01"),
                                   anchor.FACTOR: 0.5}])
        self.assertEqual(len(anchor.add(anchor.empty_table(), 2024, ev, reason="x")), 0)

    def test_store_applies_pending_factors_and_the_file_round_trips(self):
        writer.persist(frame(), root=self.lake, provenance=ALPACA)
        ev = anchor.events_frame([{anchor.SYMBOL: "AAA", anchor.BEFORE: pd.Timestamp("2026-01-01"),
                                   anchor.FACTOR: 0.25}])
        anchor.save(anchor.add(anchor.empty_table(), 2024, ev, reason="test"), self.lake)
        loaded = store.load_prices(["AAA"], root=self.lake)
        self.assertAlmostEqual(float(loaded[schema.ADJ_CLOSE].iloc[0]), 25.0)
        self.assertEqual(manifest.verify(self.lake), [])  # bytes unchanged
        anchor.save(anchor.empty_table(), self.lake)
        self.assertFalse(anchor.anchor_path(self.lake).exists())


class QuarantineTests(LakeCase):
    def setUp(self):
        super().setUp()
        full = frame(periods=60)
        # Canonical lake with a five-session hole.
        self.hole = full.iloc[30:35]
        writer.persist(full.drop(self.hole.index), root=self.lake, provenance=ALPACA)

    def lake_bytes(self):
        return {p.name: p.read_bytes() for p in schema.discover_year_files(self.lake)}

    def test_holding_never_touches_the_lake(self):
        before = self.lake_bytes()
        path = quarantine.hold(frame(periods=60, adj_scale=1.2), "yahoo", "pilot",
                               root=self.lake, reason="cross-check")
        self.assertEqual(self.lake_bytes(), before)
        self.assertTrue(str(path).startswith(str(quarantine.quarantine_dir(self.lake))))
        listed = quarantine.list_held(self.lake)
        self.assertEqual(list(listed["label"]), ["pilot"])

    def test_promotion_fills_only_gaps_rescaled_and_records_the_exception(self):
        # Same economics on another vendor's price scale and adjustment anchor.
        secondary = frame(periods=60, adj_scale=1.2, base=100.0 / 0.97)
        quarantine.hold(secondary, "tiingo", "gap-aaa", root=self.lake, reason="gap")
        result = quarantine.promote("tiingo", "gap-aaa", root=self.lake, reason="class C gap")
        self.assertEqual(result["promoted_symbols"], ["AAA"])
        self.assertEqual(result["rows"], 5)

        lake = store.load_prices(["AAA"], root=self.lake)
        self.assertEqual(len(lake), 60)
        filled = lake.set_index(schema.DATE).loc[self.hole[schema.DATE]]
        expected = self.hole.set_index(schema.DATE)
        # Held bars are stored at the CSV's six decimals, hence the tolerance.
        pd.testing.assert_series_equal(filled[schema.ADJ_CLOSE], expected[schema.ADJ_CLOSE],
                                       check_names=False, rtol=1e-6)
        pd.testing.assert_series_equal(filled[schema.CLOSE], expected[schema.CLOSE],
                                       check_names=False, rtol=1e-6)
        entry = manifest.partition(2024, self.lake)
        self.assertEqual(entry["provider"], "alpaca")  # owner unchanged
        self.assertEqual(entry["exceptions"][0]["provider"], "tiingo")
        self.assertEqual(entry["exceptions"][0]["symbol"], "AAA")
        self.assertTrue(Path(entry["exceptions"][0]["reconciliation"]).exists())
        self.assertEqual(manifest.verify(self.lake), [])

    def test_inconsistent_scale_is_refused_and_the_lake_is_untouched(self):
        secondary = frame(periods=60)
        # A "split" inside the held window that the canonical data does not have.
        secondary.loc[40:, schema.CLOSE] = secondary.loc[40:, schema.CLOSE] / 2
        secondary.loc[40:, schema.ADJ_CLOSE] = secondary.loc[40:, schema.ADJ_CLOSE] / 2
        quarantine.hold(secondary, "stooq", "bad", root=self.lake)
        before = self.lake_bytes()
        with self.assertRaises(quarantine.ReconciliationError):
            quarantine.promote("stooq", "bad", root=self.lake, reason="test")
        self.assertEqual(self.lake_bytes(), before)
        evidence = json.loads((quarantine.quarantine_dir(self.lake) / "stooq" /
                               "bad.reconciliation.json").read_text())
        self.assertFalse(evidence["symbols"]["AAA"]["ok"])


if __name__ == "__main__":
    unittest.main()

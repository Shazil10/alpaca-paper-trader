"""End-to-end: canonical provider -> sync / rebuild -> temporary lake.

Covers the pipeline-level guarantees of the Alpaca migration, offline, against
``FakeMarket`` (tests/fake_market.py):

* provider selection by ``MARKET_DATA_PROVIDER``; non-canonical refused; no
  implicit Yahoo fallback even when the canonical provider fails
* manifest generation on every write; one provider per year
* new-symbol full backfill, idempotent reruns, interrupted-run resume
* removed symbols retained; dead symbols go dormant, history intact
* a split between runs does not become a fake crash (anchor maintenance),
  for the hot year in place and for an immutable cold year via pending factors
* staged rebuild: stage-only leaves the lake alone, swap replaces whole years,
  pre-start years keep their provider and get a measured splice, a failed
  swap rolls everything back, and a refresh cannot mix providers

Run with: PYTHONPATH=src ./venv/bin/python -m pytest tests/test_canonical_pipeline.py -v
"""

from __future__ import annotations

import hashlib
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
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, SRC_DIR, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_pipeline import (  # noqa: E402
    anchor, fetch, manifest, providers, rebuild_prices, registry, schema, store,
    sync_prices, writer,
)
from data_pipeline.providers.alpaca import AlpacaProvider  # noqa: E402
from data_pipeline.providers.base import ProviderAuthError  # noqa: E402
from data_pipeline.providers.yahoo import YahooProvider  # noqa: E402
from fake_market import FakeMarket  # noqa: E402

THIS_YEAR = pd.Timestamp.now().year
LAST_YEAR = THIS_YEAR - 1


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def max_abs_return(frame: pd.DataFrame, symbol: str) -> float:
    series = frame[frame[schema.SYMBOL] == symbol].sort_values(schema.DATE)[schema.ADJ_CLOSE]
    return float(series.pct_change().abs().max())


class LakeCase(unittest.TestCase):
    """A temporary lake laid out like the real one (``.../prices/daily``)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.lake = base / "prices" / "daily"
        self.registry_path = base / "master_tickers.csv"
        self.universe_path = base / "universe.csv"
        self.checkpoint = base / "sync_state.json"
        self.staging = base / "staging"
        self.audits = base / "audits"
        pd.DataFrame(columns=["Symbol", "Sector", "Industry"]).to_csv(self.universe_path, index=False)

    def tearDown(self):
        self._tmp.cleanup()

    def sync(self, market, **kwargs):
        params = dict(
            lake_root=self.lake, registry_path=self.registry_path,
            universe_path=self.universe_path, checkpoint_path=self.checkpoint,
            lookback_start=market.sessions[0], batch_size=2, provider=market,
            repair=False,
        )
        params.update(kwargs)
        return sync_prices.sync(**params)


# ---------------------------------------------------------------------------
# Provider selection and the no-fallback rule
# ---------------------------------------------------------------------------

class ProviderSelectionTests(unittest.TestCase):
    def test_default_canonical_provider_is_alpaca(self):
        with mock.patch.dict(os.environ, {providers.SETTING: ""}):
            self.assertIsInstance(providers.canonical_provider(), AlpacaProvider)

    def test_setting_selects_the_provider(self):
        with mock.patch.dict(os.environ, {providers.SETTING: "ALPACA"}):
            self.assertEqual(providers.configured_name(), "alpaca")

    def test_cross_check_providers_cannot_be_made_canonical_by_configuration(self):
        for name in ("yahoo", "stooq"):
            with mock.patch.dict(os.environ, {providers.SETTING: name}):
                with self.assertRaises(providers.NonCanonicalProviderError):
                    providers.canonical_provider()

    def test_unknown_provider_is_refused(self):
        with self.assertRaises(providers.ProviderConfigurationError):
            providers.get_provider("bloomberg")

    def test_tiingo_is_disabled_unless_explicitly_enabled(self):
        with mock.patch.dict(os.environ, {"TIINGO_ENABLED": "", "TIINGO_API_KEY": "x"}):
            tiingo = providers.get_provider("tiingo")
            self.assertFalse(tiingo.canonical)
            with self.assertRaises(providers.ProviderConfigurationError):
                tiingo.fetch_batch(["AAPL"], pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"))

    def test_price_source_does_not_select_the_pipeline_provider(self):
        with mock.patch.dict(os.environ, {"PRICE_SOURCE": "yfinance", providers.SETTING: ""}):
            self.assertIsInstance(providers.canonical_provider(), AlpacaProvider)


class NoFallbackTests(LakeCase):
    def test_sync_refuses_a_non_canonical_provider_before_any_request(self):
        yahoo = YahooProvider()
        with mock.patch.object(yahoo, "fetch_batch") as fetch_call:
            with self.assertRaises(providers.NonCanonicalProviderError):
                sync_prices.sync(lake_root=self.lake, provider=yahoo,
                                 registry_path=self.registry_path,
                                 universe_path=self.universe_path,
                                 symbols_override=["AAA"])
        fetch_call.assert_not_called()

    def test_canonical_failure_does_not_fall_back_to_yahoo(self):
        class Rejecting:
            def get(self, *a, **k):
                class R:
                    status_code = 401
                    headers = {}
                    text = "unauthorized"

                    def json(self):
                        return {"message": "unauthorized"}
                return R()

        alpaca = AlpacaProvider(key="k", secret="s", session=Rejecting(), min_interval_s=0)
        yf = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"yfinance": yf}):
            with self.assertRaises(ProviderAuthError):
                sync_prices.sync(lake_root=self.lake, provider=alpaca,
                                 registry_path=self.registry_path,
                                 universe_path=self.universe_path,
                                 checkpoint_path=self.checkpoint,
                                 symbols_override=["AAA"], repair=False)
            with self.assertRaises(ProviderAuthError):
                fetch.fetch_batch(["AAA"], pd.Timestamp("2024-01-02"),
                                  pd.Timestamp("2024-01-05"), provider=alpaca)
        yf.download.assert_not_called()
        # An auth failure is not a per-symbol failure: nothing is retired.
        self.assertFalse(self.registry_path.exists())
        self.assertEqual(schema.discover_year_files(self.lake), [])


# ---------------------------------------------------------------------------
# Sync, end to end
# ---------------------------------------------------------------------------

class SyncEndToEndTests(LakeCase):
    def market(self, **kwargs):
        sessions = pd.bdate_range(f"{THIS_YEAR}-01-02", periods=60)
        return FakeMarket(sessions, ["AAA", "BBB", "CCC", "SPY"],
                          cutoff=sessions[40], **kwargs)

    def test_writes_carry_provider_provenance_in_the_manifest(self):
        market = self.market()
        self.sync(market, symbols_override=["AAA", "BBB"])
        entry = manifest.partition(THIS_YEAR, self.lake)
        self.assertEqual(entry["provider"], "fake-sip")
        self.assertEqual(entry["feed"], "sip")
        self.assertEqual(entry["row_count"], 80)
        self.assertEqual(manifest.verify(self.lake), [])
        self.assertTrue((self.lake.parent / "manifest.json").exists())

    def test_a_second_provider_cannot_write_into_an_owned_year(self):
        self.sync(self.market(), symbols_override=["AAA"])

        class Other(FakeMarket):
            name = "other-sip"

        other = Other(pd.bdate_range(f"{THIS_YEAR}-01-02", periods=60), ["AAA"],
                      cutoff=pd.bdate_range(f"{THIS_YEAR}-01-02", periods=60)[45])
        with self.assertRaises(manifest.MixedProviderError):
            self.sync(other, symbols_override=["AAA"])

    def test_new_symbol_gets_full_history_from_the_lookback_start(self):
        market = self.market()
        self.sync(market, extra_symbols=["AAA"])
        market.calls.clear()
        self.sync(market, extra_symbols=["AAA", "CCC"])
        starts = {s: c["start"] for c in market.calls for s in c["symbols"]}
        self.assertEqual(starts["CCC"], market.sessions[0])
        self.assertGreater(starts["AAA"], market.sessions[0])
        coverage = store.coverage(root=self.lake).set_index("symbol")
        self.assertEqual(coverage.loc["CCC", "first_date"], market.sessions[0])

    def test_rerun_with_nothing_new_is_idempotent(self):
        market = self.market()
        self.sync(market, extra_symbols=["AAA", "BBB"])
        path = schema.resolve_year_path(THIS_YEAR, self.lake)
        before = (file_hash(path), manifest.partition(THIS_YEAR, self.lake)["content_hash"])
        self.sync(market, extra_symbols=["AAA", "BBB"])
        after = (file_hash(path), manifest.partition(THIS_YEAR, self.lake)["content_hash"])
        self.assertEqual(before, after)

    def test_interrupted_run_resumes_from_its_checkpoint(self):
        market = self.market()
        market.raise_on_call = 2  # first batch lands, the second blows up
        with self.assertRaises(RuntimeError):
            self.sync(market, extra_symbols=["AAA", "BBB", "CCC", "SPY"])
        self.assertTrue(self.checkpoint.exists())
        done = set(json.loads(self.checkpoint.read_text())["done"])
        self.assertTrue(done)

        market.raise_on_call = None
        market.calls.clear()
        self.sync(market, extra_symbols=["AAA", "BBB", "CCC", "SPY"])
        requested = {s for c in market.calls for s in c["symbols"]}
        self.assertFalse(done & requested, "resumed run re-fetched finished symbols")
        self.assertEqual(set(store.load_prices(root=self.lake)[schema.SYMBOL]),
                         {"AAA", "BBB", "CCC", "SPY"})
        self.assertFalse(self.checkpoint.exists())

    def test_removed_member_keeps_its_history_and_dead_names_go_dormant(self):
        sessions = pd.bdate_range(f"{THIS_YEAR}-01-02", periods=80)
        market = FakeMarket(sessions, ["AAA", "BBB"], cutoff=sessions[20],
                            listed={"BBB": (None, str(sessions[25].date()))})
        self.sync(market, extra_symbols=["AAA", "BBB"], protected_symbols=["AAA", "BBB"])
        history = len(store.load_prices(["BBB"], root=self.lake))

        # BBB leaves the index (no longer current or protected) and then stops
        # trading. Its history stays; after 30 quiet days it is not requested.
        market.cutoff = sessions[79]
        self.sync(market, extra_symbols=["AAA", "BBB"], protected_symbols=["AAA"])
        self.assertGreater(len(store.load_prices(["BBB"], root=self.lake)), history)
        market.calls.clear()
        summary = self.sync(market, extra_symbols=["AAA", "BBB"], protected_symbols=["AAA"])
        requested = {s for c in market.calls for s in c["symbols"]}
        self.assertNotIn("BBB", requested)
        self.assertEqual(summary["dormant"], 1)
        self.assertEqual(len(store.load_prices(["BBB"], root=self.lake)), 26)

    def test_a_dead_current_member_is_lagging_not_dormant(self):
        sessions = pd.bdate_range(f"{THIS_YEAR}-01-02", periods=80)
        market = FakeMarket(sessions, ["AAA", "BBB"], cutoff=sessions[79],
                            listed={"BBB": (None, str(sessions[10].date()))})
        self.sync(market, extra_symbols=["AAA", "BBB"])
        market.calls.clear()
        self.sync(market, extra_symbols=["AAA", "BBB"], protected_symbols=["BBB"])
        requested = {s for c in market.calls for s in c["symbols"]}
        self.assertIn("BBB", requested)
        # ...in its own group: it must not drag AAA's window back.
        aaa_start = min(c["start"] for c in market.calls if "AAA" in c["symbols"])
        self.assertGreater(aaa_start, sessions[60])


class AnchorMaintenanceTests(LakeCase):
    def test_split_between_runs_is_not_a_fake_crash_in_the_hot_year(self):
        sessions = pd.bdate_range(f"{THIS_YEAR}-01-02", periods=60)
        market = FakeMarket(sessions, ["NVDA"], cutoff=sessions[30])
        self.sync(market, extra_symbols=["NVDA"])

        market.split("NVDA", str(sessions[33].date()), 10.0)
        market.cutoff = sessions[50]
        self.sync(market, extra_symbols=["NVDA"])

        frame = store.load_prices(["NVDA"], root=self.lake)
        self.assertLess(max_abs_return(frame, "NVDA"), 0.01)
        # The raw print does fall tenfold on the split: that is what raw means.
        raw = frame.set_index(schema.DATE)[schema.CLOSE]
        self.assertAlmostEqual(raw[sessions[33]] / raw[sessions[32]], 0.1001, places=3)
        self.assertFalse(anchor.anchor_path(self.lake).exists())

    def test_split_reanchors_an_immutable_cold_year_without_rewriting_it(self):
        sessions = pd.bdate_range(f"{LAST_YEAR}-11-03", f"{THIS_YEAR}-03-31")
        market = FakeMarket(sessions, ["NVDA", "AAA"], cutoff=pd.Timestamp(f"{THIS_YEAR}-02-02"))
        self.sync(market, extra_symbols=["NVDA", "AAA"])
        cold = schema.cold_year_path(LAST_YEAR, self.lake)
        self.assertTrue(cold.exists())
        cold_bytes = file_hash(cold)

        market.split("NVDA", f"{THIS_YEAR}-02-10", 4.0)
        market.cutoff = pd.Timestamp(f"{THIS_YEAR}-03-02")
        self.sync(market, extra_symbols=["NVDA", "AAA"])

        self.assertEqual(file_hash(cold), cold_bytes, "cold partition was rewritten")
        pending = anchor.load(self.lake)
        self.assertEqual(list(pending[anchor.SYMBOL]), ["NVDA"])
        # Measured from the hot CSV's six-decimal rows, so exact to ~1e-8.
        self.assertAlmostEqual(float(pending[anchor.FACTOR].iloc[0]), 0.25, places=6)

        frame = store.load_prices(root=self.lake)
        self.assertLess(max_abs_return(frame, "NVDA"), 0.01)
        self.assertLess(max_abs_return(frame, "AAA"), 0.01)
        # Readers see exactly what a full re-download would give them.
        fresh, _ = market.fetch_batch(["NVDA"], sessions[0], market.cutoff)
        merged = frame[frame[schema.SYMBOL] == "NVDA"].merge(fresh, on=[schema.DATE, schema.SYMBOL])
        ratio = merged["adj_close_x"] / merged["adj_close_y"]
        self.assertLess(float((ratio - 1).abs().max()), 1e-6)
        self.assertEqual(manifest.verify(self.lake), [])

    def test_rewriting_a_cold_year_folds_its_pending_factors(self):
        sessions = pd.bdate_range(f"{LAST_YEAR}-11-03", f"{THIS_YEAR}-03-31")
        market = FakeMarket(sessions, ["NVDA"], cutoff=pd.Timestamp(f"{THIS_YEAR}-02-02"))
        self.sync(market, extra_symbols=["NVDA"])
        market.dividend("NVDA", f"{THIS_YEAR}-02-10", 0.98)
        market.cutoff = pd.Timestamp(f"{THIS_YEAR}-03-02")
        self.sync(market, extra_symbols=["NVDA"])
        self.assertEqual(len(anchor.load(self.lake)), 1)
        seen = store.load_prices(["NVDA"], root=self.lake)

        # Touch the cold year (an overlap write): its factor moves into the bytes.
        cold_rows = schema.read_frame(schema.cold_year_path(LAST_YEAR, self.lake)).tail(1)
        folded = anchor.apply(cold_rows, LAST_YEAR, anchor.load(self.lake))
        writer.persist(folded, root=self.lake, provenance=market.provenance())
        self.assertEqual(len(anchor.load(self.lake)), 0)
        after = store.load_prices(["NVDA"], root=self.lake)
        pd.testing.assert_series_equal(seen[schema.ADJ_CLOSE], after[schema.ADJ_CLOSE],
                                       check_exact=False, rtol=1e-9)


# ---------------------------------------------------------------------------
# Staged rebuild
# ---------------------------------------------------------------------------

class RebuildTests(LakeCase):
    """Yahoo-era lake -> canonical rebuild of the later years."""

    def setUp(self):
        super().setUp()
        self.sessions = pd.bdate_range(f"{LAST_YEAR - 1}-10-01", f"{THIS_YEAR}-02-27")
        # The "Yahoo" lake: same economics, a different adj_close anchor.
        yahoo_like = FakeMarket(self.sessions, ["SPY", "AAA", "BBB"], scale=1.1)
        bars, _ = yahoo_like.fetch_batch(["SPY", "AAA", "BBB"], self.sessions[0],
                                         self.sessions[-1] + pd.Timedelta(days=1))
        # Legacy year: ETFs only, as in the real 2005-2015 partitions.
        legacy = bars[(bars[schema.DATE].dt.year == LAST_YEAR - 1) & (bars[schema.SYMBOL] == "SPY")]
        rest = bars[bars[schema.DATE].dt.year >= LAST_YEAR]
        yahoo = YahooProvider().provenance()
        writer.persist(pd.concat([legacy, rest]), root=self.lake, provenance=yahoo)
        self.market = FakeMarket(self.sessions, ["SPY", "AAA", "BBB"],
                                 cutoff=self.sessions[-1] + pd.Timedelta(days=1))

    def rebuild(self, **kwargs):
        params = dict(
            start=pd.Timestamp(f"{LAST_YEAR}-01-01"), lake_root=self.lake,
            registry_path=self.registry_path, universe_path=self.universe_path,
            staging_root=self.staging, audit_dir=self.audits, batch_size=2,
            provider=self.market,
        )
        params.update(kwargs)
        return rebuild_prices.rebuild(**params)

    def lake_hashes(self):
        return {p.name: file_hash(p) for p in schema.discover_year_files(self.lake)}

    def test_stage_only_leaves_the_lake_untouched_and_reports(self):
        before = self.lake_hashes()
        manifest_before = manifest.load(self.lake)
        report = self.rebuild(stage_only=True)
        self.assertEqual(report["errors"], [])
        self.assertFalse(report["swapped"])
        self.assertEqual(self.lake_hashes(), before)
        self.assertEqual(manifest.load(self.lake), manifest_before)
        self.assertTrue(Path(report["report_path"]).exists())
        self.assertIn(str(LAST_YEAR), report["comparison"]["years"])
        self.assertEqual(report["comparison"]["years"][str(LAST_YEAR)]["symbols_lost"], [])

    def test_swap_replaces_whole_years_and_splices_the_legacy_seam(self):
        staged = self.rebuild(stage_only=True)
        report = self.rebuild(resume=staged["run_id"])
        self.assertTrue(report["swapped"], report["errors"])

        by_year = manifest.providers_by_year(self.lake)
        self.assertEqual(by_year[LAST_YEAR - 1], "yahoo")   # preserved legacy year
        self.assertEqual(by_year[LAST_YEAR], "fake-sip")
        self.assertEqual(by_year[THIS_YEAR], "fake-sip")
        self.assertEqual(manifest.verify(self.lake), [])

        seam = report["seam"]["SPY"]
        self.assertEqual(seam["status"], "splice")
        self.assertAlmostEqual(seam["factor"], 1 / 1.1, places=6)
        pending = anchor.load(self.lake)
        self.assertEqual(set(pending[anchor.YEAR]), {LAST_YEAR - 1})

        frame = store.load_prices(["SPY"], root=self.lake)
        self.assertLess(max_abs_return(frame, "SPY"), 0.01)

    def test_resumed_run_does_not_refetch_staged_batches(self):
        self.market.raise_on_call = 2
        with self.assertRaises(RuntimeError):
            self.rebuild(stage_only=True)
        landed = set(self.market.calls[0]["symbols"])
        self.market.raise_on_call = None
        first = len(self.market.calls)
        report = self.rebuild(stage_only=True)
        refetched = {s for c in self.market.calls[first:] for s in c["symbols"]}
        self.assertFalse(landed & refetched, "a staged batch was fetched again")
        self.assertEqual(report["errors"], [])

    def test_failed_swap_restores_lake_manifest_and_factors(self):
        before = self.lake_hashes()
        manifest_before = manifest.load(self.lake)
        with mock.patch.object(manifest, "verify", return_value=[f"{LAST_YEAR}: forced failure"]):
            with self.assertRaises(RuntimeError):
                self.rebuild()
        self.assertEqual(self.lake_hashes(), before)
        self.assertEqual(manifest.load(self.lake), manifest_before)
        self.assertFalse(anchor.anchor_path(self.lake).exists())

    def test_losing_too_many_symbols_blocks_the_swap(self):
        self.market.symbols = ["SPY", "AAA"]  # BBB unavailable from the new provider
        before = self.lake_hashes()
        report = self.rebuild()
        self.assertFalse(report["swapped"])
        self.assertTrue(any("would lose" in e for e in report["errors"]))
        self.assertEqual(self.lake_hashes(), before)

    def test_symbol_refresh_cannot_mix_providers_in_a_year(self):
        with self.assertRaises(manifest.MixedProviderError):
            self.rebuild(symbols_override=["AAA"])

    def test_mid_year_start_cannot_take_over_another_providers_year(self):
        with self.assertRaises(manifest.MixedProviderError):
            self.rebuild(start=pd.Timestamp(f"{LAST_YEAR}-06-01"))


if __name__ == "__main__":
    unittest.main()

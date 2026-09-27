"""Alpaca SIP provider: request shape, pagination, retries, errors, the join.

No network. ``FakeAlpaca`` stands in for ``requests.Session`` and serves bars
from an in-memory dataset exactly the way the v2 bars endpoint does: sorted by
symbol then time, ``limit`` counted across symbols, ``next_page_token`` until
exhausted, daily bars stamped at midnight New York time, ``end`` inclusive.

Run with: PYTHONPATH=src ./venv/bin/python -m pytest tests/test_alpaca_provider.py -v
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest import mock

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_pipeline import aliases, schema  # noqa: E402
from data_pipeline.providers import alpaca  # noqa: E402
from data_pipeline.providers.alpaca import AlpacaProvider  # noqa: E402
from data_pipeline.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderConfigurationError,
    ProviderError,
    ProviderPermissionError,
)

KEY = "PKTESTKEY0123456789"
SECRET = "sEcReTvAlUe-should-never-appear-anywhere"

#: Monday 2024-06-10 16:00 ET, during a normal session.
MID_SESSION_UTC = pd.Timestamp("2024-06-10 20:00", tz="UTC")
#: Monday 2024-06-10 20:30 ET, after the extended session.
AFTER_CLOSE_UTC = pd.Timestamp("2024-06-11 00:30", tz="UTC")


def ny_midnight_utc(day: str) -> str:
    stamp = pd.Timestamp(day).tz_localize("America/New_York").tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeResponse:
    def __init__(self, status: int, body: Optional[dict] = None, headers: Optional[dict] = None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = str(self._body)

    def json(self):
        return self._body


class FakeAlpaca:
    """A minimal /v2/stocks/bars and /v1/corporate-actions server.

    ``data[adjustment][alpaca_symbol] = [(date, o, h, l, c, v), ...]``.
    ``script`` is a list of responses served before the dataset (for retry and
    error tests); ``reject`` names symbols that make a request fail with 400.
    """

    def __init__(self, data: Dict[str, Dict[str, List[Tuple]]], *, page_size: int = 10_000,
                 script: Optional[List[FakeResponse]] = None, reject=(),
                 corporate_actions: Optional[dict] = None):
        self.data = data
        self.page_size = page_size
        self.script = list(script or [])
        self.reject = set(reject)
        self.corporate_actions = corporate_actions or {}
        self.calls: List[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append({"url": url, "params": params, "headers": dict(headers or {})})
        if self.script:
            return self.script.pop(0)
        if url.endswith(alpaca.CORPORATE_ACTIONS_PATH):
            return FakeResponse(200, {"corporate_actions": self.corporate_actions,
                                      "next_page_token": None})

        symbols = params["symbols"].split(",")
        if self.reject & set(symbols):
            return FakeResponse(400, {"message": f"invalid symbol: {sorted(self.reject & set(symbols))[0]}"})

        start = pd.Timestamp(params["start"])
        end = pd.Timestamp(params["end"])
        series = self.data.get(params["adjustment"], {})
        flat = []
        for sym in sorted(symbols):
            for (day, o, h, l, c, v) in series.get(sym, []):
                t = pd.Timestamp(ny_midnight_utc(day))
                if start <= t <= end:
                    flat.append((sym, {"t": ny_midnight_utc(day), "o": o, "h": h, "l": l, "c": c, "v": v}))

        offset = int(params.get("page_token") or 0)
        page = flat[offset: offset + self.page_size]
        bars: Dict[str, list] = {}
        for sym, bar in page:
            bars.setdefault(sym, []).append(bar)
        nxt = offset + self.page_size
        token = str(nxt) if nxt < len(flat) else None
        return FakeResponse(200, {"bars": bars, "next_page_token": token})


def dataset(days, symbols=("AAPL",), *, adj_factor=0.5, raw_base=100.0):
    """Raw and all-adjusted series for ``symbols`` over ``days``."""
    raw: Dict[str, List[Tuple]] = {}
    adj: Dict[str, List[Tuple]] = {}
    for i, sym in enumerate(symbols):
        raw[sym] = []
        adj[sym] = []
        for j, day in enumerate(days):
            c = raw_base + i * 10 + j
            raw[sym].append((day, c - 1, c + 1, c - 2, c, 1_000 + j))
            a = c * adj_factor
            adj[sym].append((day, a - 1, a + 1, a - 2, a, 2_000 + j))
    return {"raw": raw, "all": adj}


def provider(fake: FakeAlpaca, *, now=MID_SESSION_UTC, **kwargs) -> AlpacaProvider:
    return AlpacaProvider(
        key=KEY, secret=SECRET, session=fake, min_interval_s=0,
        sleep=kwargs.pop("sleep", lambda s: None), now=lambda: now, **kwargs,
    )


DAYS = ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07"]


class RequestShapeTests(unittest.TestCase):
    def test_every_request_pins_the_sip_feed_and_daily_timeframe(self):
        fake = FakeAlpaca(dataset(DAYS))
        provider(fake).fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual({c["params"]["feed"] for c in fake.calls}, {"sip"})
        self.assertEqual({c["params"]["timeframe"] for c in fake.calls}, {"1Day"})

    def test_raw_and_all_adjustments_are_both_requested(self):
        fake = FakeAlpaca(dataset(DAYS))
        provider(fake).fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual(sorted(c["params"]["adjustment"] for c in fake.calls), ["all", "raw"])

    def test_share_class_symbols_use_dot_form_on_the_wire_and_hyphen_in_the_lake(self):
        fake = FakeAlpaca(dataset(DAYS, symbols=("BRK.B",)))
        bars, failed = provider(fake).fetch_batch(
            ["BRK-B"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08")
        )
        self.assertEqual({c["params"]["symbols"] for c in fake.calls}, {"BRK.B"})
        self.assertEqual(set(bars[schema.SYMBOL]), {"BRK-B"})
        self.assertEqual(failed, [])

    def test_symbol_round_trip(self):
        self.assertEqual(alpaca.to_alpaca_symbol("brk-b"), "BRK.B")
        self.assertEqual(alpaca.from_alpaca_symbol("BRK.B"), "BRK-B")
        self.assertEqual(alpaca.to_alpaca_symbol("AAPL"), "AAPL")

    def test_asof_is_passed_through_when_set(self):
        fake = FakeAlpaca(dataset(DAYS))
        provider(fake, asof="-").fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual({c["params"].get("asof") for c in fake.calls}, {"-"})

    def test_request_end_is_exclusive_of_the_end_date(self):
        fake = FakeAlpaca(dataset(DAYS))
        bars, _ = provider(fake, now=AFTER_CLOSE_UTC).fetch_batch(
            ["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-06")
        )
        self.assertEqual(bars[schema.DATE].max(), pd.Timestamp("2024-06-05"))


class PaginationTests(unittest.TestCase):
    def test_pages_are_followed_until_the_token_runs_out(self):
        fake = FakeAlpaca(dataset(DAYS, symbols=("AAPL", "MSFT")), page_size=3)
        prov = provider(fake)
        bars, failed = prov.fetch_batch(
            ["AAPL", "MSFT"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08")
        )
        self.assertEqual(len(bars), 10)
        self.assertEqual(failed, [])
        # 10 bars at 3 per page is 4 pages, for each of the two adjustments.
        self.assertEqual(prov.last_report.pages, 8)
        tokens = [c["params"].get("page_token") for c in fake.calls]
        self.assertIn("3", tokens)
        self.assertIn("9", tokens)

    def test_a_repeated_token_raises_instead_of_looping(self):
        stuck = FakeResponse(200, {"bars": {}, "next_page_token": "same"})
        fake = FakeAlpaca(dataset(DAYS), script=[stuck, stuck])
        with self.assertRaises(ProviderError):
            provider(fake).fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))


class RetryTests(unittest.TestCase):
    def test_429_is_retried_honouring_retry_after(self):
        sleeps: List[float] = []
        fake = FakeAlpaca(dataset(DAYS), script=[FakeResponse(429, {"message": "slow down"},
                                                              {"Retry-After": "2"})])
        prov = provider(fake, sleep=sleeps.append)
        bars, _ = prov.fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual(len(bars), 5)
        self.assertIn(2.0, sleeps)
        self.assertEqual(prov.last_report.retries, 1)

    def test_backoff_is_exponential_without_a_header(self):
        sleeps: List[float] = []
        script = [FakeResponse(429), FakeResponse(503), FakeResponse(502)]
        fake = FakeAlpaca(dataset(DAYS), script=script)
        provider(fake, sleep=sleeps.append).fetch_batch(
            ["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08")
        )
        self.assertEqual(sleeps[:3], [1.0, 2.0, 4.0])

    def test_persistent_429_gives_up_with_an_error(self):
        fake = FakeAlpaca(dataset(DAYS), script=[FakeResponse(429)] * 10)
        with self.assertRaises(ProviderError):
            provider(fake, max_retries=3).fetch_batch(
                ["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08")
            )
        self.assertEqual(len(fake.calls), 4)  # first try + 3 retries


class AuthErrorTests(unittest.TestCase):
    def test_401_raises_auth_error_without_retrying(self):
        fake = FakeAlpaca(dataset(DAYS), script=[FakeResponse(401, {"message": "forbidden"})])
        with self.assertRaises(ProviderAuthError):
            provider(fake).fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual(len(fake.calls), 1)

    def test_403_raises_permission_error_and_never_tries_iex(self):
        fake = FakeAlpaca(dataset(DAYS), script=[FakeResponse(403, {"message": "subscription"})])
        with self.assertRaises(ProviderPermissionError):
            provider(fake).fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("iex", {c["params"]["feed"] for c in fake.calls})

    def test_missing_credentials_are_a_configuration_error(self):
        names = alpaca.KEY_ENV_NAMES + alpaca.SECRET_ENV_NAMES
        with mock.patch.dict(os.environ, {n: "" for n in names}):
            prov = AlpacaProvider(session=FakeAlpaca(dataset(DAYS)), min_interval_s=0)
            with self.assertRaises(ProviderConfigurationError):
                prov.fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))

    def test_secrets_never_reach_logs_or_exceptions(self):
        echo = FakeResponse(401, {"message": f"bad key {KEY} with secret {SECRET}"})
        fake = FakeAlpaca(dataset(DAYS), script=[FakeResponse(503, {"message": SECRET}), echo])
        with self.assertLogs(level=logging.DEBUG) as captured:
            logging.getLogger("secret-probe").debug("probe")
            with self.assertRaises(ProviderAuthError) as ctx:
                provider(fake).fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"),
                                           pd.Timestamp("2024-06-08"))
        text = "\n".join(captured.output) + str(ctx.exception)
        self.assertNotIn(KEY, text)
        self.assertNotIn(SECRET, text)
        self.assertNotIn(SECRET, repr(provider(fake)))


class JoinTests(unittest.TestCase):
    def test_raw_ohlcv_and_adjusted_close_come_from_their_own_requests(self):
        fake = FakeAlpaca(dataset(DAYS, adj_factor=0.5))
        bars, _ = provider(fake).fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        first = bars.iloc[0]
        self.assertEqual(first[schema.CLOSE], 100.0)
        self.assertEqual(first[schema.ADJ_CLOSE], 50.0)
        self.assertEqual(first[schema.OPEN], 99.0)
        self.assertEqual(int(first[schema.VOLUME]), 1_000)  # raw volume, not adjusted

    def test_session_date_is_the_new_york_date(self):
        fake = FakeAlpaca(dataset(["2024-01-02", "2024-07-01"]))  # EST and EDT
        bars, _ = provider(fake, now=pd.Timestamp("2024-07-10 20:00", tz="UTC")).fetch_batch(["AAPL"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-02"))
        self.assertEqual(list(bars[schema.DATE]), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-07-01")])

    def test_missing_adjusted_rows_are_dropped_and_reported(self):
        data = dataset(DAYS, symbols=("AAPL", "MSFT"))
        data["all"]["AAPL"] = [row for row in data["all"]["AAPL"] if row[0] != "2024-06-05"]
        prov = provider(FakeAlpaca(data))
        bars, failed = prov.fetch_batch(["AAPL", "MSFT"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual(prov.last_report.missing_adjusted, {"AAPL": 1})
        self.assertNotIn(pd.Timestamp("2024-06-05"),
                         set(bars.loc[bars[schema.SYMBOL] == "AAPL", schema.DATE]))
        self.assertEqual(len(bars), 9)
        self.assertEqual(failed, [])
        self.assertIn("lacked adjusted close", prov.last_report.summary())

    def test_missing_raw_rows_are_dropped_and_reported(self):
        data = dataset(DAYS)
        data["raw"]["AAPL"] = data["raw"]["AAPL"][1:]
        prov = provider(FakeAlpaca(data))
        bars, _ = prov.fetch_batch(["AAPL"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual(prov.last_report.missing_raw, {"AAPL": 1})
        self.assertEqual(len(bars), 4)

    def test_symbol_with_no_bars_is_a_failure(self):
        prov = provider(FakeAlpaca(dataset(DAYS)))
        _, failed = prov.fetch_batch(["AAPL", "GONE"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08"))
        self.assertEqual(failed, ["GONE"])

    def test_one_rejected_symbol_does_not_fail_its_batch(self):
        fake = FakeAlpaca(dataset(DAYS, symbols=("AAPL", "MSFT", "NVDA")), reject={"BAD^"})
        bars, failed = provider(fake).fetch_batch(
            ["AAPL", "BAD^", "MSFT", "NVDA"], pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-08")
        )
        self.assertEqual(failed, ["BAD^"])
        self.assertEqual(set(bars[schema.SYMBOL]), {"AAPL", "MSFT", "NVDA"})


class SessionCutoffTests(unittest.TestCase):
    TODAY = "2024-06-10"

    def _data(self):
        return dataset(["2024-06-06", "2024-06-07", self.TODAY])

    def test_current_session_is_excluded_during_the_day(self):
        fake = FakeAlpaca(self._data())
        bars, _ = provider(fake, now=MID_SESSION_UTC).fetch_batch(
            ["AAPL"], pd.Timestamp("2024-06-06"), pd.Timestamp("2024-06-12")
        )
        self.assertEqual(bars[schema.DATE].max(), pd.Timestamp("2024-06-07"))

    def test_completed_session_is_included_after_the_extended_close(self):
        fake = FakeAlpaca(self._data())
        bars, _ = provider(fake, now=AFTER_CLOSE_UTC).fetch_batch(
            ["AAPL"], pd.Timestamp("2024-06-06"), pd.Timestamp("2024-06-12")
        )
        self.assertEqual(bars[schema.DATE].max(), pd.Timestamp(self.TODAY))

    def test_cutoff_is_new_york_time_not_the_runner_date(self):
        # 00:30 UTC Tuesday is 20:30 ET Monday: Monday is complete, Tuesday is not.
        prov = provider(FakeAlpaca({}), now=AFTER_CLOSE_UTC)
        self.assertEqual(prov.session_cutoff(), pd.Timestamp("2024-06-11"))
        # 23:00 UTC Monday is 19:00 ET Monday: Monday is not yet final.
        prov = provider(FakeAlpaca({}), now=pd.Timestamp("2024-06-10 23:00", tz="UTC"))
        self.assertEqual(prov.session_cutoff(), pd.Timestamp("2024-06-10"))

    def test_request_end_respects_the_recent_sip_embargo(self):
        now = pd.Timestamp("2024-06-11 00:05", tz="UTC")  # 20:05 ET
        fake = FakeAlpaca(self._data())
        provider(fake, now=now).fetch_batch(["AAPL"], pd.Timestamp("2024-06-06"), pd.Timestamp("2024-06-12"))
        ends = {pd.Timestamp(c["params"]["end"]) for c in fake.calls}
        self.assertTrue(all(end <= now - alpaca.RECENT_SIP_EMBARGO for end in ends))


class SymbolChangeTests(unittest.TestCase):
    def test_name_changes_become_an_explicit_alias_table(self):
        fake = FakeAlpaca({}, corporate_actions={"name_changes": [
            {"old_symbol": "FB", "new_symbol": "META", "process_date": "2022-06-09"},
            {"old_symbol": "BRK.A", "new_symbol": "BRK.A", "process_date": "2022-01-01"},
        ]})
        changes = provider(fake).fetch_symbol_changes(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
        self.assertEqual(len(changes), 1)  # the self-rename carries no mapping
        self.assertEqual(changes.iloc[0]["old_symbol"], "FB")
        self.assertEqual(changes.iloc[0]["new_symbol"], "META")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "symbol_aliases.csv"
            result = aliases.merge(changes, path)
            self.assertEqual(result["added"], 1)
            self.assertEqual(aliases.merge(changes, path)["added"], 0)  # idempotent
            table = aliases.load(path)
            self.assertEqual(aliases.successor("FB", table), "META")
            self.assertEqual(aliases.successor("AAPL", table), "AAPL")

    def test_alias_chains_are_followed_and_cycles_stop(self):
        table = pd.DataFrame([
            {"old_symbol": "A", "new_symbol": "B", "effective_date": "2020-01-01", "source": "t", "recorded_at": ""},
            {"old_symbol": "B", "new_symbol": "C", "effective_date": "2021-01-01", "source": "t", "recorded_at": ""},
            {"old_symbol": "X", "new_symbol": "Y", "effective_date": "2020-01-01", "source": "t", "recorded_at": ""},
            {"old_symbol": "Y", "new_symbol": "X", "effective_date": "2021-01-01", "source": "t", "recorded_at": ""},
        ])
        self.assertEqual(aliases.successor("A", table), "C")
        self.assertIn(aliases.successor("X", table), {"X", "Y"})


if __name__ == "__main__":
    unittest.main()

"""A deterministic canonical provider for end-to-end pipeline tests.

``FakeMarket`` implements the provider contract over an in-memory market in
which corporate actions happen on known dates and the vendor back-anchors
``adj_close`` to whatever it knows at fetch time -- which is exactly the
behaviour that makes incremental lakes drift and that ``anchor.py`` corrects.

Price model per symbol: a smooth "economic" path ``p_t``. A split of ratio
``r`` on ``E`` makes the raw print ``p_t / r`` from ``E`` on; once the split is
known (``E < cutoff``) every earlier ``adj_close`` is divided by ``r`` too, so
the adjusted series is continuous. A dividend factor ``f`` on ``E`` multiplies
earlier ``adj_close`` by ``f`` once known and leaves the raw print alone.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline import schema
from data_pipeline.providers.base import FetchReport, MarketDataProvider


class FakeMarket(MarketDataProvider):
    name = "fake-sip"
    feed = "sip"
    adjustment = "ohlcv=raw, adj_close=all"
    canonical = True

    def __init__(
        self,
        sessions: Sequence[pd.Timestamp],
        symbols: Sequence[str],
        *,
        cutoff: Optional[pd.Timestamp] = None,
        listed: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.sessions = pd.DatetimeIndex(sorted(pd.to_datetime(list(sessions))))
        self.symbols = list(symbols)
        self.cutoff = pd.Timestamp(cutoff) if cutoff is not None else self.sessions[-1] + pd.Timedelta(days=1)
        #: symbol -> (first session, last session) it trades; None = unbounded
        self.listed = dict(listed or {})
        #: multiplies every adj_close -- a different vendor's anchor convention
        self.scale = float(scale)
        self.splits: List[Tuple[str, pd.Timestamp, float]] = []
        self.dividends: List[Tuple[str, pd.Timestamp, float]] = []
        self.calls: List[dict] = []
        self.raise_on_call: Optional[int] = None
        self.raise_with: Exception = RuntimeError("simulated outage")

    # -- market events -------------------------------------------------------

    def split(self, symbol: str, date: str, ratio: float) -> None:
        self.splits.append((symbol, pd.Timestamp(date), float(ratio)))

    def dividend(self, symbol: str, date: str, factor: float) -> None:
        self.dividends.append((symbol, pd.Timestamp(date), float(factor)))

    # -- contract ------------------------------------------------------------

    def session_cutoff(self, now=None) -> pd.Timestamp:
        return self.cutoff

    def _bars(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        i = self.symbols.index(symbol) if symbol in self.symbols else 0
        positions = self.sessions.get_indexer(dates)
        economic = (50.0 + 10 * i) * (1.001 ** positions)
        raw = pd.Series(economic, index=dates, dtype="float64")
        # The print: after a split, the raw price is on the new share count.
        for sym, when, ratio in self.splits:
            if sym == symbol:
                raw[dates >= when] = raw[dates >= when] / ratio
        # The vendor's adjusted close: back-anchored using only the actions it
        # knows about at the cutoff.
        adj = raw.copy()
        for sym, when, ratio in self.splits:
            if sym == symbol and when < self.cutoff:
                adj[dates < when] = adj[dates < when] / ratio
        for sym, when, factor in self.dividends:
            if sym == symbol and when < self.cutoff:
                adj[dates < when] = adj[dates < when] * factor
        adj = adj * self.scale
        return pd.DataFrame({
            schema.DATE: dates,
            schema.SYMBOL: symbol,
            schema.OPEN: raw.values * 0.999,
            schema.HIGH: raw.values * 1.01,
            schema.LOW: raw.values * 0.99,
            schema.CLOSE: raw.values,
            schema.ADJ_CLOSE: adj.values,
            schema.VOLUME: 1_000_000,
        })

    def fetch_batch(self, symbols, start, end_exclusive):
        self.calls.append({"symbols": list(symbols), "start": pd.Timestamp(start),
                           "end": pd.Timestamp(end_exclusive)})
        if self.raise_on_call is not None and len(self.calls) >= self.raise_on_call:
            raise self.raise_with
        report = FetchReport(provider=self.name, requested=len(symbols))
        self.last_report = report

        end = min(pd.Timestamp(end_exclusive), self.cutoff)
        window = self.sessions[(self.sessions >= pd.Timestamp(start)) & (self.sessions < end)]
        frames = []
        for symbol in symbols:
            if symbol not in self.symbols:
                continue
            first, last = self.listed.get(symbol, (None, None))
            dates = window
            if first is not None:
                dates = dates[dates >= pd.Timestamp(first)]
            if last is not None:
                dates = dates[dates <= pd.Timestamp(last)]
            if len(dates):
                frames.append(self._bars(symbol, dates))
        bars = schema.coerce(pd.concat(frames, ignore_index=True)) if frames else schema.empty_frame()
        returned = set(bars[schema.SYMBOL]) if len(bars) else set()
        report.rows = len(bars)
        report.symbols_returned = len(returned)
        return bars, sorted(set(symbols) - returned)


def year_sessions(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)

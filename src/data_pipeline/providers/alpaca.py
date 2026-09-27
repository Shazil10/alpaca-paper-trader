"""Alpaca Market Data, SIP feed. The canonical daily-bar provider.

    GET https://data.alpaca.markets/v2/stocks/bars
        ?symbols=AAPL,BRK.B&timeframe=1Day&feed=sip&adjustment=raw|all

Why SIP and not IEX: SIP is the consolidated tape across every US exchange. IEX is
one venue carrying a few percent of volume, so its "daily bar" is a different,
thinner number under the same column names. The feed is pinned to ``sip`` and a
403 is raised as ``ProviderPermissionError`` -- there is no code path that retries
on IEX, because a lake that quietly switched would look complete and be wrong.

Two requests per batch
----------------------
The lake stores the raw print *and* an adjusted close, and Alpaca returns one or
the other per request. So each batch is fetched twice:

* ``adjustment=raw`` supplies open, high, low, close and volume, exactly as
  printed. Unlike Yahoo's ``Close``, this is genuinely unadjusted -- a pre-split
  bar carries the pre-split price.
* ``adjustment=all`` supplies ``adj_close`` (splits, dividends, spin-offs),
  back-anchored to today.

The two are joined on (date, symbol). A row present in one and absent from the
other cannot be stored honestly -- a raw close with no adjusted close would break
every signal that reads ``adj_close`` -- so it is dropped and **counted** in
``last_report.missing_adjusted`` / ``missing_raw``. Nothing is filled in.

Symbol mapping
--------------
Alpaca's ``asof`` parameter resolves a ticker to its underlying entity, so asking
for META from 2016 returns FB's history labelled META. That is left on (Alpaca's
default) because it is what makes a renamed company's history continuous. But the
mapping must not live invisibly inside fetch code, so ``fetch_symbol_changes``
exposes the rename events themselves, and the pipeline writes them to
``data/universe/symbol_aliases.csv`` where they can be audited.

Secrets
-------
Credentials travel only in request headers. Headers are never logged, no
exception message is built from them, and every message that does leave this
module passes through ``_redact`` first -- a belt for the braces, since a
``requests`` error can embed a URL and a future change could embed more.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from data_pipeline import schema
from data_pipeline.providers.base import (
    MARKET_TZ,
    FetchReport,
    MarketDataProvider,
    ProviderAuthError,
    ProviderConfigurationError,
    ProviderError,
    ProviderPermissionError,
    ProviderRequestError,
    completed_session_cutoff,
    market_now,
)
from data_pipeline.registry import normalize_symbol

logger = logging.getLogger(__name__)

BASE_URL = "https://data.alpaca.markets"
BARS_PATH = "/v2/stocks/bars"
CORPORATE_ACTIONS_PATH = "/v1/corporate-actions"

#: Pinned. See the module docstring for why there is no IEX fallback.
FEED = "sip"

#: Alpaca's per-page maximum. The limit is across all symbols in the request.
PAGE_LIMIT = 10_000

#: Symbols per request. Pagination handles volume; this only bounds URL length
#: and the blast radius of a single bad symbol.
DEFAULT_BATCH_SIZE = 100

#: Free-plan accounts may not query SIP data newer than 15 minutes. One extra
#: minute of margin keeps a run at 20:15 from tripping the 403 on clock skew.
RECENT_SIP_EMBARGO = pd.Timedelta(minutes=16)

#: Basic plan: 200 requests/minute. Pace just under it so the 429 path is the
#: exception rather than the steady state.
DEFAULT_MIN_INTERVAL_S = 60.0 / 190.0

MAX_RETRIES = 6
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 60.0

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

KEY_ENV_NAMES = ("ALPACA_KEY", "APCA_API_KEY_ID")
SECRET_ENV_NAMES = ("ALPACA_SECRET", "APCA_API_SECRET_KEY")


# ---------------------------------------------------------------------------
# Symbol form
# ---------------------------------------------------------------------------

def to_alpaca_symbol(symbol: str) -> str:
    """Lake form -> Alpaca form. ``BRK-B`` -> ``BRK.B``.

    The lake stores the Yahoo/Alpaca-trading hyphen form everywhere; the market
    data API wants the dot form for share classes. Converting at the boundary
    keeps the lake's keys stable regardless of provider.
    """
    return normalize_symbol(symbol).replace("-", ".")


def from_alpaca_symbol(symbol: str) -> str:
    """Alpaca form -> lake form. ``BRK.B`` -> ``BRK-B``."""
    return normalize_symbol(symbol)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class AlpacaProvider(MarketDataProvider):
    """Daily SIP bars, raw OHLCV joined to all-adjusted close."""

    name = "alpaca"
    feed = FEED
    adjustment = "ohlcv=raw, adj_close=all (split+dividend+spin-off)"
    canonical = True

    def __init__(
        self,
        *,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        session: Any = None,
        base_url: str = BASE_URL,
        asof: Optional[str] = None,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        max_retries: int = MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        super().__init__()
        self._key = key if key is not None else _first_env(KEY_ENV_NAMES)
        self._secret = secret if secret is not None else _first_env(SECRET_ENV_NAMES)
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._asof = asof
        self._min_interval_s = max(float(min_interval_s), 0.0)
        self._max_retries = max(int(max_retries), 0)
        self._sleep = sleep
        self._now = now
        self._last_request_at = 0.0

    # -- plumbing ----------------------------------------------------------

    @property
    def has_credentials(self) -> bool:
        return bool(self._key and self._secret)

    def _clock(self) -> Optional[datetime]:
        return self._now() if self._now is not None else None

    def session_cutoff(self, now: Optional[datetime] = None) -> pd.Timestamp:
        return completed_session_cutoff(now or self._clock())

    def _http(self):
        if self._session is None:
            import requests  # local: keep the dependency off the read path
            self._session = requests.Session()
        return self._session

    def _headers(self) -> Dict[str, str]:
        if not self.has_credentials:
            raise ProviderConfigurationError(
                "Alpaca credentials not found. Set ALPACA_KEY and ALPACA_SECRET "
                "(the same names the trading workflow uses)."
            )
        return {
            "APCA-API-KEY-ID": str(self._key),
            "APCA-API-SECRET-KEY": str(self._secret),
            "Accept": "application/json",
        }

    def _redact(self, text: object) -> str:
        """Replace any credential that appears in ``text``."""
        message = str(text)
        for value in (self._key, self._secret):
            if value:
                message = message.replace(str(value), "***")
        return message

    def _pace(self) -> None:
        """Keep under the per-minute request budget."""
        if self._min_interval_s <= 0:
            return
        wait = self._last_request_at + self._min_interval_s - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_request_at = time.monotonic()

    def _backoff(self, attempt: int, response: Any = None) -> float:
        """Seconds to wait before retry ``attempt`` (1-based).

        Honours ``Retry-After`` and Alpaca's ``X-RateLimit-Reset`` (an epoch
        second) when present, otherwise exponential from ``BACKOFF_BASE_S``.
        """
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return min(float(retry_after), BACKOFF_MAX_S)
            except (TypeError, ValueError):
                pass
        reset = headers.get("X-RateLimit-Reset") if hasattr(headers, "get") else None
        if reset:
            try:
                delta = float(reset) - time.time()
                if 0 < delta <= BACKOFF_MAX_S:
                    return delta + 0.25
            except (TypeError, ValueError):
                pass
        return min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)

    def _get(self, path: str, params: Dict[str, Any], report: Optional[FetchReport]) -> Dict:
        """One GET with pacing, retries, and error classification."""
        url = f"{self._base_url}{path}"
        headers = self._headers()

        attempt = 0
        while True:
            attempt += 1
            self._pace()
            try:
                response = self._http().get(url, params=params, headers=headers, timeout=60)
            except Exception as exc:  # connection reset, timeout, DNS
                if attempt > self._max_retries:
                    raise ProviderError(
                        f"Alpaca request failed after {attempt} attempt(s): "
                        f"{self._redact(type(exc).__name__)}: {self._redact(exc)}"
                    ) from None
                delay = self._backoff(attempt)
                logger.warning(
                    "Alpaca network error (%s), retry %d/%d in %.1fs",
                    self._redact(type(exc).__name__), attempt, self._max_retries, delay,
                )
                if report is not None:
                    report.retries += 1
                self._sleep(delay)
                continue

            status = int(getattr(response, "status_code", 0))
            if status == 200:
                try:
                    return response.json() or {}
                except ValueError:
                    raise ProviderError("Alpaca returned a non-JSON 200 response") from None

            message = self._error_message(response)

            if status == 401:
                raise ProviderAuthError(
                    "Alpaca rejected the credentials (401). Check ALPACA_KEY and "
                    f"ALPACA_SECRET. Server said: {message}"
                )
            if status == 403:
                raise ProviderPermissionError(
                    "Alpaca refused the request (403): the account's data "
                    "subscription does not permit this SIP query. Not retrying on "
                    f"IEX. Server said: {message}"
                )
            if status in RETRYABLE_STATUS and attempt <= self._max_retries:
                delay = self._backoff(attempt, response)
                logger.warning(
                    "Alpaca HTTP %d, retry %d/%d in %.1fs",
                    status, attempt, self._max_retries, delay,
                )
                if report is not None:
                    report.retries += 1
                self._sleep(delay)
                continue
            if status in RETRYABLE_STATUS:
                raise ProviderError(
                    f"Alpaca HTTP {status} persisted through {self._max_retries} "
                    f"retries: {message}"
                )
            raise ProviderRequestError(f"Alpaca HTTP {status}: {message}")

    def _error_message(self, response: Any) -> str:
        try:
            body = response.json()
            text = body.get("message") if isinstance(body, dict) else body
        except Exception:
            text = getattr(response, "text", "")
        return self._redact(str(text or "")[:300])

    # -- bars --------------------------------------------------------------

    def _window(
        self, start: pd.Timestamp, end_exclusive: pd.Timestamp
    ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """Clamp the request to completed sessions and the SIP embargo.

        Returns ``(start, end_exclusive, cutoff)``. The request ``end`` is sent
        as midnight UTC of ``end_exclusive``: Alpaca's ``end`` is inclusive and a
        daily bar is stamped at midnight New York time (04:00 or 05:00 UTC), so
        midnight UTC of a date always falls before that date's bar and after the
        previous one -- an exact exclusive bound without guessing at time zones.
        """
        cutoff = self.session_cutoff()
        start = pd.Timestamp(start).normalize()
        end_exclusive = min(pd.Timestamp(end_exclusive).normalize(), cutoff)
        return start, end_exclusive, cutoff

    def _request_end(self, end_exclusive: pd.Timestamp) -> str:
        end = pd.Timestamp(end_exclusive).tz_localize("UTC")
        embargo = market_now(self._clock()).tz_convert("UTC") - RECENT_SIP_EMBARGO
        return min(end, embargo).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fetch_adjustment(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
        adjustment: str,
        report: FetchReport,
    ) -> pd.DataFrame:
        """All pages of one adjustment for one batch, in lake symbol form."""
        params: Dict[str, Any] = {
            "symbols": ",".join(to_alpaca_symbol(s) for s in symbols),
            "timeframe": "1Day",
            "start": start.strftime("%Y-%m-%dT00:00:00Z"),
            "end": self._request_end(end_exclusive),
            "adjustment": adjustment,
            "feed": FEED,
            "limit": PAGE_LIMIT,
            "sort": "asc",
        }
        if self._asof is not None:
            params["asof"] = self._asof

        rows: List[Dict[str, Any]] = []
        seen_tokens = set()
        token: Optional[str] = None

        while True:
            if token:
                params["page_token"] = token
            else:
                params.pop("page_token", None)

            payload = self._get(BARS_PATH, params, report)
            report.pages += 1

            bars = payload.get("bars") or {}
            for alpaca_symbol, series in bars.items():
                lake_symbol = from_alpaca_symbol(alpaca_symbol)
                for bar in series or []:
                    rows.append({
                        "t": bar.get("t"),
                        schema.SYMBOL: lake_symbol,
                        schema.OPEN: bar.get("o"),
                        schema.HIGH: bar.get("h"),
                        schema.LOW: bar.get("l"),
                        schema.CLOSE: bar.get("c"),
                        schema.VOLUME: bar.get("v"),
                    })

            token = payload.get("next_page_token")
            if not token:
                break
            if token in seen_tokens:
                # A repeated token means the server is not advancing; looping
                # on it would never terminate and never error.
                raise ProviderError("Alpaca returned a repeated next_page_token")
            seen_tokens.add(token)

        if not rows:
            return pd.DataFrame(columns=[schema.DATE, schema.SYMBOL, schema.OPEN,
                                         schema.HIGH, schema.LOW, schema.CLOSE,
                                         schema.VOLUME])

        frame = pd.DataFrame(rows)
        # A daily bar is stamped at midnight New York time; its session date is
        # that timestamp's date *in New York*, not in UTC.
        stamps = pd.to_datetime(frame.pop("t"), utc=True).dt.tz_convert(MARKET_TZ)
        frame[schema.DATE] = stamps.dt.tz_localize(None).dt.normalize()
        return frame

    def _join(
        self, raw: pd.DataFrame, adjusted: pd.DataFrame, report: FetchReport
    ) -> pd.DataFrame:
        """Inner-join raw OHLCV to all-adjusted close, counting what fails."""
        key = [schema.DATE, schema.SYMBOL]
        adj = adjusted[key + [schema.CLOSE]].rename(columns={schema.CLOSE: schema.ADJ_CLOSE})

        merged = raw.merge(adj, on=key, how="outer", indicator=True)

        for side, bucket in (("left_only", report.missing_adjusted),
                             ("right_only", report.missing_raw)):
            orphans = merged[merged["_merge"] == side]
            for symbol, count in orphans.groupby(schema.SYMBOL).size().items():
                bucket[str(symbol)] = bucket.get(str(symbol), 0) + int(count)

        if report.missing_adjusted:
            logger.warning(
                "Alpaca: %d raw row(s) across %d symbol(s) had no adjusted close "
                "and were dropped: %s",
                sum(report.missing_adjusted.values()), len(report.missing_adjusted),
                ", ".join(sorted(report.missing_adjusted)[:10]),
            )
        if report.missing_raw:
            logger.warning(
                "Alpaca: %d adjusted row(s) across %d symbol(s) had no raw bar "
                "and were dropped: %s",
                sum(report.missing_raw.values()), len(report.missing_raw),
                ", ".join(sorted(report.missing_raw)[:10]),
            )

        both = merged[merged["_merge"] == "both"].drop(columns="_merge")
        return both[schema.COLUMNS]

    def _fetch_isolating(
        self,
        symbols: List[str],
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
        report: FetchReport,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Fetch a batch; on a 4xx that names no one, bisect to find the culprit.

        Alpaca rejects the *whole* request when one symbol is malformed. Without
        isolation one bad ticker in a historical membership list would fail every
        symbol that shared its batch, and after enough runs retire all of them.
        """
        try:
            raw = self._fetch_adjustment(symbols, start, end_exclusive, "raw", report)
            adjusted = self._fetch_adjustment(symbols, start, end_exclusive, "all", report)
        except ProviderRequestError as exc:
            if len(symbols) == 1:
                logger.warning("Alpaca rejected %s: %s", symbols[0], exc)
                report.notes.append(f"rejected {symbols[0]}: {exc}")
                return pd.DataFrame(columns=schema.COLUMNS), list(symbols)
            mid = len(symbols) // 2
            left, left_failed = self._fetch_isolating(symbols[:mid], start, end_exclusive, report)
            right, right_failed = self._fetch_isolating(symbols[mid:], start, end_exclusive, report)
            parts = [f for f in (left, right) if len(f)]
            joined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=schema.COLUMNS)
            return joined, left_failed + right_failed

        return self._join(raw, adjusted, report), []

    def fetch_batch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, List[str]]:
        wanted = sorted({normalize_symbol(s) for s in symbols if str(s).strip()})
        report = FetchReport(provider=self.name, requested=len(wanted))
        self.last_report = report
        if not wanted:
            return schema.empty_frame(), []

        start, end_exclusive, cutoff = self._window(start, end_exclusive)
        if start >= end_exclusive:
            report.notes.append(
                f"empty window {start.date()}..{end_exclusive.date()} "
                f"(cutoff {cutoff.date()})"
            )
            return schema.empty_frame(), []

        joined, rejected = self._fetch_isolating(wanted, start, end_exclusive, report)
        bars = schema.coerce(joined) if len(joined) else schema.empty_frame()

        # Belt and braces: the window already stops at the cutoff, but the
        # contract is enforced on the data, not on the request.
        late = bars[bars[schema.DATE] >= cutoff]
        if len(late):
            report.dropped_after_cutoff = len(late)
            bars = bars[bars[schema.DATE] < cutoff]

        bars = bars.dropna(subset=[schema.CLOSE, schema.ADJ_CLOSE])
        bars = schema.coerce(bars)

        returned = set(bars[schema.SYMBOL]) if len(bars) else set()
        failed = sorted((set(wanted) - returned) | set(rejected))

        report.rows = len(bars)
        report.symbols_returned = len(returned)
        logger.info(report.summary())
        return bars, failed

    # -- symbol changes and corporate actions ------------------------------

    def fetch_corporate_actions(
        self,
        *,
        types: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Raw corporate-action records grouped by Alpaca's plural type key."""
        params: Dict[str, Any] = {
            "types": ",".join(types),
            "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "limit": 1000,
        }
        if symbols:
            params["symbols"] = ",".join(to_alpaca_symbol(s) for s in symbols)

        out: Dict[str, List[Dict[str, Any]]] = {}
        seen = set()
        token: Optional[str] = None
        while True:
            if token:
                params["page_token"] = token
            payload = self._get(CORPORATE_ACTIONS_PATH, params, None)
            for kind, records in (payload.get("corporate_actions") or {}).items():
                out.setdefault(kind, []).extend(records or [])
            token = payload.get("next_page_token")
            if not token:
                break
            if token in seen:
                raise ProviderError("Alpaca returned a repeated next_page_token")
            seen.add(token)
        return out

    def fetch_symbol_changes(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Rename events as an explicit alias table.

        Columns: old_symbol, new_symbol, effective_date, source. Written by the
        pipeline to ``data/universe/symbol_aliases.csv`` so the entity mapping that
        ``asof`` applies inside the bars request is visible and auditable.
        """
        actions = self.fetch_corporate_actions(
            types=["name_change"], start=start, end=end, symbols=symbols
        )
        rows = []
        for record in actions.get("name_changes", []):
            old = record.get("old_symbol")
            new = record.get("new_symbol")
            when = record.get("process_date") or record.get("ex_date")
            if not old or not new or from_alpaca_symbol(old) == from_alpaca_symbol(new):
                continue  # a "rename" to itself carries no mapping
            rows.append({
                "old_symbol": from_alpaca_symbol(old),
                "new_symbol": from_alpaca_symbol(new),
                "effective_date": pd.Timestamp(when).normalize() if when else pd.NaT,
                "source": "alpaca:corporate-actions:name_change",
            })
        columns = ["old_symbol", "new_symbol", "effective_date", "source"]
        frame = pd.DataFrame(rows, columns=columns)
        if len(frame):
            frame = frame.drop_duplicates().sort_values(["effective_date", "old_symbol"])
        return frame.reset_index(drop=True)


def _first_env(names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None

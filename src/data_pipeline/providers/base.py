"""The contract every market-data provider implements.

One method does the work::

    fetch_batch(symbols, start, end_exclusive) -> (bars, failures)

``bars`` is a frame already in the lake schema (``schema.COLUMNS``), ``failures``
the requested symbols that came back with nothing. Everything above this line --
``sync_prices``, the rebuild, the pilot -- is written against the contract and not
against any vendor, which is what makes it possible to change the canonical source
without touching the code that decides *what* to fetch.

Three properties live here rather than in each provider, because getting any of
them wrong in one provider corrupts the lake no matter how good the others are:

**Which providers may write the canonical lake.** ``canonical`` is a class
attribute, not a setting. A cross-check source cannot be promoted by an
environment variable; it has to go through quarantine and reconciliation. That is
the mechanism behind "Yahoo cannot silently enter the canonical lake".

**When a session is complete.** Two cutoffs, deliberately different:

* ``strict_cutoff`` excludes the current market date outright. Right for a source
  that may hand back a partial intraday bar and that is only ever called during
  the trading day -- Yahoo at 09:30 ET.
* ``completed_session_cutoff`` includes today's session once the extended-hours
  session has ended at 20:00 ET. Right for the after-close workflow, whose entire
  purpose is to capture the session that just closed so the next morning's
  trading run reads completed data. Before 20:00 ET it behaves exactly like the
  strict cutoff, so a daytime run still cannot store a partial bar.

Both are computed in America/New_York, not in the machine's local zone. A GitHub
runner is on UTC; at 21:00 ET on a Monday its local date is already Tuesday, and a
local-date cutoff would silently change what counts as "today".

**Errors that must stop the run.** Most fetch problems are recoverable -- a
symbol with no data this window is a failure to record, not a reason to abort.
Bad credentials and a missing data entitlement are not recoverable, and treating
them as per-symbol failures would slowly retire every ticker in the registry.
They raise instead.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

MARKET_TZ = "America/New_York"

#: Extended-hours trading ends at 20:00 ET. After that the day's consolidated bar
#: no longer changes, so the session counts as complete.
SESSION_FINAL_TIME_ET = time(20, 0)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProviderError(RuntimeError):
    """Base class for provider failures that should stop the run."""


class ProviderConfigurationError(ProviderError):
    """The provider is not usable here: disabled, or credentials absent."""


class ProviderAuthError(ProviderError):
    """Credentials were rejected (HTTP 401)."""


class ProviderPermissionError(ProviderError):
    """Credentials are valid but lack the entitlement requested (HTTP 403).

    For Alpaca this is almost always the SIP feed. Deliberately *not* handled by
    retrying on IEX: IEX is a single exchange carrying a few percent of volume,
    and a lake that quietly switched to it would record thin, unrepresentative
    bars under the same column names as consolidated ones.
    """


class ProviderRequestError(ProviderError):
    """A non-retryable request error (4xx other than 401/403/429)."""


class NonCanonicalProviderError(ProviderError):
    """A cross-check provider was asked to write the canonical lake."""


# ---------------------------------------------------------------------------
# Session cutoffs
# ---------------------------------------------------------------------------

def market_now(now: Optional[datetime] = None) -> pd.Timestamp:
    """The current instant in America/New_York.

    ``now`` may be naive (taken as UTC) or aware. Injectable so tests can pin the
    clock on either side of the close without sleeping or patching time.
    """
    if now is None:
        stamp = pd.Timestamp.now(tz="UTC")
    else:
        stamp = pd.Timestamp(now)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(MARKET_TZ)


def market_date(now: Optional[datetime] = None) -> pd.Timestamp:
    """Today's date in New York, as a naive midnight timestamp."""
    return pd.Timestamp(market_now(now).date())


def strict_cutoff(now: Optional[datetime] = None) -> pd.Timestamp:
    """Bars dated on or after this are never stored. Excludes today outright."""
    return market_date(now)


def completed_session_cutoff(now: Optional[datetime] = None) -> pd.Timestamp:
    """Like ``strict_cutoff``, but admits today once the session has ended.

    Before 20:00 ET: today excluded, identical to the strict cutoff.
    From 20:00 ET: today included, because its bar is final.
    """
    local = market_now(now)
    today = pd.Timestamp(local.date())
    if local.time() >= SESSION_FINAL_TIME_ET:
        return today + pd.Timedelta(days=1)
    return today


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class FetchReport:
    """What one ``fetch_batch`` call did, beyond the rows it returned.

    Kept on the provider as ``last_report`` rather than returned, so the
    two-value contract stays stable while a caller that cares -- the pilot, the
    rebuild -- can still see pages, retries, and which rows could not be joined.
    """
    provider: str
    requested: int = 0
    symbols_returned: int = 0
    rows: int = 0
    pages: int = 0
    retries: int = 0
    dropped_after_cutoff: int = 0
    #: symbol -> raw rows that had no adjusted counterpart (dropped)
    missing_adjusted: Dict[str, int] = field(default_factory=dict)
    #: symbol -> adjusted rows that had no raw counterpart (dropped)
    missing_raw: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"{self.provider}: {self.rows} row(s) for "
            f"{self.symbols_returned}/{self.requested} symbol(s)",
            f"{self.pages} page(s)",
        ]
        if self.retries:
            parts.append(f"{self.retries} retr{'y' if self.retries == 1 else 'ies'}")
        if self.missing_adjusted:
            parts.append(
                f"{sum(self.missing_adjusted.values())} row(s) lacked adjusted close"
            )
        if self.missing_raw:
            parts.append(f"{sum(self.missing_raw.values())} row(s) lacked raw OHLC")
        if self.dropped_after_cutoff:
            parts.append(f"{self.dropped_after_cutoff} incomplete-session row(s) dropped")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

class MarketDataProvider(abc.ABC):
    """A source of daily bars in the lake schema."""

    #: Stable identifier recorded in the partition manifest.
    name: str = ""

    #: Vendor-specific feed, recorded in the manifest ("sip", "iex", "" ...).
    feed: str = ""

    #: How the columns were produced, recorded in the manifest.
    adjustment: str = ""

    #: Only canonical providers may write the canonical lake. A class attribute
    #: on purpose -- see the module docstring.
    canonical: bool = False

    def __init__(self) -> None:
        self.last_report: Optional[FetchReport] = None

    @abc.abstractmethod
    def fetch_batch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Return ``(bars, failed_symbols)`` for ``[start, end_exclusive)``."""

    def session_cutoff(self, now: Optional[datetime] = None) -> pd.Timestamp:
        """First date whose bar must not be stored. Strict unless overridden."""
        return strict_cutoff(now)

    def provenance(self) -> Dict[str, str]:
        """What the manifest records about rows this provider wrote."""
        return {
            "provider": self.name,
            "feed": self.feed,
            "adjustment": self.adjustment,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} canonical={self.canonical}>"

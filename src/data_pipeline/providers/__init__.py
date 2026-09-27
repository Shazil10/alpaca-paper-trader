"""Market-data providers behind one contract.

    fetch_batch(symbols, start, end_exclusive) -> (bars, failures)

| provider | role                      | writes canonical lake |
|----------|---------------------------|-----------------------|
| alpaca   | canonical (SIP feed)      | yes                   |
| yahoo    | cross-check               | no -- quarantine only |
| stooq    | cross-check               | no -- quarantine only |
| tiingo   | optional gap repair, off  | no -- quarantine only |

Which provider the pipeline uses is ``MARKET_DATA_PROVIDER`` (default ``alpaca``).
That is a separate setting from ``PRICE_SOURCE`` on purpose: ``PRICE_SOURCE``
controls what *live strategies read* (the lake or a live download), while this
controls what *the pipeline writes into the lake*. Conflating them would mean
flipping a strategy's read path could change the lake's provenance.

There is no fallback chain. ``canonical_provider()`` returns exactly the configured
provider or raises; it never substitutes another when the first fails, because a
silent substitution is exactly how a second provider's rows end up interleaved
with the first's under the same keys.
"""

from __future__ import annotations

import os
from typing import Dict, Type

from data_pipeline.providers.base import (
    MarketDataProvider,
    NonCanonicalProviderError,
    ProviderAuthError,
    ProviderConfigurationError,
    ProviderError,
    ProviderPermissionError,
    ProviderRequestError,
    completed_session_cutoff,
    market_date,
    strict_cutoff,
)

SETTING = "MARKET_DATA_PROVIDER"
DEFAULT_PROVIDER = "alpaca"


def _registry() -> Dict[str, Type[MarketDataProvider]]:
    # Imported lazily so reading this module never imports yfinance or requests.
    from data_pipeline.providers.alpaca import AlpacaProvider
    from data_pipeline.providers.stooq import StooqProvider
    from data_pipeline.providers.tiingo import TiingoProvider
    from data_pipeline.providers.yahoo import YahooProvider

    return {
        "alpaca": AlpacaProvider,
        "yahoo": YahooProvider,
        "stooq": StooqProvider,
        "tiingo": TiingoProvider,
    }


def available() -> Dict[str, bool]:
    """{provider name: canonical?}"""
    return {name: cls.canonical for name, cls in _registry().items()}


def configured_name() -> str:
    """The provider named by ``MARKET_DATA_PROVIDER``, lower-cased."""
    return (os.getenv(SETTING) or DEFAULT_PROVIDER).strip().lower()


def get_provider(name: str = None, **kwargs) -> MarketDataProvider:
    """Instantiate a provider by name. Unknown names raise; nothing is guessed."""
    key = (name or configured_name()).strip().lower()
    registry = _registry()
    if key not in registry:
        raise ProviderConfigurationError(
            f"unknown market-data provider {key!r}; expected one of "
            f"{sorted(registry)}"
        )
    return registry[key](**kwargs)


def canonical_provider(name: str = None, **kwargs) -> MarketDataProvider:
    """The provider allowed to write the canonical lake, or an error.

    Setting ``MARKET_DATA_PROVIDER=yahoo`` does not make Yahoo canonical. It makes
    the canonical sync refuse to run, which is the point: promoting a
    cross-check source is a reconciliation decision, not a configuration one.
    """
    provider = get_provider(name, **kwargs)
    if not provider.canonical:
        raise NonCanonicalProviderError(
            f"{provider.name!r} is a cross-check provider and cannot write the "
            f"canonical lake. Its output goes to data/prices/quarantine/ and is "
            f"promoted only after reconciliation. Set {SETTING}=alpaca for the "
            f"canonical sync."
        )
    return provider


__all__ = [
    "SETTING",
    "DEFAULT_PROVIDER",
    "MarketDataProvider",
    "NonCanonicalProviderError",
    "ProviderAuthError",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderPermissionError",
    "ProviderRequestError",
    "available",
    "canonical_provider",
    "completed_session_cutoff",
    "configured_name",
    "get_provider",
    "market_date",
    "strict_cutoff",
]

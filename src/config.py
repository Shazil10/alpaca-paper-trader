"""config.py

Project configuration.

This file keeps (1) Alpaca credentials loading and (2) the list of strategies + budgets.
Other modules import from here so the rest of the codebase stays simple.
"""

import os

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv


load_dotenv()  # Loads ALPACA_KEY / ALPACA_SECRET from a local .env


# Strategy import path -> dollars to allocate for that strategy.
# NOTE: For rotation strategies (ranked_asset_alloc), the budget is a lifetime
# cap — sell proceeds recycle into it.  The strategy internally sizes
# positions from its own BASE_BUDGET × DAF leverage (1–2×).
STRATEGY_ALLOCATIONS: dict[str, float] = {
    "strategies.momentum.clenow_trend": 10_000,
    "strategies.ranks.ranked_asset_alloc": 15_000,
    # "strategies.mean_reversion.rsi_dip": 10_000,
}


def get_client() -> TradingClient:
    """Create and return an authenticated Alpaca `TradingClient` (paper trading)."""
    api_key = os.getenv("ALPACA_KEY")
    secret_key = os.getenv("ALPACA_SECRET")
    if not api_key or not secret_key:
        raise ValueError("API keys not found. Set ALPACA_KEY and ALPACA_SECRET.")

    return TradingClient(api_key, secret_key, paper=True)
"""Transaction cost model.

Default: spread_bps/2 + slippage_bps one-way, $0 commission (Alpaca equities).
Stress scenario: $0.005/share commission.
"""
from __future__ import annotations

import math
from backtest.types import CostConfig, OrderSide


def compute_fill_price(
    raw_price: float,
    side: OrderSide,
    config: CostConfig,
) -> float:
    """Apply spread and slippage to get the effective fill price.
    
    BUY: pay more (price * (1 + cost_bps / 10000))
    SELL: receive less (price * (1 - cost_bps / 10000))
    """
    total_bps = config.total_one_way_bps
    factor = total_bps / 10_000

    if side == OrderSide.BUY:
        return raw_price * (1 + factor)
    else:
        return raw_price * (1 - factor)


def compute_commission(
    shares: float,
    config: CostConfig,
) -> float:
    """Compute commission for a trade."""
    raw = abs(shares) * config.commission_per_share
    return max(raw, config.min_commission)


def max_shares_for_volume(
    daily_volume: float,
    config: CostConfig,
) -> float:
    """Maximum shares that can be traded as fraction of daily volume."""
    if daily_volume <= 0 or config.participation_rate <= 0:
        return float("inf")
    return daily_volume * config.participation_rate


def estimate_market_impact_bps(
    notional: float,
    daily_dollar_volume: float,
) -> float:
    """Estimate additional market impact beyond the flat slippage.
    
    Square-root model: impact ~ k * sqrt(notional / ADV).
    For v1, this returns 0 — the flat slippage_bps covers it.
    Placeholder for future enhancement.
    """
    return 0.0

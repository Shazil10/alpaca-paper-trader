"""Risk manager — position and exposure limits.

Applied between the strategy's target weights and the order generation step.
Clips weights that would violate constraints, logs adjustments.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from backtest.types import RiskConfig, PortfolioSnapshot

logger = logging.getLogger(__name__)


def apply_risk_limits(
    target_weights: Dict[str, float],
    snapshot: PortfolioSnapshot,
    config: RiskConfig,
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Clip target weights to respect risk limits.
    
    Returns adjusted weights. Logs every adjustment.
    """
    adjusted = dict(target_weights)
    sector_map = sector_map or {}

    # 1. Cap individual position size
    for sym, w in list(adjusted.items()):
        if abs(w) > config.max_position_pct:
            old = w
            adjusted[sym] = config.max_position_pct if w > 0 else -config.max_position_pct
            logger.debug(
                "Risk: clipped %s from %.1f%% to %.1f%%",
                sym, old * 100, adjusted[sym] * 100,
            )

    # 2. Cap number of positions
    if len(adjusted) > config.max_positions:
        sorted_by_weight = sorted(adjusted.items(), key=lambda x: abs(x[1]), reverse=True)
        keep = dict(sorted_by_weight[:config.max_positions])
        dropped = len(adjusted) - len(keep)
        logger.debug("Risk: dropped %d positions exceeding max_positions=%d", dropped, config.max_positions)
        adjusted = keep

    # 3. Enforce cash reserve.
    # Measured against max_leverage, not against 1.0. Hard-coding 1.0 here made
    # max_leverage unreachable: a deliberately levered sleeve (the ranked
    # sleeve's DAF 2x) was scaled back to gross 1.0 before the leverage check
    # ever ran, so the backtest quietly simulated an unlevered strategy. At the
    # default max_leverage=1.0 this expression is identical to the old one.
    total_weight = sum(adjusted.values())
    max_invested = config.max_leverage - config.cash_reserve_pct
    if total_weight > max_invested:
        scale = max_invested / total_weight
        adjusted = {sym: w * scale for sym, w in adjusted.items()}
        logger.debug(
            "Risk: scaled weights by %.3f to enforce %.0f%% cash reserve",
            scale, config.cash_reserve_pct * 100,
        )

    # 4. Sector exposure limits
    if sector_map and config.max_sector_pct < 1.0:
        sector_exposure: Dict[str, float] = {}
        for sym, w in adjusted.items():
            sector = sector_map.get(sym, "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(w)

        for sector, exposure in sector_exposure.items():
            if exposure > config.max_sector_pct:
                scale = config.max_sector_pct / exposure
                for sym in list(adjusted.keys()):
                    if sector_map.get(sym, "Unknown") == sector:
                        adjusted[sym] *= scale
                logger.debug(
                    "Risk: scaled sector %s by %.3f (was %.1f%%, cap %.1f%%)",
                    sector, scale, exposure * 100, config.max_sector_pct * 100,
                )

    # 5. Leverage limit
    gross = sum(abs(w) for w in adjusted.values())
    if gross > config.max_leverage:
        scale = config.max_leverage / gross
        adjusted = {sym: w * scale for sym, w in adjusted.items()}
        logger.debug("Risk: scaled gross exposure from %.2f to %.2f", gross, config.max_leverage)

    return adjusted

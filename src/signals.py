"""signals.py

Small shared data structures used between strategies and the trade runner.

Idea:
- Strategies should *describe* what they want to do (signals).
- The trade runner (`trade.py`) decides sizing/risk limits and sends orders.

This keeps strategies broker-agnostic and makes it easy to add long/short,
mean reversion, etc. later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Signal:
    """A strategy's intent.

    Notes:
    - For now we focus on BUY signals only (per your request).
    - Later we can add SELL/SHORT and target weights without changing `trade.py` much.
    """

    symbol: str
    side: Side = Side.BUY
    reason: str | None = None
    notional: float | None = None
    client_order_id: str | None = None
    strategy_id: str | None = None

    def normalized_symbol(self) -> str:
        return str(self.symbol).strip().upper()

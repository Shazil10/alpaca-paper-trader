"""trade_models.py

Shared data structures and pure helpers for the trade runner.

Contains:
- Signal / Side: broker-agnostic signal objects returned by strategies
- committed_dollars_from_orders: lifetime budget accounting
- safe_float: tiny numeric helper
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol


# ---------------------------------------------------------------------------
# Signal data structures
# ---------------------------------------------------------------------------

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Signal:
    """A strategy's intent.

    Notes:
    - For now we focus on BUY signals only.
    - Later we can add SELL/SHORT and target weights without changing trade.py.
    """

    symbol: str
    side: Side = Side.BUY
    reason: str | None = None
    notional: float | None = None
    client_order_id: str | None = None
    strategy_id: str | None = None

    def normalized_symbol(self) -> str:
        return str(self.symbol).strip().upper()


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------

class OrderLike(Protocol):
    id: object
    client_order_id: object
    side: object
    status: object
    filled_qty: object
    filled_avg_price: object
    notional: object
    qty: object
    limit_price: object


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _enum_str(value: object) -> str:
    """Extract the raw string from an Alpaca SDK enum (or plain string).

    The alpaca-py SDK wraps side/status in Enum subclasses whose ``str()``
    gives ``'OrderSide.BUY'`` instead of ``'BUY'``.  ``.value`` gives ``'buy'``.
    """
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    s = str(value)
    if "." in s:
        return s.rsplit(".", 1)[-1]
    return s


def committed_dollars_from_orders(orders: Iterable[OrderLike], *, strategy_id: str) -> float:
    """Sum dollars committed by a strategy based on tagged orders.

    Attribution rule: ``client_order_id`` must start with ``f"{strategy_id}:"``.
    Counting rule ("strict"):
      - include BUY orders that are filled or still open/pending
      - exclude canceled/rejected/expired

    Dollar estimation:
      1) if any fills exist, count filled_qty * filled_avg_price
      2) else if order has notional, count it
      3) else if order has qty and limit_price, count qty * limit_price
      4) else count 0
    """

    prefix = f"{strategy_id}:"
    committed = 0.0

    for o in orders:
        cid = str(getattr(o, "client_order_id", "") or "")
        if not cid.startswith(prefix):
            continue

        side = _enum_str(getattr(o, "side", "")).lower()
        if side != "buy":
            continue

        status = _enum_str(getattr(o, "status", "")).lower()
        if status in {"canceled", "rejected", "expired"}:
            continue

        filled_qty = safe_float(getattr(o, "filled_qty", 0.0))
        filled_avg_price = safe_float(getattr(o, "filled_avg_price", 0.0))
        if filled_qty > 0 and filled_avg_price > 0:
            committed += filled_qty * filled_avg_price
            continue

        notional = safe_float(getattr(o, "notional", 0.0))
        if notional > 0:
            committed += notional
            continue

        qty = safe_float(getattr(o, "qty", 0.0))
        limit_price = safe_float(getattr(o, "limit_price", 0.0))
        if qty > 0 and limit_price > 0:
            committed += qty * limit_price

    return committed

"""trade_models.py

Shared data structures and pure helpers for the trade runner.

Contains:
- Signal / Side: broker-agnostic signal objects returned by strategies
- committed_dollars_from_orders: lifetime budget accounting
- safe_float: tiny numeric helper
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Protocol

logger = logging.getLogger(__name__)


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
    @property
    def id(self) -> object: ...
    @property
    def client_order_id(self) -> object: ...
    @property
    def side(self) -> object: ...
    @property
    def status(self) -> object: ...
    @property
    def filled_qty(self) -> object: ...
    @property
    def filled_avg_price(self) -> object: ...
    @property
    def notional(self) -> object: ...
    @property
    def qty(self) -> object: ...
    @property
    def limit_price(self) -> object: ...


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
        return str(getattr(value, "value"))
    s = str(value)
    if "." in s:
        return s.rsplit(".", 1)[-1]
    return s


def fetch_all_orders(client, *, status: str = "all", page_size: int = 500) -> List[object]:
    """Fetch all orders by paginating through the Alpaca API.

    Uses the ``after`` parameter (oldest order ID of previous page) to walk
    forward through pages until a page returns fewer than ``page_size`` results.
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
    except ImportError:
        GetOrdersRequest = None  # type: ignore[assignment]
        QueryOrderStatus = None  # type: ignore[assignment]

    all_orders: List[object] = []
    after_id = None

    for _ in range(50):  # safety cap: 50 pages x 500 = 25k orders max
        kwargs = {"status": status, "limit": page_size, "direction": "asc"}
        if after_id:
            kwargs["after"] = after_id

        page: list = []
        if GetOrdersRequest is not None and QueryOrderStatus is not None:
            try:
                req_kwargs = {
                    "status": QueryOrderStatus(status),
                    "limit": page_size,
                    "nested": True,
                }
                if after_id:
                    req_kwargs["after"] = after_id
                req = GetOrdersRequest(**req_kwargs)
                page = list(client.get_orders(req))
            except Exception:
                pass

        if not page:
            try:
                page = list(client.get_orders(**kwargs, nested=True))
            except TypeError:
                try:
                    page = list(client.get_orders(limit=page_size))
                except TypeError:
                    page = list(client.get_orders())

        if not page:
            break

        all_orders.extend(page)

        if len(page) < page_size:
            break

        last_id = getattr(page[-1], "id", None)
        if last_id is None or str(last_id) == str(after_id):
            break
        after_id = str(last_id)

    logger.info("fetch_all_orders: retrieved %d orders across pages", len(all_orders))
    return all_orders


def committed_dollars_from_orders(orders: Iterable[OrderLike], *, strategy_id: str) -> float:
    """Return net dollars currently deployed by a strategy.

    Attribution rule: ``client_order_id`` must start with ``f"{strategy_id}:"``.

    BUY counting ("strict"):
      - include filled or still-open/pending BUY orders
      - exclude canceled/rejected/expired

    SELL counting:
      - subtract filled SELL proceeds so that capital is recycled.
      - e.g. if the budget is $10,000, all is deployed, then $2,000 is sold,
        the net committed becomes $8,000, freeing $2,000 for new buys.

    Dollar estimation for BUYs:
      1) if any fills exist, use filled_qty * filled_avg_price
      2) else if order has notional, use it
      3) else if order has qty and limit_price, use qty * limit_price
      4) else 0

    Result is clamped to 0 (can never go negative).
    """

    prefix = f"{strategy_id}:"
    committed = 0.0

    for o in orders:
        cid = str(getattr(o, "client_order_id", "") or "")
        if not cid.startswith(prefix):
            continue

        side = _enum_str(getattr(o, "side", "")).lower()
        status = _enum_str(getattr(o, "status", "")).lower()

        if side == "buy":
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

        elif side == "sell":
            # Subtract filled proceeds — capital is freed up for redeployment.
            filled_qty = safe_float(getattr(o, "filled_qty", 0.0))
            filled_avg_price = safe_float(getattr(o, "filled_avg_price", 0.0))
            if filled_qty > 0 and filled_avg_price > 0:
                committed -= filled_qty * filled_avg_price

    return max(committed, 0.0)

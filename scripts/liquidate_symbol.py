#!/usr/bin/env python3
"""Cancel open orders for a symbol and sell the full paper position.

Usage:
    PYTHONPATH=src python scripts/liquidate_symbol.py NVO
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alpaca.common.exceptions import APIError

import config
import orders

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPEN_STATUSES = {"new", "accepted", "pending_new", "held", "partially_filled", "pending_cancel"}


def _status_str(order) -> str:
    status = getattr(order, "status", "")
    return str(getattr(status, "value", status)).lower()


def cancel_open_orders(client, symbol: str) -> int:
    symbol = symbol.upper()
    cancelled = 0
    for order in client.get_orders():
        sym = str(getattr(order, "symbol", "")).upper()
        if sym != symbol:
            continue
        status = _status_str(order)
        if status not in OPEN_STATUSES:
            continue
        if status == "pending_cancel":
            logger.info("Order %s already pending cancel (%s)", getattr(order, "id", ""), sym)
            continue
        order_id = getattr(order, "id", None)
        if not order_id:
            continue
        try:
            client.cancel_order_by_id(order_id)
            logger.info("Cancelled open order %s (%s %s)", order_id, sym, status)
            cancelled += 1
        except APIError as exc:
            logger.warning("Could not cancel order %s: %s", order_id, exc)
    return cancelled


def cancel_all_open_orders(client) -> int:
    """Cancel every open order on the account (used to clear stuck holds)."""
    try:
        client.delete("/orders")
        logger.info("Requested cancel of all open orders")
        return 1
    except APIError as exc:
        logger.warning("Could not cancel all orders: %s", exc)
        return 0


def close_position(client, symbol: str) -> bool:
    symbol = symbol.upper()
    try:
        order = client.close_position(symbol)
        logger.info("Closed position via Alpaca close_position: %s", getattr(order, "id", order))
        return True
    except APIError as exc:
        logger.warning("close_position failed for %s: %s", symbol, exc)
        return False


def sell_full_position(client, symbol: str) -> bool:
    symbol = symbol.upper()
    for pos in client.get_all_positions():
        if str(pos.symbol).upper() != symbol:
            continue
        qty = float(pos.qty)
        if qty <= 0:
            logger.info("No long position to sell for %s (qty=%.4f)", symbol, qty)
            return False
        try:
            order = orders.sell_market_qty(client, symbol, qty)
            logger.info(
                "Submitted SELL %s qty=%.4f order_id=%s",
                symbol,
                qty,
                getattr(order, "id", order),
            )
            return True
        except APIError as exc:
            logger.error("SELL failed for %s: %s", symbol, exc)
            return False
    logger.info("No open position for %s", symbol)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Cancel open orders and sell full position.")
    parser.add_argument("symbol", help="Ticker symbol, e.g. NVO")
    parser.add_argument(
        "--force-cancel-all",
        action="store_true",
        help="Cancel all open account orders before closing (clears stuck held shares)",
    )
    args = parser.parse_args()
    symbol = args.symbol.strip().upper()

    client = config.get_client()
    n_cancelled = cancel_open_orders(client, symbol)
    if args.force_cancel_all:
        cancel_all_open_orders(client)
    sold = close_position(client, symbol) or sell_full_position(client, symbol)
    logger.info("Done: cancelled=%d sold=%s", n_cancelled, sold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

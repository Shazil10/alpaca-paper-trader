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
        if _status_str(order) not in OPEN_STATUSES:
            continue
        order_id = getattr(order, "id", None)
        if not order_id:
            continue
        client.cancel_order_by_id(order_id)
        logger.info("Cancelled open order %s (%s %s)", order_id, sym, _status_str(order))
        cancelled += 1
    return cancelled


def sell_full_position(client, symbol: str) -> bool:
    symbol = symbol.upper()
    for pos in client.get_all_positions():
        if str(pos.symbol).upper() != symbol:
            continue
        qty = float(pos.qty)
        if qty <= 0:
            logger.info("No long position to sell for %s (qty=%.4f)", symbol, qty)
            return False
        side = "long"
        if qty < 0:
            logger.warning("%s is a short position (qty=%.4f); this script only sells long qty", symbol, qty)
            return False
        order = orders.sell_market_qty(client, symbol, qty)
        logger.info("Submitted SELL %s qty=%.4f order_id=%s", symbol, qty, getattr(order, "id", order))
        return True
    logger.info("No open position for %s", symbol)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Cancel open orders and sell full position.")
    parser.add_argument("symbol", help="Ticker symbol, e.g. NVO")
    args = parser.parse_args()
    symbol = args.symbol.strip().upper()

    client = config.get_client()
    n_cancelled = cancel_open_orders(client, symbol)
    sold = sell_full_position(client, symbol)
    logger.info("Done: cancelled=%d sold=%s", n_cancelled, sold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

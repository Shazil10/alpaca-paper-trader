"""orders.py

Small wrappers around Alpaca order placement.

`trade.py` (and any other module) calls these helpers instead of building Alpaca request
objects everywhere. That keeps order logic in one place.
"""

import logging

from typing import Optional

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


logger = logging.getLogger(__name__)

def buy_market_notional(client, symbol, dollars, *, client_order_id: Optional[str] = None):
    """Buy using a dollar amount (notional), e.g. buy $500 of AAPL."""
    if dollars is None or float(dollars) <= 0:
        raise ValueError("dollars must be > 0")

    symbol = str(symbol).strip().upper()
    dollars = round(float(dollars), 2)  # Alpaca rejects more than 2 decimal places
    order_data = MarketOrderRequest(
        symbol=symbol,
        notional=dollars,  # 'notional' means dollar amount
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )

    logger.debug("Submitting BUY notional order: %s $%.2f", symbol, float(dollars))
    return client.submit_order(order_data)


def buy_market_qty(client, symbol, qty=1):
    """Buy using a share quantity, e.g. buy 10 shares of AAPL."""
    if qty is None or float(qty) <= 0:
        raise ValueError("qty must be > 0")

    symbol = str(symbol).strip().upper()
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )

    logger.debug("Submitting BUY qty order: %s qty=%s", symbol, qty)
    return client.submit_order(order_data)


def sell_market_notional(client, symbol, dollars):
    """Sell using a dollar amount (notional), e.g. sell $500 of AAPL."""
    if dollars is None or float(dollars) <= 0:
        raise ValueError("dollars must be > 0")

    symbol = str(symbol).strip().upper()
    dollars = round(float(dollars), 2)  # Alpaca rejects more than 2 decimal places
    order_data = MarketOrderRequest(
        symbol=symbol,
        notional=dollars,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )

    logger.debug("Submitting SELL notional order: %s $%.2f", symbol, float(dollars))
    return client.submit_order(order_data)


def sell_market_qty(client, symbol, qty=1, *, client_order_id: Optional[str] = None):
    """Sell using a share quantity, e.g. sell 10 shares of AAPL."""
    if qty is None or float(qty) <= 0:
        raise ValueError("qty must be > 0")

    symbol = str(symbol).strip().upper()
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )

    logger.debug("Submitting SELL qty order: %s qty=%s", symbol, qty)
    return client.submit_order(order_data)
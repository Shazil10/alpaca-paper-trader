from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def buy_market(client, symbol, qty=1):
    """
    Places a simple market buy order for a specific symbol.
    """
    print(f"Submitting Buy Order: {qty} {symbol}...")
    
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    
    return client.submit_order(order_data)

def sell_market(client, symbol, qty=1):
    """
    Places a simple market sell order.
    """
    print(f"Submitting Sell Order: {qty} {symbol}...")

    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    
    return client.submit_order(order_data)
import config
import orders

def main():
    # 1. Connect to the Alpaca API
    try:
        client = config.get_client()
    except ValueError as error:
        print(f"Connection Error: {error}")
        return

    # 2. Check current portfolio
    # We want to avoid buying SPY if we already own it.
    positions = client.get_all_positions()
    owns_spy = False

    for position in positions:
        if position.symbol == 'SPY':
            owns_spy = True
            break

    # 3. Execute Trade Logic
    if owns_spy:
        print("We already own SPY. No new order placed.")
    else:
        print("SPY not found in portfolio. Placing buy order for 1 share...")
        try:
            order = orders.buy_market(client, "SPY", 1)
            print(f"Order submitted successfully. ID: {order.id}")
        except Exception as error:
            print(f"Order failed: {error}")

if __name__ == "__main__":
    main()
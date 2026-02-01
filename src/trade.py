import os
from dotenv import load_dotenv  # secret keys coming from .env file
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def main():
    # 1. Load the secret .env file directly
    load_dotenv() 

    # 2. Get our keys
    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET")

    # Debug: Check if keys were found (We print "Found" or "Not Found" to stay safe)
    if not api_key:
        print("Error: ALPACA_KEY is missing from .env file")
        return
    if not api_secret:
        print("Error: ALPACA_SECRET is missing from .env file")
        return
    
    print("Keys found successfully!")

    # 3. Connect to Alpaca
    print("Starting Trading Bot")
    try:
        trading_client = TradingClient(api_key, api_secret, paper=True)
        account = trading_client.get_account()
        
        print(f"Connected to Alpaca!")
        print(f"Cash Balance: ${account.cash}")
        print(f"Portfolio Value: ${account.portfolio_value}")
        
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    main()
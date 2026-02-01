import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load environment variables once when this module is imported
load_dotenv()

API_KEY = os.getenv("ALPACA_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET")

def get_client():
    """
    Returns an authenticated Alpaca TradingClient.
    Raises an error if keys are missing.
    """
    if not API_KEY or not SECRET_KEY:
        # Stop execution immediately if we don't have keys
        raise ValueError("API keys not found. Check .env or GitHub Secrets.")
    
    return TradingClient(API_KEY, SECRET_KEY, paper=True)
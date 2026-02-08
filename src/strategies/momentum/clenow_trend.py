import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
import os

def get_momentum_score(prices):
    """Calculates exponential regression slope * r2."""
    log_prices = np.log(prices)
    x = np.arange(len(log_prices))
    slope, _, r_value, _, _ = linregress(x, log_prices)
    
    annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
    return annualized_slope * (r_value ** 2)

def get_top_picks(universe_file="universe.csv", top_n=20):
    if not os.path.exists(universe_file):
        # Fallback for different execution contexts
        universe_file = os.path.join(os.path.dirname(__file__), "../../../universe.csv")

    try:
        tickers = pd.read_csv(universe_file)['Symbol'].tolist()
    except FileNotFoundError:
        print("Universe file not found.")
        return []

    print(f"Analyzing {len(tickers)} stocks...")
    scores = []
    
    # Download in bulk
    chunk_size = 300
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i+chunk_size]
        try:
            data = yf.download(batch, period="6mo", group_by='ticker', progress=False, threads=True)
            
            for symbol in batch:
                try:
                    hist = data[symbol]['Close'].dropna()
                    if len(hist) < 90: continue

                    score = get_momentum_score(hist.iloc[-90:].values)
                    scores.append((symbol, score))
                except KeyError:
                    pass
        except Exception as e:
            print(f"Chunk error: {e}")

    # Sort and return top N symbols
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_n]]

if __name__ == "__main__":
    picks = get_top_picks()
    print(f"Top Picks: {picks}")
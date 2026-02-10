import os

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

from signals import Signal, Side

"""
STRATEGY EXPLANATION: CLENOW MOMENTUM
-------------------------------------
This strategy identifies stocks with the smoothest, most consistent upward trends.
It is based on Andreas Clenow's 'Stocks on the Move'.

Logic:
1. Calculate the Regression Slope (How fast is it going up?)
2. Calculate the R-Squared (How smooth is the line?)
3. Score = Slope * (R^2)
"""

# Keep defaults in one place so notebooks/scripts can reuse them.
DEFAULT_TOP_N = 20

def get_momentum_score(prices):
    """Return Clenow momentum score for a 1D array of prices."""
    log_prices = np.log(prices)
    x = np.arange(len(log_prices))
    
    slope, _, r_value, _, _ = linregress(x, log_prices)
    
    # Annualize slope (252 trading days)
    annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
    
    return annualized_slope * (r_value ** 2)

def get_top_picks(universe_file="universe.csv", top_n: int = DEFAULT_TOP_N, *, return_scores: bool = False):
    """Return the top N symbols based on Clenow momentum.

    Args:
        universe_file: CSV containing a `Symbol` column.
        top_n: Number of symbols to return.
        return_scores: If True, return (symbols, scores_df) for notebook analysis.
    """
    if not os.path.exists(universe_file):
        # Fallback for different execution contexts
        universe_file = os.path.join(os.path.dirname(__file__), "../../../universe.csv")

    if not os.path.exists(universe_file):
        # Caller can decide how to handle this.
        return ([], pd.DataFrame(columns=["Symbol", "Score"])) if return_scores else []

    tickers = pd.read_csv(universe_file)['Symbol'].tolist()
    # This can take a while on large universes.

    scores = []
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
                    continue
        except Exception:
            # yfinance can rate-limit or fail intermittently; skip this batch.
            continue

    # Rank by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    scores_df = pd.DataFrame(scores, columns=["Symbol", "Score"])
    top_picks = [s[0] for s in scores[:top_n]]
    return (top_picks, scores_df) if return_scores else top_picks


def generate_signals(
    *,
    universe_file: str = "universe.csv",
    top_n: int = DEFAULT_TOP_N,
    budget: float | None = None,
    strategy_id: str = "strategies.momentum.clenow_trend",
) -> list[Signal]:
    """Main entry point used by `trade.py`.

    This strategy owns *selection* and *sizing*.

    Args:
        universe_file: CSV containing a `Symbol` column.
        top_n: Number of symbols to consider.
        budget: Max dollars this strategy is allowed to deploy *today*.
            If omitted/<=0, signals are returned with no sizing.
        strategy_id: Stable id used for attribution/tagging/ledgers.
    """
    symbols = get_top_picks(universe_file=universe_file, top_n=top_n)
    if not symbols:
        return []

    notional_each: float | None = None
    try:
        if budget is not None and float(budget) > 0:
            notional_each = float(budget) / len(symbols)
    except Exception:
        notional_each = None

    return [
        Signal(
            symbol=s,
            side=Side.BUY,
            reason="clenow_momentum",
            notional=notional_each,
            strategy_id=strategy_id,
        )
        for s in symbols
    ]
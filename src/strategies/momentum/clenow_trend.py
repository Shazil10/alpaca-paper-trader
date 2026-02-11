"""
STRATEGY: CLENOW MOMENTUM (Enhanced)
======================================
Based on Andreas Clenow's 'Stocks on the Move', enhanced with:

1. **Regime Filter** (Gatekeeper)
   - Download SPY, IJH, IJR (large/mid/small cap ETFs).
   - If at least 2 of 3 have Price > 200-day SMA → risk-on → proceed.
   - Otherwise → "go to beach" → no new buys (may still exit).

2. **Entry Filter**
   - Only score stocks where current Price > 200-day SMA.

3. **Ranking**
   - Clenow Score = annualized_slope × R² on 90-day log-prices.
   - Top 20 by score.

4. **Sizing** (inverse-volatility weighted)
   - Weight ∝ 1 / σ_20d for each pick.
   - Normalize so weights sum to 1, then clamp to [2%, 10%].
   - After clamping, renormalize to sum to 1.

5. **Exit Logic** (SELL signals)
   - Sell any currently-held position where:
     a) Rank dropped beyond 30, OR
     b) Price < 100-day SMA.
   - Requires current portfolio positions to be passed in.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

from trade_models import Signal, Side


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOP_N = 20
EXIT_RANK_CUTOFF = 30          # sell if rank drops beyond this
REGIME_ETFS = ("SPY", "IJH", "IJR")
REGIME_SMA_PERIOD = 200
ENTRY_SMA_PERIOD = 200
EXIT_SMA_PERIOD = 100
LOOKBACK_DAYS = 90
VOL_LOOKBACK = 20              # 20-day realized vol for sizing
MIN_WEIGHT = 0.02              # 2% floor per position
MAX_WEIGHT = 0.10              # 10% cap per position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_momentum_score(prices: np.ndarray) -> float:
    """Return Clenow momentum score for a 1-D array of prices."""
    log_prices = np.log(prices)
    x = np.arange(len(log_prices))

    slope, _, r_value, _, _ = linregress(x, log_prices)

    # Annualize slope (252 trading days)
    annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100

    return annualized_slope * (r_value ** 2)


def _sma(series: pd.Series, period: int) -> float:
    """Return the simple moving average of the last *period* values."""
    if len(series) < period:
        return float("nan")
    return float(series.iloc[-period:].mean())


def _realized_vol(series: pd.Series, period: int = VOL_LOOKBACK) -> float:
    """Annualized realized volatility from daily log returns."""
    if len(series) < period + 1:
        return float("nan")
    log_ret = np.log(series.iloc[-period:] / series.iloc[-period - 1 : -1].values)
    return float(np.std(log_ret, ddof=1) * np.sqrt(252))


# ---------------------------------------------------------------------------
# 1. Regime filter
# ---------------------------------------------------------------------------

def check_regime(etfs: Tuple[str, ...] = REGIME_ETFS, sma_period: int = REGIME_SMA_PERIOD) -> bool:
    """Return True (risk-on) if ≥2 of 3 ETFs close above their 200-day SMA."""
    try:
        # Need enough history for the SMA
        data = yf.download(list(etfs), period="1y", group_by="ticker", progress=False, threads=True)
    except Exception:
        logger.warning("Regime check failed (download error). Defaulting to risk-ON.")
        return True

    above = 0
    for etf in etfs:
        try:
            close = data[etf]["Close"].dropna()
            if len(close) < sma_period:
                continue
            if float(close.iloc[-1]) > _sma(close, sma_period):
                above += 1
        except (KeyError, IndexError):
            continue

    risk_on = above >= 2
    logger.info("Regime check: %d / %d ETFs above %d-SMA → %s", above, len(etfs), sma_period, "RISK-ON" if risk_on else "RISK-OFF")
    return risk_on


# ---------------------------------------------------------------------------
# 2 + 3. Score universe (with entry filter) and rank
# ---------------------------------------------------------------------------

def score_universe(
    universe_file: str = "universe.csv",
    *,
    lookback: int = LOOKBACK_DAYS,
    entry_sma: int = ENTRY_SMA_PERIOD,
) -> pd.DataFrame:
    """Score every stock in the universe and return a DataFrame.

    Only stocks with current price > *entry_sma*-day SMA are scored.
    Returns columns: Symbol, Score, Close, Vol20 (annualized), SMA200, SMA100.
    """
    if not os.path.exists(universe_file):
        universe_file = os.path.join(os.path.dirname(__file__), "../../../universe.csv")
    if not os.path.exists(universe_file):
        return pd.DataFrame(columns=["Symbol", "Score", "Close", "Vol20", "SMA200", "SMA100"])

    tickers = pd.read_csv(universe_file)["Symbol"].tolist()

    records: List[Dict[str, object]] = []
    chunk_size = 300

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i : i + chunk_size]
        try:
            data = yf.download(batch, period="1y", group_by="ticker", progress=False, threads=True)

            for symbol in batch:
                try:
                    close = data[symbol]["Close"].dropna()
                    if len(close) < max(lookback, entry_sma):
                        continue

                    current_price = float(close.iloc[-1])
                    sma200 = _sma(close, ENTRY_SMA_PERIOD)
                    sma100 = _sma(close, EXIT_SMA_PERIOD)
                    vol20 = _realized_vol(close, VOL_LOOKBACK)

                    # Entry filter: only score if price > 200-day SMA
                    if current_price <= sma200:
                        continue

                    score = get_momentum_score(close.iloc[-lookback:].values)

                    records.append(
                        {
                            "Symbol": symbol,
                            "Score": score,
                            "Close": current_price,
                            "Vol20": vol20,
                            "SMA200": sma200,
                            "SMA100": sma100,
                        }
                    )
                except (KeyError, IndexError):
                    continue
        except Exception:
            continue

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1  # 1-based rank
    return df


def get_top_picks(
    universe_file: str = "universe.csv",
    top_n: int = DEFAULT_TOP_N,
    *,
    return_scores: bool = False,
) -> "Tuple[List[str], pd.DataFrame] | List[str]":  # type: ignore[override]
    """Convenience wrapper that returns the top N symbols (backward-compatible).

    If *return_scores* is True, returns ``(symbols, scores_df)``.
    """
    df = score_universe(universe_file=universe_file)
    if df.empty:
        return ([], df) if return_scores else []

    top = df.head(top_n)
    symbols = top["Symbol"].tolist()
    return (symbols, df) if return_scores else symbols


# ---------------------------------------------------------------------------
# 4. Inverse-volatility sizing
# ---------------------------------------------------------------------------

def _inverse_vol_weights(
    vol_series: pd.Series,
    min_w: float = MIN_WEIGHT,
    max_w: float = MAX_WEIGHT,
) -> pd.Series:
    """Compute inverse-volatility weights, clamped to [min_w, max_w].

    After clamping, weights are renormalized to sum to 1.
    """
    inv = 1.0 / vol_series.replace(0, np.nan).dropna()
    if inv.empty:
        return inv

    raw = inv / inv.sum()

    # Iterative clamp-and-renormalize (converges fast for reasonable bounds).
    for _ in range(10):
        clamped = raw.clip(lower=min_w, upper=max_w)
        total = clamped.sum()
        if total == 0:
            break
        clamped = clamped / total
        if clamped.equals(raw):
            break
        raw = clamped

    return raw


# ---------------------------------------------------------------------------
# 5. Exit logic
# ---------------------------------------------------------------------------

def _generate_exit_signals(
    scores_df: pd.DataFrame,
    held_symbols: Set[str],
    strategy_id: str,
) -> List[Signal]:
    """Return SELL signals for held positions that violate exit rules.

    Exit if:
      a) Symbol's rank > EXIT_RANK_CUTOFF (30), OR
      b) Symbol's current price < 100-day SMA.

    Stocks not found in the scored universe at all are also exited (they failed
    the entry filter, which is even worse).
    """
    signals: List[Signal] = []
    scored_symbols: Set[str] = set(scores_df["Symbol"].tolist()) if not scores_df.empty else set()

    for sym in held_symbols:
        reason: Optional[str] = None

        if sym not in scored_symbols:
            # Symbol didn't pass the 200 SMA entry filter → definitely exit.
            reason = "exit:below_200sma_filter"
        else:
            row = scores_df.loc[scores_df["Symbol"] == sym].iloc[0]
            rank = int(row["Rank"])
            price = float(row["Close"])
            sma100 = float(row["SMA100"])

            if rank > EXIT_RANK_CUTOFF and price < sma100:
                reason = f"exit:rank_{rank}_and_below_100sma"
            elif rank > EXIT_RANK_CUTOFF:
                reason = f"exit:rank_{rank}_gt_{EXIT_RANK_CUTOFF}"
            elif price < sma100:
                reason = "exit:below_100sma"

        if reason:
            signals.append(
                Signal(
                    symbol=sym,
                    side=Side.SELL,
                    reason=reason,
                    notional=None,  # runner will close entire position
                    strategy_id=strategy_id,
                )
            )
            logger.info("EXIT signal: %s (%s)", sym, reason)

    return signals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_signals(
    *,
    universe_file: str = "universe.csv",
    top_n: int = DEFAULT_TOP_N,
    budget: Optional[float] = None,
    strategy_id: str = "strategies.momentum.clenow_trend",
    held_symbols: Optional[Set[str]] = None,
) -> List[Signal]:
    """Main entry point used by ``trade.py``.

    This strategy owns *selection*, *sizing*, and *exit logic*.

    Args:
        universe_file: CSV with a ``Symbol`` column.
        top_n: Number of symbols to buy.
        budget: Max dollars this strategy may deploy today.
        strategy_id: Stable id for attribution / tagging / ledgers.
        held_symbols: Symbols currently held by this strategy (for exit logic).
    """
    if held_symbols is None:
        held_symbols = set()

    # ── Step 1: Score the full universe (entry filter is built-in) ──
    scores_df = score_universe(universe_file=universe_file)

    # ── Step 2: Generate EXIT signals (always, even in risk-off) ──
    exit_signals = _generate_exit_signals(scores_df, held_symbols, strategy_id)

    # ── Step 3: Regime gate (only affects new BUY signals) ──
    if not check_regime():
        logger.info("Regime is RISK-OFF → no new buys. Returning %d exit signals only.", len(exit_signals))
        return exit_signals

    # ── Step 4: Pick top N and size with inverse-vol ──
    if scores_df.empty:
        return exit_signals

    top = scores_df.head(top_n).copy()
    if top.empty:
        return exit_signals

    # Inverse-vol weights
    weights = _inverse_vol_weights(top.set_index("Symbol")["Vol20"])
    if weights.empty:
        return exit_signals

    # ── Step 5: Build BUY signals ──
    buy_signals: List[Signal] = []
    for sym in weights.index:
        w = float(weights[sym])
        notional: Optional[float] = None
        if budget is not None and float(budget) > 0:
            notional = float(budget) * w

        buy_signals.append(
            Signal(
                symbol=sym,
                side=Side.BUY,
                reason="clenow_momentum",
                notional=notional,
                strategy_id=strategy_id,
            )
        )

    logger.info(
        "Generated %d BUY signals and %d EXIT signals (budget=%.2f)",
        len(buy_signals),
        len(exit_signals),
        float(budget or 0),
    )

    return exit_signals + buy_signals
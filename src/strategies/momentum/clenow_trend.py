"""
STRATEGY: CLENOW MOMENTUM (Enhanced v2)
=========================================
Based on Andreas Clenow's 'Stocks on the Move', enhanced with:

1. **Regime Filter** (Gatekeeper)
   - Download SPY, IJH, IJR (large/mid/small cap ETFs).
   - If at least 2 of 3 have Price > 200-day SMA → risk-on → proceed.
   - Otherwise → "go to beach" → no new buys (may still exit).

2. **Entry Filters** (three gates)
   a) Price > 200-day SMA (trend intact).
   b) Momentum deceleration guard: 30-day score ≥ 50% of 60-day score.
      (Rejects stocks that ran hard months ago but are now stalling.)
   c) 52-week high proximity: price must be within 25% of its 52-week high.
      (Avoids buying broken stocks that have merely stopped falling.)

3. **Ranking**
   - Clenow Score = annualized_slope × R² on 60-day log-prices (was 90).
   - Fresher signal — avoids buying into already-exhausted trends.

4. **Sizing** (inverse-volatility weighted)
   - Weight ∝ 1 / σ_20d for each pick.
   - Normalize so weights sum to 1, then clamp to [2%, 10%].
   - After clamping, renormalize to sum to 1.

5. **Exit Logic** (SELL signals) — tightened vs v1
   - Sell any currently-held position where:
     a) Rank dropped beyond 20 (was 30), OR
     b) Price < 50-day SMA (was 100-day — much faster exit).
   - Requires current portfolio positions to be passed in.

v2 changes vs v1
-----------------
  LOOKBACK_DAYS      : 90  → 60   (fresher momentum score)
  EXIT_SMA_PERIOD    : 100 → 50   (exit sooner when trend breaks)
  EXIT_RANK_CUTOFF   : 30  → 20   (tighter rank tolerance)
  Deceleration filter: added      (30d score must be ≥ 50% of 60d score)
  52-week proximity  : added      (must be within 25% of 52wk high)
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
DEFAULT_TOP_N = 7
EXIT_RANK_CUTOFF = 20          # sell if rank drops beyond this (was 30)
REGIME_ETFS = ("SPY", "IJH", "IJR")
REGIME_SMA_PERIOD = 200
ENTRY_SMA_PERIOD = 200
EXIT_SMA_PERIOD = 50           # exit sooner when trend breaks (was 100)
LOOKBACK_DAYS = 60             # fresher momentum signal (was 90)
DECEL_LOOKBACK = 30            # short window for deceleration check
DECEL_MIN_RATIO = 0.50         # 30d score must be ≥ 50% of 60d score
HIGH_PROX_PCT = 0.25           # must be within 25% of 52-week high
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

def _high_52w(series: pd.Series) -> float:
    """Return the 52-week (252-day) rolling high."""
    if len(series) < 2:
        return float("nan")
    n = min(252, len(series))
    return float(series.iloc[-n:].max())


def score_universe(
    universe_file: str = "universe.csv",
    *,
    lookback: int = LOOKBACK_DAYS,
    entry_sma: int = ENTRY_SMA_PERIOD,
    decel_lookback: int = DECEL_LOOKBACK,
    decel_min_ratio: float = DECEL_MIN_RATIO,
    high_prox_pct: float = HIGH_PROX_PCT,
) -> pd.DataFrame:
    """Score every stock in the universe and return a DataFrame.

    Entry gates applied (all must pass):
      1. Price > *entry_sma*-day SMA.
      2. Deceleration guard: 30-day score ≥ *decel_min_ratio* × 60-day score.
      3. 52-week proximity: price ≥ (1 - *high_prox_pct*) × 52wk high.

    Returns columns: Symbol, Score, Close, Vol20 (annualized), SMA200, SMA50.
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
                    sma50  = _sma(close, EXIT_SMA_PERIOD)
                    vol20  = _realized_vol(close, VOL_LOOKBACK)
                    high52 = _high_52w(close)

                    # Gate 1: price > 200-day SMA
                    if current_price <= sma200:
                        continue

                    # Gate 3: 52-week high proximity
                    if not np.isnan(high52) and high52 > 0:
                        if current_price < (1.0 - high_prox_pct) * high52:
                            continue

                    score = get_momentum_score(close.iloc[-lookback:].values)

                    # Gate 2: deceleration guard (short-term momentum not collapsing)
                    if len(close) >= decel_lookback and score > 0:
                        score_short = get_momentum_score(close.iloc[-decel_lookback:].values)
                        if score_short < decel_min_ratio * score:
                            continue

                    records.append(
                        {
                            "Symbol": symbol,
                            "Score": score,
                            "Close": current_price,
                            "Vol20": vol20,
                            "SMA200": sma200,
                            "SMA50": sma50,
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


def _col_exit_sma(df: pd.DataFrame) -> str:
    """Return whichever SMA column exists in *df* (SMA50 in v2, SMA100 in v1)."""
    for col in ("SMA50", "SMA100"):
        if col in df.columns:
            return col
    return "SMA50"


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
      a) Symbol's rank > EXIT_RANK_CUTOFF (20), OR
      b) Symbol's current price < 50-day SMA.

    Stocks not found in the scored universe are also exited — they failed
    one or more of the three entry gates.
    """
    signals: List[Signal] = []
    scored_symbols: Set[str] = set(scores_df["Symbol"].tolist()) if not scores_df.empty else set()

    sma_col = _col_exit_sma(scores_df)
    sma_label = sma_col.lower()  # e.g. "sma50"

    for sym in held_symbols:
        reason: Optional[str] = None

        if sym not in scored_symbols:
            # Symbol didn't pass entry filters → definitely exit.
            reason = "exit:failed_entry_filters"
        else:
            row   = scores_df.loc[scores_df["Symbol"] == sym].iloc[0]
            rank  = int(row["Rank"])
            price = float(row["Close"])
            sma_x = float(row[sma_col])

            if rank > EXIT_RANK_CUTOFF and price < sma_x:
                reason = f"exit:rank_{rank}_and_below_{sma_label}"
            elif rank > EXIT_RANK_CUTOFF:
                reason = f"exit:rank_{rank}_gt_{EXIT_RANK_CUTOFF}"
            elif price < sma_x:
                reason = f"exit:below_{sma_label}"

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

    slots_available = max(top_n - len(held_symbols), 0)
    if slots_available == 0:
        logger.info("Momentum slots full (%d held / %d target) — no new buys.", len(held_symbols), top_n)
        return exit_signals

    available = scores_df[~scores_df["Symbol"].isin(list(held_symbols))].head(slots_available).copy()
    if available.empty:
        return exit_signals

    # Inverse-vol weights
    weights = _inverse_vol_weights(available.set_index("Symbol")["Vol20"])
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

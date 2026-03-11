"""
STRATEGY: V10 DAF 2× — Ranked Sector Allocation
=================================================
50/50 ensemble of V4-Best (sector momentum rotation with conviction overweight)
and V8-AW (all-weather: bull→sectors, bear→hedge ETFs by momentum),
with Dual-Filter Leverage (DAF 2×/35).

Monthly rebalancing on the first trading day of each month.

Target portfolio: 4–6 ETF positions.
Base budget: $30k.  DAF may scale to 2× ($60k) in low-vol bull regimes.
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yfinance as yf

from trade_models import Signal, Side


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
HEDGE_ETFS  = ["TLT", "GLD", "UUP", "FXY", "FXF"]
BENCHMARK   = "SPY"
CASH_ETF    = "SHY"
ALL_TICKERS = list(dict.fromkeys(SECTOR_ETFS + HEDGE_ETFS + [BENCHMARK, CASH_ETF]))

BASE_BUDGET = 30_000  # unleveraged dollars to deploy

# ---------------------------------------------------------------------------
# V4-Best parameters (fixed from back-test optimisation)
# ---------------------------------------------------------------------------
V4_WEIGHTS      = (0.10, 0.20, 0.25, 0.45)   # mom, vol, corr, trend
V4_TOP_BULL     = 6
V4_TOP_NEUTRAL  = 5
V4_TOP_BEAR     = 2
V4_BREADTH      = 0.5
V4_CONVICTION   = 2.0    # z-score threshold for #1-sector overweight
V4_CONV_PCT     = 0.5    # weight given to conviction pick
V4_MOM_THR      = -0.03  # neutral-regime cash-filter momentum threshold

# ---------------------------------------------------------------------------
# V8-AW parameters (best from grid search)
# ---------------------------------------------------------------------------
V8_WEIGHTS      = (0.15, 0.15, 0.20, 0.50)
V8_BULL_N       = 4
V8_NEUTRAL_N    = 4      # min(bull_n, 4)
V8_HEDGE_N      = 3
V8_BREADTH      = 0.6
V8_HEDGE_LB     = 42     # hedge-ETF momentum lookback (days)
V8_MOM_THR      = -0.03

# ---------------------------------------------------------------------------
# DAF (Dual-Filter Leverage) parameters
# ---------------------------------------------------------------------------
DAF_MAX_LEV     = 2.0
DAF_VOL_LB      = 21     # rolling-vol window
DAF_VOL_PCT     = 35     # percentile threshold


# ═══════════════════════════════════════════════════════════════════════════
# Data download
# ═══════════════════════════════════════════════════════════════════════════

def _download_data() -> tuple:
    """Download ~2 years of daily data.  Returns (closes_df, ohlc_dict)."""
    raw = yf.download(ALL_TICKERS, period="2y", progress=False, group_by="ticker")

    closes = pd.DataFrame()
    for t in ALL_TICKERS:
        try:
            c = raw[t]["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            closes[t] = c
        except Exception:
            pass
    closes = closes.sort_index().dropna(how="all")

    # OHLC for sector ETFs (ATR trend signal needs High / Low)
    ohlc_dict: Dict[str, pd.DataFrame] = {}
    for t in SECTOR_ETFS:
        try:
            df = raw[t][["Open", "High", "Low", "Close"]].dropna()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            ohlc_dict[t] = df
        except Exception:
            pass

    logger.info(
        "Downloaded %d days, %d tickers, OHLC for %d sectors",
        len(closes), len(closes.columns), len(ohlc_dict),
    )
    return closes, ohlc_dict


# ═══════════════════════════════════════════════════════════════════════════
# Signal functions (replicated from research notebook)
# ═══════════════════════════════════════════════════════════════════════════

def _momentum(closes_df: pd.DataFrame, idx: int, lookback: int = 84) -> pd.Series:
    """Rate of change over *lookback* days.  Higher = better."""
    if idx < lookback:
        return pd.Series(dtype=float)
    return ((closes_df.iloc[idx] - closes_df.iloc[idx - lookback])
            / closes_df.iloc[idx - lookback]).dropna()


def _ewma_vol(closes_df: pd.DataFrame, idx: int, lam: float = 0.94, window: int = 84) -> pd.Series:
    """EWMA volatility (RiskMetrics).  Lower = better."""
    if idx < window:
        return pd.Series(dtype=float)
    rets = closes_df.iloc[idx - window : idx + 1].pct_change().iloc[1:]
    if len(rets) < 2:
        return pd.Series(dtype=float)
    var = rets.iloc[0] ** 2
    for t in range(1, len(rets)):
        var = lam * var + (1 - lam) * rets.iloc[t] ** 2
    return (np.sqrt(var * 252)).dropna()


def _correlation(closes_df: pd.DataFrame, idx: int, lookback: int = 84) -> pd.Series:
    """Average pairwise correlation.  Lower = better."""
    if idx < lookback:
        return pd.Series(dtype=float)
    rets = closes_df.iloc[idx - lookback : idx + 1].pct_change().iloc[1:]
    if len(rets) < 10:
        return pd.Series(dtype=float)
    cm = rets.corr()
    return ((cm.sum(axis=1) - 1) / (len(cm) - 1)).dropna()


def _trend_atr(ohlc_dict: dict, ticker: str, date,
               atr_period: int = 42, high_period: int = 63, low_period: int = 105) -> int:
    """ATR breakout trend: +1 up, -1 down, 0 neutral."""
    try:
        df = ohlc_dict[ticker].loc[:date]
        if len(df) < max(atr_period, high_period, low_period):
            return 0
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        upper = c.rolling(high_period).max() + atr
        lower = l.rolling(low_period).min() - atr
        lc, lu, ll = float(c.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])
        if lc >= lu:
            return 1
        if lc <= ll:
            return -1
        return 0
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Regime detection
# ═══════════════════════════════════════════════════════════════════════════

def _regime(closes: pd.DataFrame, idx: int, breadth_thr: float) -> str:
    """Return 'B' (bull), 'N' (neutral), or 'D' (bear/down)."""
    spy = closes[BENCHMARK].iloc[: idx + 1].dropna()
    if len(spy) < 200:
        return "N"
    spy_sma = float(spy.iloc[-200:].mean())

    sec = closes[SECTOR_ETFS].iloc[: idx + 1]
    sma50 = sec.rolling(50).mean().iloc[-1]
    current = sec.iloc[-1]
    breadth = float((current > sma50).sum()) / max(float(current.notna().sum()), 1)

    price = float(spy.iloc[-1])
    if price > spy_sma and breadth >= breadth_thr:
        return "B"
    if price > spy_sma:
        return "N"
    return "D"


# ═══════════════════════════════════════════════════════════════════════════
# Sector composite ranking (shared by V4 and V8-AW for bull/neutral)
# ═══════════════════════════════════════════════════════════════════════════

def _sector_composite(sector_closes: pd.DataFrame, ohlc_dict: dict, idx: int,
                      weights: tuple) -> Optional[pd.Series]:
    """Compute composite rank for sector ETFs at *idx*.  Lower = better.

    Returns None if insufficient data.
    """
    mom = _momentum(sector_closes, idx)
    vol = _ewma_vol(sector_closes, idx)
    corr = _correlation(sector_closes, idx)
    common = mom.index.intersection(vol.index).intersection(corr.index)
    if len(common) < 3:
        return None

    m_r = mom[common].rank(ascending=False)
    v_r = vol[common].rank(ascending=True)
    c_r = corr[common].rank(ascending=True)
    date = sector_closes.index[idx]
    t_vals = pd.Series({t: -_trend_atr(ohlc_dict, t, date) for t in common})

    wM, wV, wC, wT = weights
    comp = wM * m_r + wV * v_r + wC * c_r + wT * t_vals
    return comp


# ═══════════════════════════════════════════════════════════════════════════
# V4-Best allocation for today
# ═══════════════════════════════════════════════════════════════════════════

def _v4_today(sector_closes: pd.DataFrame, ohlc_dict: dict,
              closes: pd.DataFrame, idx: int) -> Dict[str, float]:
    comp = _sector_composite(sector_closes, ohlc_dict, idx, V4_WEIGHTS)
    if comp is None:
        return {CASH_ETF: 1.0}

    regime = _regime(closes, idx, V4_BREADTH)
    if regime == "B":
        top_n, use_cf = V4_TOP_BULL, False
    elif regime == "N":
        top_n, use_cf = V4_TOP_NEUTRAL, True
    else:
        top_n, use_cf = V4_TOP_BEAR, True

    ranked = comp.sort_values().index.tolist()[:top_n]
    if not ranked:
        return {CASH_ETF: 1.0}

    n = len(ranked)
    weights: Dict[str, float] = {s: 1.0 / n for s in ranked}

    # Conviction overweight: if #1 sector is >2σ better than the group, give it 50%
    if n >= 2:
        scores = comp[ranked]
        std_s = float(scores.std())
        if std_s > 0:
            best_z = (float(scores.mean()) - float(scores.iloc[0])) / std_s
            if best_z > V4_CONVICTION:
                rest_w = (1.0 - V4_CONV_PCT) / (n - 1)
                weights = {s: rest_w for s in ranked}
                weights[ranked[0]] = V4_CONV_PCT

    # Cash filter (neutral / bear): sectors with momentum < threshold → cash
    if use_cf:
        mom = _momentum(sector_closes, idx)
        to_cash: List[str] = []
        for s in list(weights):
            m = float(mom.get(s, np.nan))
            if np.isnan(m) or m < V4_MOM_THR:
                to_cash.append(s)
        if to_cash:
            cash_w = sum(weights[s] for s in to_cash)
            for s in to_cash:
                del weights[s]
            weights[CASH_ETF] = weights.get(CASH_ETF, 0) + cash_w

    return weights


# ═══════════════════════════════════════════════════════════════════════════
# V8-AW allocation for today
# ═══════════════════════════════════════════════════════════════════════════

def _v8aw_today(sector_closes: pd.DataFrame, ohlc_dict: dict,
                closes: pd.DataFrame, idx: int) -> Dict[str, float]:
    regime = _regime(closes, idx, V8_BREADTH)

    # ── Bear: rotate into best hedge ETFs by momentum ──
    if regime == "D":
        available = [t for t in HEDGE_ETFS if t in closes.columns]
        if not available or idx < V8_HEDGE_LB:
            return {CASH_ETF: 1.0}
        hc = closes[available]
        cur = hc.iloc[idx]
        past = hc.iloc[idx - V8_HEDGE_LB]
        h_mom = ((cur - past) / past).dropna()
        if len(h_mom) < V8_HEDGE_N:
            return {CASH_ETF: 1.0}
        top = h_mom.nlargest(V8_HEDGE_N).index.tolist()
        w = 1.0 / len(top)
        return {t: w for t in top}

    # ── Bull / Neutral: concentrated sectors ──
    comp = _sector_composite(sector_closes, ohlc_dict, idx, V8_WEIGHTS)
    if comp is None:
        return {CASH_ETF: 1.0}

    top_n = V8_BULL_N if regime == "B" else V8_NEUTRAL_N
    use_cf = regime == "N"

    ranked = comp.sort_values().index.tolist()[:top_n]
    if not ranked:
        return {CASH_ETF: 1.0}

    n = len(ranked)
    weights: Dict[str, float] = {s: 1.0 / n for s in ranked}

    if use_cf:
        mom = _momentum(sector_closes, idx)
        to_cash: List[str] = []
        for s in list(weights):
            m = float(mom.get(s, np.nan))
            if np.isnan(m) or m < V8_MOM_THR:
                to_cash.append(s)
        if to_cash:
            cash_w = sum(weights[s] for s in to_cash)
            for s in to_cash:
                del weights[s]
            weights[CASH_ETF] = weights.get(CASH_ETF, 0) + cash_w

    return weights


# ═══════════════════════════════════════════════════════════════════════════
# Blend + DAF leverage
# ═══════════════════════════════════════════════════════════════════════════

def _blend(a1: Dict[str, float], a2: Dict[str, float]) -> Dict[str, float]:
    """50/50 blend of two allocation dicts."""
    out: Dict[str, float] = {}
    for s in set(a1) | set(a2):
        v = 0.5 * a1.get(s, 0.0) + 0.5 * a2.get(s, 0.0)
        if v > 1e-6:
            out[s] = v
    return out


def _daf_leverage(closes: pd.DataFrame) -> float:
    """Return leverage factor (1.0 or DAF_MAX_LEV).

    Lever up only when BOTH:
      1) 21-day realised vol of sector-EW portfolio < expanding 35th-pctile
      2) SPY > 200-SMA  (bull)
    """
    sector_cols = [c for c in SECTOR_ETFS if c in closes.columns]
    if not sector_cols:
        return 1.0
    ew_rets = closes[sector_cols].pct_change().mean(axis=1).dropna()
    if len(ew_rets) < 200:
        return 1.0

    rv = ew_rets.rolling(DAF_VOL_LB, min_periods=5).std() * np.sqrt(252)
    pctile = float(rv.expanding().quantile(DAF_VOL_PCT / 100).iloc[-1])
    cur_vol = float(rv.iloc[-1])
    low_vol = cur_vol < pctile

    spy = closes[BENCHMARK].dropna()
    spy_sma = float(spy.rolling(200, min_periods=50).mean().iloc[-1])
    bull = float(spy.iloc[-1]) > spy_sma

    lev = DAF_MAX_LEV if (low_vol and bull) else 1.0
    logger.info(
        "DAF: vol=%.3f pctile_35=%.3f low_vol=%s | SPY>SMA200=%s → %.1f×",
        cur_vol, pctile, low_vol, bull, lev,
    )
    return lev


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point  (called by trade.py)
# ═══════════════════════════════════════════════════════════════════════════

def generate_signals(
    *,
    budget: Optional[float] = None,
    strategy_id: str = "strategies.ranks.ranked_asset_alloc",
    held_symbols: Optional[Set[str]] = None,
) -> List[Signal]:
    """Generate BUY / SELL signals for the ranked-sector-allocation strategy.

    Rebalances monthly (first trading day of each month) or on first run.
    Between rebalances no signals are emitted.
    """
    if held_symbols is None:
        held_symbols = set()

    today = datetime.date.today()
    first_run = len(held_symbols) == 0
    is_rebalance = today.day <= 3 or first_run

    if not is_rebalance:
        logger.info(
            "Ranked-alloc: not a rebalance day (day=%d, held=%d). Skipping.",
            today.day, len(held_symbols),
        )
        return []

    # ── Download data ─────────────────────────────────────────────────────
    try:
        closes, ohlc_dict = _download_data()
    except Exception:
        logger.exception("Ranked-alloc: data download failed")
        return []
    if closes.empty:
        logger.error("Ranked-alloc: no data")
        return []

    sector_closes = closes[SECTOR_ETFS]  # keep same row index as closes
    idx = len(closes) - 1

    # ── Compute target allocations ────────────────────────────────────────
    alloc_v4  = _v4_today(sector_closes, ohlc_dict, closes, idx)
    alloc_v8  = _v8aw_today(sector_closes, ohlc_dict, closes, idx)
    target    = _blend(alloc_v4, alloc_v8)

    lev       = _daf_leverage(closes)
    effective = BASE_BUDGET * lev
    if budget is not None and budget > 0:
        effective = min(effective, float(budget))

    target_syms = {s for s, w in target.items() if s != CASH_ETF and w > 1e-6}

    logger.info(
        "V4 regime=%s → %s",
        _regime(closes, idx, V4_BREADTH),
        {k: f"{v:.0%}" for k, v in alloc_v4.items() if k != CASH_ETF},
    )
    logger.info(
        "V8 regime=%s → %s",
        _regime(closes, idx, V8_BREADTH),
        {k: f"{v:.0%}" for k, v in alloc_v8.items() if k != CASH_ETF},
    )
    logger.info(
        "Blended target: %s | lev=%.1f× deploy=$%.0f",
        {k: f"{v:.0%}" for k, v in target.items() if k != CASH_ETF},
        lev, effective,
    )

    # ── Generate signals ──────────────────────────────────────────────────
    signals: List[Signal] = []

    # SELL anything held but NOT in today's target
    for sym in held_symbols:
        if sym not in target_syms:
            signals.append(Signal(
                symbol=sym, side=Side.SELL,
                reason="exit:not_in_target",
                strategy_id=strategy_id,
            ))
            logger.info("SELL %s (rotation exit)", sym)

    # BUY anything in target but NOT yet held
    for sym in sorted(target_syms):
        if sym in held_symbols:
            continue
        w = target[sym]
        notional = round(effective * w, 2)
        if notional < 1.0:
            continue
        signals.append(Signal(
            symbol=sym, side=Side.BUY,
            reason="ranked_alloc",
            notional=notional,
            strategy_id=strategy_id,
        ))
        logger.info("BUY %s $%.2f (w=%.1f%%)", sym, notional, w * 100)

    logger.info(
        "Ranked-alloc: %d SELL + %d BUY signals (held=%d, target=%d)",
        sum(1 for s in signals if s.side == Side.SELL),
        sum(1 for s in signals if s.side == Side.BUY),
        len(held_symbols), len(target_syms),
    )
    return signals

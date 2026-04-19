"""
V10 DAF research backtest pipeline (ported from ranked_2_asset_allocation.ipynb).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as scipy_stats

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
HEDGE_ETFS  = ['TLT', 'GLD', 'UUP', 'FXY', 'FXF']
BENCHMARK   = 'SPY'
CASH_ETF    = 'SHY'
ALL_TICKERS = list(dict.fromkeys(SECTOR_ETFS + HEDGE_ETFS + [BENCHMARK, CASH_ETF]))


def download_v10_data(start: str = "2016-01-01", end: Optional[str] = None, progress: bool = False):
    """Download closes + sector OHLC (same parsing pattern as the research notebook)."""
    kwargs: Dict[str, Any] = dict(
        start=start, progress=progress, auto_adjust=True, group_by="ticker", threads=True
    )
    if end:
        kwargs["end"] = end
    raw = yf.download(ALL_TICKERS, **kwargs)

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

    okw = dict(start=start, progress=progress, group_by="ticker", auto_adjust=True, threads=True)
    if end:
        okw["end"] = end
    ohlc_raw = yf.download(SECTOR_ETFS, **okw)
    ohlc_dict = {}
    for t in SECTOR_ETFS:
        try:
            df = ohlc_raw[t][["Open", "High", "Low", "Close"]].dropna()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            ohlc_dict[t] = df
        except Exception:
            pass
    return closes, ohlc_dict



# -- Signal computation functions --
# Each takes a DataFrame of closes (columns = tickers) and an integer row index,
# and returns a Series of raw signal values indexed by ticker.

def momentum_signal(closes, idx, lookback=84):
    """4-month rate of change. Higher = better."""
    if idx < lookback:
        return pd.Series(dtype=float)
    current = closes.iloc[idx]
    past    = closes.iloc[idx - lookback]
    return ((current - past) / past).dropna()


def ewma_volatility(closes, idx, lam=0.94, window=84):
    """EWMA volatility (RiskMetrics style). Lower = better."""
    if idx < window:
        return pd.Series(dtype=float)
    rets = closes.iloc[idx - window:idx + 1].pct_change().iloc[1:]  # drop first NaN row
    if len(rets) < 2:
        return pd.Series(dtype=float)
    var = rets.iloc[0] ** 2
    for t in range(1, len(rets)):
        var = lam * var + (1 - lam) * rets.iloc[t] ** 2
    return (np.sqrt(var * 252)).dropna()


def correlation_signal(closes, idx, lookback=84):
    """Average pairwise correlation over lookback window. Lower = better."""
    if idx < lookback:
        return pd.Series(dtype=float)
    rets = closes.iloc[idx - lookback:idx + 1].pct_change().iloc[1:]
    if len(rets) < 10:
        return pd.Series(dtype=float)
    corr_matrix = rets.corr()
    n = len(corr_matrix)
    avg_corr = (corr_matrix.sum(axis=1) - 1) / (n - 1)
    return avg_corr.dropna()


def trend_signal_atr(ohlc_dict, ticker, date, atr_period=42, high_period=63, low_period=105):
    """ATR breakout trend signal: +1 (uptrend), -1 (downtrend), 0 (neutral)."""
    try:
        df = ohlc_dict[ticker].loc[:date]
        if len(df) < max(atr_period, high_period, low_period):
            return 0
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        upper = c.rolling(high_period).max() + atr
        lower = l.rolling(low_period).min() - atr
        last_c, last_u, last_l = float(c.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])
        if last_c >= last_u: return 1
        elif last_c <= last_l: return -1
        else: return 0
    except Exception:
        return 0


# -- Multi-horizon signal functions (article author's modifications) --
HORIZONS = [21, 63, 126, 252]

def multi_momentum(closes, idx, horizons=HORIZONS):
    """Average momentum (ROC) across multiple lookback windows."""
    if idx < max(horizons):
        return pd.Series(dtype=float)
    current = closes.iloc[idx]
    rocs = [(current - closes.iloc[idx - h]) / closes.iloc[idx - h] for h in horizons]
    return pd.concat(rocs, axis=1).mean(axis=1).dropna()


def multi_volatility(closes, idx, horizons=HORIZONS):
    """Average historical volatility across multiple windows."""
    if idx < max(horizons):
        return pd.Series(dtype=float)
    rets = closes.pct_change()
    vols = [rets.iloc[idx - h:idx + 1].std() * np.sqrt(252) for h in horizons]
    return pd.concat(vols, axis=1).mean(axis=1).dropna()


def multi_correlation(closes, idx, horizons=HORIZONS):
    """Average pairwise correlation across multiple windows."""
    if idx < max(horizons):
        return pd.Series(dtype=float)
    rets = closes.pct_change()
    avg_corrs = []
    for h in horizons:
        cm = rets.iloc[idx - h:idx + 1].corr()
        avg_corrs.append((cm.sum(axis=1) - 1) / (len(cm) - 1))
    return pd.concat(avg_corrs, axis=1).mean(axis=1).dropna()


def multi_trend(closes, idx, horizons=HORIZONS):
    """Price / rolling high, averaged across windows. Higher = closer to highs."""
    if idx < max(horizons):
        return pd.Series(dtype=float)
    current = closes.iloc[idx]
    ratios = [current / closes.iloc[idx - h:idx + 1].max() for h in horizons]
    return pd.concat(ratios, axis=1).mean(axis=1).dropna()




def build_rank_matrices(closes_df, ohlc_dict=None, mode="original", spy_shy_closes=None):
    """Build (days x tickers) rank matrices for all 4 signals + raw momentum.

    Returns:
        tickers: list of ticker names (column order)
        dates: DatetimeIndex of valid rows
        M_ranks, V_ranks, C_ranks, T_ranks: np.ndarray (n_days, n_tickers), float
        mom_raw: np.ndarray (n_days, n_tickers) -- raw momentum values (for cash filter)
        fwd_rets: np.ndarray (n_days, n_tickers) -- next-day returns
        cash_rets: np.ndarray (n_days,) -- next-day SHY returns
        bench_rets_arr: np.ndarray (n_days,) -- next-day SPY returns
        valid_mask: np.ndarray (n_days, n_tickers) bool -- True if data exists for that day
    """
    tickers = list(closes_df.columns)
    n_tickers = len(tickers)
    n_days = len(closes_df)

    # Determine warmup based on mode
    min_idx = 260 if mode == "multi" else 110

    # Preallocate NaN matrices
    M_ranks = np.full((n_days, n_tickers), np.nan)
    V_ranks = np.full((n_days, n_tickers), np.nan)
    C_ranks = np.full((n_days, n_tickers), np.nan)
    T_ranks = np.full((n_days, n_tickers), np.nan)
    mom_raw = np.full((n_days, n_tickers), np.nan)

    print(f"Building {mode} rank matrices ({n_days - min_idx} days, {n_tickers} tickers)...")

    for idx in range(min_idx, n_days):
        available = closes_df.columns[closes_df.iloc[idx].notna()].tolist()
        if len(available) < 3:
            continue
        sc = closes_df[available]

        if mode == "original":
            mom  = momentum_signal(sc, idx)
            vol  = ewma_volatility(sc, idx)
            corr = correlation_signal(sc, idx)
            common = mom.index.intersection(vol.index).intersection(corr.index)
            if len(common) < 3:
                continue
            # Ranks: lower = better
            m_r = mom[common].rank(ascending=False)
            v_r = vol[common].rank(ascending=True)
            c_r = corr[common].rank(ascending=True)
            # Trend: ATR signal (+1/0/-1), negate so that uptrend = lower "rank"
            date = sc.index[idx]
            t_vals = pd.Series({t: -trend_signal_atr(ohlc_dict, t, date) for t in common})
            for t in common:
                j = tickers.index(t)
                M_ranks[idx, j] = m_r[t]
                V_ranks[idx, j] = v_r[t]
                C_ranks[idx, j] = c_r[t]
                T_ranks[idx, j] = t_vals[t]
                mom_raw[idx, j] = mom[t]
        else:
            mom   = multi_momentum(sc, idx)
            vol   = multi_volatility(sc, idx)
            corr  = multi_correlation(sc, idx)
            trend = multi_trend(sc, idx)
            common = mom.index.intersection(vol.index).intersection(corr.index).intersection(trend.index)
            if len(common) < 3:
                continue
            m_r = mom[common].rank(ascending=False)
            v_r = vol[common].rank(ascending=True)
            c_r = corr[common].rank(ascending=True)
            t_r = trend[common].rank(ascending=False)
            for t in common:
                j = tickers.index(t)
                M_ranks[idx, j] = m_r[t]
                V_ranks[idx, j] = v_r[t]
                C_ranks[idx, j] = c_r[t]
                T_ranks[idx, j] = t_r[t]
                mom_raw[idx, j] = mom[t]

        if (idx - min_idx) % 500 == 0:
            print(f"  {idx - min_idx}/{n_days - min_idx}", end="\r")

    # Forward returns: tomorrow's return for each sector ETF
    rets_df = closes_df.pct_change().shift(-1).fillna(0)
    fwd_rets = rets_df.values  # (n_days, n_tickers)

    # Cash (SHY) and benchmark (SPY) next-day returns
    _ss = spy_shy_closes if spy_shy_closes is not None else closes_df
    cash_r  = _ss[CASH_ETF].pct_change().shift(-1).fillna(0).reindex(closes_df.index).fillna(0).values
    bench_r = _ss[BENCHMARK].pct_change().shift(-1).fillna(0).reindex(closes_df.index).fillna(0).values

    valid_mask = ~np.isnan(M_ranks)  # True where we have valid ranks

    print(f"  Done. Valid days: {valid_mask.any(axis=1).sum()}")

    return {
        "tickers": tickers, "dates": closes_df.index,
        "M": M_ranks, "V": V_ranks, "C": C_ranks, "T": T_ranks,
        "mom": mom_raw, "fwd_rets": fwd_rets,
        "cash_rets": cash_r, "bench_rets": bench_r, "valid": valid_mask,
    }

def compute_stats(equity, benchmark):
    """Compute key performance metrics."""
    sr = equity.pct_change().dropna()
    br = benchmark.pct_change().dropna()
    n_yr = len(sr) / 252
    if n_yr <= 0:
        return pd.DataFrame()

    cagr_s = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_yr) - 1
    cagr_b = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (1 / n_yr) - 1
    vol_s, vol_b = sr.std() * np.sqrt(252), br.std() * np.sqrt(252)
    sh_s = cagr_s / vol_s if vol_s > 0 else 0
    sh_b = cagr_b / vol_b if vol_b > 0 else 0

    def mdd(eq):
        pk = eq.cummax()
        return ((eq - pk) / pk).min()
    md_s, md_b = mdd(equity), mdd(benchmark)
    cal_s = -cagr_s / md_s if md_s < 0 else 0
    cal_b = -cagr_b / md_b if md_b < 0 else 0
    wr = (sr.resample("ME").sum() > 0).mean()

    return pd.DataFrame({
        "Strategy": [f"{cagr_s:.1%}", f"{vol_s:.1%}", f"{sh_s:.2f}",
                     f"{md_s:.1%}", f"{cal_s:.2f}", f"{wr:.1%}", f"{equity.iloc[-1]:.2f}"],
        "SPY": [f"{cagr_b:.1%}", f"{vol_b:.1%}", f"{sh_b:.2f}",
                f"{md_b:.1%}", f"{cal_b:.2f}", "-", f"{benchmark.iloc[-1]:.2f}"],
    }, index=["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar",
             "Monthly Win Rate", "Final Value ($1)"])


def compute_full_stats(equity, benchmark):
    """Extended stats including Sortino, Skew, Kurtosis, PSR."""
    sr = equity.pct_change().dropna()
    br = benchmark.pct_change().dropna()
    n_yr = len(sr) / 252

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_yr) - 1
    cagr_b = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (1 / n_yr) - 1
    vol = sr.std() * np.sqrt(252)
    vol_b = br.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    sharpe_b = cagr_b / vol_b if vol_b > 0 else 0

    pk = equity.cummax()
    dd = ((equity - pk) / pk).min()
    pk_b = benchmark.cummax()
    dd_b = ((benchmark - pk_b) / pk_b).min()
    calmar = -cagr / dd if dd < 0 else 0

    down = sr[sr < 0]
    down_std = down.std() * np.sqrt(252)
    sortino = cagr / down_std if down_std > 0 else 0

    wr = (sr.resample("ME").sum() > 0).mean()
    skew = float(scipy_stats.skew(sr))
    kurt = float(scipy_stats.kurtosis(sr))

    # PSR: probability true Sharpe > 0
    n = len(sr)
    sr_hat = sharpe / np.sqrt(252)  # daily Sharpe
    denom = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2)
    psr = float(scipy_stats.norm.cdf(sr_hat * np.sqrt(n - 1) / denom)) if denom > 0 else 0.5

    return {
        "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Max DD": dd,
        "Calmar": calmar, "Sortino": sortino, "Win Rate": wr,
        "Skew": skew, "Kurtosis": kurt, "PSR": psr, "Final": equity.iloc[-1],
    }


# Rebuild adaptive results


def apply_transaction_costs(equity, allocations, cost_bps=10):
    """Apply round-trip transaction costs at each rebalance."""
    eq = equity.copy()
    cost = cost_bps / 10000
    prev_alloc = {}
    for date, alloc in allocations:
        if date not in eq.index:
            continue
        all_tickers = set(list(prev_alloc.keys()) + list(alloc.keys()))
        turnover = sum(abs(alloc.get(t, 0) - prev_alloc.get(t, 0)) for t in all_tickers) / 2
        cost_drag = 1 - cost * turnover * 2
        idx = eq.index.get_loc(date)
        eq.iloc[idx:] *= cost_drag
        prev_alloc = alloc
    return eq


def quick_stats(r, cost_bps=10):
    """Return (sharpe_10bp, cagr_10bp, maxdd, beat_spy_count, total_years, vol)."""
    if r is None:
        return (-999, 0, 0, 0, 0, 0)
    eq, bm = r["equity"], r["benchmark"]
    allocs = r.get("allocations", [])
    eq_c = apply_transaction_costs(eq, allocs, cost_bps=cost_bps)
    sr = eq_c.pct_change().dropna()
    n_yr = len(sr) / 252
    if n_yr <= 0:
        return (-999, 0, 0, 0, 0, 0)
    cagr = (eq_c.iloc[-1] / eq_c.iloc[0]) ** (1 / n_yr) - 1
    vol = sr.std() * np.sqrt(252)
    sh = cagr / vol if vol > 0 else 0
    pk = eq_c.cummax(); dd = ((eq_c - pk) / pk).min()
    strat_yr = eq_c.resample("YE").last().pct_change().dropna()
    spy_yr   = bm.resample("YE").last().pct_change().dropna()
    common = strat_yr.index.intersection(spy_yr.index)
    beats = int((strat_yr[common] > spy_yr[common]).sum())
    return (sh, cagr, dd, beats, len(common), vol)


def fmt_full_stats(stats, label='Strategy'):
    """Format a compute_full_stats dict into a display-ready DataFrame."""
    d = {}
    for k, v in stats.items():
        if k in ('CAGR', 'Vol', 'Win Rate'):
            d[k] = f'{v:.1%}'
        elif k == 'Max DD':
            d[k] = f'{v:.1%}'
        elif k in ('Sharpe', 'Calmar', 'Sortino'):
            d[k] = f'{v:.3f}'
        elif k == 'PSR':
            d[k] = f'{v:.1%}'
        elif k == 'Final':
            d[k] = f'${v:.2f}'
        else:
            d[k] = f'{v:.2f}'
    return pd.Series(d, name=label)


def adaptive_backtest_v4(mat, closes, weights=(0.25, 0.25, 0.25, 0.25),
                         top_n_bull=8, top_n_neutral=5, top_n_bear=3,
                         rebal_freq="M", breadth_thr=0.6,
                         sizing="equal", vol_lookback=63,
                         cash_filter_bull=False, cash_filter_neutral=True, cash_filter_bear=True,
                         # V4 new parameters
                         conviction_overweight=False, conviction_threshold=2.0,
                         conviction_top_pct=0.40,
                         sector_sma50_filter=False,
                         recovery_accel=False, recovery_extra_n=3, recovery_months=3,
                         neutral_mom_threshold=0.0):
    """
    V4: Targeted improvements based on year-loss attribution.

    New features:
      conviction_overweight: If True, #1 ranked sector gets conviction_top_pct weight
                             when its score is >conviction_threshold σ below mean
      sector_sma50_filter:   If True, skip sectors trading below their 50-day SMA
                             (replace with next-best sector, NOT cash)
      recovery_accel:        If True, expand top_n by recovery_extra_n during first
                             recovery_months months after bear→bull transition
      neutral_mom_threshold: Cash filter threshold in neutral regime (default 0 = any
                             negative momentum → cash; set to e.g. -0.05 for softer)
    """
    wM, wV, wC, wT = weights
    M, V, C, T = mat["M"], mat["V"], mat["C"], mat["T"]
    mom_mat, fwd, cash_r, bench_r = mat["mom"], mat["fwd_rets"], mat["cash_rets"], mat["bench_rets"]
    valid = mat["valid"]
    mat_dates = mat["dates"]
    tickers = mat["tickers"]
    n_days, n_tickers = M.shape

    # Regime indicators
    spy_p = closes["SPY"].reindex(mat_dates).ffill().values
    spy_sma_arr = pd.Series(spy_p, index=mat_dates).rolling(200).mean().values
    sec_p = closes[SECTOR_ETFS].reindex(mat_dates).ffill()
    sec_sma50 = sec_p.rolling(50).mean()
    breadth_arr = (sec_p > sec_sma50).sum(axis=1).values / sec_p.notna().sum(axis=1).values

    # Per-sector SMA50 (for sector_sma50_filter)
    if sector_sma50_filter:
        sector_above_sma50 = (sec_p > sec_sma50).values  # (n_days, n_sectors)
        sec_to_idx = {t: SECTOR_ETFS.index(t) for t in tickers if t in SECTOR_ETFS}
    else:
        sector_above_sma50 = None
        sec_to_idx = None

    # Precompute for inv_vol
    sec_rets = closes[SECTOR_ETFS].reindex(mat_dates).ffill().pct_change()
    rolling_vol = sec_rets.rolling(vol_lookback, min_periods=20).std().values * np.sqrt(252)
    tick_to_sec = {t: SECTOR_ETFS.index(t) for t in tickers if t in SECTOR_ETFS}

    # Composite
    big = 9999.0
    composite = (wM * np.where(valid, M, big) +
                 wV * np.where(valid, V, big) +
                 wC * np.where(valid, C, big) +
                 wT * np.where(valid, T, big))

    valid_days_arr = np.where(valid.any(axis=1))[0]
    if len(valid_days_arr) == 0:
        return None
    first, last = valid_days_arr[0], valid_days_arr[-1]
    if last - first < 20:
        return None

    # Rebalance schedule
    active_dates = mat_dates[first:last + 1]
    rebal_mask = np.zeros(n_days, dtype=bool)
    if rebal_freq == "W":
        rd = pd.Series(active_dates).groupby(active_dates.to_period("W")).last().values
    elif rebal_freq == "2W":
        wk = pd.Series(active_dates).groupby(active_dates.to_period("W")).last()
        rd = wk.iloc[::2].values
    else:
        rd = pd.Series(active_dates).groupby(active_dates.to_period("M")).last().values
    for d in rd:
        loc = mat_dates.get_indexer([d], method="ffill")[0]
        if first <= loc <= last:
            rebal_mask[loc] = True
    rebal_mask[first] = True

    # Recovery tracking: detect bear→bull transitions
    last_bear_end = -9999  # day index when bear last ended

    # Main loop
    port_weights = np.zeros((n_days, n_tickers))
    cash_weight = np.zeros(n_days)
    alloc_list = []
    current_w = np.zeros(n_tickers)
    current_cash = 1.0
    prev_regime = "N"

    for i in range(first, last + 1):
        if rebal_mask[i]:
            s_p, s_sma, br = spy_p[i], spy_sma_arr[i], breadth_arr[i]
            if np.isnan(s_sma):
                regime = "N"
            elif s_p > s_sma and br >= breadth_thr:
                regime = "B"
            elif s_p > s_sma:
                regime = "N"
            else:
                regime = "D"

            # Track bear→bull transitions for recovery acceleration
            if prev_regime == "D" and regime in ("B", "N"):
                last_bear_end = i
            prev_regime = regime

            # Determine top_n based on regime
            if regime == "B":
                top_n = top_n_bull
                use_cf = cash_filter_bull
                mom_thr = 0.0  # not used when cf is off
            elif regime == "N":
                top_n = top_n_neutral
                use_cf = cash_filter_neutral
                mom_thr = neutral_mom_threshold
            else:
                top_n = top_n_bear
                use_cf = cash_filter_bear
                mom_thr = 0.0

            # Recovery acceleration: expand top_n after bear→bull
            if recovery_accel and (i - last_bear_end) < recovery_months * 21:
                top_n = min(top_n + recovery_extra_n, n_tickers)

            row = composite[i]
            valid_count = valid[i].sum()
            if valid_count < top_n:
                current_w = np.zeros(n_tickers)
                current_cash = 1.0
            else:
                # Sort all valid tickers by composite score
                valid_idx = np.where(valid[i])[0]
                sorted_valid = valid_idx[np.argsort(row[valid_idx])]

                # Apply sector SMA50 filter: skip sectors below their SMA50
                if sector_sma50_filter and sector_above_sma50 is not None:
                    filtered = []
                    for j in sorted_valid:
                        t = tickers[j]
                        if t in sec_to_idx:
                            if sector_above_sma50[i, sec_to_idx[t]]:
                                filtered.append(j)
                            # Skip this sector, try next best
                        else:
                            filtered.append(j)  # non-sector tickers pass through
                        if len(filtered) >= top_n:
                            break
                    # If we couldn't fill top_n, take what we have
                    top_idx = np.array(filtered[:top_n]) if len(filtered) > 0 else sorted_valid[:top_n]
                else:
                    top_idx = sorted_valid[:top_n]

                n_selected = len(top_idx)

                # ── Sizing ──
                if sizing == "inv_vol":
                    raw_w = np.zeros(n_selected)
                    for k, j in enumerate(top_idx):
                        t = tickers[j]
                        if t in tick_to_sec:
                            v = rolling_vol[i, tick_to_sec[t]]
                            raw_w[k] = 1.0 / v if (not np.isnan(v) and v > 0) else 1.0
                        else:
                            raw_w[k] = 1.0
                elif sizing == "mom_prop":
                    raw_w = np.zeros(n_selected)
                    for k, j in enumerate(top_idx):
                        m = mom_mat[i, j]
                        raw_w[k] = max(m, 0.001) if not np.isnan(m) else 0.001
                elif sizing == "ranked":
                    raw_w = np.zeros(n_selected)
                    for k in range(n_selected):
                        raw_w[k] = n_selected - k  # best rank gets highest weight
                else:  # equal
                    raw_w = np.ones(n_selected)

                # ── Conviction overweight ──
                if conviction_overweight and n_selected >= 2:
                    scores = row[top_idx]
                    mean_score = scores.mean()
                    std_score = scores.std()
                    if std_score > 0:
                        best_z = (mean_score - scores[0]) / std_score  # positive = better
                        if best_z > conviction_threshold:
                            # Top-1 gets conviction_top_pct, rest share the remainder
                            raw_w = np.ones(n_selected)
                            total_rest = (1.0 - conviction_top_pct) / (n_selected - 1) if n_selected > 1 else 0
                            raw_w[0] = conviction_top_pct
                            raw_w[1:] = total_rest
                            # Skip normal normalization
                            total_w = raw_w.sum()
                            if abs(total_w - 1.0) > 0.001:
                                raw_w /= total_w

                total_w = raw_w.sum()
                if total_w > 0:
                    raw_w /= total_w
                else:
                    raw_w = np.full(n_selected, 1.0 / n_selected)

                new_w = np.zeros(n_tickers)
                new_cash = 0.0
                holdings = {}
                for k, j in enumerate(top_idx):
                    m_val = mom_mat[i, j]
                    if use_cf and (np.isnan(m_val) or m_val < mom_thr):
                        new_cash += raw_w[k]
                        holdings[CASH_ETF] = holdings.get(CASH_ETF, 0) + raw_w[k]
                    else:
                        new_w[j] = raw_w[k]
                        holdings[tickers[j]] = raw_w[k]
                current_w = new_w
                current_cash = new_cash
                alloc_list.append((mat_dates[i], holdings))

        port_weights[i] = current_w
        cash_weight[i] = current_cash

    port_daily = (port_weights[first:last+1] * fwd[first:last+1]).sum(axis=1) + \
                  cash_weight[first:last+1] * cash_r[first:last+1]
    bench_daily = bench_r[first:last+1]
    eq_vals = np.cumprod(1 + port_daily)
    bm_vals = np.cumprod(1 + bench_daily)
    eq_vals = np.insert(eq_vals, 0, 1.0)
    bm_vals = np.insert(bm_vals, 0, 1.0)
    dt_index = mat_dates[first:last + 1]
    dt_index = dt_index.insert(0, mat_dates[max(0, first - 1)])

    equity = pd.Series(eq_vals, index=dt_index[:len(eq_vals)], name="Strategy")
    bench  = pd.Series(bm_vals, index=dt_index[:len(bm_vals)], name="SPY")

    return {"equity": equity, "benchmark": bench,
            "allocations": alloc_list, "stats": compute_stats(equity, bench)}

def adaptive_backtest_v8aw(mat, mat_hedge_data, closes, weights=(0.1, 0.2, 0.25, 0.45),
                           top_n_bull=3, top_n_neutral=3, top_n_bear_hedge=2,
                           rebal_freq="M", breadth_thr=0.5,
                           neutral_mom_threshold=-0.03, cash_filter_neutral=True,
                           hedge_lookback=63):
    """
    V8-AW: "All-Weather" variant.
    Bull: concentrated top sectors (like V8)
    Bear: rotate into best HEDGE ETFs by momentum (TLT, GLD, etc.)
    instead of just parking in SHY.
    
    Key difference from V8: bear markets earn TLT/GLD returns (~5-15%)
    instead of SHY returns (~2-3%). This could boost CAGR significantly.
    """
    wM, wV, wC, wT = weights
    M, V, C, T = mat["M"], mat["V"], mat["C"], mat["T"]
    mom_mat = mat["mom"]
    fwd, cash_r, bench_r = mat["fwd_rets"], mat["cash_rets"], mat["bench_rets"]
    valid = mat["valid"]
    mat_dates = mat["dates"]
    tickers = mat["tickers"]
    n_days, n_tickers = M.shape

    # Hedge ETF data (precomputed)
    hedge_tickers = mat_hedge_data["tickers"]
    hedge_fwd = mat_hedge_data["fwd_rets"]
    hedge_dates = mat_hedge_data["dates"]
    n_hedge = len(hedge_tickers)
    
    # Build hedge momentum for ranking
    hedge_closes_df = closes[hedge_tickers].reindex(mat_dates).ffill()
    hedge_mom = (hedge_closes_df / hedge_closes_df.shift(hedge_lookback) - 1).values

    spy_p = closes["SPY"].reindex(mat_dates).ffill().values
    spy_sma_arr = pd.Series(spy_p, index=mat_dates).rolling(200).mean().values
    sec_p = closes[SECTOR_ETFS].reindex(mat_dates).ffill()
    sec_sma50 = sec_p.rolling(50).mean()
    breadth_arr = (sec_p > sec_sma50).sum(axis=1).values / sec_p.notna().sum(axis=1).values

    # Sector composite
    big = 9999.0
    composite = (wM * np.where(valid, M, big) +
                 wV * np.where(valid, V, big) +
                 wC * np.where(valid, C, big) +
                 wT * np.where(valid, T, big))

    # Hedge forward returns aligned to mat_dates
    hedge_fwd_aligned = np.zeros((n_days, n_hedge))
    for hi, ht in enumerate(hedge_tickers):
        h_rets = closes[ht].reindex(mat_dates).ffill().pct_change().shift(-1).fillna(0).values
        hedge_fwd_aligned[:, hi] = h_rets

    valid_days_arr = np.where(valid.any(axis=1))[0]
    if len(valid_days_arr) == 0:
        return None
    first, last = valid_days_arr[0], valid_days_arr[-1]
    if last - first < 20:
        return None

    active_dates = mat_dates[first:last + 1]
    rebal_mask = np.zeros(n_days, dtype=bool)
    if rebal_freq == "W":
        rd = pd.Series(active_dates).groupby(active_dates.to_period("W")).last().values
    elif rebal_freq == "2W":
        wk = pd.Series(active_dates).groupby(active_dates.to_period("W")).last()
        rd = wk.iloc[::2].values
    else:
        rd = pd.Series(active_dates).groupby(active_dates.to_period("M")).last().values
    for d in rd:
        loc = mat_dates.get_indexer([d], method="ffill")[0]
        if first <= loc <= last:
            rebal_mask[loc] = True
    rebal_mask[first] = True

    port_weights = np.zeros((n_days, n_tickers))
    hedge_weights = np.zeros((n_days, n_hedge))
    cash_weight = np.zeros(n_days)
    alloc_list = []
    current_w = np.zeros(n_tickers)
    current_hw = np.zeros(n_hedge)
    current_cash = 1.0

    for i in range(first, last + 1):
        if rebal_mask[i]:
            s_p, s_sma, br = spy_p[i], spy_sma_arr[i], breadth_arr[i]
            if np.isnan(s_sma):
                regime = "N"
            elif s_p > s_sma and br >= breadth_thr:
                regime = "B"
            elif s_p > s_sma:
                regime = "N"
            else:
                regime = "D"

            new_w = np.zeros(n_tickers)
            new_hw = np.zeros(n_hedge)
            new_cash = 0.0
            holdings = {}

            if regime == "D":
                # BEAR: rotate into best hedge ETFs by momentum
                h_mom = hedge_mom[i]
                valid_h = ~np.isnan(h_mom)
                if valid_h.sum() >= top_n_bear_hedge:
                    sorted_h = np.argsort(-h_mom)  # descending by momentum
                    top_h = [j for j in sorted_h if valid_h[j]][:top_n_bear_hedge]
                    w_per = 1.0 / len(top_h)
                    for j in top_h:
                        new_hw[j] = w_per
                        holdings[hedge_tickers[j]] = w_per
                else:
                    new_cash = 1.0
                    holdings[CASH_ETF] = 1.0
            else:
                # BULL / NEUTRAL: concentrated sectors (like V8)
                if regime == "B":
                    top_n = top_n_bull
                    use_cf = False
                    mom_thr = 0.0
                else:
                    top_n = top_n_neutral
                    use_cf = cash_filter_neutral
                    mom_thr = neutral_mom_threshold

                row = composite[i]
                valid_count = valid[i].sum()
                if valid_count < top_n:
                    new_cash = 1.0
                else:
                    valid_idx = np.where(valid[i])[0]
                    sorted_valid = valid_idx[np.argsort(row[valid_idx])]
                    top_idx = sorted_valid[:top_n]
                    w_per = 1.0 / top_n
                    for j in top_idx:
                        m_val = mom_mat[i, j]
                        if use_cf and (np.isnan(m_val) or m_val < mom_thr):
                            new_cash += w_per
                            holdings[CASH_ETF] = holdings.get(CASH_ETF, 0) + w_per
                        else:
                            new_w[j] = w_per
                            holdings[tickers[j]] = w_per

            current_w = new_w
            current_hw = new_hw
            current_cash = new_cash
            alloc_list.append((mat_dates[i], holdings))

        port_weights[i] = current_w
        hedge_weights[i] = current_hw
        cash_weight[i] = current_cash

    # Portfolio daily returns = sector returns + hedge returns + cash
    port_daily = (
        (port_weights[first:last+1] * fwd[first:last+1]).sum(axis=1) +
        (hedge_weights[first:last+1] * hedge_fwd_aligned[first:last+1]).sum(axis=1) +
        cash_weight[first:last+1] * cash_r[first:last+1]
    )
    bench_daily = bench_r[first:last+1]
    eq_vals = np.cumprod(1 + port_daily)
    bm_vals = np.cumprod(1 + bench_daily)
    eq_vals = np.insert(eq_vals, 0, 1.0)
    bm_vals = np.insert(bm_vals, 0, 1.0)
    dt_index = mat_dates[first:last + 1]
    dt_index = dt_index.insert(0, mat_dates[max(0, first - 1)])
    equity = pd.Series(eq_vals, index=dt_index[:len(eq_vals)], name="Strategy")
    bench  = pd.Series(bm_vals, index=dt_index[:len(bm_vals)], name="SPY")
    return {"equity": equity, "benchmark": bench,
            "allocations": alloc_list, "stats": compute_stats(equity, bench)}

def ensemble_blend(results_dict, blend_weights=None):
    """
    V10: Equal-weight (or custom-weight) ensemble of multiple strategies.
    
    Diversifying ACROSS strategies reduces strategy-specific risk:
    if V4 loses in 2023 but V6 wins, the blend is more stable.
    """
    names = list(results_dict.keys())
    if blend_weights is None:
        blend_weights = {n: 1.0 / len(names) for n in names}
    
    # Align all equity/benchmark series to common dates
    all_eq = pd.DataFrame({n: r["equity"] for n, r in results_dict.items()})
    all_bm = pd.DataFrame({n: r["benchmark"] for n, r in results_dict.items()})
    
    # Drop dates where any strategy has NaN
    all_eq = all_eq.dropna()
    all_bm = all_bm.dropna()
    
    # Compute daily returns for each
    eq_rets = all_eq.pct_change().dropna()
    bm_rets = all_bm.iloc[:, 0].pct_change().dropna()
    
    # Blend returns
    blended_rets = sum(blend_weights[n] * eq_rets[n] for n in names)
    
    # Rebuild equity
    common_start = eq_rets.index[0]
    eq_start_idx = all_eq.index.get_loc(common_start) - 1 if common_start != all_eq.index[0] else 0
    blended_eq = (1 + blended_rets).cumprod()
    blended_eq = pd.concat([pd.Series([1.0], index=[all_eq.index[eq_start_idx]]), blended_eq])
    blended_eq.name = "Strategy"
    
    bench_eq = (1 + bm_rets).cumprod()
    bench_eq = pd.concat([pd.Series([1.0], index=[all_eq.index[eq_start_idx]]), bench_eq])
    bench_eq.name = "SPY"
    
    # Merge all allocations for cost estimation
    all_allocs = []
    for n, r in results_dict.items():
        w = blend_weights[n]
        for date, alloc in r["allocations"]:
            scaled_alloc = {k: v * w for k, v in alloc.items()}
            all_allocs.append((date, scaled_alloc))
    all_allocs.sort(key=lambda x: x[0])
    
    return {"equity": blended_eq, "benchmark": bench_eq,
            "allocations": all_allocs,
            "stats": compute_stats(blended_eq, bench_eq)}

def apply_dual_filter_leverage(result, closes, max_lev=1.5, vol_lb=21, vol_pct=50,
                                spread_annual=0.005):
    """
    Apply max_lev only when BOTH:
      (1) Realized vol < rolling vol_pct-th percentile (quiet markets)
      (2) SPY > SMA200 (bull regime)
    1-day lag — no look-ahead.
    """
    eq = result["equity"]; bm = result["benchmark"]
    sr = eq.pct_change().dropna()
    rv      = sr.rolling(vol_lb, min_periods=5).std() * np.sqrt(252)
    low_vol = rv < rv.expanding().quantile(vol_pct / 100)
    spy_p   = closes["SPY"].reindex(sr.index).ffill()
    bull    = spy_p > spy_p.rolling(200, min_periods=50).mean()
    lev_s = pd.Series(1.0, index=sr.index)
    lev_s[low_vol & bull] = max_lev
    lev_s = lev_s.shift(1).fillna(1.0)
    shy    = closes[CASH_ETF].ffill().pct_change().fillna(0).reindex(sr.index).fillna(0)
    borrow = (lev_s - 1).clip(lower=0) * (shy + spread_annual / 252)
    lev_r  = lev_s * sr - borrow
    lev_eq = pd.concat([pd.Series([1.0], index=[eq.index[0]]), (1 + lev_r).cumprod()])
    lev_eq.name = "Strategy"
    bm_eq  = pd.concat([pd.Series([1.0], index=[bm.index[0]]),
                        (1 + bm.pct_change().dropna()).cumprod()]); bm_eq.name = "SPY"
    return {"equity": lev_eq, "benchmark": bm_eq,
            "allocations": result.get("allocations", []),
            "stats": compute_stats(lev_eq, bm_eq), "avg_leverage": float(lev_s.mean())}



def run_v10_daf_backtest(
    start: str = "2004-07-01",
    display_start: Optional[str] = None,
    rebase_at_display: bool = False,
    end: Optional[str] = None,
    apply_tc: bool = True,
    cost_bps: float = 10.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """V4-Best + V8-AW (research grid winner) + V10 return-blend + DAF 2×/35.

    Matches analysis/strategies/momentum/ranked_2_asset_allocation.ipynb logic.

    Parameters
    ----------
    start :
        First date for downloads & simulation ("warmup / history" start). Same role
        as ``data_start`` in older notebook versions (see ``**kwargs``).
    display_start :
        If set, slice ``rot_result`` and ``closes`` to this date onward for plotting.
    rebase_at_display :
        If True, renormalize all series to 100 on the first displayed row.

    **Why long history:** DAF uses an *expanding* vol quantile and monthly path
    dependence. Downloading only from 2016 changes leverage vs the research notebook.
    """
    # Alias used in some notebook cells / older API
    if "data_start" in kwargs:
        start = str(kwargs.pop("data_start"))
    if kwargs:
        raise TypeError(
            "run_v10_daf_backtest() got unexpected keyword arguments: "
            + f"{sorted(kwargs)!r}"
        )

    data_start = start

    closes, ohlc_dict = download_v10_data(start=data_start, end=end)
    if closes.empty or len(closes) < 300:
        raise RuntimeError("Insufficient price data for V10 backtest")

    print(
        f"V10 research pipeline: {len(closes)} days, "
        f"{closes.index[0].date()} -> {closes.index[-1].date()}"
    )

    sector_closes = closes[SECTOR_ETFS].dropna(how="all")
    print("Building sector rank matrices (original mode)...")
    mat_orig = build_rank_matrices(
        sector_closes, ohlc_dict, mode="original", spy_shy_closes=closes
    )

    available_hedge = [t for t in HEDGE_ETFS if t in closes.columns]
    hedge_closes = closes[available_hedge].dropna(how="all")
    print("Building hedge rank matrices (multi-horizon)...")
    mat_hedge = build_rank_matrices(hedge_closes, None, mode="multi", spy_shy_closes=closes)

    hedge_data = {
        "tickers": available_hedge,
        "fwd_rets": mat_hedge["fwd_rets"],
        "dates": mat_hedge["dates"],
    }

    print("Running V4-Best...")
    r_v4_best = adaptive_backtest_v4(
        mat_orig,
        closes,
        weights=(0.1, 0.2, 0.25, 0.45),
        top_n_bull=6,
        top_n_neutral=5,
        top_n_bear=2,
        rebal_freq="M",
        breadth_thr=0.5,
        sizing="equal",
        cash_filter_bull=False,
        cash_filter_neutral=True,
        cash_filter_bear=True,
        conviction_overweight=True,
        conviction_threshold=2.0,
        conviction_top_pct=0.5,
        neutral_mom_threshold=-0.03,
    )

    print("Running V8-AW (research grid-search winner)...")
    r_v8aw_best = adaptive_backtest_v8aw(
        mat_orig,
        hedge_data,
        closes,
        weights=(0.15, 0.15, 0.2, 0.5),
        top_n_bull=4,
        top_n_neutral=4,
        top_n_bear_hedge=3,
        rebal_freq="M",
        breadth_thr=0.6,
        neutral_mom_threshold=-0.03,
        cash_filter_neutral=True,
        hedge_lookback=42,
    )

    if r_v4_best is None or r_v8aw_best is None:
        raise RuntimeError("V4 or V8-AW backtest returned None")

    print("Blending V10 (50/50 daily returns)...")
    r_v10 = ensemble_blend({"V4-Best": r_v4_best, "V8-AW": r_v8aw_best})

    # DAF operates on the raw V10 equity (before transaction costs), matching the research notebook.
    print("Applying DAF 2×/35 on return stream...")
    r_daf = apply_dual_filter_leverage(
        {
            "equity": r_v10["equity"],
            "benchmark": r_v10["benchmark"],
            "allocations": r_v10.get("allocations", []),
        },
        closes,
        max_lev=2.0,
        vol_pct=35,
    )

    allocs = r_v10.get("allocations", [])
    if apply_tc:
        r_v10_tc = apply_transaction_costs(r_v10["equity"], allocs, cost_bps=cost_bps)
        r_v10_tc = r_v10_tc / r_v10_tc.iloc[0]
        r_daf_tc = apply_transaction_costs(r_daf["equity"], allocs, cost_bps=cost_bps)
        r_daf_tc = r_daf_tc / r_daf_tc.iloc[0]
    else:
        r_v10_tc = r_v10["equity"] / r_v10["equity"].iloc[0]
        r_daf_tc = r_daf["equity"] / r_daf["equity"].iloc[0]

    bm = r_v10["benchmark"].reindex(r_v10_tc.index).ffill()
    spy_norm = bm / bm.iloc[0]

    # Align all three to common index
    idx = r_v10_tc.index.intersection(r_daf_tc.index).intersection(spy_norm.index)
    r_v10_tc = r_v10_tc.reindex(idx).ffill()
    r_daf_tc = r_daf_tc.reindex(idx).ffill()
    spy_norm = spy_norm.reindex(idx).ffill()

    rot_result = pd.DataFrame(
        {
            "V10 DAF 2x (Leveraged)": r_daf_tc * 100,
            "V10 Unleveraged": r_v10_tc * 100,
            "SPY Benchmark": spy_norm * 100,
        },
        index=idx,
    )

    closes_out = closes
    if display_start:
        ds = pd.Timestamp(display_start)
        rot_result = rot_result.loc[rot_result.index >= ds].copy()
        if rot_result.empty:
            raise ValueError(f"No rows on/after display_start={display_start!r}")
        if rebase_at_display:
            rot_result = rot_result / rot_result.iloc[0] * 100.0
        closes_out = closes.loc[closes.index >= ds].copy()
        print(
            f"Display window: {rot_result.index[0].date()} -> {rot_result.index[-1].date()} "
            f"(engine history from {data_start}"
            + ("; rebased at display start" if rebase_at_display else "; cumulative levels as in research zoom")
            + ")"
        )

    print(f"Average DAF leverage: {float(r_daf['avg_leverage']):.2f}×")
    print(f"V10 (1×) end mult (full series): {float(r_v10_tc.iloc[-1]):.2f}")
    print(f"V10 DAF end mult (full series): {float(r_daf_tc.iloc[-1]):.2f}")
    print(f"SPY end mult (full series): {float(spy_norm.iloc[-1]):.2f}")

    return {
        "rot_result": rot_result,
        "r_v10": r_v10,
        "r_daf": r_daf,
        "closes": closes_out,
        "data_start": data_start,
        "display_start": display_start,
        "rebase_at_display": rebase_at_display,
    }

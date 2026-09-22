"""Centralized performance metrics for the backtesting platform.

Consolidates and extends the metrics from v10_research_pipeline.py.
All strategies get the same metrics automatically.

Sharpe convention: mean_daily_return / std_daily_return * sqrt(252)
(the standard convention, not CAGR/vol as in the v10 pipeline).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # can be overridden


# ---------------------------------------------------------------------------
# Core return metrics
# ---------------------------------------------------------------------------

def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate.

    Elapsed time is ``len(equity) - 1`` periods, not ``len(equity)``: a curve
    of N marks spans N-1 returns. Using N overstates the holding period and so
    understates CAGR -- by ~0.25% relative over a 400-session run, but by far
    more on the short windows that walk-forward folds are made of.
    """
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    n_years = (len(equity) - 1) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized volatility from daily returns."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio (standard: mean/std * sqrt(252))."""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS_PER_YEAR
    std = float(excess.std())
    # A constant series lands on float residue (~1e-18) rather than exactly 0,
    # so a `std <= 0` guard lets it through and divides by it: a flat 1%/day
    # stream scored a Sharpe of 9e16 and won every selection it entered.
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sortino ratio (downside deviation only).

    Mirrors `sharpe_ratio`: a daily ratio scaled by sqrt(252). The previous
    form multiplied numerator *and* denominator by sqrt(252), so the factors
    cancelled and it returned a daily Sortino sitting next to an annualized
    Sharpe in the same report -- reading ~16x too pessimistic.

    Downside deviation is the root-mean-square shortfall below the target over
    *all* periods. Using `excess[excess < 0].std()` instead would de-mean the
    losses and divide by the loss count, so a strategy that loses rarely but
    deeply would score the same as one that bleeds every day.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS_PER_YEAR
    shortfall = np.minimum(excess.to_numpy(dtype=float), 0.0)
    down_std = float(np.sqrt(np.mean(np.square(shortfall))))
    if not np.isfinite(down_std) or down_std < 1e-12:
        # No downside at all: unbounded if it made money, undefined if flat.
        return float("inf") if float(excess.mean()) > 0 else 0.0
    return float(excess.mean() / down_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (negative number). Returns 0 if no drawdown."""
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Full drawdown time series (negative values during drawdowns)."""
    peak = equity.cummax()
    return (equity - peak) / peak


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar ratio = CAGR / |max_drawdown|."""
    dd = max_drawdown(equity)
    if dd >= 0:
        return 0.0
    return -cagr(equity) / dd


def ulcer_index(equity: pd.Series, window: int = 14) -> float:
    """Ulcer Index — RMS of drawdowns (measures pain of drawdowns)."""
    dd = drawdown_series(equity) * 100  # percentage
    if len(dd) < window:
        return 0.0
    return float(np.sqrt((dd ** 2).rolling(window).mean().iloc[-1]))


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (negative number)."""
    if len(returns) < 10:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall / CVaR (average loss beyond VaR)."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def skewness(returns: pd.Series) -> float:
    if len(returns) < 3:
        return 0.0
    return float(scipy_stats.skew(returns))


def kurtosis(returns: pd.Series) -> float:
    if len(returns) < 4:
        return 0.0
    return float(scipy_stats.kurtosis(returns))


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Portfolio beta vs benchmark."""
    if len(returns) < 10 or len(benchmark_returns) < 10:
        return 0.0
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    var_bench = cov[1, 1]
    if var_bench <= 0:
        return 0.0
    return float(cov[0, 1] / var_bench)


def alpha_jensen(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = RISK_FREE_RATE,
) -> float:
    """Jensen's alpha (annualized)."""
    b = beta(returns, benchmark_returns)
    r_p = float(returns.mean()) * TRADING_DAYS_PER_YEAR
    r_b = float(benchmark_returns.mean()) * TRADING_DAYS_PER_YEAR
    rf_annual = rf
    return r_p - rf_annual - b * (r_b - rf_annual)


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Information ratio = mean(excess) / std(excess) * sqrt(252)."""
    if len(returns) < 10 or len(benchmark_returns) < 10:
        return 0.0
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    std = excess.std()
    if std <= 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def capture_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    up: bool = True,
) -> float:
    """Upside or downside capture ratio."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0
    if up:
        mask = aligned.iloc[:, 1] > 0
    else:
        mask = aligned.iloc[:, 1] < 0

    subset = aligned[mask]
    if len(subset) < 5:
        return 0.0

    port_cum = (1 + subset.iloc[:, 0]).prod() ** (TRADING_DAYS_PER_YEAR / len(subset)) - 1
    bench_cum = (1 + subset.iloc[:, 1]).prod() ** (TRADING_DAYS_PER_YEAR / len(subset)) - 1

    if bench_cum == 0:
        return 0.0
    return float(port_cum / bench_cum)


# ---------------------------------------------------------------------------
# Trade / turnover statistics
# ---------------------------------------------------------------------------

def turnover_from_weights(
    weight_history: List[Dict[str, float]],
) -> float:
    """Average one-way turnover from a list of weight dicts."""
    if len(weight_history) < 2:
        return 0.0

    total_turnover = 0.0
    for i in range(1, len(weight_history)):
        prev = weight_history[i - 1]
        curr = weight_history[i]
        all_syms = set(prev.keys()) | set(curr.keys())
        turnover = sum(abs(curr.get(s, 0) - prev.get(s, 0)) for s in all_syms) / 2
        total_turnover += turnover

    return total_turnover / (len(weight_history) - 1)


def win_rate(returns: pd.Series, frequency: str = "D") -> float:
    """Fraction of periods with positive returns."""
    if frequency == "D":
        positive = returns > 0
    elif frequency == "M":
        monthly = returns.resample("ME").sum()
        positive = monthly > 0
    elif frequency == "Y":
        yearly = returns.resample("YE").sum()
        positive = yearly > 0
    else:
        positive = returns > 0

    if len(positive) == 0:
        return 0.0
    return float(positive.mean())


def profit_factor(returns: pd.Series) -> float:
    """Sum of positive returns / |sum of negative returns|."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Calendar analysis
# ---------------------------------------------------------------------------

def annual_returns(equity: pd.Series) -> pd.Series:
    """Year-by-year returns."""
    yearly = equity.resample("YE").last()
    return yearly.pct_change().dropna()


def monthly_returns(equity: pd.Series) -> pd.Series:
    """Month-by-month returns."""
    monthly = equity.resample("ME").last()
    return monthly.pct_change().dropna()


def monthly_return_table(equity: pd.Series) -> pd.DataFrame:
    """Calendar heatmap: rows=years, columns=months (Jan-Dec)."""
    monthly = equity.resample("ME").last().pct_change().dropna()
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    table = df.pivot_table(index="year", columns="month", values="return")
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(table.columns)]
    return table


def annual_return_table(
    equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Year-by-year returns table with optional benchmark comparison."""
    strat = annual_returns(equity)
    result = pd.DataFrame({"strategy": strat})

    if benchmark_equity is not None and len(benchmark_equity) > 0:
        bench = annual_returns(benchmark_equity)
        result["benchmark"] = bench
        result["excess"] = result["strategy"] - result.get("benchmark", 0)

    return result


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def regime_returns(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    vol_lookback: int = 63,
    vol_percentile: float = 0.75,
) -> Dict[str, Dict[str, float]]:
    """Performance broken down by market regime.

    Regimes:
      - Bull: benchmark return > 0 over lookback
      - Bear: benchmark return <= 0 over lookback
      - High-vol: realized vol in top quartile
      - Low-vol: realized vol in bottom quartile
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < vol_lookback * 2:
        return {}

    bench = aligned.iloc[:, 1]
    port = aligned.iloc[:, 0]

    rolling_ret = bench.rolling(vol_lookback).sum()
    rolling_vol = bench.rolling(vol_lookback).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    vol_threshold = rolling_vol.quantile(vol_percentile)

    regimes = {
        "bull": rolling_ret > 0,
        "bear": rolling_ret <= 0,
        "high_vol": rolling_vol >= vol_threshold,
        "low_vol": rolling_vol < rolling_vol.quantile(1 - vol_percentile),
    }

    result = {}
    for name, mask in regimes.items():
        mask = mask.reindex(port.index).fillna(False)
        subset = port[mask]
        if len(subset) < 20:
            continue
        result[name] = {
            "sharpe": sharpe_ratio(subset),
            "cagr": float(subset.mean() * TRADING_DAYS_PER_YEAR),
            "vol": float(subset.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "max_dd": max_drawdown((1 + subset).cumprod()),
            "days": len(subset),
        }

    return result


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _sharpe_standard_error(returns: pd.Series, sr_daily: float) -> float:
    """Standard error of a daily Sharpe estimate (Bailey & Lopez de Prado).

    se = sqrt((1 - g3*SR + (g4 - 1)/4 * SR^2) / (n - 1))

    g4 is the *non-excess* kurtosis (3 for a Gaussian), so scipy's default
    Fisher (excess) convention has to be turned off — leaving it on flips the
    sign of the SR^2 term and understates the error, inflating confidence.
    """
    n = len(returns)
    if n < 2:
        return 0.0

    skew = float(scipy_stats.skew(returns))
    kurt = float(scipy_stats.kurtosis(returns, fisher=False))

    variance = 1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2
    if variance <= 0:
        return 0.0
    return float(np.sqrt(variance / (n - 1)))


def _expected_max_sharpe(num_trials: int) -> float:
    """E[max] of `num_trials` independent standard-normal draws.

    Bailey & Lopez de Prado's approximation:
      (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/(N*e))
    """
    if num_trials < 2:
        return 0.0
    euler_gamma = 0.5772156649015329
    a = float(scipy_stats.norm.ppf(1.0 - 1.0 / num_trials))
    b = float(scipy_stats.norm.ppf(1.0 - 1.0 / (num_trials * np.e)))
    return (1.0 - euler_gamma) * a + euler_gamma * b


def probabilistic_sharpe(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio — P(true Sharpe > benchmark_sharpe).

    Accounts for skewness and kurtosis of returns.
    Bailey & Lopez de Prado (2012).
    """
    n = len(returns)
    if n < 10:
        return 0.5

    sr = sharpe_ratio(returns) / np.sqrt(TRADING_DAYS_PER_YEAR)  # daily
    sr_bench = benchmark_sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)

    se = _sharpe_standard_error(returns, sr)
    if se <= 0:
        return 0.5

    test_stat = (sr - sr_bench) / se
    return float(scipy_stats.norm.cdf(test_stat))


def deflated_sharpe(
    returns: pd.Series,
    num_trials: int,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio — PSR corrected for multiple testing.

    Bailey & Lopez de Prado (2014). Adjusts the benchmark Sharpe
    based on the expected maximum Sharpe from `num_trials` independent tests.
    """
    if num_trials <= 1:
        return probabilistic_sharpe(returns, benchmark_sharpe)

    n = len(returns)
    if n < 10:
        return 0.5

    sr_daily = sharpe_ratio(returns) / np.sqrt(TRADING_DAYS_PER_YEAR)
    se = _sharpe_standard_error(returns, sr_daily)
    if se <= 0:
        return 0.5

    # The hurdle is the best Sharpe `num_trials` coin flips would produce,
    # measured in standard errors of the Sharpe estimate. Scaling by the
    # return volatility instead would make the multiple-testing penalty depend
    # on leverage: the same strategy run at 3x would face a 3x hurdle despite
    # an identical Sharpe.
    hurdle_daily = _expected_max_sharpe(num_trials) * se
    adjusted_bench = max(
        benchmark_sharpe, hurdle_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    return probabilistic_sharpe(returns, adjusted_bench)


# ---------------------------------------------------------------------------
# Comprehensive report
# ---------------------------------------------------------------------------

def compute_full_metrics(
    equity: pd.Series,
    returns: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    num_trials: int = 1,
) -> Dict[str, Any]:
    """Compute the complete metrics suite.

    Returns a dict with all metrics organized by category.
    """
    metrics: Dict[str, Any] = {}

    # Returns
    metrics["cagr"] = cagr(equity)
    metrics["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0.0
    metrics["annualized_vol"] = annualized_volatility(returns)
    metrics["sharpe"] = sharpe_ratio(returns)
    metrics["sortino"] = sortino_ratio(returns)
    metrics["calmar"] = calmar_ratio(equity)

    # Risk
    metrics["max_drawdown"] = max_drawdown(equity)
    metrics["var_95"] = value_at_risk(returns, 0.95)
    metrics["cvar_95"] = conditional_var(returns, 0.95)
    metrics["skewness"] = skewness(returns)
    metrics["kurtosis"] = kurtosis(returns)
    metrics["ulcer_index"] = ulcer_index(equity)

    # Trade stats
    metrics["daily_win_rate"] = win_rate(returns, "D")
    metrics["monthly_win_rate"] = win_rate(returns, "M")
    metrics["profit_factor"] = profit_factor(returns)

    # Calendar
    metrics["best_day"] = float(returns.max()) if len(returns) > 0 else 0.0
    metrics["worst_day"] = float(returns.min()) if len(returns) > 0 else 0.0
    metrics["best_month"] = float(monthly_returns(equity).max()) if len(equity) > 10 else 0.0
    metrics["worst_month"] = float(monthly_returns(equity).min()) if len(equity) > 10 else 0.0

    # Benchmark
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        metrics["beta"] = beta(returns, benchmark_returns)
        metrics["alpha_jensen"] = alpha_jensen(returns, benchmark_returns)
        metrics["information_ratio"] = information_ratio(returns, benchmark_returns)
        metrics["upside_capture"] = capture_ratio(returns, benchmark_returns, up=True)
        metrics["downside_capture"] = capture_ratio(returns, benchmark_returns, up=False)

        if benchmark_equity is not None:
            bench_cagr = cagr(benchmark_equity)
            metrics["excess_return"] = metrics["cagr"] - bench_cagr
            metrics["benchmark_cagr"] = bench_cagr
            metrics["benchmark_sharpe"] = sharpe_ratio(benchmark_returns)
            metrics["benchmark_max_dd"] = max_drawdown(benchmark_equity)

        # Regime
        metrics["regime"] = regime_returns(returns, benchmark_returns)

    # Statistical
    metrics["psr"] = probabilistic_sharpe(returns)
    metrics["deflated_sharpe"] = deflated_sharpe(returns, num_trials)

    # Metadata
    metrics["n_days"] = len(returns)
    metrics["n_years"] = len(returns) / TRADING_DAYS_PER_YEAR
    metrics["start_date"] = str(equity.index[0].date()) if len(equity) > 0 else ""
    metrics["end_date"] = str(equity.index[-1].date()) if len(equity) > 0 else ""
    metrics["final_equity"] = float(equity.iloc[-1]) if len(equity) > 0 else 0.0

    return metrics


def format_metrics(metrics: Dict[str, Any], label: str = "Strategy") -> pd.DataFrame:
    """Format metrics dict into a display-ready DataFrame."""
    rows = {}

    fmt_pct = lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else str(v)
    fmt_f2 = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
    fmt_f3 = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)

    rows["CAGR"] = fmt_pct(metrics.get("cagr", 0))
    rows["Total Return"] = fmt_pct(metrics.get("total_return", 0))
    rows["Volatility"] = fmt_pct(metrics.get("annualized_vol", 0))
    rows["Sharpe"] = fmt_f3(metrics.get("sharpe", 0))
    rows["Sortino"] = fmt_f3(metrics.get("sortino", 0))
    rows["Calmar"] = fmt_f3(metrics.get("calmar", 0))
    rows["Max Drawdown"] = fmt_pct(metrics.get("max_drawdown", 0))
    rows["Beta"] = fmt_f3(metrics.get("beta", 0))
    rows["Alpha (Jensen)"] = fmt_pct(metrics.get("alpha_jensen", 0))
    rows["Info Ratio"] = fmt_f3(metrics.get("information_ratio", 0))
    rows["PSR"] = fmt_pct(metrics.get("psr", 0))
    rows["Daily Win Rate"] = fmt_pct(metrics.get("daily_win_rate", 0))
    rows["Monthly Win Rate"] = fmt_pct(metrics.get("monthly_win_rate", 0))
    rows["Profit Factor"] = fmt_f2(metrics.get("profit_factor", 0))
    rows["VaR 95%"] = fmt_pct(metrics.get("var_95", 0))
    rows["Best Day"] = fmt_pct(metrics.get("best_day", 0))
    rows["Worst Day"] = fmt_pct(metrics.get("worst_day", 0))

    return pd.DataFrame({label: rows})


# ---------------------------------------------------------------------------
# Notebook API
#
# Four research notebooks each carried their own `perf_metrics`, which is how
# `Sortino` ended up meaning two different things in the same repository. These
# two functions are the single implementation they now import. The return shapes
# are deliberately the ones those notebooks already consumed -- including the
# pre-formatted strings, which belong in a notebook and not in a library, but
# moving the formatting would mean editing every display cell downstream.
# ---------------------------------------------------------------------------

def _equity_from_returns(returns: pd.Series) -> pd.Series:
    """Growth of $1, with the $1 included.

    The leading 1.0 matters: ``cagr`` measures elapsed time as
    ``len(equity) - 1`` because N marks span N-1 returns. Compounding without
    the starting mark would hand it N-1 returns and call that N, understating
    CAGR.
    """
    equity = (1.0 + returns).cumprod()
    if len(equity) == 0:
        return equity
    step = equity.index[1] - equity.index[0] if len(equity) > 1 else pd.Timedelta(days=1)
    head = pd.Series([1.0], index=[equity.index[0] - step])
    return pd.concat([head, equity])


def summary_from_returns(
    returns: pd.Series,
    turnover: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Numeric performance summary from a daily return series.

    Replaces the ``perf_metrics(r, turn=None)`` defined in
    ``analysis/strategies/momentum/tsmom_etf.ipynb``.

    One number moves relative to that local copy. ``Sortino`` now uses the
    root-mean-square shortfall below zero over *every* period
    (``sortino_ratio``), not ``r[r < 0].std()``. The old form de-means the losses
    and divides by the loss count, so a strategy that loses rarely but deeply
    scored the same as one that bleeds daily. Expect the value to change; the
    ranking between streams rarely does.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    equity = _equity_from_returns(r)
    years = len(r) / TRADING_DAYS_PER_YEAR
    monthly = monthly_returns(equity)

    return {
        "CAGR": cagr(equity),
        "Ann vol": annualized_volatility(r),
        "Sharpe": sharpe_ratio(r),
        "Sortino": sortino_ratio(r),
        "Max DD": max_drawdown(equity),
        "Calmar": calmar_ratio(equity),
        "Monthly win rate": float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "Ann turnover": (
            float(turnover.reindex(r.index).sum() / years)
            if turnover is not None and years > 0
            else float("nan")
        ),
    }


def summary_from_equity(
    equity: pd.Series,
    label: str,
    initial: Optional[float] = None,
) -> Dict[str, str]:
    """Display-ready performance summary from an equity curve.

    Replaces the identical ``perf_metrics(series, label, initial)`` that
    ``triple_trigger_momentum``, ``high_pullback_reversion`` and
    ``Draft_52W_mean_reversion`` each defined separately.

    ``initial`` sets the denominator of ``Total Return`` only, and defaults to
    the curve's first value. ``CAGR`` comes from the curve itself via ``cagr``.
    Those agree exactly whenever the curve starts at the funding level, which is
    how all three notebooks build it. They diverge if the first mark is already
    above ``initial``: the old local copies measured the return from ``initial``
    but the elapsed time from the first mark to the last, which charges the
    initial-to-first-mark move to a zero-length period.

    The win-rate key is ``Win Rate`` for all three; ``triple_trigger_momentum``
    previously labelled its column ``Daily Win Rate``. Same number, one name.
    """
    series = equity.dropna()
    if len(series) < 2:
        return {"Label": label}

    base = float(initial) if initial else float(series.iloc[0])
    returns = series.pct_change().dropna()
    total_return = float(series.iloc[-1]) / base - 1.0 if base else float("nan")

    return {
        "Label": label,
        "Total Return": f"{total_return:.1%}",
        "CAGR": f"{cagr(series):.1%}",
        "Sharpe Ratio": f"{sharpe_ratio(returns):.2f}",
        "Max Drawdown": f"{max_drawdown(series):.1%}",
        "Calmar Ratio": f"{calmar_ratio(series):.2f}",
        "Win Rate": f"{win_rate(returns):.1%}",
    }


# ---------------------------------------------------------------------------
# Research pipeline API
#
# Canonical home for these five. strategies.ranks.v10_research_pipeline defined
# them first and now re-imports them from here, so the signatures and return
# shapes must stay stable for the notebooks that call them.
# ---------------------------------------------------------------------------

def compute_stats(equity: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Back-compat wrapper matching v10_research_pipeline.compute_stats."""
    sr = equity.pct_change().dropna()
    br = benchmark.pct_change().dropna()
    n_yr = len(sr) / TRADING_DAYS_PER_YEAR
    if n_yr <= 0:
        return pd.DataFrame()

    cagr_s = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_yr) - 1
    cagr_b = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (1 / n_yr) - 1
    vol_s = annualized_volatility(sr)
    vol_b = annualized_volatility(br)
    sh_s = sharpe_ratio(sr)
    sh_b = sharpe_ratio(br)
    md_s = max_drawdown(equity)
    md_b = max_drawdown(benchmark)
    cal_s = calmar_ratio(equity)
    cal_b = -cagr_b / md_b if md_b < 0 else 0
    wr = win_rate(sr, "M")

    return pd.DataFrame({
        "Strategy": [f"{cagr_s:.1%}", f"{vol_s:.1%}", f"{sh_s:.2f}",
                     f"{md_s:.1%}", f"{cal_s:.2f}", f"{wr:.1%}", f"{equity.iloc[-1]:.2f}"],
        "SPY": [f"{cagr_b:.1%}", f"{vol_b:.1%}", f"{sh_b:.2f}",
                f"{md_b:.1%}", f"{cal_b:.2f}", "-", f"{benchmark.iloc[-1]:.2f}"],
    }, index=["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar",
             "Monthly Win Rate", "Final Value ($1)"])


def compute_full_stats(equity: pd.Series, benchmark: pd.Series) -> Dict[str, Any]:
    """Back-compat wrapper matching v10_research_pipeline.compute_full_stats.

    Returns the *original* capitalised keys (``CAGR``, ``Vol``, ``Sharpe``,
    ``Max DD``, ...), not the lowercase ones from ``compute_full_metrics``.

    A wrapper that keeps the old name but changes the dict schema is not
    back-compatible -- it just moves the breakage from import time, where you
    would see it, to ``stats['Sharpe']``, where you would not. Notebooks index
    these keys directly. Use ``compute_full_metrics`` for the full modern set.
    """
    returns = equity.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()
    m = compute_full_metrics(equity, returns, benchmark, bench_returns)

    return {
        "CAGR": m["cagr"],
        "Vol": m["annualized_vol"],
        "Sharpe": m["sharpe"],
        "Max DD": m["max_drawdown"],
        "Calmar": m["calmar"],
        "Sortino": m["sortino"],
        "Win Rate": m["monthly_win_rate"],
        "Skew": m["skewness"],
        "Kurtosis": m["kurtosis"],
        "PSR": m["psr"],
        "Final": m["final_equity"],
    }


def apply_transaction_costs(equity, allocations, cost_bps=10):
    """Back-compat wrapper matching v10_research_pipeline.apply_transaction_costs."""
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
    """Back-compat wrapper matching v10_research_pipeline.quick_stats."""
    if r is None:
        return (-999, 0, 0, 0, 0, 0)
    eq, bm = r["equity"], r["benchmark"]
    allocs = r.get("allocations", [])
    eq_c = apply_transaction_costs(eq, allocs, cost_bps=cost_bps)
    sr = eq_c.pct_change().dropna()
    n_yr = len(sr) / TRADING_DAYS_PER_YEAR
    if n_yr <= 0:
        return (-999, 0, 0, 0, 0, 0)
    cagr_val = (eq_c.iloc[-1] / eq_c.iloc[0]) ** (1 / n_yr) - 1
    vol = annualized_volatility(sr)
    sh = sharpe_ratio(sr)
    dd = max_drawdown(eq_c)
    strat_yr = eq_c.resample("YE").last().pct_change().dropna()
    spy_yr = bm.resample("YE").last().pct_change().dropna()
    common = strat_yr.index.intersection(spy_yr.index)
    beats = int((strat_yr[common] > spy_yr[common]).sum())
    return (sh, cagr_val, dd, beats, len(common), vol)


#: Keys rendered as percentages, in both the legacy capitalised schema from
#: ``compute_full_stats`` and the lowercase schema from ``compute_full_metrics``.
_PERCENT_KEYS = frozenset({
    'CAGR', 'Vol', 'Win Rate', 'Max DD', 'PSR',
    'cagr', 'annualized_vol', 'daily_win_rate', 'monthly_win_rate',
    'total_return', 'excess_return', 'benchmark_cagr',
    'max_drawdown', 'var_95', 'cvar_95', 'best_day', 'worst_day',
    'best_month', 'worst_month', 'psr', 'deflated_sharpe',
})

#: Keys rendered to three decimal places.
_RATIO_KEYS = frozenset({
    'Sharpe', 'Sortino', 'Calmar',
    'sharpe', 'sortino', 'calmar', 'beta', 'information_ratio',
})

#: Keys rendered as dollar amounts.
_MONEY_KEYS = frozenset({'Final', 'final_equity'})


def fmt_full_stats(stats, label='Strategy'):
    """Format a stats dict for display.

    Accepts either schema: the legacy capitalised keys from
    ``compute_full_stats`` or the lowercase keys from ``compute_full_metrics``.
    Handling both means callers are not silently rendered as bare floats when
    they pass the one this function was not written against.
    """
    d = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            continue  # nested sections (e.g. regime) have their own renderer
        if not isinstance(v, (int, float)):
            d[k] = str(v)
        elif k in _PERCENT_KEYS:
            d[k] = f'{v:.1%}'
        elif k in _RATIO_KEYS:
            d[k] = f'{v:.3f}'
        elif k in _MONEY_KEYS:
            d[k] = f'${v:,.2f}'
        else:
            d[k] = f'{v:.2f}'
    return pd.Series(d, name=label)

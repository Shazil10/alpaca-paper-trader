"""Quantify the Sharpe convention change between v10_research_pipeline and backtest.metrics.

v10_research_pipeline defined Sharpe = CAGR / annualized_vol.
backtest.metrics uses the standard Sharpe = mean(daily) / std(daily) * sqrt(252).

Both divide by the same annualized volatility, so the entire delta comes from the
numerator: CAGR is geometric (compounded), while mean(daily) * 252 is arithmetic.
To second order, CAGR ~= arith + (arith^2 - vol^2) / 2, so the sign of the delta is
not fixed. Variance drag pushes CAGR below the arithmetic mean, while compounding
convexity pushes it above. Low-return/high-vol curves get a higher Sharpe under the
new convention; high-return/low-vol curves get a lower one.

Run: PYTHONPATH=src ./venv/bin/python scripts/compare_sharpe_conventions.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import annualized_volatility, cagr, sharpe_ratio

TRADING_DAYS = 252


def sharpe_old(equity: pd.Series) -> float:
    """The v10_research_pipeline convention: CAGR / annualized vol."""
    r = equity.pct_change().dropna()
    n_yr = len(r) / TRADING_DAYS
    if n_yr <= 0:
        return 0.0
    c = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_yr) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return float(c / vol) if vol > 0 else 0.0


def sharpe_new(equity: pd.Series) -> float:
    """The backtest.metrics convention: mean(daily) / std(daily) * sqrt(252)."""
    return sharpe_ratio(equity.pct_change().dropna())


def random_walk(n_days: int, ann_drift: float, ann_vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    mu = ann_drift / TRADING_DAYS
    sigma = ann_vol / np.sqrt(TRADING_DAYS)
    rets = rng.normal(mu, sigma, n_days)
    dates = pd.bdate_range("2020-01-01", periods=n_days + 1)
    return pd.Series(np.concatenate([[1.0], np.cumprod(1 + rets)]), index=dates)


def spy_from_lake() -> pd.Series | None:
    """Real SPY adj_close from the price lake, if present."""
    lake = Path("data/prices/daily")
    frames = []
    for path in sorted(lake.glob("*.parquet")) + sorted(lake.glob("*.csv")):
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if "symbol" not in df.columns or "adj_close" not in df.columns:
            continue
        spy = df[df["symbol"] == "SPY"][["date", "adj_close"]]
        if not spy.empty:
            frames.append(spy)
    if not frames:
        return None
    all_spy = pd.concat(frames)
    all_spy["date"] = pd.to_datetime(all_spy["date"])
    s = all_spy.set_index("date")["adj_close"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s / s.iloc[0]


def report(label: str, equity: pd.Series) -> tuple[float, float]:
    old, new = sharpe_old(equity), sharpe_new(equity)
    r = equity.pct_change().dropna()
    # A relative delta is meaningless when the old Sharpe is near zero.
    rel = f"{(new - old) / abs(old):>+7.1%}" if abs(old) > 0.2 else "      -"
    print(
        f"{label:<34} n={len(r):>5}  CAGR={cagr(equity):>7.2%}  "
        f"vol={annualized_volatility(r):>6.2%}  "
        f"old={old:>6.3f}  new={new:>6.3f}  "
        f"delta={new - old:>+7.3f} ({rel})"
    )
    return old, new


def main() -> None:
    print("Sharpe convention comparison: CAGR/vol (old) vs mean/std*sqrt(252) (new)\n")

    print("Synthetic random walks with drift (1000 days, fixed seeds):")
    deltas = []
    cases = [
        ("drift 10%, vol 15%", 0.10, 0.15),
        ("drift 15%, vol 20%", 0.15, 0.20),
        ("drift 20%, vol 25%", 0.20, 0.25),
        ("drift 25%, vol 35% (2x levered)", 0.25, 0.35),
        ("drift 5%, vol 10% (low vol)", 0.05, 0.10),
        ("drift 30%, vol 12% (high Sharpe)", 0.30, 0.12),
    ]
    for label, drift, vol in cases:
        old, new = report(label, random_walk(1000, drift, vol, seed=42))
        deltas.append(new - old)

    print("\nSeed stability (drift 15%, vol 20%, seeds 0-9):")
    seed_deltas = []
    for seed in range(10):
        eq = random_walk(1000, 0.15, 0.20, seed=seed)
        seed_deltas.append(sharpe_new(eq) - sharpe_old(eq))
    print(
        f"  delta mean={np.mean(seed_deltas):+.3f}  "
        f"min={np.min(seed_deltas):+.3f}  max={np.max(seed_deltas):+.3f}"
    )

    spy = spy_from_lake()
    if spy is not None:
        print(f"\nReal SPY from the price lake ({spy.index[0].date()} -> {spy.index[-1].date()}):")
        report("SPY buy & hold", spy)
    else:
        print("\n(No SPY series found in data/prices/daily; skipping real-data check.)")

    print(
        f"\nSummary: absolute Sharpe delta across synthetic cases "
        f"{min(deltas):+.3f} to {max(deltas):+.3f}; "
        f"largest magnitude {max(deltas, key=abs):+.3f}."
    )
    print(
        "Sign is not fixed: variance drag raises the new Sharpe on low-return/high-vol\n"
        "curves, while compounding convexity lowers it on high-return/low-vol curves."
    )


if __name__ == "__main__":
    main()

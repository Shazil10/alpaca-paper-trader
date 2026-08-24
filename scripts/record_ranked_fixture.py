"""Record the Ranked sleeve's decisions from the pre-lake yfinance path.

This is the canary baseline for the lake migration. It must be run against the
strategy BEFORE it is wired to the lake, otherwise it records post-change
behaviour and proves nothing.

    PYTHONPATH=src python scripts/record_ranked_fixture.py

Writes tests/fixtures/ranked_decisions.json. Decisions only -- symbols, weights,
regime letters, DAF factor -- never price levels: adj_close agrees between
sources to float32 precision, not bit-exactly, so a price assertion would fail
for a correct reason.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from strategies.ranks import ranked_asset_alloc as ra  # noqa: E402

#: Last completed session at the time of recording. Pinned so the fixture is
#: reproducible; "today" would drift on every run.
AS_OF = pd.Timestamp("2026-08-19")

#: The rotation sleeve fetches ``period="2y"``. Pinning it to an explicit day
#: count lets the lake path request the identical window, so the DAF
#: ``expanding()`` quantile sees the same history and cannot shift.
LOOKBACK_DAYS = 730

OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "ranked_decisions.json"


def download_pinned(as_of: pd.Timestamp, lookback_days: int) -> tuple:
    """Reproduce ``ranked_asset_alloc._download_data`` over a fixed window.

    Mirrors the original call exactly, including the omitted ``auto_adjust``
    argument (which defaults to True, so OHLC comes back adjusted).
    """
    start = as_of - pd.Timedelta(days=lookback_days)
    raw = yf.download(
        ra.ALL_TICKERS,
        start=start.strftime("%Y-%m-%d"),
        end=(as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        group_by="ticker",
    )

    closes = pd.DataFrame()
    for t in ra.ALL_TICKERS:
        try:
            c = raw[t]["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            closes[t] = c
        except Exception:
            pass
    closes = closes.sort_index().dropna(how="all")

    ohlc: dict = {}
    for t in ra.SECTOR_ETFS:
        try:
            df = raw[t][["Open", "High", "Low", "Close"]].dropna()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            ohlc[t] = df
        except Exception:
            pass

    return closes, ohlc


def decisions(closes: pd.DataFrame, ohlc: dict) -> dict:
    """Run the sleeve's decision functions and capture the outcome."""
    sector_closes = closes[ra.SECTOR_ETFS]
    idx = len(closes) - 1

    alloc_v4 = ra._v4_today(sector_closes, ohlc, closes, idx)
    alloc_v8 = ra._v8aw_today(sector_closes, ohlc, closes, idx)
    target = ra._blend(alloc_v4, alloc_v8)
    lev = ra._daf_leverage(closes)

    def rounded(d: dict) -> dict:
        # 6dp: weights are exact rationals (1/n, 0.5), so this is lossless for
        # them while absorbing any float noise.
        return {k: round(float(v), 6) for k, v in sorted(d.items())}

    return {
        "as_of": str(AS_OF.date()),
        "lookback_days": LOOKBACK_DAYS,
        "sessions": int(len(closes)),
        "last_session": str(pd.Timestamp(closes.index[-1]).date()),
        "regime_v4": ra._regime(closes, idx, ra.V4_BREADTH),
        "regime_v8": ra._regime(closes, idx, ra.V8_BREADTH),
        "alloc_v4": rounded(alloc_v4),
        "alloc_v8": rounded(alloc_v8),
        "target": rounded(target),
        "daf_leverage": round(float(lev), 6),
        "target_symbols": sorted(
            s for s, w in target.items() if s != ra.CASH_ETF and w > 1e-6
        ),
    }


def main() -> int:
    print(f"recording Ranked decisions as of {AS_OF.date()} "
          f"({LOOKBACK_DAYS}d lookback) from the yfinance path")

    closes, ohlc = download_pinned(AS_OF, LOOKBACK_DAYS)
    if closes.empty:
        print("no data returned; aborting")
        return 1

    print(f"  {len(closes)} sessions x {len(closes.columns)} tickers, "
          f"OHLC for {len(ohlc)} sectors")

    result = decisions(closes, ohlc)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n  regime V4={result['regime_v4']}  V8={result['regime_v8']}")
    print(f"  DAF leverage: {result['daf_leverage']}x")
    print(f"  target: {result['target']}")
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

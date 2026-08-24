"""Record the Pullback sleeve's decisions from the pre-lake yfinance path.

Canary baseline for the lake migration. Must run BEFORE the strategy is wired to
the lake, or it records post-change behaviour and proves nothing.

    PYTHONPATH=src python scripts/record_pullback_fixture.py

Writes tests/fixtures/pullback_decisions.json.

Records which symbols clear the gates and in what rank order -- the decisions --
plus each candidate's pullback depth to 4dp. Prices agree between sources only to
float32, so exact price equality is never asserted. The 40% entry gate is a hard
threshold, so a name sitting exactly on it can legitimately flip; the canary
reports such boundary cases rather than pretending they cannot happen.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "pullback_decisions.json"

#: Latest completed session held by the lake, so both paths can reach it.
AS_OF = pd.Timestamp("2026-08-21")

#: Mirrors DATA_PERIOD="2y" in the strategy.
LOOKBACK_DAYS = 730

#: Mirrors period="1y" in check_regime().
REGIME_LOOKBACK_DAYS = 365

CHUNK = 300


def _mod():
    """Import the number-prefixed implementation module."""
    from importlib import import_module

    return import_module("strategies.mean_reversion.52W_mean_reversion_strat")


def regime_detail(m, as_of: pd.Timestamp) -> dict:
    """Reproduce check_regime() over a pinned window."""
    start = as_of - pd.Timedelta(days=REGIME_LOOKBACK_DAYS)
    data = yf.download(
        list(m.REGIME_ETFS),
        start=start.strftime("%Y-%m-%d"),
        end=(as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    above, detail = 0, {}
    for etf in m.REGIME_ETFS:
        try:
            s = data[etf]["Close"].dropna()
        except (KeyError, IndexError):
            detail[etf] = None
            continue
        if len(s) < m.REGIME_SMA_PERIOD:
            detail[etf] = None
            continue
        is_above = bool(float(s.iloc[-1]) > m._sma(s, m.REGIME_SMA_PERIOD))
        detail[etf] = is_above
        above += int(is_above)

    return {
        "above_count": above,
        "per_etf": detail,
        "risk_on": bool(above >= m.REGIME_MIN_ABOVE),
    }


def score_universe(m, as_of: pd.Timestamp) -> pd.DataFrame:
    """Reproduce _score_universe() over a pinned window."""
    start = as_of - pd.Timedelta(days=LOOKBACK_DAYS)
    tickers = pd.read_csv(REPO_ROOT / "universe.csv")["Symbol"].tolist()

    records = []
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i : i + CHUNK]
        print(f"  scoring {i + 1}..{min(i + CHUNK, len(tickers))} of {len(tickers)}")
        try:
            data = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=(as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception:
            continue

        for sym in batch:
            try:
                s = data[sym]["Close"].dropna()
                if len(s) < max(m.HIGH_LOOKBACK, m.MA_LOOKBACK):
                    continue
                cur = float(s.iloc[-1])
                high52 = m._high_52w(s)
                ma200 = m._sma(s, m.MA_LOOKBACK)
                if cur <= 0 or np.isnan(high52) or np.isnan(ma200):
                    continue

                pullback = (high52 - cur) / high52
                if pullback < m.ENTRY_DEPTH:
                    continue
                if cur < m.MA_FREEFALL_RATIO * high52:
                    continue
                if cur < m.MA_BROKEN_RATIO * ma200:
                    continue

                records.append(
                    {
                        "symbol": sym,
                        "pullback": round(pullback, 4),
                        "price": round(cur, 4),
                        "high52": round(high52, 4),
                        "ma200": round(ma200, 4),
                    }
                )
            except (KeyError, IndexError):
                continue

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values("pullback", ascending=False).reset_index(drop=True)


def main() -> int:
    m = _mod()
    print(f"recording Pullback decisions as of {AS_OF.date()} from the yfinance path")

    regime = regime_detail(m, AS_OF)
    print(f"  regime: {regime['above_count']}/3 above SMA -> "
          f"{'RISK-ON' if regime['risk_on'] else 'RISK-OFF'}")

    scores = score_universe(m, AS_OF)
    print(f"  {len(scores)} candidate(s) cleared the gates")

    # Names within a hair of the 40% gate: these can legitimately flip between
    # sources, so the canary treats them as advisory rather than binding.
    boundary = []
    if not scores.empty:
        near = scores[(scores["pullback"] - m.ENTRY_DEPTH).abs() < 0.005]
        boundary = sorted(near["symbol"].tolist())

    payload = {
        "as_of": str(AS_OF.date()),
        "lookback_days": LOOKBACK_DAYS,
        "regime_lookback_days": REGIME_LOOKBACK_DAYS,
        "entry_depth": m.ENTRY_DEPTH,
        "top_n": m.TOP_N,
        "regime": regime,
        "candidate_count": int(len(scores)),
        "candidates": [] if scores.empty else scores.to_dict("records"),
        "ranked_symbols": [] if scores.empty else scores["symbol"].tolist(),
        "top_picks": [] if scores.empty else scores["symbol"].head(m.TOP_N).tolist(),
        "boundary_symbols": boundary,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"  top {m.TOP_N}: {payload['top_picks']}")
    if boundary:
        print(f"  within 0.5% of the {m.ENTRY_DEPTH:.0%} gate: {boundary}")
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

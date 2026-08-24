"""Decide, objectively, whether PRICE_SOURCE can be flipped to `lake`.

    PYTHONPATH=src python scripts/check_lake_readiness.py

Exit code 0 means every check passed and the flip is safe. Exit code 1 means it
is not, and says which check failed and why it matters.

This exists so the Phase 6 cutover is a command rather than a feeling. Each check
below corresponds to a failure that was actually observed and diagnosed during
the migration -- none of them are hypothetical.
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

from data_pipeline import registry, store  # noqa: E402

#: Symbols known not to heal: renamed or taken private during the window.
#: Treated as an accepted baseline rather than a failure.
KNOWN_UNRESOLVED = {"EA", "EQR", "FISV", "MNST", "RUSHA"}

#: A weekend plus a public holiday.
MAX_SESSION_AGE_DAYS = 4

#: Fraction of *present* equities that must carry a full 52-week lookback.
MIN_EQUITY_COVERAGE = 0.98

#: Fraction of universe.csv that must exist in the lake at all. Below 100%
#: because delistings and renames are continuous and expected -- yfinance simply
#: has no history to give for those names.
MIN_EQUITY_PRESENCE = 0.95


class Check:
    def __init__(self):
        self.results = []
        self.failed = 0

    def add(self, ok: bool, name: str, detail: str, why: str = "") -> None:
        self.results.append((ok, name, detail, why))
        if not ok:
            self.failed += 1

    def report(self) -> int:
        width = max(len(n) for _, n, _, _ in self.results)
        print()
        for ok, name, detail, why in self.results:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<{width}}  {detail}")
            if not ok and why:
                print(f"         why it matters: {why}")

        print()
        if self.failed:
            print(f"NOT READY — {self.failed} check(s) failed. Keep PRICE_SOURCE=yfinance.")
        else:
            print("READY — safe to set PRICE_SOURCE=lake in .github/workflows/run_bot.yml")
        return 1 if self.failed else 0


def main() -> int:
    check = Check()

    frame = store.load_prices()
    if len(frame) == 0:
        print("Price lake is empty. Run: PYTHONPATH=src python src/data_pipeline/sync_prices.py")
        return 1

    symbols = int(frame["symbol"].nunique())
    newest = pd.Timestamp(frame["date"].max())
    age = int((pd.Timestamp.now().normalize() - newest).days)

    check.add(
        len(frame) > 0 and symbols > 500,
        "lake populated",
        f"{len(frame):,} rows across {symbols:,} symbols",
    )

    check.add(
        age <= MAX_SESSION_AGE_DAYS,
        "bars are current",
        f"newest session {newest.date()} ({age} day(s) old, limit {MAX_SESSION_AGE_DAYS})",
        "stale bars mean the sync is not actually running in CI",
    )

    # No bar may be dated today: the trading job runs at 09:30 ET, so a
    # same-day bar would be a partial intraday quote stored as a close.
    today = pd.Timestamp.now().normalize()
    check.add(
        newest < today,
        "no partial session",
        f"newest {newest.date()} < today {today.date()}",
        "a 09:35 quote persisted as a daily close corrupts every rolling window",
    )

    gaps = store.find_gaps()
    unexpected = {s: d for s, d in gaps.items() if s not in KNOWN_UNRESOLVED}
    check.add(
        not unexpected,
        "no unexpected gaps",
        f"{len(gaps)} gapped ({len(unexpected)} unexpected)"
        + (f": {sorted(unexpected)[:8]}" if unexpected else ""),
        "one missing bar makes rolling(50) NaN, silently dropping a symbol "
        "from threshold comparisons and changing signals",
    )

    # --- Rotation sleeve ---
    from strategies.ranks import ranked_asset_alloc as ra

    etf_cov = store.coverage(ra.ALL_TICKERS)
    missing_etfs = [t for t in ra.ALL_TICKERS if t not in set(etf_cov["symbol"])]
    check.add(
        not missing_etfs,
        "rotation ETFs present",
        f"{len(etf_cov)}/{len(ra.ALL_TICKERS)} present"
        + (f", missing {missing_etfs}" if missing_etfs else ""),
        "the sleeve declines to act if any sector ETF is absent",
    )

    bar_counts = sorted(set(etf_cov["bars"])) if len(etf_cov) else []
    check.add(
        len(bar_counts) == 1,
        "rotation ETFs aligned",
        f"bar counts: {bar_counts}",
        "unequal counts mean NaNs in the panel; this is exactly what flipped "
        "the regime from bull to neutral during migration",
    )

    matrix = store.load_close_matrix(ra.ALL_TICKERS)
    if not matrix.empty:
        sma50 = matrix[[c for c in ra.SECTOR_ETFS if c in matrix.columns]].rolling(50).mean().iloc[-1]
        bad = sorted(sma50[sma50.isna()].index)
        check.add(
            not bad,
            "sector averages usable",
            f"{len(bad)} sector(s) with NaN 50-day average"
            + (f": {bad}" if bad else ""),
            "a NaN average makes the sector fail its breadth test invisibly",
        )

    window_start = pd.Timestamp(newest) - pd.Timedelta(days=ra.LOOKBACK_DAYS)
    ranked_window = store.load_close_matrix(ra.ALL_TICKERS, start=window_start, end=newest)
    check.add(
        len(ranked_window) >= ra.MIN_SESSIONS,
        "rotation lookback covered",
        f"{len(ranked_window)} sessions (needs {ra.MIN_SESSIONS})",
        "below this the sleeve emits nothing rather than rotating on bad data",
    )

    # --- Pullback sleeve ---
    #
    # Measured against symbols the lake actually holds, not against the raw
    # universe. Some universe names are mid-delisting and yfinance itself
    # returns a single bar for them, so demanding full coverage of the whole
    # universe would fail permanently on a condition nothing can fix -- and
    # would say nothing about whether the lake matches its source.
    universe_path = REPO_ROOT / "universe.csv"
    if universe_path.exists():
        tickers = {
            registry.normalize_symbol(t)
            for t in pd.read_csv(universe_path)["Symbol"].tolist()
        }
        eq_cov = store.coverage(sorted(tickers))
        present = len(eq_cov)
        absent = len(tickers) - present

        check.add(
            present / max(len(tickers), 1) >= MIN_EQUITY_PRESENCE,
            "equity names present",
            f"{present}/{len(tickers)} in lake, {absent} absent "
            f"({present / max(len(tickers), 1):.1%}, needs {MIN_EQUITY_PRESENCE:.0%})",
            "absent names are usually delisted or renamed; a sharp rise means "
            "universe.csv has drifted from what the broker can trade",
        )

        enough = int((eq_cov["bars"] >= 252).sum()) if present else 0
        frac = enough / max(present, 1)
        check.add(
            frac >= MIN_EQUITY_COVERAGE,
            "equity lookback covered",
            f"{enough}/{present} present names have >=252 bars ({frac:.1%}, "
            f"needs {MIN_EQUITY_COVERAGE:.0%})",
            "the 52-week high gate silently skips any name short of 252 bars",
        )

        # Informational: a single-bar symbol is invisible to gap detection,
        # since gaps are only measured between a symbol's own first and last
        # bar. Verified against yfinance -- these reflect the source, not a
        # lake fault -- but worth surfacing so a real regression here is not
        # mistaken for normal churn.
        stubs = sorted(eq_cov.loc[eq_cov["bars"] <= 2, "symbol"]) if present else []
        if stubs:
            print(f"\n  note: {len(stubs)} symbol(s) hold <=2 bars "
                  f"(delisting/renaming, matches yfinance): {stubs[:12]}")

    # --- Canaries ---
    for label, fixture in (
        ("rotation canary fixture", "ranked_decisions.json"),
        ("pullback canary fixture", "pullback_decisions.json"),
    ):
        path = REPO_ROOT / "tests" / "fixtures" / fixture
        ok = path.exists()
        detail = "present" if ok else f"missing {path.relative_to(REPO_ROOT)}"
        if ok:
            try:
                payload = json.loads(path.read_text())
                detail = f"recorded as of {payload.get('as_of')}"
            except Exception:
                ok, detail = False, "unreadable"
        check.add(ok, label, detail, "the canary is the only proof of parity")

    code = check.report()
    if code == 0:
        print("\nThen confirm parity one more time:")
        print("  PYTHONPATH=src python -m pytest tests/test_ranked_canary.py "
              "tests/test_pullback_canary.py -q")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

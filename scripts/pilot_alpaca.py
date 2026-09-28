#!/usr/bin/env python
"""Phase 7 pilot: Alpaca SIP against Yahoo on hand-picked hard cases.

    PYTHONPATH=src ./venv/bin/python scripts/pilot_alpaca.py
    PYTHONPATH=src ./venv/bin/python scripts/pilot_alpaca.py --symbols AAPL NVDA

Nothing here writes the canonical lake. Both providers' bars are held in
``data/prices/quarantine/{alpaca,yahoo}/pilot_<date>.csv`` and the findings go
to ``data/audits/pilot_alpaca_<date>.{json,txt}``. The rebuild is only worth
running once this report says the provider behaves as the pipeline assumes.

The symbols, and what each one tests
------------------------------------
======== =============================================================
AAPL     the easy case: long history, 4:1 split in 2020
KO       a steady dividend payer: adj_close vs close drift, both vendors
NVDA     4:1 (2021-07-20) and 10:1 (2024-06-10) splits: is ``close`` raw?
META     renamed from FB on 2022-06-09: does ``asof`` carry FB's history?
FB       the old ticker, queried directly
SPY      the ETF seam: how far apart are the two vendors' adj anchors?
ARM      a recent IPO (2023-09-14): first bar on the listing date?
<SP400>  a current S&P 400 member, from the capture (or a fixed default)
<SP600>  a current S&P 600 member, from the capture (or a fixed default)
ATVI     delisted on its acquisition (2023-10-13): is history retained?
BRK-B    share-class symbol: BRK.B on the wire, BRK-B in the lake
======== =============================================================

Questions answered per symbol
-----------------------------
* coverage: first/last session and row count from each vendor, sessions one
  has and the other does not
* raw close: median Alpaca/Yahoo ratio by year. ~1 means both are raw or both
  split-adjusted; the split ratio before a split means Yahoo's ``close`` is
  split-adjusted and Alpaca's is the print
* adjusted returns: share of daily adj_close returns that differ by >0.5%
* anchor: Alpaca/Yahoo adj_close ratio at the first and last common session --
  the size of the splice the 2016 seam needs
* Alpaca's own diagnostics: rows dropped for a missing adjusted or raw side
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_pipeline import membership_capture as capture  # noqa: E402
from data_pipeline import quarantine, schema  # noqa: E402
from data_pipeline.providers.alpaca import AlpacaProvider  # noqa: E402
from data_pipeline.providers.base import ProviderError, market_date  # noqa: E402
from data_pipeline.providers.yahoo import YahooProvider  # noqa: E402
from data_pipeline.sync_prices import LOOKBACK_START  # noqa: E402

logger = logging.getLogger("pilot_alpaca")

AUDIT_DIR = REPO_ROOT / "data" / "audits"

FIXED = ["AAPL", "KO", "NVDA", "META", "FB", "SPY", "ARM", "ATVI", "BRK-B"]
DEFAULT_SP400 = "RGLD"
DEFAULT_SP600 = "AEIS"

#: Known splits the report checks the raw close against.
KNOWN_SPLITS = {
    "NVDA": [("2021-07-20", 4.0), ("2024-06-10", 10.0)],
    "AAPL": [("2020-08-31", 4.0)],
}

RETURN_TOLERANCE = 0.005


def pilot_symbols(current_path: Optional[Path] = None) -> List[str]:
    """The fixed hard cases plus one current S&P 400 and one S&P 600 member."""
    current = capture.load_current(current_path)
    picks = []
    for index, default in (("SP400", DEFAULT_SP400), ("SP600", DEFAULT_SP600)):
        members = sorted(current.loc[current["index"] == index, "symbol"])
        picks.append(members[len(members) // 2] if members else default)
    return FIXED + picks


def compare_symbol(symbol: str, a: pd.DataFrame, y: pd.DataFrame) -> Dict[str, object]:
    """Alpaca (``a``) against Yahoo (``y``) for one symbol."""
    out: Dict[str, object] = {}
    for label, frame in (("alpaca", a), ("yahoo", y)):
        out[label] = {
            "rows": int(len(frame)),
            "first": str(frame[schema.DATE].min().date()) if len(frame) else None,
            "last": str(frame[schema.DATE].max().date()) if len(frame) else None,
        }
    if len(a) == 0 or len(y) == 0:
        out["verdict"] = "one vendor returned nothing"
        return out

    key = [schema.DATE, schema.SYMBOL]
    both = a.merge(y, on=key, suffixes=("_a", "_y"))
    a_dates, y_dates = set(a[schema.DATE]), set(y[schema.DATE])
    out["sessions_only_alpaca"] = sorted(str(d.date()) for d in a_dates - y_dates)[:20]
    out["sessions_only_yahoo"] = sorted(str(d.date()) for d in y_dates - a_dates)[:20]
    out["sessions_only_alpaca_count"] = len(a_dates - y_dates)
    out["sessions_only_yahoo_count"] = len(y_dates - a_dates)
    out["common_sessions"] = int(len(both))
    if len(both) == 0:
        out["verdict"] = "no common sessions"
        return out

    ratio = both["close_a"] / both["close_y"]
    out["raw_close_ratio_by_year"] = {
        str(year): round(float(v), 4)
        for year, v in ratio.groupby(both[schema.DATE].dt.year).median().items()
    }
    vol = both["volume_a"].astype("float64") / both["volume_y"].astype("float64").replace(0, np.nan)
    out["volume_ratio_median"] = round(float(vol.median()), 4) if vol.notna().any() else None

    both = both.sort_values(schema.DATE)
    ret_a = both["adj_close_a"].pct_change()
    ret_y = both["adj_close_y"].pct_change()
    diff = (ret_a - ret_y).abs().dropna()
    out["adj_return_pairs"] = int(len(diff))
    out["adj_return_disagreement_rate"] = round(float((diff > RETURN_TOLERANCE).mean()), 5) if len(diff) else None
    worst = both.assign(_d=(ret_a - ret_y).abs()).nlargest(5, "_d")
    out["adj_return_worst"] = [
        {"date": str(r[schema.DATE].date()), "alpaca": round(float(r["adj_close_a"]), 4),
         "yahoo": round(float(r["adj_close_y"]), 4)}
        for _, r in worst.iterrows()
    ]
    anchor = both["adj_close_a"] / both["adj_close_y"]
    out["adj_anchor_ratio_first"] = round(float(anchor.iloc[0]), 6)
    out["adj_anchor_ratio_last"] = round(float(anchor.iloc[-1]), 6)

    checks = []
    for when, r in KNOWN_SPLITS.get(symbol, []):
        before = both[both[schema.DATE] < pd.Timestamp(when)].tail(1)
        if len(before) == 0:
            continue
        pre_ratio = float(before["close_a"].iloc[0] / before["close_y"].iloc[0])
        checks.append({
            "split": when, "ratio": r,
            "alpaca_over_yahoo_close_before": round(pre_ratio, 4),
            # A pre-split ratio equal to the split ratio means Alpaca carries the
            # print and Yahoo the split-adjusted price; ~1 means they agree.
            "yahoo_close_is_split_adjusted": bool(abs(pre_ratio - r) / r < 0.02),
            "closes_agree": bool(abs(pre_ratio - 1.0) < 0.02),
        })
    if checks:
        out["split_checks"] = checks
    return out


def run(symbols: List[str], start: pd.Timestamp) -> Dict[str, object]:
    alpaca = AlpacaProvider()
    if not alpaca.has_credentials:
        raise SystemExit(
            "Alpaca credentials not found. Add ALPACA_KEY and ALPACA_SECRET to the "
            "environment (or to a local, gitignored .env) and re-run."
        )
    yahoo = YahooProvider()
    end = alpaca.session_cutoff()
    label = f"pilot_{market_date():%Y%m%d}"

    a_frames, y_frames, diagnostics = [], [], {}
    for symbol in symbols:
        bars, failed = alpaca.fetch_batch([symbol], start, end)
        report = alpaca.last_report
        diagnostics[symbol] = {
            "failed": failed,
            "missing_adjusted": dict(report.missing_adjusted) if report else {},
            "missing_raw": dict(report.missing_raw) if report else {},
            "pages": report.pages if report else 0,
            "retries": report.retries if report else 0,
        }
        a_frames.append(bars)
        try:
            ybars, _ = yahoo.fetch_batch([symbol], start, end)
        except Exception as exc:  # the comparison source failing is a finding, not a stop
            logger.warning("yahoo failed for %s: %s", symbol, exc)
            ybars = schema.empty_frame()
        y_frames.append(ybars)

    a_all = schema.coerce(pd.concat(a_frames, ignore_index=True))
    y_all = schema.coerce(pd.concat(y_frames, ignore_index=True))
    quarantine.hold(a_all, "alpaca", label, reason="phase 7 pilot", provenance=alpaca.provenance())
    quarantine.hold(y_all, "yahoo", label, reason="phase 7 pilot comparison",
                    provenance=yahoo.provenance())

    per_symbol = {}
    for symbol in symbols:
        per_symbol[symbol] = compare_symbol(
            symbol,
            a_all[a_all[schema.SYMBOL] == symbol],
            y_all[y_all[schema.SYMBOL] == symbol],
        )
        per_symbol[symbol]["alpaca_diagnostics"] = diagnostics[symbol]

    corporate: Dict[str, object] = {}
    try:
        changes = alpaca.fetch_symbol_changes(start, end, symbols=["META", "FB"])
        corporate["name_changes_meta_fb"] = changes.astype(str).to_dict("records")
        splits = alpaca.fetch_corporate_actions(
            types=["forward_split", "reverse_split"], start=start, end=end,
            symbols=["NVDA", "AAPL"],
        )
        corporate["splits"] = {k: len(v) for k, v in splits.items()}
    except ProviderError as exc:
        corporate["error"] = str(exc)

    return {
        "label": label,
        "window": {"start": str(start.date()), "end_exclusive": str(end.date())},
        "alpaca_provenance": alpaca.provenance(),
        "symbols": per_symbol,
        "corporate_actions": corporate,
        "canonical_lake_written": False,
    }


def render(result: Dict[str, object]) -> str:
    lines = [f"Alpaca SIP pilot {result['label']}  window {result['window']}", ""]
    for symbol, info in result["symbols"].items():
        a, y = info["alpaca"], info["yahoo"]
        lines.append(f"{symbol}")
        lines.append(f"  alpaca {a['rows']:>5} rows {a['first']}..{a['last']}")
        lines.append(f"  yahoo  {y['rows']:>5} rows {y['first']}..{y['last']}")
        if "adj_return_disagreement_rate" in info:
            lines.append(
                f"  adj returns disagree >0.5% on {info['adj_return_disagreement_rate']:.2%} "
                f"of {info['adj_return_pairs']} pairs; anchor ratio "
                f"{info['adj_anchor_ratio_first']} -> {info['adj_anchor_ratio_last']}"
            )
            lines.append(f"  raw close ratio by year {info['raw_close_ratio_by_year']}")
        for check in info.get("split_checks", []):
            lines.append(
                f"  split {check['split']} {check['ratio']}:1 -> alpaca/yahoo close before "
                f"= {check['alpaca_over_yahoo_close_before']} "
                f"(yahoo split-adjusted: {check['yahoo_close_is_split_adjusted']})"
            )
        diag = info["alpaca_diagnostics"]
        if diag["missing_adjusted"] or diag["missing_raw"]:
            lines.append(f"  alpaca join orphans: adj {diag['missing_adjusted']} raw {diag['missing_raw']}")
        lines.append("")
    lines.append(f"corporate actions: {result['corporate_actions']}")
    lines.append("canonical lake written: no")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Alpaca SIP pilot (no canonical writes).")
    parser.add_argument("--symbols", nargs="+", help="Override the pilot list.")
    parser.add_argument("--start", default=str(LOOKBACK_START.date()))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    symbols = args.symbols or pilot_symbols()
    result = run([s.upper() for s in symbols], pd.Timestamp(args.start))

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    base = AUDIT_DIR / f"pilot_alpaca_{result['label'].split('_', 1)[1]}"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    text = render(result)
    base.with_suffix(".txt").write_text(text)
    print(text)
    print(f"wrote {base}.json and {base}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

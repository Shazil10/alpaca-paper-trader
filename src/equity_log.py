"""equity_log.py

Append one row per run to reports/equity_daily.csv with account-level and
per-sleeve mark-to-market estimates.

Call ``log_equity()`` at the end of trade.py and report.py so each daily
run produces a time-stamped snapshot.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import config
from trade_models import _enum_str

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
CSV_PATH = REPORTS_DIR / "equity_daily.csv"

STRATEGY_PREFIXES: Dict[str, str] = {
    "strategies.momentum.clenow_trend": "Clenow Trend",
    "strategies.ranks.ranked_asset_alloc": "Ranked Asset Alloc",
    "strategies.mean_reversion.high_pullback_reversion": "High Pullback Reversion",
}

HEADER = [
    "timestamp_et",
    "date",
    "equity",
    "cash",
    "buying_power",
    "clenow_mv",
    "ranked_mv",
    "mean_rev_mv",
    "discretionary_mv",
    "total_position_mv",
]


def _classify_position(client, symbol: str) -> str:
    """Attribute a position to a sleeve by scanning recent orders."""
    try:
        for order in client.get_orders():
            sym = str(getattr(order, "symbol", "")).strip().upper()
            if sym != symbol:
                continue
            cid = str(getattr(order, "client_order_id", "") or "")
            side = _enum_str(getattr(order, "side", "")).upper()
            status = _enum_str(getattr(order, "status", "")).lower()
            if side != "BUY" or status != "filled":
                continue
            for prefix, name in STRATEGY_PREFIXES.items():
                if cid.startswith(f"{prefix}:"):
                    return name
    except Exception:
        pass
    return "Discretionary"


def log_equity() -> None:
    """Fetch account + positions and append a row to equity_daily.csv."""
    try:
        client = config.get_client()
    except Exception:
        logger.exception("equity_log: cannot create Alpaca client")
        return

    try:
        acct = client.get_account()
    except Exception:
        logger.exception("equity_log: cannot fetch account")
        return

    equity = float(getattr(acct, "equity", 0) or 0)
    cash = float(getattr(acct, "cash", 0) or 0)
    buying_power = float(getattr(acct, "buying_power", 0) or 0)

    sleeve_mv: Dict[str, float] = {
        "Clenow Trend": 0.0,
        "Ranked Asset Alloc": 0.0,
        "High Pullback Reversion": 0.0,
        "Discretionary": 0.0,
    }
    total_mv = 0.0

    try:
        positions = client.get_all_positions()
        all_orders = list(client.get_orders())

        for pos in positions:
            sym = str(getattr(pos, "symbol", "")).strip().upper()
            mv = float(getattr(pos, "market_value", 0) or 0)
            total_mv += mv

            sleeve = "Discretionary"
            for order in all_orders:
                o_sym = str(getattr(order, "symbol", "")).strip().upper()
                if o_sym != sym:
                    continue
                cid = str(getattr(order, "client_order_id", "") or "")
                o_side = _enum_str(getattr(order, "side", "")).upper()
                o_status = _enum_str(getattr(order, "status", "")).lower()
                if o_side == "BUY" and o_status == "filled":
                    for prefix, name in STRATEGY_PREFIXES.items():
                        if cid.startswith(f"{prefix}:"):
                            sleeve = name
                            break
                    if sleeve != "Discretionary":
                        break
            sleeve_mv[sleeve] += mv
    except Exception:
        logger.exception("equity_log: cannot fetch positions")

    et = ZoneInfo("America/New_York")
    now = datetime.now(tz=et)

    row = [
        now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        now.strftime("%Y-%m-%d"),
        f"{equity:.2f}",
        f"{cash:.2f}",
        f"{buying_power:.2f}",
        f"{sleeve_mv['Clenow Trend']:.2f}",
        f"{sleeve_mv['Ranked Asset Alloc']:.2f}",
        f"{sleeve_mv['High Pullback Reversion']:.2f}",
        f"{sleeve_mv['Discretionary']:.2f}",
        f"{total_mv:.2f}",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(HEADER)
        w.writerow(row)

    logger.info(
        "equity_log: equity=$%.2f cash=$%.2f positions_mv=$%.2f → %s",
        equity, cash, total_mv, CSV_PATH,
    )

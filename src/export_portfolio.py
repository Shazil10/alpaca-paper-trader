"""export_portfolio.py

Regenerate the microsite's data feed from the live Alpaca account.

Outputs two JSON artifacts from the same live read:

    reports/microsite_snapshot.json
        Public snapshot consumed by the Lovable site (calm-execution-engine).
        Fetched at runtime from raw.githubusercontent.com on this repo.

    microsite/client/public/data/portfolio.json
        Legacy contract for the in-repo React microsite, when present.

Read-only with respect to trading — it fetches account state and never
submits an order.

Inputs
    portfolio.meta.json          narrative copy, human-edited
    config.STRATEGY_ALLOCATIONS  lifetime budget per strategy
    Alpaca                       open positions + full order history

Usage
    PYTHONPATH=src python src/export_portfolio.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from order_ledger import build_ledger, held_symbols_by_strategy, realized_pnl_by_strategy
from trade_models import committed_dollars_from_orders, fetch_all_orders, safe_float

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python 3.8 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "reports" / "microsite_snapshot.json"
MICROSITE_DATA = REPO_ROOT / "microsite" / "client" / "public" / "data"
MICROSITE_OUT = MICROSITE_DATA / "portfolio.json"

# Meta may live in reports/ (CI) or beside the local microsite dev tree.
META_CANDIDATES = (
    REPO_ROOT / "reports" / "portfolio.meta.json",
    MICROSITE_DATA / "portfolio.meta.json",
)

PNL_BASIS = "realized_closed_trades"
PNL_SCOPE = "full_alpaca_order_history"
SCHEMA_VERSION = 1


def _resolve_meta_path() -> Optional[Path]:
    for path in META_CANDIDATES:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Alpaca snapshot
# ---------------------------------------------------------------------------

def _fetch_snapshot(client) -> Tuple[Dict[str, object], list, list]:
    """Return (positions_by_symbol, ledger_rows, raw_orders)."""
    positions: Dict[str, object] = {}
    try:
        for p in client.get_all_positions():
            symbol = str(getattr(p, "symbol", "")).strip().upper()
            if symbol:
                positions[symbol] = p
    except Exception:
        logger.exception("Could not fetch positions")

    try:
        orders = fetch_all_orders(client, status="all")
    except Exception:
        logger.exception("Could not fetch order history")
        orders = []

    return positions, build_ledger(orders), orders


def _order_window(orders: list) -> Tuple[Optional[str], Optional[str]]:
    """Earliest and latest submitted_at dates across fetched orders."""
    dates: List[str] = []
    for o in orders:
        ts = getattr(o, "submitted_at", None)
        if ts is None:
            continue
        if hasattr(ts, "strftime"):
            dates.append(ts.strftime("%Y-%m-%d"))
        else:
            s = str(ts)
            dates.append(s[:10] if len(s) >= 10 else s)
    if not dates:
        return None, None
    return min(dates), max(dates)


def _money_display(value: float) -> str:
    return f"${round(value):,}"


def _attributed_holdings(
    module: str,
    bought_by_strategy: Dict[str, Set[str]],
    positions: Dict[str, object],
) -> Set[str]:
    held: Set[str] = set()
    for symbol in bought_by_strategy.get(module, set()) & set(positions.keys()):
        qty = safe_float(getattr(positions[symbol], "qty", 0.0))
        if qty > 0:
            held.add(symbol)
        else:
            logger.info("Omitting %s from %s book (qty=%.4f)", symbol, module, qty)
    return held


def _holding_rows(symbol: str, pos: object) -> dict:
    qty = round(safe_float(getattr(pos, "qty", 0.0)), 4)
    market_value = round(safe_float(getattr(pos, "market_value", 0.0)), 2)
    current_price = safe_float(getattr(pos, "current_price", 0.0))
    unrealized = safe_float(getattr(pos, "unrealized_pl", 0.0))
    row = {
        "symbol": symbol,
        "qty": qty,
        "market_value": market_value,
    }
    if current_price:
        row["current_price"] = round(current_price, 4)
    if unrealized:
        row["unrealized_pl"] = round(unrealized, 2)
    return row


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_payload(
    meta: dict,
    allocations: Dict[str, float],
    positions: Dict[str, object],
    ledger_rows: list,
    *,
    as_of: str,
) -> dict:
    """Merge narrative meta with live account state into the legacy microsite payload."""
    bought_by_strategy = held_symbols_by_strategy(ledger_rows)
    pnl_by_strategy = realized_pnl_by_strategy(ledger_rows)

    strategies: List[dict] = []
    site_positions: List[dict] = []
    sleeve_pnl: List[dict] = []

    for entry in meta.get("strategies", []):
        sleeve = {k: v for k, v in entry.items() if not k.startswith("_")}
        module = sleeve.pop("module", None)

        if not module:
            sleeve.setdefault("budget", None)
            strategies.append(sleeve)
            continue

        sleeve["budget"] = allocations.get(module)
        if sleeve["budget"] is None:
            logger.warning(
                "%s has no entry in STRATEGY_ALLOCATIONS — budget will render as unset",
                module,
            )

        held = _attributed_holdings(module, bought_by_strategy, positions)
        sleeve["holdings"] = sorted(held)

        short = sleeve.get("short", sleeve.get("name", module))

        for symbol in sorted(held):
            pos = positions[symbol]
            mv = safe_float(getattr(pos, "market_value", 0.0))
            site_positions.append(
                {
                    "strategy": short,
                    "symbol": symbol,
                    "qty": round(safe_float(getattr(pos, "qty", 0.0)), 4),
                    "notional": round(mv, 2),
                    "notional_display": _money_display(mv),
                    "note": "",
                }
            )

        sleeve_pnl.append({"label": short, "pnl": round(pnl_by_strategy.get(module, 0.0), 2)})
        strategies.append(sleeve)

    sleeve_pnl.sort(key=lambda row: row["pnl"], reverse=True)

    return {
        "_generated": "Written by src/export_portfolio.py — do not hand-edit.",
        "as_of": as_of,
        "disclaimer": meta.get("disclaimer", ""),
        "repo": meta.get("repo", ""),
        "portfolio_home": meta.get("portfolio_home", ""),
        "pnl_basis": PNL_BASIS,
        "pnl_scope": PNL_SCOPE,
        "strategies": strategies,
        "positions": site_positions,
        "sleeve_realized_pnl": sleeve_pnl,
    }


def build_microsite_snapshot(
    meta: dict,
    allocations: Dict[str, float],
    positions: Dict[str, object],
    ledger_rows: list,
    orders: list,
    *,
    as_of: str,
    generated_at: str,
) -> dict:
    """Public snapshot schema consumed by calm-execution-engine (src/lib/snapshot.ts)."""
    bought_by_strategy = held_symbols_by_strategy(ledger_rows)
    pnl_by_strategy = realized_pnl_by_strategy(ledger_rows)

    live_strategies: List[dict] = []
    total_committed = 0.0
    total_remaining = 0.0
    total_market_value = 0.0
    total_realized = 0.0

    for entry in meta.get("strategies", []):
        module = entry.get("module")
        if not module:
            continue

        cap = allocations.get(module)
        if cap is None:
            continue

        committed = round(committed_dollars_from_orders(orders, strategy_id=module), 2)
        remaining = round(max(float(cap) - committed, 0.0), 2)
        held = _attributed_holdings(module, bought_by_strategy, positions)

        holdings: List[dict] = []
        market_value = 0.0
        for symbol in sorted(held):
            pos = positions[symbol]
            row = _holding_rows(symbol, pos)
            holdings.append(row)
            market_value += safe_float(row["market_value"])

        market_value = round(market_value, 2)
        realized = round(pnl_by_strategy.get(module, 0.0), 2)
        utilization = round((market_value / cap) * 100, 1) if cap > 0 else 0.0

        evidence = entry.get("evidence") or {}
        live_strategies.append(
            {
                "id": entry.get("id", module),
                "name": entry.get("name", module),
                "status": entry.get("status", "active"),
                "summary": entry.get("thesis") or entry.get("plain", ""),
                "citation": evidence.get("label", ""),
                "cadence_note": entry.get("sizing", ""),
                "capital_cap": cap,
                "committed": committed,
                "remaining": remaining,
                "market_value_attributed": market_value,
                "realized_pnl_closed": realized,
                "utilization_by_market_value": utilization,
                "holdings": holdings,
            }
        )

        total_committed += committed
        total_remaining += remaining
        total_market_value += market_value
        total_realized += realized

    capital_configured = round(sum(allocations.values()), 2)
    window_start, window_end = _order_window(orders)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "snapshot_date": as_of,
        "last_report_date": as_of,
        "status_label": "active",
        "source": "Alpaca paper account",
        "data_source": "Alpaca paper account",
        "broker": "Alpaca paper",
        "order_window_start": window_start,
        "order_window_end": window_end,
        "order_count_fetched": len(orders),
        "positions_included": True,
        "caveat": meta.get("disclaimer", ""),
        "capital_configured": capital_configured,
        "strategy_count": len(live_strategies),
        "total_committed": round(total_committed, 2),
        "total_remaining": round(total_remaining, 2),
        "total_market_value_attributed": round(total_market_value, 2),
        "total_realized_pnl_closed": round(total_realized, 2),
        "pnl_basis": PNL_BASIS,
        "pnl_scope": PNL_SCOPE,
        "strategies": live_strategies,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    meta_path = _resolve_meta_path()
    if meta_path is None:
        logger.error(
            "Missing portfolio.meta.json — looked in: %s",
            ", ".join(str(p) for p in META_CANDIDATES),
        )
        return 2

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("Using narrative meta from %s", meta_path)

    import config

    try:
        client = config.get_client()
    except Exception:
        logger.exception("Could not create Alpaca client (ALPACA_KEY / ALPACA_SECRET set?)")
        return 2

    positions, ledger_rows, orders = _fetch_snapshot(client)

    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz=tz)
    as_of = now.strftime("%Y-%m-%d")
    generated_at = now.isoformat(timespec="seconds")

    snapshot = build_microsite_snapshot(
        meta,
        config.STRATEGY_ALLOCATIONS,
        positions,
        ledger_rows,
        orders,
        as_of=as_of,
        generated_at=generated_at,
    )

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s — snapshot_date=%s, %d strategies, %d orders",
        SNAPSHOT_PATH,
        as_of,
        len(snapshot["strategies"]),
        snapshot["order_count_fetched"],
    )

    for s in snapshot["strategies"]:
        logger.info(
            "  %-28s cap=%s committed=%s mv=%s realized=%+.2f",
            s["name"],
            _money_display(s["capital_cap"]),
            _money_display(s["committed"]),
            _money_display(s["market_value_attributed"]),
            s["realized_pnl_closed"],
        )

    if MICROSITE_DATA.exists() or MICROSITE_OUT.parent.exists():
        payload = build_payload(
            meta,
            config.STRATEGY_ALLOCATIONS,
            positions,
            ledger_rows,
            as_of=as_of,
        )
        MICROSITE_OUT.parent.mkdir(parents=True, exist_ok=True)
        MICROSITE_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", MICROSITE_OUT)

    untagged = sorted(
        symbol
        for symbol in positions
        if not any(
            symbol in {h["symbol"] for h in s.get("holdings", [])}
            for s in snapshot["strategies"]
        )
    )
    if untagged:
        logger.info(
            "Excluded %d untagged position(s) from the book: %s",
            len(untagged),
            ", ".join(untagged),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

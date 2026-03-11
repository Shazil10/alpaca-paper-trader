"""report.py

Generate a simple, readable orders report from Alpaca and persist it to `reports/`.

This is meant to be run at the end of each daily bot run.

Outputs:
- reports/orders_latest.csv
- reports/orders_latest.md

The report attributes each order to a strategy using the `client_order_id` prefix:
    "{strategy_id}:..."

Usage:
    python src/report.py

Optional env vars:
- REPORT_LIMIT (default: 200)
"""

from __future__ import annotations

import csv
import logging
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config


logger = logging.getLogger(__name__)


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
CSV_PATH = REPORTS_DIR / "orders_latest.csv"
MD_PATH = REPORTS_DIR / "orders_latest.md"
HTML_PATH = REPORTS_DIR / "orders_latest.html"


@dataclass(frozen=True)
class ReportRow:
    submitted_at: str
    symbol: str
    side: str
    status: str
    notional: float
    filled_qty: float
    filled_avg_price: float
    filled_value: float
    client_order_id: str
    strategy_type: str
    strategy_name: str
    pnl: float | None = None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _parse_strategy_id(client_order_id: str) -> tuple[str, str]:
    """Parse strategy_id from client_order_id and split into (type, name).
    
    Example: 'strategies.momentum.clenow_trend:...' -> ('Momentum', 'Clenow Trend')
    """
    if not client_order_id:
        return ("", "")
    # Our convention: "{strategy_id}:{random}".
    left, _, _ = client_order_id.partition(":")
    
    # Parse strategies.{type}.{name} format
    parts = left.split(".")
    if len(parts) >= 3 and parts[0] == "strategies":
        strategy_type = parts[1].replace("_", " ").title()
        strategy_name = parts[2].replace("_", " ").title()
        return (strategy_type, strategy_name)
    
    # Fallback: if format doesn't match, return the whole thing as type
    return (left.replace("_", " ").title(), "")


def _iso(value: object) -> str:
    # Alpaca returns timestamps that are often already strings.
    if value is None:
        return ""
    s = str(value)
    return s


def _format_money(x: float) -> str:
    return f"{x:,.2f}"


def _format_pnl(pnl: float | None) -> str:
    if pnl is None:
        return ""
    sign = "+" if pnl >= 0 else ""
    return f"{sign}{pnl:,.2f}"


def fetch_orders(*, limit: int = 200) -> list[object]:
    client = config.get_client()

    # Try GetOrdersRequest object first (works reliably across SDK versions)
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=int(limit), nested=True)
        return list(client.get_orders(req))
    except Exception:
        pass

    # Fallback: raw keyword args (older SDK versions)
    try:
        return list(client.get_orders(status="all", limit=int(limit), direction="desc", nested=True))
    except TypeError:
        pass

    try:
        return list(client.get_orders(limit=int(limit), direction="desc", nested=True))
    except TypeError:
        pass

    # Minimal fallback
    try:
        return list(client.get_orders(limit=int(limit)))
    except TypeError:
        return list(client.get_orders())


def build_rows(orders_list: list[object]) -> list[ReportRow]:
    # Sort chronologically (oldest-first) for FIFO cost-basis matching.
    chrono = sorted(orders_list, key=lambda o: str(getattr(o, "submitted_at", "") or ""))

    # FIFO cost basis per symbol: symbol → deque of entry avg_price
    cost_basis: dict[str, deque[float]] = {}

    rows: list[ReportRow] = []
    for o in chrono:
        symbol = str(getattr(o, "symbol", "") or "").strip().upper()

        # Strip enum prefixes: OrderSide.BUY -> BUY
        side_raw = str(getattr(o, "side", "") or "")
        side = side_raw.replace("OrderSide.", "").upper()

        # Strip enum prefixes: OrderStatus.FILLED -> FILLED
        status_raw = str(getattr(o, "status", "") or "")
        status = status_raw.replace("OrderStatus.", "").upper()

        notional = _safe_float(getattr(o, "notional", 0.0))
        filled_qty = _safe_float(getattr(o, "filled_qty", 0.0))
        filled_avg_price = _safe_float(getattr(o, "filled_avg_price", 0.0))
        filled_value = filled_qty * filled_avg_price if filled_qty > 0 and filled_avg_price > 0 else 0.0

        client_order_id = str(getattr(o, "client_order_id", "") or "")
        strategy_type, strategy_name = _parse_strategy_id(client_order_id)

        submitted_at = _iso(getattr(o, "submitted_at", ""))

        # Compute PnL for filled SELL orders using FIFO cost basis.
        pnl: float | None = None
        if side == "BUY" and status == "FILLED" and filled_qty > 0 and filled_avg_price > 0:
            if symbol not in cost_basis:
                cost_basis[symbol] = deque()
            cost_basis[symbol].append(filled_avg_price)
        elif side == "SELL" and status == "FILLED" and filled_qty > 0 and filled_avg_price > 0:
            if symbol in cost_basis and cost_basis[symbol]:
                entry_price = cost_basis[symbol].popleft()
                pnl = round((filled_avg_price - entry_price) * filled_qty, 2)

        rows.append(
            ReportRow(
                submitted_at=submitted_at,
                symbol=symbol,
                side=side,
                status=status,
                notional=notional,
                filled_qty=filled_qty,
                filled_avg_price=filled_avg_price,
                filled_value=filled_value,
                client_order_id=client_order_id,
                strategy_type=strategy_type,
                strategy_name=strategy_name,
                pnl=pnl,
            )
        )

    # Restore newest-first ordering for display.
    rows.sort(key=lambda r: r.submitted_at, reverse=True)
    return rows


def write_csv(rows: list[ReportRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Submitted At",
                "Symbol",
                "Side",
                "Status",
                "Notional",
                "Filled Qty",
                "Filled Avg Price",
                "Filled Value",
                "PnL ($)",
                "Client Order ID",
                "Strategy Type",
                "Strategy Name",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.submitted_at,
                    r.symbol,
                    r.side,
                    r.status,
                    f"{r.notional:.6f}",
                    f"{r.filled_qty:.6f}",
                    f"{r.filled_avg_price:.6f}",
                    f"{r.filled_value:.6f}",
                    "" if r.pnl is None else f"{r.pnl:.2f}",
                    r.client_order_id,
                    r.strategy_type,
                    r.strategy_name,
                ]
            )


def write_markdown(rows: list[ReportRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = []
    lines.append(f"# Orders report\n\nGenerated: {now} (UTC)\n")

    # Keep the table human-sized.
    max_rows = 200
    preview = rows[:max_rows]

    header = [
        "Submitted At",
        "Symbol",
        "Side",
        "Status",
        "Notional ($)",
        "Filled Qty",
        "Filled Avg Price",
        "Filled Value ($)",
        "PnL ($)",
        "Strategy Type",
        "Strategy Name",
        "Client Order ID",
    ]

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for r in preview:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.submitted_at,
                    r.symbol,
                    str(r.side),
                    str(r.status),
                    _format_money(r.notional),
                    f"{r.filled_qty:.4f}",
                    _format_money(r.filled_avg_price),
                    _format_money(r.filled_value),
                    _format_pnl(r.pnl),
                    r.strategy_type,
                    r.strategy_name,
                    r.client_order_id,
                ]
            )
            + " |"
        )

    if len(rows) > max_rows:
        lines.append(f"\nShowing first {max_rows} rows out of {len(rows)}.\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(rows: list[ReportRow], path: Path) -> None:
        """Write a self-contained HTML table (easy to open in VS Code or download)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

        # Simple, readable table. (No JS; keeps it robust in artifacts.)
        def esc(s: str) -> str:
                return (
                        s.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace('"', "&quot;")
                )

        header = [
                "Submitted At",
                "Symbol",
                "Side",
                "Status",
                "Notional ($)",
                "Filled Qty",
                "Filled Avg Price",
                "Filled Value ($)",
                "PnL ($)",
                "Strategy Type",
                "Strategy Name",
                "Client Order ID",
        ]

        rows_html: list[str] = []
        for r in rows[:200]:
                cells = [
                        r.submitted_at,
                        r.symbol,
                        str(r.side),
                        str(r.status),
                        _format_money(r.notional),
                        f"{r.filled_qty:.4f}",
                        _format_money(r.filled_avg_price),
                        _format_money(r.filled_value),
                        r.strategy_type,
                        r.strategy_name,
                        r.client_order_id,
                ]
                pnl_str = _format_pnl(r.pnl)
                if pnl_str.startswith("+"):
                        pnl_style = ' style="color:#16a34a;font-weight:600"'
                elif pnl_str.startswith("-"):
                        pnl_style = ' style="color:#dc2626;font-weight:600"'
                else:
                        pnl_style = ""
                pnl_cell = f"<td{pnl_style}>{esc(pnl_str)}</td>"
                rows_html.append(
                        "<tr>"
                        + "".join(f"<td>{esc(c)}</td>" for c in cells[:8])
                        + pnl_cell
                        + "".join(f"<td>{esc(c)}</td>" for c in cells[8:])
                        + "</tr>"
                )

        html = f"""<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Orders report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
            h1 {{ margin: 0 0 8px 0; }}
            .meta {{ color: #666; margin-bottom: 16px; }}
            table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
            th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
            th {{ background: #f6f6f6; text-align: left; position: sticky; top: 0; }}
            tr:nth-child(even) td {{ background: #fcfcfc; }}
            code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }}
            .wrap {{ max-width: 100%; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>Orders report</h1>
        <div class=\"meta\">Generated: {esc(now)} (UTC). Showing first {min(200, len(rows))} rows out of {len(rows)}.</div>
        <div class=\"wrap\">
            <table>
                <thead>
                    <tr>{''.join(f'<th>{esc(h)}</th>' for h in header)}</tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
    </body>
</html>
"""

        path.write_text(html, encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    limit = int(os.getenv("REPORT_LIMIT", "200"))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        orders_list = fetch_orders(limit=limit)
    except Exception:
        logger.exception("Failed to fetch orders")
        return 2

    rows = build_rows(orders_list)

    # Persist artifacts
    write_csv(rows, CSV_PATH)
    write_markdown(rows, MD_PATH)
    write_html(rows, HTML_PATH)

    logger.info("Wrote %d rows -> %s", len(rows), CSV_PATH)
    logger.info("Wrote %d rows -> %s", len(rows), MD_PATH)
    logger.info("Wrote %d rows -> %s", len(rows), HTML_PATH)

    # Also print a small preview to stdout so GH Actions logs show something useful.
    preview_n = min(10, len(rows))
    if preview_n:
        logger.info("Preview (first %d rows):", preview_n)
        for r in rows[:preview_n]:
            logger.info(
                "%s %s %s %s notional=%.2f filled=%.4f@%.2f pnl=%s strategy=%s/%s cid=%s",
                r.submitted_at,
                r.symbol,
                r.side,
                r.status,
                r.notional,
                r.filled_qty,
                r.filled_avg_price,
                _format_pnl(r.pnl) or "—",
                r.strategy_type,
                r.strategy_name,
                r.client_order_id,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

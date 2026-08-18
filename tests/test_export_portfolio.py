"""Tests for the microsite data export and the shared order ledger.

No Alpaca connection: orders and positions are plain fixture objects that
duck-type the SDK attributes the code reads.

Run with: python -m pytest tests/test_export_portfolio.py -v
"""

from __future__ import annotations

import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


CLENOW = "strategies.momentum.clenow_trend"
PULLBACK = "strategies.mean_reversion.high_pullback_reversion"
RANKED = "strategies.ranks.ranked_asset_alloc"


class _Order:
    def __init__(
        self,
        *,
        submitted_at: str,
        symbol: str,
        side: str,
        client_order_id: str,
        status: str = "filled",
        filled_qty: float = 0.0,
        filled_avg_price: float = 0.0,
        notional: float = 0.0,
    ):
        self.submitted_at = submitted_at
        self.symbol = symbol
        self.side = side
        self.client_order_id = client_order_id
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.notional = notional
        self.qty = filled_qty
        self.limit_price = 0.0
        self.id = client_order_id


class _Position:
    def __init__(self, symbol: str, qty: float, market_value: float):
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value


def _buy(ts, symbol, strategy, qty, price):
    return _Order(
        submitted_at=ts,
        symbol=symbol,
        side="buy",
        client_order_id=f"{strategy}:{ts}",
        filled_qty=qty,
        filled_avg_price=price,
    )


def _sell(ts, symbol, strategy, qty, price):
    return _Order(
        submitted_at=ts,
        symbol=symbol,
        side="sell",
        client_order_id=f"{strategy}:{ts}",
        filled_qty=qty,
        filled_avg_price=price,
    )


class PerStrategyFifoTests(unittest.TestCase):
    """The bug this guards: two sleeves holding one ticker."""

    def test_shared_symbol_does_not_cross_attribute(self):
        from order_ledger import build_ledger, realized_pnl_by_strategy

        # Both sleeves buy GEO at very different prices, then both sell at $30.
        # A symbol-only FIFO queue would book Clenow's sell against whichever
        # entry happened to be first in the queue.
        orders = [
            _buy("2026-01-01", "GEO", CLENOW, 10, 10.0),
            _buy("2026-01-02", "GEO", PULLBACK, 10, 20.0),
            _sell("2026-01-03", "GEO", CLENOW, 10, 30.0),
            _sell("2026-01-04", "GEO", PULLBACK, 10, 30.0),
        ]

        pnl = realized_pnl_by_strategy(build_ledger(orders))

        # Clenow: (30-10)*10 = +200. Pullback: (30-20)*10 = +100.
        self.assertEqual(pnl[CLENOW], 200.0)
        self.assertEqual(pnl[PULLBACK], 100.0)

    def test_fifo_is_quantity_aware(self):
        """One sell can span several buy lots and consume them partially."""
        from order_ledger import build_ledger, realized_pnl_by_strategy

        orders = [
            _buy("2026-01-01", "AAPL", CLENOW, 10, 100.0),
            _buy("2026-01-02", "AAPL", CLENOW, 10, 200.0),
            # Sell 15 at 250: 10 from the $100 lot, 5 from the $200 lot.
            _sell("2026-01-03", "AAPL", CLENOW, 15, 250.0),
        ]

        pnl = realized_pnl_by_strategy(build_ledger(orders))

        # (250-100)*10 + (250-200)*5 = 1500 + 250 = 1750
        self.assertEqual(pnl[CLENOW], 1750.0)

    def test_untagged_orders_are_not_attributed(self):
        """Hand-placed dashboard trades must not land in a sleeve's PnL."""
        from order_ledger import build_ledger, realized_pnl_by_strategy

        orders = [
            _Order(
                submitted_at="2026-01-01",
                symbol="NVO",
                side="buy",
                client_order_id="ac5b0675-1635-4df0-a11b-e598bf6e7b52",
                filled_qty=10,
                filled_avg_price=50.0,
            ),
            _Order(
                submitted_at="2026-01-02",
                symbol="NVO",
                side="sell",
                client_order_id="510fc858-774e-468c-91c9-559056cabb0b",
                filled_qty=10,
                filled_avg_price=60.0,
            ),
        ]

        pnl = realized_pnl_by_strategy(build_ledger(orders))
        self.assertEqual(pnl, {})

    def test_unfilled_orders_carry_no_pnl(self):
        from order_ledger import build_ledger

        orders = [
            _buy("2026-01-01", "AAPL", CLENOW, 10, 100.0),
            _Order(
                submitted_at="2026-01-02",
                symbol="AAPL",
                side="sell",
                client_order_id=f"{CLENOW}:x",
                status="canceled",
                filled_qty=0,
                filled_avg_price=0,
            ),
        ]

        rows = build_ledger(orders)
        self.assertTrue(all(r.pnl is None for r in rows))


class PayloadTests(unittest.TestCase):
    META = {
        "disclaimer": "paper account",
        "repo": "https://example.com/repo",
        "portfolio_home": "https://example.com",
        "strategies": [
            {
                "id": "clenow-trend",
                "module": CLENOW,
                "name": "Clenow Trend Following",
                "short": "Clenow Trend",
                "status": "Running",
                "thesis": "narrative that must survive export",
                "chips": ["Cross-Sectional Momentum"],
            },
            {
                "id": "deep-momentum",
                "name": "Deep Momentum",
                "short": "Deep Momentum",
                "status": "Research",
                "budget": None,
                "thesis": "still in research",
            },
        ],
    }

    ALLOCATIONS = {CLENOW: 10_000, RANKED: 15_000}

    def _payload(self, positions, orders):
        from export_portfolio import build_payload
        from order_ledger import build_ledger

        return build_payload(
            self.META,
            self.ALLOCATIONS,
            positions,
            build_ledger(orders),
            as_of="2026-08-17",
        )

    def test_budget_comes_from_config(self):
        payload = self._payload({}, [])
        clenow = payload["strategies"][0]
        self.assertEqual(clenow["budget"], 10_000)

    def test_holdings_require_fill_and_open_position(self):
        """A sleeve holds a symbol only when both facts line up."""
        positions = {
            "AAPL": _Position("AAPL", 10, 1500.0),   # bought by sleeve, still open
            "MSFT": _Position("MSFT", 5, 900.0),     # open, but never bought by sleeve
        }
        orders = [
            _buy("2026-01-01", "AAPL", CLENOW, 10, 100.0),
            # Bought and already sold — must not show as a holding.
            _buy("2026-01-01", "TSLA", CLENOW, 5, 200.0),
            _sell("2026-02-01", "TSLA", CLENOW, 5, 220.0),
        ]

        payload = self._payload(positions, orders)

        self.assertEqual(payload["strategies"][0]["holdings"], ["AAPL"])
        symbols = [p["symbol"] for p in payload["positions"]]
        self.assertEqual(symbols, ["AAPL"])

    def test_untagged_position_excluded_from_book(self):
        positions = {"NVO": _Position("NVO", 10, 500.0)}
        payload = self._payload(positions, [])
        self.assertEqual(payload["positions"], [])

    def test_position_notional_uses_live_market_value(self):
        positions = {"AAPL": _Position("AAPL", 10, 1234.56)}
        orders = [_buy("2026-01-01", "AAPL", CLENOW, 10, 100.0)]

        payload = self._payload(positions, orders)
        position = payload["positions"][0]

        self.assertEqual(position["notional"], 1234.56)
        self.assertEqual(position["notional_display"], "$1,235")
        self.assertEqual(position["strategy"], "Clenow Trend")

    def test_meta_narrative_survives_and_module_is_stripped(self):
        payload = self._payload({}, [])
        clenow = payload["strategies"][0]

        self.assertEqual(clenow["thesis"], "narrative that must survive export")
        self.assertEqual(clenow["chips"], ["Cross-Sectional Momentum"])
        # `module` is internal plumbing; the site never needs it.
        self.assertNotIn("module", clenow)

    def test_research_sleeve_passes_through_with_null_budget(self):
        payload = self._payload({}, [])
        research = payload["strategies"][1]

        self.assertIsNone(research["budget"])
        self.assertNotIn("holdings", research)
        # Research sleeves hold no capital, so they get no PnL row.
        labels = [r["label"] for r in payload["sleeve_realized_pnl"]]
        self.assertNotIn("Deep Momentum", labels)

    def test_realized_pnl_row_per_live_sleeve(self):
        orders = [
            _buy("2026-01-01", "AAPL", CLENOW, 10, 100.0),
            _sell("2026-02-01", "AAPL", CLENOW, 10, 120.0),
        ]
        payload = self._payload({}, orders)

        self.assertEqual(
            payload["sleeve_realized_pnl"],
            [{"label": "Clenow Trend", "pnl": 200.0}],
        )

    def test_as_of_and_honesty_labels_present(self):
        payload = self._payload({}, [])
        self.assertEqual(payload["as_of"], "2026-08-17")
        self.assertEqual(payload["pnl_basis"], "realized_closed_trades")
        self.assertEqual(payload["pnl_scope"], "full_alpaca_order_history")


class SnapshotDisplayTests(unittest.TestCase):
    META = {
        "disclaimer": "paper account",
        "strategies": [
            {
                "id": "clenow-trend",
                "module": CLENOW,
                "name": "Clenow Trend Following",
                "status": "Running",
            },
            {
                "id": "high-pullback-reversion",
                "module": PULLBACK,
                "name": "High Pullback Mean Reversion",
                "status": "Running",
            },
            {
                "id": "ranked-asset-allocation",
                "module": RANKED,
                "name": "Ranked Asset Allocation",
                "status": "Running",
            },
        ],
    }

    ALLOCATIONS = {CLENOW: 10_000, PULLBACK: 15_000, RANKED: 15_000}

    def _snapshot(self, positions, orders):
        from export_portfolio import build_microsite_snapshot
        from order_ledger import build_ledger

        return build_microsite_snapshot(
            self.META,
            self.ALLOCATIONS,
            positions,
            build_ledger(orders),
            orders,
            as_of="2026-08-17",
            generated_at="2026-08-17T16:30:00-04:00",
        )

    def test_schema_version_is_four(self):
        snapshot = self._snapshot({}, [])
        self.assertEqual(snapshot["schema_version"], 4)

    def test_public_display_labels_present(self):
        snapshot = self._snapshot({}, [])
        self.assertEqual(snapshot["attribution_label"], "Strategy-prefixed order IDs")
        self.assertEqual(snapshot["closing_report_label"], "Closing report and JSON snapshot")
        self.assertEqual(snapshot["report_artifacts"], ["CSV", "HTML", "JSON"])

    def test_live_since_from_first_filled_order(self):
        orders = [
            _buy("2026-03-10", "AAPL", CLENOW, 10, 100.0),
            _buy("2026-02-02", "FOUR", PULLBACK, 5, 50.0),
        ]
        snapshot = self._snapshot({}, orders)
        clenow = next(s for s in snapshot["strategies"] if s["id"] == "clenow-trend")
        pullback = next(s for s in snapshot["strategies"] if s["id"] == "high-pullback-reversion")

        self.assertEqual(clenow["first_filled_order_date"], "2026-03-10")
        self.assertEqual(clenow["live_since_date"], "2026-03-10")
        self.assertEqual(clenow["live_since"], "Mar 2026")
        self.assertEqual(pullback["first_filled_order_date"], "2026-02-02")
        self.assertEqual(pullback["live_since"], "Feb 2026")

    def test_realized_pnl_rank_and_strategy_order(self):
        orders = [
            _buy("2026-01-01", "AAPL", CLENOW, 10, 100.0),
            _sell("2026-02-01", "AAPL", CLENOW, 10, 120.0),
            _buy("2026-01-01", "FOUR", PULLBACK, 10, 50.0),
            _sell("2026-02-01", "FOUR", PULLBACK, 10, 100.0),
            _buy("2026-01-01", "XLE", RANKED, 10, 80.0),
            _sell("2026-02-01", "XLE", RANKED, 10, 70.0),
        ]
        snapshot = self._snapshot({}, orders)

        self.assertEqual(
            snapshot["strategy_order_realized_pnl_desc"],
            ["high-pullback-reversion", "clenow-trend", "ranked-asset-allocation"],
        )
        ranks = {s["id"]: s["realized_pnl_rank"] for s in snapshot["strategies"]}
        self.assertEqual(ranks["high-pullback-reversion"], 1)
        self.assertEqual(ranks["clenow-trend"], 2)
        self.assertEqual(ranks["ranked-asset-allocation"], 3)
        self.assertEqual(snapshot["strategies"][0]["id"], "high-pullback-reversion")


if __name__ == "__main__":
    unittest.main()

"""Minimal tests for lifetime budget accounting.

These are intentionally tiny and dependency-free.
Run with: python -m pytest tests/ -v   (or python -m unittest discover tests)
"""

from __future__ import annotations

import os
import sys
import unittest


# Ensure src/ is importable when running from repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


class _Order:
    def __init__(
        self,
        *,
        id: str,
        client_order_id: str,
        side: str = "buy",
        status: str = "filled",
        filled_qty: float = 0,
        filled_avg_price: float = 0,
        notional: float = 0,
        qty: float = 0,
        limit_price: float = 0,
    ):
        self.id = id
        self.client_order_id = client_order_id
        self.side = side
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.notional = notional
        self.qty = qty
        self.limit_price = limit_price


class StrategyBudgetTests(unittest.TestCase):
    def test_counts_only_matching_prefix(self):
        """Only orders with the strategy prefix are counted; orders from other
        strategies are ignored.  BUYs add to committed, SELLs subtract."""
        from trade_models import committed_dollars_from_orders

        strategy_id = "strategies.momentum.clenow_trend"
        prefix = strategy_id + ":"

        orders = [
            _Order(id="1", client_order_id=prefix + "aaa", side="buy", status="filled", filled_qty=2, filled_avg_price=100),
            _Order(id="2", client_order_id=prefix + "bbb", side="sell", status="filled", filled_qty=1, filled_avg_price=50),
            _Order(id="3", client_order_id="other:ccc", side="buy", status="filled", filled_qty=1, filled_avg_price=999),
        ]
        # buy #1: +200, sell #2: -50, other prefix #3: ignored
        committed = committed_dollars_from_orders(orders, strategy_id=strategy_id)
        self.assertEqual(committed, 150.0)

    def test_ignores_canceled_rejected(self):
        from trade_models import committed_dollars_from_orders

        strategy_id = "s1"
        prefix = strategy_id + ":"

        orders = [
            _Order(id="1", client_order_id=prefix + "a", side="buy", status="canceled", notional=500),
            _Order(id="2", client_order_id=prefix + "b", side="buy", status="rejected", notional=500),
            _Order(id="3", client_order_id=prefix + "c", side="buy", status="accepted", notional=250),
        ]
        committed = committed_dollars_from_orders(orders, strategy_id=strategy_id)
        self.assertEqual(committed, 250.0)

    def test_prefers_fill_dollars_over_notional(self):
        from trade_models import committed_dollars_from_orders

        strategy_id = "s1"
        prefix = strategy_id + ":"

        orders = [
            _Order(
                id="1",
                client_order_id=prefix + "a",
                side="buy",
                status="partially_filled",
                filled_qty=1,
                filled_avg_price=100,
                notional=500,
            ),
        ]
        committed = committed_dollars_from_orders(orders, strategy_id=strategy_id)
        self.assertEqual(committed, 100.0)


    def test_sell_proceeds_recycle_budget(self):
        """Full deploy-then-sell cycle: selling frees the exact fill value."""
        from trade_models import committed_dollars_from_orders

        strategy_id = "s1"
        prefix = strategy_id + ":"

        orders = [
            # Deploy the full $10,000 budget across two buys.
            _Order(id="1", client_order_id=prefix + "a", side="buy", status="filled", filled_qty=10, filled_avg_price=600),   # $6,000
            _Order(id="2", client_order_id=prefix + "b", side="buy", status="filled", filled_qty=5,  filled_avg_price=800),   # $4,000
            # Sell one position at market — proceeds recycle into budget.
            _Order(id="3", client_order_id=prefix + "c", side="sell", status="filled", filled_qty=10, filled_avg_price=550),  # -$5,500
        ]
        # committed = 6000 + 4000 - 5500 = 4500
        committed = committed_dollars_from_orders(orders, strategy_id=strategy_id)
        self.assertEqual(committed, 4500.0)

    def test_unfilled_sell_does_not_recycle(self):
        """A SELL that hasn't filled yet must not reduce committed dollars."""
        from trade_models import committed_dollars_from_orders

        strategy_id = "s1"
        prefix = strategy_id + ":"

        orders = [
            _Order(id="1", client_order_id=prefix + "a", side="buy",  status="filled", filled_qty=5, filled_avg_price=200),  # +$1,000
            _Order(id="2", client_order_id=prefix + "b", side="sell", status="new",    filled_qty=0, filled_avg_price=0),    # pending — no subtraction
        ]
        committed = committed_dollars_from_orders(orders, strategy_id=strategy_id)
        self.assertEqual(committed, 1000.0)

    def test_committed_never_goes_negative(self):
        """If sell proceeds somehow exceed buy fills, result is clamped to 0."""
        from trade_models import committed_dollars_from_orders

        strategy_id = "s1"
        prefix = strategy_id + ":"

        orders = [
            _Order(id="1", client_order_id=prefix + "a", side="buy",  status="filled", filled_qty=1, filled_avg_price=100),  # +$100
            _Order(id="2", client_order_id=prefix + "b", side="sell", status="filled", filled_qty=2, filled_avg_price=200),  # -$400
        ]
        committed = committed_dollars_from_orders(orders, strategy_id=strategy_id)
        self.assertEqual(committed, 0.0)


if __name__ == "__main__":
    unittest.main()

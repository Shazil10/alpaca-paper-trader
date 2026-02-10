"""Minimal tests for lifetime budget accounting.

These are intentionally tiny and dependency-free.
Run with: python -m unittest -v
"""

from __future__ import annotations

import os
import sys

import unittest


# Ensure `import src.trade` works when running from repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
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
        filled_qty: str | float = 0,
        filled_avg_price: str | float = 0,
        notional: str | float = 0,
        qty: str | float = 0,
        limit_price: str | float = 0,
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


class _Client:
    def __init__(self, batches):
        self._batches = batches
        self.calls = 0

    def get_orders(self, **kwargs):
        # simple pagination simulator
        i = self.calls
        self.calls += 1
        return list(self._batches[i]) if i < len(self._batches) else []


class StrategyBudgetTests(unittest.TestCase):
    def test_counts_only_matching_prefix_and_buy(self):
        from src.budget import committed_dollars_from_orders

        strategy_id = "strategies.momentum.clenow_trend"
        prefix = strategy_id + ":"

        batches = [
            [
                _Order(id="1", client_order_id=prefix + "aaa", side="buy", status="filled", filled_qty=2, filled_avg_price=100),
                _Order(id="2", client_order_id=prefix + "bbb", side="sell", status="filled", filled_qty=1, filled_avg_price=50),
                _Order(id="3", client_order_id="other:ccc", side="buy", status="filled", filled_qty=1, filled_avg_price=999),
            ]
        ]
        client = _Client(batches)
        committed = committed_dollars_from_orders(batches[0], strategy_id=strategy_id)
        self.assertEqual(committed, 200.0)

    def test_ignores_canceled_rejected(self):
        from src.budget import committed_dollars_from_orders

        strategy_id = "s1"
        prefix = strategy_id + ":"

        batches = [
            [
                _Order(id="1", client_order_id=prefix + "a", side="buy", status="canceled", notional=500),
                _Order(id="2", client_order_id=prefix + "b", side="buy", status="rejected", notional=500),
                _Order(id="3", client_order_id=prefix + "c", side="buy", status="accepted", notional=250),
            ]
        ]
        client = _Client(batches)
        committed = committed_dollars_from_orders(batches[0], strategy_id=strategy_id)
        self.assertEqual(committed, 250.0)

    def test_prefers_fill_dollars_over_notional(self):
        from src.budget import committed_dollars_from_orders

        strategy_id = "s1"
        prefix = strategy_id + ":"

        batches = [
            [
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
        ]
        client = _Client(batches)
        committed = committed_dollars_from_orders(batches[0], strategy_id=strategy_id)
        self.assertEqual(committed, 100.0)


if __name__ == "__main__":
    unittest.main()

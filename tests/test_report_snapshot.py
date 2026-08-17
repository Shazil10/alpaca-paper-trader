"""Tests for the public microsite snapshot generated from reports."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
import tempfile


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


class MicrositeSnapshotTests(unittest.TestCase):
    def test_snapshot_is_generated_from_current_report_rows(self):
        import report

        strategy_id = "strategies.momentum.clenow_trend"
        rows = [
            report.ReportRow(
                submitted_at="2026-08-14T13:30:00Z",
                symbol="AAA",
                side="BUY",
                status="FILLED",
                notional=100.0,
                filled_qty=2.0,
                filled_avg_price=50.0,
                filled_value=100.0,
                client_order_id=f"{strategy_id}:buy",
                strategy_type="Momentum",
                strategy_name="Clenow Trend",
            ),
            report.ReportRow(
                submitted_at="2026-08-15T13:30:00Z",
                symbol="AAA",
                side="SELL",
                status="FILLED",
                notional=0.0,
                filled_qty=1.0,
                filled_avg_price=60.0,
                filled_value=60.0,
                client_order_id=f"{strategy_id}:sell",
                strategy_type="Momentum",
                strategy_name="Clenow Trend",
                pnl=10.0,
            ),
        ]

        snapshot = report.build_microsite_snapshot(rows, limit=200)
        strategy = next(s for s in snapshot["strategies"] if s["id"] == strategy_id)

        self.assertEqual(snapshot["order_window_limit"], 200)
        self.assertEqual(strategy["committed"], 40.0)
        self.assertEqual(strategy["realized_pnl_order_window"], 10.0)
        self.assertEqual(strategy["holdings_inferred_from_order_window"], [{"symbol": "AAA", "qty": 1.0}])

    def test_snapshot_writer_outputs_json(self):
        import json
        import report

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "microsite_snapshot.json"
            report.write_microsite_snapshot({"ok": True}, path)
            self.assertEqual(json.loads(path.read_text()), {"ok": True})


if __name__ == "__main__":
    unittest.main()

"""Targeted regression tests for live strategy wiring and sizing behavior."""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


class StrategyImportTests(unittest.TestCase):
    def test_mean_reversion_alias_imports(self):
        module = importlib.import_module("strategies.mean_reversion.high_pullback_reversion")
        self.assertTrue(callable(getattr(module, "generate_signals", None)))
        self.assertTrue(callable(getattr(module, "check_regime", None)))


class MomentumSignalTests(unittest.TestCase):
    def test_generate_signals_limits_new_buys_to_open_slots(self):
        from strategies.momentum import clenow_trend as ct

        scores = pd.DataFrame(
            [
                {"Symbol": "BBB", "Vol20": 0.20},
                {"Symbol": "CCC", "Vol20": 0.30},
                {"Symbol": "DDD", "Vol20": 0.40},
                {"Symbol": "AAA", "Vol20": 0.50},
            ]
        )
        weights = pd.Series({"BBB": 0.5, "CCC": 0.5})

        with patch.object(ct, "score_universe", return_value=scores), patch.object(
            ct, "_generate_exit_signals", return_value=[]
        ), patch.object(ct, "check_regime", return_value=True), patch.object(
            ct, "_inverse_vol_weights", return_value=weights
        ):
            signals = ct.generate_signals(
                top_n=3,
                budget=300.0,
                held_symbols={"AAA"},
            )

        self.assertEqual([s.symbol for s in signals], ["BBB", "CCC"])
        self.assertEqual([round(float(s.notional or 0.0), 2) for s in signals], [150.0, 150.0])


if __name__ == "__main__":
    unittest.main()

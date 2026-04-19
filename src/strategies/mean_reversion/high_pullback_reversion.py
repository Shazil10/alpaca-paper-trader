"""Stable import alias for the deployed 52-week pullback strategy.

The live config refers to ``strategies.mean_reversion.high_pullback_reversion``.
Keep that strategy id stable while delegating implementation to the
number-prefixed research module.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_impl: ModuleType = import_module("strategies.mean_reversion.52W_mean_reversion_strat")

check_regime = _impl.check_regime
generate_signals = _impl.generate_signals

__all__ = ["check_regime", "generate_signals"]

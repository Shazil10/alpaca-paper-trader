"""Strategy protocol and registry.

Every backtestable strategy must be expressible as a function:
    target_weights(ctx: StrategyContext) -> dict[str, float]

This module defines the protocol, a registry for discovering strategies,
and parameter schema validation.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable

from backtest.context import StrategyContext

logger = logging.getLogger(__name__)


# The canonical strategy type
TargetWeightsFn = Callable[[StrategyContext], Dict[str, float]]


@runtime_checkable
class BacktestStrategy(Protocol):
    """Protocol for a backtestable strategy module."""

    def target_weights(self, ctx: StrategyContext) -> Dict[str, float]:
        """Return target portfolio weights.

        {symbol: fraction_of_sleeve_equity}
        Unlisted symbols are exited. Weights need not sum to 1;
        the remainder is cash.
        """
        ...

    @property
    def name(self) -> str: ...

    @property
    def param_schema(self) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, TargetWeightsFn] = {}


def register(name: str, fn: TargetWeightsFn) -> None:
    """Register a strategy function by name."""
    _REGISTRY[name] = fn
    logger.debug("Registered strategy: %s", name)


#: Opt-in prefix for backtesting an unmigrated live strategy through the
#: generate_signals bridge, e.g. "legacy:strategies.momentum.clenow_trend".
LEGACY_PREFIX = "legacy:"


def get_strategy(name: str) -> TargetWeightsFn:
    """Look up a registered strategy. Falls back to module import.

    A module exposing only the live `generate_signals` interface is *not*
    resolved automatically. The bridge that adapts it cannot enforce the
    point-in-time firewall -- the strategy keeps doing its own data fetches
    and can still see the future -- so silently falling back would hand back
    a contaminated equity curve that looks exactly like a clean one. Callers
    must opt in with the `legacy:` prefix and get a warning when they do.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]

    if name.startswith(LEGACY_PREFIX):
        module_path = name[len(LEGACY_PREFIX):]
        from backtest.adapter import wrap_generate_signals

        logger.warning(
            "Backtesting %s through the legacy generate_signals bridge. "
            "The point-in-time firewall is NOT enforced: this strategy loads "
            "its own data and may look ahead. Treat these results as "
            "indicative only, and migrate it to target_weights(ctx).",
            module_path,
        )
        fn = wrap_generate_signals(module_path)
        register(name, fn)
        return fn

    try:
        module = importlib.import_module(name)
    except ImportError as e:
        raise KeyError(f"Strategy {name!r} could not be imported: {e}") from e

    if hasattr(module, "target_weights"):
        fn = module.target_weights
        register(name, fn)
        return fn

    if hasattr(module, "generate_signals"):
        raise KeyError(
            f"Strategy {name!r} has no target_weights(); it only implements the "
            f"live generate_signals() interface. To backtest it as-is, request "
            f"'{LEGACY_PREFIX}{name}' -- but that bridge does not enforce the "
            f"point-in-time firewall, so the strategy can still see the future. "
            f"Port it to target_weights(ctx) for a result you can trust."
        )

    raise KeyError(
        f"Strategy {name!r} implements neither target_weights() nor "
        f"generate_signals()."
    )


def list_strategies() -> List[str]:
    """Return names of all registered strategies."""
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    """Clear the strategy registry (for testing)."""
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Strategy factory helpers
# ---------------------------------------------------------------------------

def make_strategy(
    fn: Callable,
    default_params: Optional[Dict[str, Any]] = None,
) -> TargetWeightsFn:
    """Wrap a plain function into a TargetWeightsFn with default params.

    The wrapped function receives ctx with params already merged.
    """
    defaults = dict(default_params or {})

    def wrapped(ctx: StrategyContext) -> Dict[str, float]:
        merged_params = {**defaults, **ctx.params}
        patched_ctx = StrategyContext(
            as_of=ctx.as_of,
            price_panel=ctx._price_panel,
            close_matrix=ctx._close_matrix,
            ohlc_adjusted=ctx._ohlc_adjusted,
            universe_fn=ctx._universe_fn,
            portfolio=ctx.portfolio,
            params=merged_params,
            trading_sessions=ctx._trading_sessions,
            sector_map=ctx._sector_map,
        )
        return fn(patched_ctx)

    wrapped.__name__ = getattr(fn, "__name__", "anonymous_strategy")
    wrapped.__doc__ = getattr(fn, "__doc__", "")
    return wrapped

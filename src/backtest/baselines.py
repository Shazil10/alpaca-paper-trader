"""Baselines a strategy must beat to be interesting.

A return number means nothing on its own. Every run is compared against:

* **SPY buy-and-hold** -- the thing you could have done with no work.
* **Equal-weight over the same universe** -- isolates selection skill from
  the universe's own drift. If equal-weight matches you, your ranking adds
  nothing and you are just holding the basket.
* **Random-portfolio placebo** -- the honest one. Draw many portfolios with
  the same position count from the same universe and see where the strategy
  lands in that distribution. Below the 95th percentile, there is no edge to
  discuss: a monkey with the same constraints does as well.

The implementations live in ``benchmark`` alongside the benchmark-relative
metrics that consume them; this module is the baseline-facing name for them
so callers read as intent ("compare to baselines") rather than mechanism.
"""

from __future__ import annotations

from backtest.benchmark import (  # noqa: F401
    buy_and_hold,
    equal_weight,
    percentile_vs_random,
    random_portfolio_placebo,
)

__all__ = [
    "buy_and_hold",
    "equal_weight",
    "random_portfolio_placebo",
    "percentile_vs_random",
]

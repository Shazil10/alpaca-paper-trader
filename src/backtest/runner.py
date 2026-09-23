"""Backtest runner — CLI entry point.

Usage:
    python -m src.backtest.runner --config configs/backtests/example.yaml

Or programmatically:
    from backtest.runner import run_backtest
    result = run_backtest(config)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml
import pandas as pd

from backtest.types import (
    BacktestConfig, BacktestResult, CostConfig, ExecutionConfig,
    FillType, RiskConfig,
)
from backtest.engine import BacktestEngine
from backtest.strategy import get_strategy
from backtest.result import save_result
from backtest import metrics

logger = logging.getLogger(__name__)


def load_panels(
    start: str,
    end: str,
    price_adjustment: str = "asof",
    *,
    root: Optional[Path] = None,
) -> tuple:
    """Load (price_panel, close_matrix, ohlc_adjusted) for a backtest window.

    ``price_adjustment="asof"`` re-anchors ``adj_close`` to ``end``, so the run
    cannot see splits or dividends that postdate the test window. The raw print
    in ``open/high/low/close`` is left alone either way -- it is the only price
    that is honest on every session, and ``StrategyContext.raw_close`` hands it
    to strategies for dollar-threshold decisions.

    ``price_adjustment="today"`` keeps the lake's stored adjustment, which is
    anchored to the download date.
    """
    from data_pipeline import store

    mode = str(price_adjustment or "asof").lower()
    panel = store.load_prices(start=start, end=end, root=root)

    if mode == "asof" and len(panel) > 0:
        panel = _reanchor_adj_close(panel, pd.Timestamp(end))
    elif mode not in ("asof", "today"):
        raise ValueError(
            f"price_adjustment must be 'asof' or 'today', got {price_adjustment!r}"
        )

    return panel, _close_matrix_from(panel), _ohlc_from(panel)


def _reanchor_adj_close(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Re-anchor ``adj_close`` to ``as_of``, leaving raw OHLC untouched."""
    from data_pipeline import adjust
    from data_pipeline.schema import ADJ_CLOSE, CLOSE, DATE, SYMBOL

    factors = adjust.adjustment_factor(panel, as_of)
    out = panel.merge(factors, on=[DATE, SYMBOL], how="left")
    out["factor"] = out["factor"].fillna(1.0)
    out[ADJ_CLOSE] = out[CLOSE].astype("float64") * out["factor"]
    return out.drop(columns=["factor"])


def _close_matrix_from(panel: pd.DataFrame) -> pd.DataFrame:
    """date x symbol matrix of adj_close, matching store.load_close_matrix."""
    from data_pipeline.schema import ADJ_CLOSE, DATE, SYMBOL

    if len(panel) == 0:
        return pd.DataFrame(dtype="float64")

    matrix = panel.pivot_table(
        index=DATE, columns=SYMBOL, values=ADJ_CLOSE, aggfunc="last"
    )
    matrix.index.name = DATE
    matrix.columns.name = None
    return matrix.dropna(how="all").astype("float64").sort_index()


def _ohlc_from(panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """{symbol: DataFrame[Open, High, Low, Close]} on the adj_close scale."""
    from data_pipeline import schema
    from data_pipeline.schema import ADJ_CLOSE, CLOSE, DATE, SYMBOL

    if len(panel) == 0:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    for symbol, group in panel.groupby(SYMBOL, sort=True):
        g = group.set_index(DATE).sort_index()
        close = g[CLOSE].astype("float64")
        adj = g[ADJ_CLOSE].astype("float64")
        factor = (adj / close.where(close > 0)).fillna(1.0)

        frame = pd.DataFrame(
            {
                "Open": g[schema.OPEN].astype("float64") * factor,
                "High": g[schema.HIGH].astype("float64") * factor,
                "Low": g[schema.LOW].astype("float64") * factor,
                "Close": adj,
            },
            index=g.index,
        ).dropna()

        if len(frame) > 0:
            out[str(symbol)] = frame

    return out


def default_universe_fn(universe_source: str):
    """Build a PIT universe callable, or None to fall back to the price panel.

    ``pit_*`` sources union the fixed ETFs into index membership: ETFs are never
    constituents, so a bare ``members_asof`` would hand a sleeve that holds
    ``SHY`` a universe excluding it. ``etf_rotation`` is the ETF set alone.
    Anything else returns None, which leaves ``StrategyContext.universe()`` on
    its panel-derived fallback -- survivorship-biased, and logged as such.
    """
    try:
        from data_pipeline.membership import (
            ALL_FIXED_ETFS, has_pit_membership, members_asof,
        )
    except ImportError:
        return None

    source = str(universe_source or "").strip().lower()

    if source == "etf_rotation":
        etfs = set(ALL_FIXED_ETFS)
        return lambda date: etfs

    if not source.startswith("pit_"):
        logger.warning(
            "universe_source=%r is not point-in-time; universe falls back to "
            "symbols present in the panel. Results carry survivorship bias.",
            universe_source,
        )
        return None

    if not has_pit_membership():
        logger.warning(
            "universe_source=%r requested but no PIT membership file exists. "
            "Run scripts/build_membership.py. Falling back to the panel, which "
            "carries survivorship bias.",
            universe_source,
        )
        return None

    index_name = source.replace("pit_", "").upper() or "SP500"
    etfs = set(ALL_FIXED_ETFS)

    def universe_fn(date):
        return set(members_asof(date, index_name)) | etfs

    return universe_fn


def default_sector_map() -> Optional[Dict[str, str]]:
    """{ticker: sector} from the security master, or None if unavailable.

    ``RiskConfig.max_sector_pct`` is a no-op when no map is supplied, so a
    config that asks for a 40% sector cap and gets nothing would report a run as
    risk-limited when it was not. Warn loudly rather than pass quietly.
    """
    try:
        from data_pipeline.securities import sector_map as load_sector_map
    except ImportError:
        return None

    mapping = load_sector_map()
    if not mapping:
        logger.warning(
            "max_sector_pct is set but no sector labels are available "
            "(run scripts/build_securities.py). The sector cap will not bind."
        )
        return None

    logger.info("Loaded sector labels for %d tickers", len(mapping))
    return mapping


def run_backtest(
    config: BacktestConfig,
    strategy_fn: Optional[Callable] = None,
    price_panel: Optional[pd.DataFrame] = None,
    close_matrix: Optional[pd.DataFrame] = None,
    ohlc_adjusted: Optional[Dict] = None,
    universe_fn: Optional[Callable] = None,
    sector_map: Optional[Dict[str, str]] = None,
    save: bool = True,
) -> BacktestResult:
    """Run a backtest with the given config.

    If price_panel/close_matrix not provided, loads from the price lake.
    If strategy_fn not provided, looks up via config.strategy_module.
    """
    # Load strategy
    if strategy_fn is None:
        strategy_fn = get_strategy(config.strategy_module)

    # Load data
    if price_panel is None or close_matrix is None:
        # Load with generous lookback for warmup
        warmup_start = (
            pd.Timestamp(config.start_date) - pd.Timedelta(days=900)
        ).strftime("%Y-%m-%d")

        price_panel, close_matrix, lake_ohlc = load_panels(
            warmup_start, config.end_date, config.price_adjustment
        )
        if ohlc_adjusted is None:
            ohlc_adjusted = lake_ohlc

    # Load universe function
    if universe_fn is None:
        universe_fn = default_universe_fn(config.universe_source)

    # Sector labels, so the sector cap is not silently inert
    if sector_map is None and config.risk.max_sector_pct < 1.0:
        sector_map = default_sector_map()

    # Run engine
    engine = BacktestEngine(config)
    result = engine.run(
        strategy_fn=strategy_fn,
        price_panel=price_panel,
        close_matrix=close_matrix,
        ohlc_adjusted=ohlc_adjusted,
        universe_fn=universe_fn,
        sector_map=sector_map,
    )

    # Compute metrics
    result.metrics = metrics.compute_full_metrics(
        result.equity_curve,
        result.returns,
        result.benchmark_equity,
        result.benchmark_returns,
        num_trials=config.trial_index,
    )

    # Save
    if save:
        save_result(result)

    return result


def research_options_from_yaml(path: str) -> Dict[str, Any]:
    """Read the optional ``research:`` block from a backtest config.

    Kept out of ``BacktestConfig`` on purpose: these are settings for *how hard
    to interrogate* a strategy, not part of the strategy's definition, and
    mixing them in would make two runs with identical trading assumptions hash to
    different configs.
    """
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}

    block = raw.get("research") or {}
    options: Dict[str, Any] = {}

    for key in (
        "holdout_years", "walk_forward_train_years",
        "walk_forward_test_years", "n_simulations",
    ):
        if key in block:
            options[key] = int(block[key])

    if "param_grid" in block and block["param_grid"]:
        options["param_grid"] = {
            str(name): list(values) for name, values in block["param_grid"].items()
        }

    return options


def run_research_from_config(
    config: BacktestConfig,
    *,
    unlock_holdout: bool = False,
    save: bool = True,
    **options,
):
    """Load the data once, then hand it to every validation section.

    The data load is the expensive part and a research pass runs dozens of
    backtests over the same window, so the panel is loaded here and threaded
    through rather than re-read per run.
    """
    from backtest import research

    warmup_start = (
        pd.Timestamp(config.start_date) - pd.Timedelta(days=900)
    ).strftime("%Y-%m-%d")
    price_panel, close_matrix, ohlc_adjusted = load_panels(
        warmup_start, config.end_date, config.price_adjustment
    )

    sector_map = (
        default_sector_map() if config.risk.max_sector_pct < 1.0 else None
    )

    return research.run_research(
        config,
        run_backtest_fn=run_backtest,
        price_panel=price_panel,
        close_matrix=close_matrix,
        ohlc_adjusted=ohlc_adjusted,
        universe_fn=default_universe_fn(config.universe_source),
        sector_map=sector_map,
        unlock_holdout=unlock_holdout,
        save=save,
        **options,
    )


def fund_from_yaml(path: str):
    """Build a FundEngine from a fund config.

    A fund config is deliberately not a BacktestConfig with a list bolted on: the
    sleeves each carry their own params and optional risk limits, and the
    fund-level cost, risk and execution blocks apply to the *combined* book.
    """
    from backtest.fund import FundEngine, SleeveConfig

    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}

    sleeve_specs = raw.get("sleeves") or []
    if not sleeve_specs:
        raise ValueError(f"{path}: a fund config needs a `sleeves:` list")

    sleeves = []
    for spec in sleeve_specs:
        module = spec.get("strategy_module")
        if not module:
            raise ValueError(f"{path}: every sleeve needs a strategy_module")
        sleeves.append(SleeveConfig(
            strategy_id=str(spec.get("strategy_id", module)),
            strategy_fn=get_strategy(module),
            allocation_pct=float(spec.get("allocation_pct", 0.0)),
            params=dict(spec.get("params") or {}),
            risk=RiskConfig(**spec["risk"]) if spec.get("risk") else None,
        ))

    exec_raw = dict(raw.get("execution") or {})
    if "fill_type" in exec_raw:
        exec_raw["fill_type"] = FillType(exec_raw["fill_type"])

    engine = FundEngine(
        sleeves=sleeves,
        start_date=str(raw.get("start_date", "2010-01-01")),
        end_date=str(raw.get("end_date", "2025-12-31")),
        initial_capital=float(raw.get("initial_capital", 100_000)),
        benchmark=raw.get("benchmark", "SPY"),
        cost_config=CostConfig(**(raw.get("cost") or {})),
        risk_config=RiskConfig(**(raw.get("risk") or {})),
        execution_config=ExecutionConfig(**exec_raw),
        reallocate=str(raw.get("reallocate", "none")),
        reallocate_every_months=int(raw.get("reallocate_every_months", 12)),
    )
    return engine, raw


def run_fund_from_config(path: str):
    """Load data once and run every sleeve over one balance sheet."""
    engine, raw = fund_from_yaml(path)

    warmup_start = (
        pd.Timestamp(engine.start_date) - pd.Timedelta(days=900)
    ).strftime("%Y-%m-%d")
    price_panel, close_matrix, ohlc_adjusted = load_panels(
        warmup_start, engine.end_date, str(raw.get("price_adjustment", "asof"))
    )

    sector_map = (
        default_sector_map() if engine.risk.max_sector_pct < 1.0 else None
    )

    return engine.run(
        price_panel=price_panel,
        close_matrix=close_matrix,
        ohlc_adjusted=ohlc_adjusted,
        universe_fn=default_universe_fn(raw.get("universe_source", "pit_sp500")),
        sector_map=sector_map,
    )


def config_from_yaml(path: str) -> BacktestConfig:
    """Load a BacktestConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    cost_cfg = CostConfig(**raw.get("cost", {})) if "cost" in raw else CostConfig()
    risk_cfg = RiskConfig(**raw.get("risk", {})) if "risk" in raw else RiskConfig()

    exec_raw = raw.get("execution", {})
    if "fill_type" in exec_raw:
        exec_raw["fill_type"] = FillType(exec_raw["fill_type"])
    exec_cfg = ExecutionConfig(**exec_raw) if exec_raw else ExecutionConfig()

    return BacktestConfig(
        strategy_id=raw.get("strategy_id", raw.get("strategy_module", "unknown")),
        strategy_module=raw.get("strategy_module", ""),
        start_date=str(raw.get("start_date", "2010-01-01")),
        end_date=str(raw.get("end_date", "2025-12-31")),
        initial_capital=float(raw.get("initial_capital", 100_000)),
        benchmark=raw.get("benchmark", "SPY"),
        cost=cost_cfg,
        risk=risk_cfg,
        execution=exec_cfg,
        params=raw.get("params", {}),
        universe_source=raw.get("universe_source", "pit_sp500"),
        price_source=raw.get("price_source", "lake"),
        price_adjustment=str(raw.get("price_adjustment", "asof")),
        trial_index=int(raw.get("trial_index", 0)),
        random_seed=int(raw.get("random_seed", 42)),
        notes=raw.get("notes", ""),
    )


def _print_fund_summary(fund) -> None:
    """Fund line first, then the sleeves that produced it."""
    if len(fund.fund_equity) == 0:
        print("No sessions in range.")
        for warning in fund.warnings[:5]:
            print(f"  - {warning}")
        return

    m = fund.fund_metrics or {}
    print(f"\n{'=' * 72}")
    print(f"FUND  {len(fund.sleeve_equity)} sleeve(s)")
    print(f"{'=' * 72}")
    print(f"Final equity   ${fund.fund_equity.iloc[-1]:,.0f}")
    print(f"CAGR           {m.get('cagr', 0):.1%}")
    print(f"Sharpe         {m.get('sharpe', 0):.3f}")
    print(f"Max drawdown   {m.get('max_drawdown', 0):.1%}")
    print(f"Beta           {m.get('beta', 0):.3f}")
    print(f"Alpha          {m.get('alpha_jensen', 0):.1%}")
    print(f"Peak gross     {fund.gross_exposure.max():.2f}x")

    crosses = len([r for r in fund.netting if r.crossed_shares > 0])
    shared = fund.shared_symbols
    print(
        f"Netting saved  ${fund.netting_saved_notional:,.0f} of flow "
        f"({crosses} internal cross(es))"
    )
    # A zero saving means one of two very different things, so say which.
    print(
        f"Sleeve overlap {len(shared)} symbol(s) traded by more than one sleeve"
        + ("" if shared else " — nothing could cross")
    )
    if fund.reallocations:
        print(f"Reallocations  {len(fund.reallocations)}")

    print(f"\n{'sleeve':<28}{'final':>12}{'CAGR':>9}{'Sharpe':>9}{'maxDD':>9}")
    for sid, equity in fund.sleeve_equity.items():
        stats = fund.sleeve_metrics.get(sid, {})
        print(
            f"{sid:<28}{equity.iloc[-1]:>12,.0f}"
            f"{stats.get('cagr', 0):>8.1%}{stats.get('sharpe', 0):>9.3f}"
            f"{stats.get('max_drawdown', 0):>9.1%}"
        )

    if fund.warnings:
        print(f"\nWarnings: {len(fund.warnings)}")
        for warning in fund.warnings[:5]:
            print(f"  - {warning}")


def main():
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--strategy", type=str, help="Strategy module path")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--research",
        action="store_true",
        help="Run the full validation suite and print a verdict card instead of "
             "a single backtest.",
    )
    parser.add_argument(
        "--fund",
        type=str,
        help="Path to a fund config: run every sleeve over one cash balance with "
             "orders netted across them.",
    )
    parser.add_argument(
        "--unlock-holdout",
        action="store_true",
        help="Spend the locked out-of-sample holdout. Every unlock is appended "
             "to runs/holdout_unlocks.log.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.fund:
        fund = run_fund_from_config(args.fund)
        _print_fund_summary(fund)
        return

    if args.config:
        config = config_from_yaml(args.config)
    elif args.strategy:
        config = BacktestConfig(
            strategy_id=args.strategy,
            strategy_module=args.strategy,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
        )
    else:
        parser.error("Either --config or --strategy is required")
        return

    if args.research:
        options = research_options_from_yaml(args.config) if args.config else {}
        report = run_research_from_config(
            config,
            unlock_holdout=args.unlock_holdout,
            save=not args.no_save,
            **options,
        )
        print()
        print(report.card)
        return

    result = run_backtest(config, save=not args.no_save)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Strategy: {config.strategy_id}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial: ${config.initial_capital:,.0f}")
    print(f"Final:   ${result.equity_curve.iloc[-1]:,.0f}" if len(result.equity_curve) > 0 else "Final: N/A")
    print(f"{'='*60}")

    if result.metrics:
        print(f"CAGR:         {result.metrics.get('cagr', 0):.1%}")
        print(f"Sharpe:       {result.metrics.get('sharpe', 0):.3f}")
        print(f"Sortino:      {result.metrics.get('sortino', 0):.3f}")
        print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.1%}")
        print(f"Beta:         {result.metrics.get('beta', 0):.3f}")
        print(f"Alpha:        {result.metrics.get('alpha_jensen', 0):.1%}")
        print(f"PSR:          {result.metrics.get('psr', 0):.1%}")

    if result.warnings:
        print(f"\nWarnings: {len(result.warnings)}")
        for w in result.warnings[:5]:
            print(f"  - {w}")

    if result.limitations:
        print(f"\nLimitations:")
        for lim in result.limitations:
            print(f"  - {lim}")


if __name__ == "__main__":
    main()

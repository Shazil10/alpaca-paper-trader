"""Save and load backtest run artifacts.

Each run gets a directory: runs/<date>_<strategy>_<hash>/
Containing: config.json, summary.json, equity.csv, returns.csv,
orders.csv, fills.csv, positions.csv, metrics.json, run.log
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def save_result(result, runs_dir: Optional[Path] = None) -> Path:
    """Save a BacktestResult to disk. Returns the run directory path."""
    base = Path(runs_dir or DEFAULT_RUNS_DIR)

    cfg = result.config
    date_str = datetime.now().strftime("%Y-%m-%d")
    strategy_name = cfg.strategy_id.replace(".", "_").replace("/", "_")
    run_hash = result.run_id[:8] if result.run_id else "000000"

    run_dir = base / f"{date_str}_{strategy_name}_{run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config_dict = _config_to_dict(cfg)
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2, default=str))

    # Equity curve
    if len(result.equity_curve) > 0:
        eq_df = pd.DataFrame({
            "date": result.equity_curve.index,
            "equity": result.equity_curve.values,
        })
        eq_df.to_csv(run_dir / "equity.csv", index=False)

    # Returns
    if len(result.returns) > 0:
        ret_df = pd.DataFrame({
            "date": result.returns.index,
            "return": result.returns.values,
        })
        ret_df.to_csv(run_dir / "returns.csv", index=False)

    # Orders
    if result.orders:
        orders_data = []
        for o in result.orders:
            orders_data.append({
                "order_id": o.order_id,
                "symbol": o.symbol,
                "side": o.side.value,
                "notional": o.notional,
                "shares": o.shares,
                "strategy_id": o.strategy_id,
                "reason": o.reason,
                "created_date": str(o.created_date) if o.created_date else "",
            })
        pd.DataFrame(orders_data).to_csv(run_dir / "orders.csv", index=False)

    # Fills
    if result.fills:
        fills_data = []
        for f in result.fills:
            fills_data.append({
                "fill_id": f.fill_id,
                "symbol": f.order.symbol,
                "side": f.order.side.value,
                "fill_price": f.fill_price,
                "fill_shares": f.fill_shares,
                "fill_date": str(f.fill_date),
                "commission": f.commission,
                "slippage_bps": f.slippage_bps,
            })
        pd.DataFrame(fills_data).to_csv(run_dir / "fills.csv", index=False)

    # Positions (snapshots)
    if result.snapshots:
        pos_records = []
        for snap in result.snapshots:
            for sym, pos in snap.positions.items():
                pos_records.append({
                    "date": str(snap.date),
                    "symbol": sym,
                    "shares": pos.shares,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "weight": pos.market_value / snap.equity if snap.equity > 0 else 0,
                })
        if pos_records:
            pd.DataFrame(pos_records).to_csv(run_dir / "positions.csv", index=False)

    # Metrics
    if result.metrics:
        _save_metrics(result.metrics, run_dir / "metrics.json")

    # Summary
    summary = {
        "run_id": result.run_id,
        "strategy_id": cfg.strategy_id,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "initial_capital": cfg.initial_capital,
        "final_equity": float(result.equity_curve.iloc[-1]) if len(result.equity_curve) > 0 else 0,
        "total_return": float(result.equity_curve.iloc[-1] / cfg.initial_capital - 1) if len(result.equity_curve) > 0 else 0,
        "n_fills": len(result.fills),
        "n_sessions": len(result.equity_curve),
        "elapsed_seconds": result.elapsed_seconds,
        "warnings": result.warnings,
        "limitations": result.limitations,
    }
    if result.metrics:
        for key in ("sharpe", "max_drawdown", "cagr"):
            if key in result.metrics:
                summary[key] = result.metrics[key]

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info("Saved run to %s", run_dir)
    return run_dir


def load_result(run_dir: Path) -> Dict[str, Any]:
    """Load saved run artifacts from a directory."""
    run_dir = Path(run_dir)
    result = {}

    for name in ("config", "summary", "metrics"):
        path = run_dir / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())

    for name in ("equity", "returns", "orders", "fills", "positions"):
        path = run_dir / f"{name}.csv"
        if path.exists():
            result[name] = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else [])

    return result


def list_runs(runs_dir: Optional[Path] = None) -> list:
    """List all saved runs with their summaries."""
    base = Path(runs_dir or DEFAULT_RUNS_DIR)
    if not base.exists():
        return []

    runs = []
    for d in sorted(base.iterdir()):
        if d.is_dir():
            summary_path = d / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    summary["path"] = str(d)
                    runs.append(summary)
                except Exception:
                    runs.append({"path": str(d), "error": "corrupt summary"})

    return runs


def data_vintage() -> Dict[str, Any]:
    """Fingerprint the data a run was built on.

    A backtest is only reproducible if the inputs are pinned, and the price lake
    is mutable by design -- ``sync_prices`` rewrites the hot year daily and
    late vendor corrections rewrite closed years. Storing a content hash per
    file makes "the lake changed under us" detectable instead of a mystery when
    the same config stops reproducing its own numbers.
    """
    import hashlib

    vintage: Dict[str, Any] = {}

    lake_dir = REPO_ROOT / "data" / "prices" / "daily"
    files = {}
    if lake_dir.exists():
        for path in sorted(lake_dir.iterdir()):
            if path.is_file() and path.suffix in (".parquet", ".csv"):
                digest = hashlib.sha256()
                with path.open("rb") as fh:
                    for block in iter(lambda: fh.read(1 << 20), b""):
                        digest.update(block)
                files[path.name] = {
                    "sha256": digest.hexdigest()[:16],
                    "bytes": path.stat().st_size,
                }
    vintage["price_lake"] = files

    membership = REPO_ROOT / "data" / "universe" / "membership.parquet"
    if membership.exists():
        try:
            rows = len(pd.read_parquet(membership))
        except Exception:
            rows = -1
        vintage["membership"] = {
            "path": str(membership.relative_to(REPO_ROOT)),
            "rows": rows,
            "bytes": membership.stat().st_size,
        }
    else:
        vintage["membership"] = None

    securities = REPO_ROOT / "data" / "universe" / "securities.parquet"
    vintage["securities"] = (
        {"rows": len(pd.read_parquet(securities))} if securities.exists() else None
    )

    actions = REPO_ROOT / "data" / "corporate_actions" / "actions.parquet"
    vintage["corporate_actions"] = (
        {"rows": len(pd.read_parquet(actions))} if actions.exists() else None
    )

    return vintage


def _config_to_dict(cfg) -> dict:
    """Convert BacktestConfig to a JSON-serializable dict."""
    import subprocess

    git_commit = ""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        pass

    try:
        vintage: Any = cfg.data_vintage or data_vintage()
    except Exception:
        logger.exception("could not fingerprint data vintage")
        vintage = cfg.data_vintage

    return {
        "strategy_id": cfg.strategy_id,
        "strategy_module": cfg.strategy_module,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "initial_capital": cfg.initial_capital,
        "benchmark": cfg.benchmark,
        "cost": {
            "spread_bps": cfg.cost.spread_bps,
            "slippage_bps": cfg.cost.slippage_bps,
            "commission_per_share": cfg.cost.commission_per_share,
            "participation_rate": cfg.cost.participation_rate,
        },
        "risk": {
            "max_position_pct": cfg.risk.max_position_pct,
            "max_sector_pct": cfg.risk.max_sector_pct,
            "cash_reserve_pct": cfg.risk.cash_reserve_pct,
            "max_leverage": cfg.risk.max_leverage,
            "max_positions": cfg.risk.max_positions,
        },
        "execution": {
            "fill_type": cfg.execution.fill_type.value,
            "fill_delay_days": cfg.execution.fill_delay_days,
            "fractional_shares": cfg.execution.fractional_shares,
            "delisting_return": cfg.execution.delisting_return,
        },
        "short": {
            "allow_shorts": cfg.short.allow_shorts,
            "borrow_rate_annual": cfg.short.borrow_rate_annual,
            "hard_to_borrow_rate_annual": cfg.short.hard_to_borrow_rate_annual,
            "hard_to_borrow": sorted(cfg.short.hard_to_borrow),
            "unborrowable": sorted(cfg.short.unborrowable),
            "maintenance_margin_long": cfg.short.maintenance_margin_long,
            "maintenance_margin_short": cfg.short.maintenance_margin_short,
            "short_proceeds_earn_interest": cfg.short.short_proceeds_earn_interest,
        },
        "params": cfg.params,
        "universe_source": cfg.universe_source,
        "price_source": cfg.price_source,
        "price_adjustment": cfg.price_adjustment,
        "git_commit": git_commit or cfg.git_commit,
        "data_vintage": vintage,
        "trial_index": cfg.trial_index,
        "random_seed": cfg.random_seed,
        "notes": cfg.notes,
    }


def _save_metrics(metrics: dict, path: Path):
    """Save metrics, handling non-JSON-serializable values."""
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            clean[k] = {
                str(kk): float(vv) if isinstance(vv, (int, float, np.floating)) else str(vv)
                for kk, vv in v.items()
            } if all(isinstance(vv, (int, float, str, type(None))) for vv in v.values()) else str(v)
        elif isinstance(v, (int, float)):
            clean[k] = v
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        else:
            clean[k] = str(v)

    path.write_text(json.dumps(clean, indent=2, default=str))

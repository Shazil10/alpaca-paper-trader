# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full daily trading cycle (requires env vars)
export PYTHONPATH=src
python src/universe.py   # Refresh S&P 500/400/600 stock universe
python src/trade.py      # Generate signals and place orders
python src/report.py     # Generate orders report (CSV/MD/HTML)

# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_budget.py -v   # Budget accounting tests only
```

Required env vars: `ALPACA_KEY`, `ALPACA_SECRET` (paper trading account).

## Architecture

This is a multi-strategy Alpaca paper trading bot. GitHub Actions runs the daily cycle: `universe.py → trade.py` at 9:30 AM ET, `report.py` at 4:30 PM ET.

### Execution flow (`src/trade.py`)

`trade.py` is the main orchestrator. It:
1. Loads strategy modules dynamically via `importlib` based on `config.STRATEGY_ALLOCATIONS`
2. Calls `generate_signals(budget, strategy_id, held_symbols)` on each strategy
3. Executes SELL signals first, then BUY signals
4. Maintains a 10% cash reserve (never deployed)
5. Prevents double-buys by checking existing positions + open orders

### Strategy contract

Every strategy module must expose exactly:
```python
def generate_signals(budget: float, strategy_id: str, held_symbols: Set[str]) -> List[Signal]
```

Strategies own selection, sizing, and exit logic. `trade.py` handles execution only.

### Budget accounting (`src/trade_models.py`)

Budgets are **lifetime caps**, not daily resets. `committed_dollars_from_orders()` computes deployed capital by scanning all orders with matching `client_order_id` prefix (`f"{strategy_id}:"`). Sell proceeds automatically recycle back into the remaining budget. Orders are tagged `client_order_id=f"{strategy_id}:{uuid().hex[:16]}"` for attribution.

### Adding a new strategy

1. Create `src/strategies/<category>/<name>.py` with a `generate_signals()` function
2. Add an entry to `STRATEGY_ALLOCATIONS` in `src/config.py` with the module path and dollar budget
3. The strategy is automatically discovered and called by `trade.py`

## Strategies

| Strategy | Budget | Module |
|---|---|---|
| Clenow Trend (momentum) | $5k | `strategies.momentum.clenow_trend` |
| Ranked Sector Allocation V10 | $15k | `strategies.ranks.ranked_asset_alloc` |
| High Pullback Reversion | $10k | `strategies.mean_reversion.high_pullback_reversion` |

**Clenow Trend**: 60-day log-price momentum (slope × R²). Entry requires price > 200-SMA, 30d score ≥ 50% of 60d score, and price ≥ 75% of 52-week high. Regime filter: ≥2 of SPY/IJH/IJR above 200-SMA. Inverse-volatility sizing, top 7 picks.

**Ranked Sector Allocation (V10 DAF)**: Monthly rebalance. 50/50 blend of V4-Best (sector momentum rotation) and V8-AW (all-weather with TLT/GLD/UUP hedges in bear). 2× leverage when sector vol is in the bottom 35th percentile and regime is bull.

**High Pullback Reversion**: Delegates to `src/strategies/mean_reversion/52W_mean_reversion_strat.py`. Targets stocks near 52-week lows.

## Key data models

- `Signal(symbol, side, reason, notional, strategy_id)` — broker-agnostic trade intent
- `Side` enum — `BUY` / `SELL`
- Universe is stored in a local file and refreshed daily by `universe.py` (scrapes S&P 500/400/600, filters by price ≥ $10 and dollar volume ≥ $10M)

## Research vs. production

Backtesting and strategy research live in Jupyter notebooks under `analysis/` and `Assignment/`. The `src/` directory is production-only. There is no shared backtesting framework — notebooks use standalone data pulls via `yfinance`.

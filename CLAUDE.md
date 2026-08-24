# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies. NOTE: venv/bin/pip has a stale shebang pointing at a
# different project, so always go through the interpreter.
./venv/bin/python -m pip install -r requirements.txt

# Run the full daily trading cycle (requires env vars)
export PYTHONPATH=src
python src/universe.py                    # Refresh S&P 500/400/600 universe
python src/data_pipeline/sync_prices.py   # Update the price lake
python src/trade.py                       # Generate signals and place orders
python src/report.py                      # Orders report (CSV/MD/HTML)

# Run tests
./venv/bin/python -m pytest tests/ -v
./venv/bin/python -m pytest tests/test_budget.py -v   # Budget accounting only

# Is it safe to switch strategies onto the price lake?
PYTHONPATH=src python scripts/check_lake_readiness.py
```

Required env vars: `ALPACA_KEY`, `ALPACA_SECRET` (paper trading account).
Optional: `PRICE_SOURCE` (`yfinance` default, or `lake`).

## Architecture

Multi-strategy Alpaca paper trading bot. GitHub Actions runs the daily cycle:
`universe.py → sync_prices.py → commit → trade.py → report.py` at 9:30 AM ET,
then `report.py → export_portfolio.py → commit` at 4:30 PM ET.

### Price data (`src/data_pipeline/`)

One shared lake of daily bars in `data/prices/daily/`, replacing the per-strategy
`yf.download` calls that each invented their own window. Full contracts live in
`data/prices/_schema.md`; the load-bearing ones:

- Strategies read **`adj_close`**, matching what `auto_adjust=True` used to give.
- The lake holds **completed sessions only**. The 9:30 job runs at market open,
  so a same-day bar would be a partial intraday quote stored as a close.
- Cold years are Parquet; the current year is **CSV**, because git deltas
  append-only text cheaply while a recompressed Parquet blob is near-unshareable
  between commits.
- **Completeness is enforced.** One missing bar makes `rolling(50)` return NaN,
  which silently drops a symbol from threshold comparisons and changes signals.
  This was observed: five sector ETFs each short 2-3 bars flipped the rotation
  sleeve's regime from bull to neutral with no error raised.

`PRICE_SOURCE` selects the backend (`yfinance` default, `lake` opt-in). There is
no fallback between them — a silent fallback would mean reasoning over a
different window while reporting success. Lake or skip.

Migrations are verified by canaries (`tests/test_ranked_canary.py`,
`tests/test_pullback_canary.py`) against fixtures recorded from the *pre-lake*
yfinance path. They assert on decisions, never prices: `adj_close` agrees across
sources to float32 precision, not bit-exactly.

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

Budgets below mirror `STRATEGY_ALLOCATIONS` in `src/config.py`, which is the single source of truth. Update both together.

| Strategy | Budget | Module |
|---|---|---|
| Clenow Trend (momentum) | $10k | `strategies.momentum.clenow_trend` |
| Ranked Sector Allocation V10 | $15k | `strategies.ranks.ranked_asset_alloc` |
| High Pullback Reversion | $15k | `strategies.mean_reversion.high_pullback_reversion` |

Total deployable cap: **$40k** (a 10% cash reserve is held back at execution time).

**Clenow Trend**: 60-day log-price momentum (slope × R²). Entry requires price > 200-SMA, 30d score ≥ 50% of 60d score, and price ≥ 75% of 52-week high. Regime filter: ≥2 of SPY/IJH/IJR above 200-SMA. Inverse-volatility sizing, top 7 picks.

**Ranked Sector Allocation (V10 DAF)**: Monthly rebalance. 50/50 blend of V4-Best (sector momentum rotation) and V8-AW (all-weather with TLT/GLD/UUP hedges in bear). 2× leverage when sector vol is in the bottom 35th percentile and regime is bull.

**High Pullback Reversion**: Delegates to `src/strategies/mean_reversion/52W_mean_reversion_strat.py`. Targets stocks near 52-week lows.

## Key data models

- `Signal(symbol, side, reason, notional, strategy_id)` — broker-agnostic trade intent
- `Side` enum — `BUY` / `SELL`
- Universe is stored in a local file and refreshed daily by `universe.py` (scrapes S&P 500/400/600, filters by price ≥ $10 and dollar volume ≥ $10M)

## Research vs. production

Backtesting and strategy research live in Jupyter notebooks under `analysis/` and `Assignment/`. The `src/` directory is production-only.

Notebooks still maintain their own `.cache/*.pkl` pulls and have **not** been migrated to the price lake. They are free to opt in via `data_pipeline.store`, but nothing forces it.

## Known hazards

- **`src/strategies/momentum/clenow_trend.py` is 0 bytes locally** (iCloud
  eviction). The GitHub copy is intact and the sleeve trades fine in CI, since
  Actions checks out from the remote. But `src/` is not gitignored, so a broad
  `git add -A` would push the empty file over the working copy and silently kill
  that sleeve — it would stop placing orders rather than erroring. Always stage
  explicit paths. Restore it from `origin/main` before touching that module.
  `tests/test_strategy_integration.py` fails locally for this reason alone.
- `venv/bin/pip` has a stale shebang pointing at a different project on disk.
  Installs must use `./venv/bin/python -m pip`, or they land in the wrong
  environment silently.
- Committing the lake requires `git add -f` for cold Parquet years: they are
  covered by the ignore-with-exceptions rules in `.gitignore`, and the exception
  list is maintained by hand at each January rollover.

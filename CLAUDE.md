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

## Backtesting platform (`src/backtest/`)

Reusable daily stock/ETF backtester. Write strategy logic once, select dates and
capital, run, and receive a verdict on whether the strategy may have alpha.

```bash
# One backtest from a YAML config
PYTHONPATH=src python -m src.backtest.runner --config configs/backtests/tsmom_etf.yaml

# The full validation suite and a verdict card
PYTHONPATH=src python -m src.backtest.runner --config configs/backtests/tsmom_etf.yaml --research

# Multi-strategy fund: one cash balance, orders netted across sleeves
PYTHONPATH=src python -m src.backtest.runner --fund configs/funds/three_sleeve.yaml

# One HTML page comparing runs against SPY, equal weight, and each other
PYTHONPATH=src python scripts/build_report.py --latest-per-strategy

# What the data does and does not support
PYTHONPATH=src python scripts/audit_data.py
```

The workflow these compose into:

```
write target_weights()  ->  YAML config  ->  --research for a verdict
                                          ->  --fund to see it beside the others
                                          ->  build_report.py to compare versions
```

**Verdict discipline.** A research pass grades eight sections and any single FAIL
caps the verdict. NOT RUN never counts as a pass, and a pass needs a quorum of five
scored sections — both rules exist because absence of evidence had twice been
scoring as good evidence. The out-of-sample holdout is locked: `--unlock-holdout`
is required to read it and every unlock appends to `runs/holdout_unlocks.log`.

**Strategy interface.** Canonical form is target weights:
```python
def target_weights(ctx: StrategyContext) -> dict[str, float]:
    """Return {symbol: fraction_of_sleeve_equity}. Remainder is cash."""
```

Existing `generate_signals()` strategies work via adapter (`src/backtest/adapter.py`).

**Key modules:**
- `engine.py` — daily clock, D+1 open fills, PIT-enforced context
- `context.py` — `StrategyContext` is the PIT firewall; requesting data past `as_of` raises `LookaheadError`
- `portfolio.py` — FIFO lot tracking, cash, realized/unrealized PnL
- `broker.py` — simulated fills with configurable slippage, commission, participation cap
- `metrics.py` — consolidated Sharpe, Sortino, alpha, beta, PSR, Deflated Sharpe, regime analysis
- `fast.py` — vectorized alpha screen for parameter sweeps
- `fund.py` — one cash balance across sleeves, orders netted before they reach the market, fund-level exposure limits, per-sleeve attribution retained internally
- `research.py` — runs every validation section in one pass and grades it
- `report.py` — self-contained HTML comparison; inline SVG, no JavaScript
- `validation/` — parameter stability, Monte Carlo, clustering, walk-forward, PBO, verdict card

**Strategies on the interface** (all five; each is decision logic only):
`strategies.ranks.ranked_target_weights`, `strategies.momentum.clenow_target_weights`,
`strategies.mean_reversion.pullback_target_weights`, `strategies.momentum.tsmom_etf`,
`strategies.momentum.triple_trigger`. Each has a parity gate under `tests/backtest/`
asserting it makes the same decisions as the implementation it was extracted from.

**Price levels vs returns.** `adj_close` is back-adjusted to the *download* date, so
historical levels know about splits that had not happened yet. Returns are fine;
levels are not. Use `ctx.prices()` for returns and momentum, and
**`ctx.raw_close(symbol)` for anything compared against an absolute dollar amount**
— minimum-price screens, round-lot sizing, dollar-volume filters. On 2010-06-30
adjusted AAPL reads ~$8 against a $251 print, so a `price >= 10` screen on the
adjusted scale silently drops a name that was never cheap. `BacktestConfig.price_adjustment`
(`"asof"` default / `"today"`) additionally re-anchors `adj_close` to `end_date`
so a run cannot see corporate actions that postdate its own window.

**Data extensions:**
- `data_pipeline/adjust.py` — as-of-date price adjustment (research/notebook tool; the engine anchors via `runner.load_panels`)
- `data_pipeline/membership.py` — PIT S&P 500 membership via `members_asof(date)`
- `data_pipeline/securities.py` — security master with permanent IDs (ticker recycling guard) and `sector_map()`, which is what makes `RiskConfig.max_sector_pct` bind
- `data_pipeline/providers/tiingo.py` — free-tier delisted stock backfill (needs an API key; not yet wired)

**Data build steps** (manual, in order; see `data/universe/_schema.md`):
```bash
PYTHONPATH=src ./venv/bin/python scripts/build_membership.py     # PIT index tape
PYTHONPATH=src ./venv/bin/python scripts/build_securities.py     # security master + sectors
PYTHONPATH=src ./venv/bin/python scripts/backfill_history.py --start 2005-01-01 --dry-run
PYTHONPATH=src ./venv/bin/python scripts/check_lake_readiness.py --backtest
```
`backfill_history.py` is the only way to deepen the lake — `sync_prices` clamps a
known symbol's window to `max(LOOKBACK_START, last_stored - 5d)`, so it fetches
nothing at all when asked for older history. Closed years land as Parquet and each
new year needs an explicit `!data/prices/daily/<year>.parquet` line in `.gitignore`.

`check_lake_readiness.py` has two modes: bare for live-trading readiness, and
`--backtest` for history depth and point-in-time coverage. They answer different
questions and a lake can pass one while failing the other.

**Run artifacts** saved to `runs/<date>_<strategy>_<hash>/`: config, equity curve, returns,
orders, fills, positions, metrics, summary.

## Research vs. production

Backtesting and strategy research live in Jupyter notebooks under `analysis/` and `Assignment/`. The `src/` directory is production-only.

Notebooks still maintain their own `.cache/*.pkl` pulls and have **not** been migrated to the price lake. They are free to opt in via `data_pipeline.store`, but nothing forces it.

Performance metrics *are* shared. Four notebooks used to define their own
`perf_metrics`, which is how `Sortino` came to mean two different things in one
repository; they now import `backtest.metrics.summary_from_returns` (from a return
series) or `summary_from_equity` (from an equity curve). Do not reintroduce a local
copy. `strategies/ranks/v10_research_pipeline.py` likewise re-exports
`compute_stats` / `compute_full_stats` / `apply_transaction_costs` / `quick_stats`
from `backtest.metrics` rather than defining them.

## Known hazards

- **`src/strategies/momentum/clenow_trend.py`** — previously reported as 0 bytes
  (iCloud eviction). Verified 2026-09-20: file is 15,460 bytes locally and matches
  the GitHub copy. If iCloud evicts it again, restore from `origin/main`. A broad
  `git add -A` with an empty file would silently kill the sleeve. Always stage
  explicit paths.
- `venv/bin/pip` has a stale shebang pointing at a different project on disk.
  Installs must use `./venv/bin/python -m pip`, or they land in the wrong
  environment silently.
- Committing the lake requires `git add -f` for cold Parquet years: they are
  covered by the ignore-with-exceptions rules in `.gitignore`, and the exception
  list is maintained by hand at each January rollover.

<!-- cce-block-version: 4 -->
## Context Engine (CCE)

This project uses Code Context Engine for intelligent code retrieval and
cross-session memory.

### Searching the codebase

**You MUST use `context_search` instead of reading files directly** when
exploring the codebase, answering questions about code, or understanding how
things work. This is a hard requirement, not a suggestion. `context_search`
returns the most relevant code chunks with confidence scores instead of whole
files, and tracks token savings automatically.

When to use `context_search`:
- Answering questions about the codebase ("how does X work?", "where is Y?")
- Exploring structure or architecture
- Finding related code, functions, or patterns
- Any time you would otherwise read a file just to understand it

When to use `Read` instead:
- You need to edit a specific file (read before editing)
- You need the exact, complete content of a known file path

Other search tools:
- `expand_chunk` — get full source for a compressed result
- `related_context` — find what calls/imports a function

### Cross-session memory — use it actively

This project has persistent memory across Claude Code sessions. **You must
use it both ways: recall before answering, record after deciding.** Memory
that is not recorded is lost; memory that is not recalled does nothing.

**Before answering a non-trivial question, call `session_recall`.**
Especially when:
- The question touches architecture, design, or naming choices
- The user asks "what / why / how did we ..."
- You are about to recommend an approach the team may have already chosen
  or already rejected

Pass a topic phrase, not a single word — e.g. `session_recall("auth flow")`,
not `session_recall("auth")`. Recall is vector-similarity-based, so paraphrases
match. If recall returns relevant entries, lead with them ("Per a prior
decision: ...") instead of re-deriving the answer.

**After making a non-obvious decision, call `record_decision`.** Especially:
- Choosing one library / pattern / approach over another
- Resolving an ambiguity in the spec or requirements
- Establishing a convention the project should follow going forward
- Anything you would not want to re-litigate next session

Format: `record_decision(decision="...", reason="...")`. Keep both fields
short and specific — they are surfaced verbatim at the start of future
sessions.

**After meaningful work in a file, call `record_code_area`.** Especially when:
- You added or substantially modified a function/class
- You traced through a non-obvious flow and want future-you to find it fast

Format: `record_code_area(file_path="...", description="...")`.

Skip recording for trivial reads, formatting changes, or one-off lookups —
the goal is durable signal, not an event log.

### Drilling deeper from a recall hit

`session_recall` results are tagged with the source session id, e.g.
`[turn sid:abc123|n:5]`. To drill in:

- `session_timeline(session_id="abc123")` — walk the per-turn summaries of
  that session in order. Use this when the user asks "what was the
  reasoning?" or "how did we get there?".
- `session_event(event_id=N)` — fetch a specific tool event's raw input
  and output (capped at 4 KB at read time). Use this when a turn summary
  references a tool result you actually need to inspect.

Both are read-only and cheap. Prefer them over re-running tool calls or
asking the user to re-paste context.

### Output style

Respond in compressed style. Drop articles (a, an, the) in prose. Use
sentence fragments over full sentences. Use short synonyms (fix not resolve,
check not investigate). Pattern: [thing] [action] [reason]. [next step].
No filler, hedging, pleasantries, trailing summaries, or restating what
the user said. One sentence if one sentence is enough.

When suggesting code changes, show only the changed lines with 3 lines of
context. Never rewrite entire files. Multiple changes in one file: show each
change separately. Never echo back unchanged code the user already has.

Code blocks, file paths, commands, error messages: always written in full.
Security warnings and destructive action confirmations: use full clarity.
<!-- /cce-block -->

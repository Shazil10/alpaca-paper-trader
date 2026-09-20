# Repository Guide

This document is a companion to `strategy_performance_review.ipynb`. It walks through the public GitHub repository at https://github.com/Shazil10/alpaca-paper-trader so you can navigate the live code, the GitHub Actions cron jobs, the historical artifacts, and the research notebooks without needing to clone anything locally.

The notebook is the analytical submission. This guide is the engineering map.

---

## TL;DR

A multi-strategy paper-trading bot that runs on Alpaca's paper API. Three algorithmic sleeves (Clenow Trend, Ranked Asset Allocation V10, 52-week pullback mean reversion) plus a discretionary book that I traded by hand from the Alpaca dashboard. Everything is automated end-to-end on GitHub Actions: signals fire at 9:30 AM ET, end-of-day reports fire at 4:30 PM ET, and the resulting CSV/MD/HTML artifacts are downloadable from each workflow run.

The repository is public. Every commit, every workflow run, and every report artifact is browsable through the GitHub UI without any credentials.

---

## Top-level layout

```
alpaca-paper-trader/
├── .github/workflows/      # GitHub Actions cron jobs (the trading "pulse")
├── src/                    # Production code only. No notebooks here.
├── analysis/               # Research notebooks. One per candidate strategy.
├── tests/                  # pytest suite (budget accounting, integration)
├── reports/                # Persisted run outputs (CSV, MD memos, HTML)
├── Assignment_final/       # This submission (the audit notebook + this guide)
├── Assignment/             # Earlier in-class assignments
├── scripts/                # One-off helpers
├── universe.csv            # Cached S&P 500 / 400 / 600 universe
├── requirements.txt        # Pinned Python deps
├── CLAUDE.md               # Architecture cheatsheet
└── README.md               # One-paragraph project intro
```

The split between `src/` (production) and `analysis/` (research) is the most important structural rule. Anything in `src/` is what GitHub Actions actually runs. Anything in `analysis/` is research that may or may not have been promoted. The deep dives in the audit notebook explicitly call out which sleeves have been promoted from `analysis/` to `src/`, and which have not.

---

## How a daily trade actually happens

The whole trading cycle is two GitHub Actions workflows. They are the only thing standing between a strategy idea and an Alpaca paper order.

### Morning workflow: `Daily Trading Bot`

File: `.github/workflows/run_bot.yml`. Cron: `30 14 * * 1-5` (9:30 AM ET in standard time).

```
GitHub Actions runner
   │
   ├── checkout repo
   ├── set up Python 3.9, install requirements.txt
   ├── inject ALPACA_KEY / ALPACA_SECRET from GitHub Secrets
   │
   ├── python src/universe.py
   │     ↓ scrapes S&P 500/400/600 from Wikipedia, filters by price ≥ $10
   │       and dollar volume ≥ $10M, writes universe.csv (~1400 names)
   │
   ├── python src/trade.py
   │     ↓ orchestrates the daily run:
   │       1. importlib-loads each module in config.STRATEGY_ALLOCATIONS
   │       2. calls strategy.generate_signals(budget, strategy_id, held_symbols)
   │       3. executes SELLs first, then BUYs
   │       4. tags every order with client_order_id = "{strategy_id}:{uuid16}"
   │       5. enforces a 10% cash reserve account-wide
   │       6. blocks duplicate-symbol buys across sleeves
   │
   └── python src/report.py
         ↓ pulls the day's order tape from Alpaca, FIFO-matches sells against
           prior buys to compute realized PnL, writes:
             reports/orders_latest.csv
             reports/orders_latest.md
             reports/orders_latest.html
         and uploads them as the workflow artifact `trading-reports`.
```

### Afternoon workflow: `Close Report`

File: `.github/workflows/close_report.yml`. Cron: `30 21 * * 1-5` (4:30 PM ET in standard time).

Just runs `python src/report.py` again to generate the end-of-day version with all intraday fills (including any discretionary trades I placed by hand during the session). Uploads the same three files as the workflow artifact `trading-reports-close`. This is the artifact whose `orders_latest.csv` is the source of truth for the audit notebook.

### How to inspect any historical run

Every workflow run keeps its artifact for 90 days. To see what happened on any given day:

1. Open https://github.com/Shazil10/alpaca-paper-trader/actions.
2. Pick a run from `Close Report` (for the most complete daily picture).
3. Scroll to the bottom of the run page to download `trading-reports-close.zip`.
4. Inside, `orders_latest.html` is human-readable; `orders_latest.csv` is the machine-readable version.

The `Assignment_final/data/orders_latest.csv` in this submission was downloaded the same way.

---

## `src/` walkthrough

Production code. This is the entire trading stack.

### `src/trade.py` (orchestrator, ~360 lines)

Everything starts here. Loads strategy modules dynamically from `config.STRATEGY_ALLOCATIONS`. For each strategy, computes how much of its lifetime budget has already been deployed (by scanning historical orders with the matching `client_order_id` prefix), passes the remaining budget plus the set of currently-held symbols into `generate_signals()`, and gets back a list of `Signal` objects. Then it executes those signals in a safe order: SELLs first (to recycle cash), then BUYs (constrained by the per-strategy remaining budget and the account-wide 10 percent cash reserve).

Key safety rails: it never lets a strategy exceed its lifetime budget; it never double-buys a symbol that already has an open position or open order; it logs every decision with full reasons.

### `src/config.py` (40 lines)

The single source of truth for which strategies are live and what their budgets are.

```python
STRATEGY_ALLOCATIONS: dict[str, float] = {
    "strategies.momentum.clenow_trend":                  5_000,
    "strategies.ranks.ranked_asset_alloc":              15_000,
    "strategies.mean_reversion.high_pullback_reversion": 10_000,
}
```

To enable or disable a strategy, comment out its line and push. To rebudget, change the dollar number and push. The next workflow run picks it up automatically.

### `src/trade_models.py` (~100 lines)

Defines the `Signal` dataclass (the broker-agnostic trade intent), the `Side` enum, and the `committed_dollars_from_orders()` helper that scans Alpaca's order history to compute "how much budget have I already used" by parsing the `client_order_id` prefix. Lifetime budgets are accounted by client-order-id prefix, not by date, so sells automatically recycle into the budget.

### `src/orders.py`

Wraps the Alpaca SDK so that `trade.py` never touches the broker API directly. Handles fractional shares, market-order placement, and order tagging.

### `src/report.py`

End-of-day report generator. Pulls the full order list from Alpaca, FIFO-matches sells against prior buys for realized-PnL computation, and writes `orders_latest.{csv,md,html}` to `reports/`. The CSV is the canonical input to the audit notebook.

### `src/universe.py`

Refreshes `universe.csv` from Wikipedia's S&P 500 / 400 / 600 component lists, filters by price floor and dollar-volume floor.

### `src/strategies/`

One folder per strategy family. Each strategy module exposes exactly one entry point:

```python
def generate_signals(*, budget: float, strategy_id: str, held_symbols: set[str]) -> list[Signal]
```

The current modules are:

| File                                                           | What it does                                                       |
| -------------------------------------------------------------- | ------------------------------------------------------------------ |
| `momentum/clenow_trend.py`                                     | Clenow score, regime gate, inverse-vol sizing, exit on rank or SMA |
| `ranks/ranked_asset_alloc.py`                                  | Live sector ETF rotation (the one that actually trades)            |
| `ranks/v10_research_pipeline.py`                               | Research version of V10 ensemble; not invoked by `trade.py`        |
| `mean_reversion/52W_mean_reversion_strat.py`                   | 40 percent pullback mean-reversion logic                           |
| `mean_reversion/high_pullback_reversion.py`                    | Stable import alias that delegates to the file above               |

The `high_pullback_reversion.py` alias is a bridge module added because `config.py` referenced one path and the implementation lived under a different filename. The alias keeps the historical strategy ID stable (so client_order_id tags from earlier weeks still attribute correctly) while letting the implementation file have its own name.

---

## `analysis/` walkthrough (research notebooks)

Where strategies are designed, backtested, and either promoted to `src/` or shelved.

```
analysis/
├── universe_analysis.ipynb              # Sector / market-cap / liquidity diagnostics
└── strategies/
    ├── momentum/
    │   ├── clenow_trend.ipynb              # Retrospective Clenow analysis (built post-deploy)
    │   ├── ranked_asset_allocation.ipynb   # V10 research, the canonical ranking backtest
    │   ├── ranked_2_asset_allocation.ipynb # Earlier two-pipeline blend (precursor to V10)
    │   ├── triple_trigger_momentum.ipynb   # Momentum variant; shelved (no edge over Clenow)
    │   └── deep_momentum.ipynb             # LSTM forecaster; not deployed (CV concerns)
    └── mean_reversion/
        ├── 52W_mean_reversion.ipynb        # 52-week pullback strategy notebook (final version)
        ├── high_pullback_reversion.ipynb   # Sister notebook with the param sweeps
        └── Draft_52W_mean_reversion.ipynb  # Earlier draft, kept for history
```

Two of these (Deep Momentum and 52W Pullback) are the "coded but not yet deployed" pair the audit notebook discusses in detail. The other notebooks are either superseded (`ranked_2_asset_allocation` evolved into V10) or shelved (`triple_trigger_momentum`).

Notebooks here are not invoked by GitHub Actions. They are research artifacts, run manually, and consume their own data downloads via `yfinance`. Promoting a notebook to live means writing a sibling `.py` module under `src/strategies/` that exposes the `generate_signals()` contract, then adding it to `config.STRATEGY_ALLOCATIONS`.

---

## `reports/` walkthrough

Persisted artifacts from runs and analyses.

| File                                       | What it is                                                    |
| ------------------------------------------ | ------------------------------------------------------------- |
| `orders_latest.csv` / `.md` / `.html`      | Latest end-of-day order tape from `src/report.py`             |
| `strategy_triage_decision_memo.md`         | The brutal sleeve-by-sleeve audit I wrote in March            |
| `strategy_comparison_metrics.csv`          | Backtest metrics for each sleeve, with comparability flags    |
| `portfolio_combination_proposal.md`        | Rationale for the 35 / 15 / 50 baseline allocation            |
| `portfolio_combination_evidence_index.md`  | Pointer index: every numeric claim → its source file/notebook |

These are checked into git so the analysis is reproducible without re-running the workflows.

---

## `tests/` walkthrough

```
tests/
├── test_budget.py                # Budget accounting (lifetime cap, recycle on sell)
└── test_strategy_integration.py  # Strategy-import + config integrity regression tests
```

Run with `python -m pytest tests/ -v`. The integration tests exist specifically to catch the class of bug that broke mean-reversion in March (configured strategy path that did not import).

---

## `Assignment_final/` (this folder)

```
Assignment_final/
├── strategy_performance_review.ipynb   # The audit notebook (the analytical submission)
├── strategy_performance_review.html    # Static HTML export of the notebook
├── REPO_GUIDE.md                       # This file
├── data/
│   └── orders_latest.csv               # Frozen snapshot of the live order tape
└── _build_notebook.py                  # Builder script that regenerates the notebook
```

The notebook is the primary submission. This guide is the secondary submission. The HTML export is a convenience: it renders the notebook in any browser, with all figures and tables, no Python install required.

---

## How the algorithmic sleeves work (one-paragraph each)

For full mathematical detail see the deep-dive sections in the audit notebook.

### Clenow Trend (momentum, single-name equities)

For each name in the universe, fit a linear regression on 60 days of log prices. Take the regression slope and the R-squared. The Clenow score is `(exp(slope) ^ 252) * R^2`. Rank names by score, take the top 7 that pass three quality filters (price above 200-SMA, recent momentum still strong, price within 25 percent of 52-week high), inverse-vol weight them within a [2 percent, 10 percent] band, and only place new buys when the regime gate is risk-on (at least two of SPY / IJH / IJR above their 200-SMA). Sell on rank fall-off or when price closes below the 50-SMA.

### Ranked Asset Allocation V10 (sector ETF rotation, monthly)

A 50/50 blend of two sector-rotation strategies. V4 picks top-K sector ETFs by a composite of momentum / vol / correlation / ATR-trend, with a conviction overweight when the top sector is more than two standard deviations better than the rest. V8 holds concentrated sectors in bull and neutral, but rotates to the top hedge ETFs (TLT, GLD, UUP, FXY, FXF) by 42-day momentum in bear regimes. The blended weights then get scaled 1x or 2x via the Dual-Filter Leverage rule (2x only when sector vol is in the bottom 35th percentile of its 21-day distribution and the regime is bull).

### High Pullback Reversion (52-week pullback mean reversion, single-name equities)

Buy stocks that have pulled back at least 40 percent from their 52-week high but still pass quality gates (price at least 50 percent of 52-week high to filter out free-fallers, price at least 90 percent of 200-day MA to filter out structurally broken names). Exit when the stock recovers to within 25 percent of its 52-week high, hits a 25 percent stop from entry, or has been held for 63 trading days. Pullback-weighted sizing: the deeper the pullback (subject to the quality gates), the larger the allocation.

---

## How to reproduce the audit notebook locally

```bash
# 1. clone the repo
git clone https://github.com/Shazil10/alpaca-paper-trader
cd alpaca-paper-trader

# 2. set up the venv
python -m venv venv
venv/bin/pip install -r requirements.txt

# 3. (optional) refresh the order tape from the latest GitHub Actions artifact
gh run download $(gh run list --workflow "Close Report" --json databaseId --jq '.[0].databaseId') \
  --dir /tmp/latest && \
cp /tmp/latest/trading-reports-close/orders_latest.csv Assignment_final/data/orders_latest.csv

# 4. rebuild the notebook end-to-end
venv/bin/python Assignment_final/_build_notebook.py
```

Step 3 is optional; the CSV in `Assignment_final/data/` is already the snapshot used for this submission.

---

## What to read in what order if you only have 20 minutes

1. The first three sections of `strategy_performance_review.ipynb` (the title, the data section, and the weekly reflections). That gives you the narrative.
2. The portfolio tear sheet section and the per-strategy forensics section. That gives you the numbers.
3. The two deep dives (Clenow, V10). That gives you the technical detail and the honest engineering admissions.
4. The final reflection. That is the forward-looking part the prompt actually asks for.
5. This guide, only if you want to dig into the live code on GitHub.

The discretionary, correlation, and "coded but not deployed" sections are useful but not essential to the assignment grading.

---

## Repository link

https://github.com/Shazil10/alpaca-paper-trader

The repo is public for the duration of grading. All workflow runs, all artifacts, all code, all commits are visible without credentials.

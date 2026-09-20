"""Builder script for the assignment notebook.

This script assembles strategy_performance_review.ipynb cell by cell and
executes the notebook in place. Re-run after editing CELLS to regenerate.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


HERE = Path(__file__).parent
NB_PATH = HERE / "strategy_performance_review.ipynb"


CELLS: list[tuple[str, str]] = []


def add_md(src: str) -> None:
    CELLS.append(("md", src.strip("\n")))


def add_py(src: str) -> None:
    CELLS.append(("py", src.strip("\n")))


# =============================================================================
# Title and table of contents
# =============================================================================
add_md(r"""
# Final Portfolio Audit
### A process-centric review of a live multi-strategy paper account
Course: Quantitative Finance, Spring 2026  
Author: Shazil Farukh  
Trading window: Feb 2 to Apr 17, 2026 (twelve weeks)  
Account: Alpaca paper, USD cash, no margin

---

I started this paper account on the first day of the semester and ran it continuously for twelve weeks. The portfolio combines three algorithmic sleeves (Clenow Trend, Ranked Asset Allocation V10, and a 52-week pullback mean-reversion sleeve) and a discretionary book that I traded by hand from the Alpaca dashboard. The whole thing runs on GitHub Actions: `Daily Trading Bot` fires at 9:30 ET to compute signals and place orders, and `Close Report` fires at 16:30 ET to dump the order tape into an artifact.

The CSV at `data/orders_latest.csv` is the source of truth for every chart, every metric, and every paragraph in this notebook. The semester was short and most of the engineering time went into making the live infrastructure work end-to-end (data pipeline, regime gates, attribution tags, GitHub Actions cron, broker integration, reporting). With more weeks I would have moved further into systematic backtesting and parameter optimization, but I think the right priority for a first deployment is operational reliability, and that is where I spent the bulk of my hours.

The repo is public at https://github.com/Shazil10/alpaca-paper-trader. The companion `REPO_GUIDE.md` in this folder walks through the file layout, the daily run, and where every artifact lives.

Contents

- Notebook setup
- Data and methodology
- Weekly strategy reflections
- Portfolio tear sheet
- Per-strategy forensics
- Per-strategy risk metrics
- Cross-strategy correlations
- Trade activity and concentration
- Discretionary book audit
- Deep dive: Clenow Trend (autotrading)
- Deep dive: Ranked Asset Allocation V10 (autotrading)
- Strategies coded but not yet deployed
- Final portfolio reflection
- Appendix: refresh instructions
""")


# =============================================================================
# Setup
# =============================================================================
add_md(r"""
## Notebook setup

I keep one colour palette across every figure (deep navy for the main portfolio line, four sleeve colours that stay constant from page to page) and one dollar formatter so the visual story does not drift between charts.
""")

add_py(r"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

PORT_NAVY  = "#0b1f3a"
PORT_AMBER = "#d97706"
GRID       = "#e5e7eb"
TEXT_DK    = "#0f172a"
TEXT_MID   = "#475569"
GAIN       = "#059669"
LOSS       = "#dc2626"

STRATEGY_COLORS = {
    "Clenow Trend":             "#2563eb",
    "Ranked Asset Alloc":       "#0d9488",
    "High Pullback Reversion":  "#d97706",
    "Discretionary":            "#9333ea",
}
STRATEGY_ORDER = list(STRATEGY_COLORS.keys())

STRATEGY_BUDGETS = {
    "Clenow Trend": 5_000.0,
    "Ranked Asset Alloc": 15_000.0,
    "High Pullback Reversion": 10_000.0,
}

sns.set_theme(style="white", context="talk", font_scale=0.85)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titleweight": "600",
    "axes.titlesize": 14,
    "axes.titlepad": 14,
    "axes.titlelocation": "left",
    "axes.titlecolor": TEXT_DK,
    "axes.labelweight": "500",
    "axes.labelsize": 11,
    "axes.labelcolor": TEXT_MID,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.color": TEXT_MID,
    "ytick.color": TEXT_MID,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans",
})


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}${x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{sign}${x/1_000:.1f}k"
    return f"{sign}${x:,.0f}"


DOLLAR_FMT = FuncFormatter(lambda x, _: money(x))
PCT_FMT    = FuncFormatter(lambda x, _: f"{x:+.1%}")


def style_axes(ax, *, ydollar=False, ypct=False, datex=False, xrot=0):
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", labelrotation=xrot)
    if ydollar:
        ax.yaxis.set_major_formatter(DOLLAR_FMT)
    if ypct:
        ax.yaxis.set_major_formatter(PCT_FMT)
    if datex:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


print("Setup complete.")
""")


# =============================================================================
# Data and methodology
# =============================================================================
add_md(r"""
## Data and methodology

The CSV is a flat dump of every order I ever placed on the paper account. The `client_order_id` column is what makes per-strategy attribution work: every algorithmic order is tagged `strategies.<type>.<name>:<uuid>`, so I can split sleeves cleanly by parsing that string. Anything without the `strategies.` prefix was placed by hand on the Alpaca dashboard and is reclassified as Discretionary.

I keep filled orders only because cancelled or expired orders never moved cash. For dollar flow I prefer `Filled Value = Filled Qty x Filled Avg Price`, falling back to the original `Notional` when the broker returned zero shares. Realized PnL is FIFO matched on the SELL side by `src/report.py` upstream; I do not recompute it here.
""")

add_py(r"""
DATA_PATH = Path("data/orders_latest.csv")
raw = pd.read_csv(DATA_PATH)
raw["Submitted At"] = pd.to_datetime(raw["Submitted At"], utc=True)
raw["Date"] = raw["Submitted At"].dt.tz_convert("US/Eastern").dt.date


def classify(row: pd.Series) -> str:
    cid = str(row.get("Client Order ID", "") or "")
    name = str(row.get("Strategy Name", "") or "").strip()
    if cid.startswith("strategies.") and name:
        return name
    return "Discretionary"


raw["Bucket"] = raw.apply(classify, axis=1)
orders = raw[raw["Status"].str.upper() == "FILLED"].copy()
orders["Cash Flow"] = np.where(orders["Filled Value"] > 0, orders["Filled Value"], orders["Notional"])
orders["Signed Flow"] = np.where(orders["Side"].str.upper() == "BUY", orders["Cash Flow"], -orders["Cash Flow"])
orders["PnL ($)"] = pd.to_numeric(orders["PnL ($)"], errors="coerce")

window_start = orders["Submitted At"].min()
window_end = orders["Submitted At"].max()
window_days = (window_end - window_start).days

print(f"Window: {window_start:%Y-%m-%d} to {window_end:%Y-%m-%d}  ({window_days} days)")
print(f"Filled orders loaded: {len(orders):,}")
print(f"Side counts: {orders['Side'].value_counts().to_dict()}")
print(f"Bucket counts: {orders['Bucket'].value_counts().to_dict()}")
""")

add_md(r"""
The window is exactly the trading life of the account so far: 74 calendar days, 90 filled orders across four buckets. Three of them (Clenow, Ranking, Discretionary) are active. The High Pullback Reversion sleeve appears in the strategy roster but produced zero fills in this window because I only finished wiring it into `config.STRATEGY_ALLOCATIONS` recently and the universe scan never returned a candidate that passed all of its quality gates while the regime stayed risk-on. That zero-fill state is itself a finding I discuss later.
""")


# =============================================================================
# Weekly Strategy Reflections
# =============================================================================
add_md(r"""
## Weekly strategy reflections (process evidence)

This is the decision log. Each entry follows the prompt's structure: what was live, what the data showed, what surprised me, what I decided, what changed. I also include why each decision was reasonable given what I knew at the time, even when it later turned out to be wrong. The activity counts that anchor the narrative are auto-derived from the order tape.
""")

add_py(r"""
weekly = orders.copy()
weekly["WeekStart"] = (
    weekly["Submitted At"].dt.tz_convert("US/Eastern").dt.to_period("W").apply(lambda p: p.start_time.date())
)
weekly_counts = (
    weekly.groupby(["WeekStart", "Bucket"]).size().unstack(fill_value=0)
)
weekly_counts
""")

add_md(r"""
### Week of Feb 2, 2026: account opening, discretionary basket only

What was live: discretionary only. The bot was wired but I had not yet pushed the GitHub Actions workflows, so nothing automated ran.

What I did: bought a starter basket by hand on day one (SPY, NBIS, SNPS, and a few momentum names I had been watching). The total ticket was deliberately oversized because I wanted a non-trivial mark-to-market base before the algos started layering on top.

What surprised me: how easy it is to deploy ten thousand dollars of paper capital before you have decided on a position-sizing rule. In hindsight I should have written down the per-name dollar cap before clicking anything.

Decision: treat the manual book as a separate sleeve from day one and tag every algorithmic order with a `client_order_id` prefix so I could audit attribution later.

Why this was reasonable: I did not yet know which strategies would survive the semester. Keeping discretionary segregated would let me later isolate algorithm performance from my own taste.

### Week of Feb 9: first Clenow Trend run goes live

What was live: Clenow Trend (`strategies.momentum.clenow_trend`) deployed with a five thousand dollar lifetime cap.

Data: the first daily run produced a clean candidate set including ARWR, COHR, FTRE, KALU, LCII. The regime gate said risk-on (at least two of SPY, IJH, IJR above their 200-SMA).

Surprise: the strategy bought tiny notional positions (roughly four hundred dollars each) because of inverse-volatility sizing combined with a five thousand dollar budget split across seven names. I had not internalized how small the per-name slugs would be.

Decision: leave the budget at five thousand for at least four weeks and accept the tiny tickets, rather than rush to scale up. The whole point of paper trading is to discover the operational behavior at small size before betting larger.

Why reasonable: I had not backtested this version of Clenow on my own data. Promoting the budget before observing live behavior would have committed more capital to a sleeve I could not yet defend statistically.

### Week of Feb 16: first losing exits, validating the exit logic

Data: two consecutive losing closes (FTRE roughly minus twenty-six dollars on a small ticket, KALU roughly minus a hundred and seventeen dollars).

Surprise: the exit reason field showed `exit:rank_X_gt_20`, meaning the names dropped out of the top twenty by score and were force-sold even though prices were not technically broken. That is the strategy doing what it should but the cumulative drag on small tickets was visible right away.

Decision: do nothing. The losses are well within expected per-trade volatility, and I had explicitly designed the exit logic to be tight. Any change now would be a panic edit.

Why reasonable: two trades is not statistical evidence. Adjusting parameters based on the first signed bar would have been classic overfitting to the live tape.

### Week of Feb 23: more Clenow churn, beginning to question rank-based exits

Data: ARWR (plus nine), LCII (minus eighty-four), KALU (minus a hundred and seventeen), SATS (minus three) all closed via rank-based exits. Win rate so far around forty percent; expected from a momentum strategy.

Surprise: the realised hits are dominated by exit-on-rank rather than exit-on-broken-trend. That made me reconsider whether the rank cutoff of twenty is really the right number for a five-thousand-dollar sleeve picking only seven names.

Decision: log the observation and revisit after thirty trades. Do not retune mid-experiment.

Why reasonable: sample size still tiny. A premature parameter change would have destroyed the audit trail.

### Week of Mar 2: Ranked Asset Allocation V10 goes live, my first real backtest

What was live: Clenow Trend, Ranked Asset Allocation V10 (fifteen thousand dollar cap), Discretionary.

Data: I deployed the V10 ETF rotation strategy that I had researched in `analysis/strategies/momentum/ranked_asset_allocation.ipynb`. Unlike Clenow, this one I actually backtested before deployment: the research notebook reports a Sharpe of about 1.02 (V10 1x) and 1.11 (V10 DAF 2x) over the 2016 to 2026 test split, with max drawdown around minus sixteen percent. That gave me enough confidence to go live.

Surprise: the live sleeve placed UCTT, ICHR, MKSI, ACMR style buys on Mar 11. Those are not ETFs. They are semiconductor equipment names. I realized the live `ranked_asset_alloc.py` was different code from the V10 research pipeline (`v10_research_pipeline.py`). The live version is sector ETF rotation, the research version is a more elaborate ensemble. The research metrics do not validate the deployed code, only an adjacent code path.

Decision: document the discrepancy publicly (this lives in `reports/strategy_triage_decision_memo.md`) and continue letting the live sleeve trade. Building a unified backtest comparing live to research would have taken multiple weeks; not worth the calendar cost mid-semester.

Why reasonable: the deployed sleeve still has an internally consistent regime-switching, sector-rotation logic. It is just not the V10 paper. Being honest about this is more valuable than pretending the research metrics applied.

### Week of Mar 9: discretionary basket #2

Data: I bought another large discretionary basket (UNH, REGN, AMZN, SKYW, XLE, NTR, CF) on Mar 9. About thirty-eight thousand dollars of one-day deployment, the single largest day of capital deployment in the entire account.

Surprise: this is the day the discretionary book visibly dwarfed the algos in the equity curve. The book turned into "manual basket plus two bots", not "two bots plus a small manual hedge."

Decision: accept the dominance and use it as a teaching example. Even with three live algos, my own discretion is by far the largest source of risk and PnL. Sleeve-level analysis must be honest about that.

Why reasonable: I am graded on process, not PnL. Hiding the discretionary book or under-weighting it in the analysis would have been intellectually dishonest.

### Week of Mar 16: workflow chaos, the GitHub Actions timezone bug

Data: I noticed `Daily Trading Bot` was firing at 9:30 UTC, not 9:30 ET. The cron in `.github/workflows/run_bot.yml` was set to `30 14 * * 1-5` which is correct for standard time, but I had not adjusted for daylight savings. So for two days the bot ran an hour late and missed the open.

Surprise: the reports still produced cleanly because `trade.py` logged everything, but my entry prices were off by an hour relative to the signal generation. Real money would have noticed.

Decision: document the bug here and add a roadmap note to convert the cron to a "fire at market open" trigger using a market-calendar Python step rather than UTC offsets.

Why reasonable: a one-hour entry slip on small notional positions is not a portfolio-killer in paper. In production it would be. The fix is operational, not mathematical.

### Week of Mar 23: strategy triage memo, deciding what to keep

Data: I sat down and wrote `reports/strategy_triage_decision_memo.md`, a brutal audit of every sleeve. It calls out that mean reversion was miswired in `src/config.py` (the configured path did not match the deployed module), that the Clenow exit logic has fail-open data handling, that the ranked sleeve does not implement target weights, and that same-run sell proceeds were not being recycled into buys.

Surprise: most of the issues were operational, not signal-related. Six weeks of live trading had taught me far more about my own infrastructure than about my strategies. That is a useful lesson on its own.

Decision: spend a week patching the mean-reversion alias module and adding regression tests in `tests/test_strategy_integration.py`. Keep all three sleeves but downgrade my confidence in the V10 sleeve because the live code is not the research code.

Why reasonable: triaging infrastructure before retuning signals is exactly what a quant team is supposed to do when paper performance is mediocre. Tuning first would have been chasing noise.

### Week of Mar 30: cleanup and resilience work

Data: I shrunk Clenow's top-N from twenty to seven on Mar 28 (`Clenow: reduce top picks from 20 to 7`). The reasoning: the budget is five thousand and twenty inverse-vol slugs were producing per-trade tickets so small that they were below sensible per-trade transaction-cost economics in real markets. Seven slots is closer to "concentrated trend portfolio" sizing.

Decision: keep the change. Add `tests/test_strategy_integration.py` to lock in that the live config still imports cleanly.

Why reasonable: before this commit my Clenow sleeve had been built for an investor with ten times the capital. Right-sizing to the actual budget is responsible.

### Week of Apr 6: mean reversion finally wired for fills

Data: I added the alias `src/strategies/mean_reversion/high_pullback_reversion.py` so that `config.STRATEGY_ALLOCATIONS` would resolve. The strategy did not produce any fills because the universe scan returned no symbols meeting the >=40% pullback gate plus the freefall and broken-MA filters.

Surprise: even with a fairly aggressive entry depth, the universe gating was strict enough that under a bull regime the candidate set is empty. The strategy is essentially a bear-market vacuum.

Decision: leave it deployed but accept that it will be quiet until a real drawdown.

Why reasonable: a strategy that does nothing in bull markets is not broken. It is doing its job. The question of whether the budget allocated to it is justified by its uncorrelated payoff in stress requires backtesting it through 2008, 2018, 2020, and 2022 stress windows. That work is on the roadmap.

### Week of Apr 13: final week, capital cleanup, and report writing

Data: closed UNH, MATX, CENX in the discretionary book. Realized roughly plus twenty-two dollars on CENX. Clenow continued to grind small trades. Ranking sat tight (no rebalance day in the window).

Decision: stop changing anything and write this audit. Every parameter change at this stage is for the next semester, not this one.

Why reasonable: twelve weeks is short. The right move now is to learn from the data I have, not generate one more week of noisy fills.
""")


# =============================================================================
# Portfolio Tear Sheet
# =============================================================================
add_md(r"""
## Portfolio tear sheet

This is the one-page artifact a PM would actually look at. I build a daily realized-PnL series across all sleeves, derive equity from it, pull SPY for the same window as a benchmark, then compute the standard set of risk and performance metrics.

One caveat I want to be explicit about: Alpaca's order tape gives me realized PnL on closed round-trips. It does not give me a true mark-to-market equity curve, because positions still open at the end of the window are not revalued. So my equity curve is conservative: it understates returns when there are open winners and understates losses when there are open losers. To partially correct for it I also compute a current-positions snapshot using last-trade prices, but the rigorous fix would be to log Alpaca's `account.equity` daily, which is on the roadmap.
""")

add_py(r"""
def daily_pnl_per_bucket(df: pd.DataFrame) -> pd.DataFrame:
    closed = df[df["PnL ($)"].notna()].copy()
    closed["Day"] = closed["Submitted At"].dt.tz_convert("US/Eastern").dt.date
    pnl = closed.pivot_table(
        index="Day", columns="Bucket", values="PnL ($)", aggfunc="sum", fill_value=0.0
    )
    return pnl


pnl_per_day = daily_pnl_per_bucket(orders)

calendar = pd.date_range(
    orders["Submitted At"].min().tz_convert("US/Eastern").normalize(),
    orders["Submitted At"].max().tz_convert("US/Eastern").normalize(),
    freq="B",
).date
pnl_per_day = pnl_per_day.reindex(calendar, fill_value=0.0)
pnl_per_day["Total"] = pnl_per_day.sum(axis=1)
equity = pnl_per_day.cumsum()
equity.tail()
""")

add_py(r"""
spy = yf.download("SPY", start=str(calendar[0]), end=str(calendar[-1] + pd.Timedelta(days=1)),
                  progress=False, auto_adjust=True)["Close"].squeeze()
spy.index = spy.index.date
spy = spy.reindex(calendar, method="ffill").dropna()
spy_ret = spy.pct_change().fillna(0.0)

gross_bought = orders.loc[orders["Side"].str.upper() == "BUY", "Cash Flow"].sum()
NOTIONAL_BASE = max(gross_bought * 0.6, 30_000)

port_ret = (pnl_per_day["Total"] / NOTIONAL_BASE).reindex(spy.index).fillna(0.0)
port_equity = (1 + port_ret).cumprod()
spy_equity = (1 + spy_ret).cumprod().reindex(port_equity.index).fillna(method="ffill")


def perf_stats(ret: pd.Series) -> dict:
    if ret.std() == 0:
        return {"Total Return": 0.0, "Annual Vol": 0.0, "Sharpe": np.nan,
                "Max Drawdown": 0.0, "Best Day": 0.0, "Worst Day": 0.0}
    cum = (1 + ret).cumprod()
    dd = (cum / cum.cummax() - 1.0)
    return {
        "Total Return": float(cum.iloc[-1] - 1),
        "Annual Vol": float(ret.std() * np.sqrt(252)),
        "Sharpe": float(ret.mean() / ret.std() * np.sqrt(252)),
        "Max Drawdown": float(dd.min()),
        "Best Day": float(ret.max()),
        "Worst Day": float(ret.min()),
    }


port_stats = perf_stats(port_ret)
spy_stats = perf_stats(spy_ret.reindex(port_ret.index).fillna(0.0))

common = pd.concat([port_ret, spy_ret.reindex(port_ret.index).fillna(0.0)], axis=1).dropna()
common.columns = ["port", "spy"]
beta = float(common.cov().loc["port", "spy"] / common["spy"].var()) if common["spy"].var() else np.nan
var_95 = float(np.percentile(port_ret.dropna(), 5))
cvar_95 = float(port_ret[port_ret <= var_95].mean()) if (port_ret <= var_95).any() else np.nan

risk_table = pd.DataFrame({
    "Portfolio (realized)": port_stats,
    "SPY (benchmark)": spy_stats,
}).T
risk_table["Beta vs SPY"] = [beta, 1.0]
risk_table["Daily VaR 95%"] = [var_95, np.percentile(spy_ret, 5)]
risk_table["Daily CVaR 95%"] = [cvar_95, spy_ret[spy_ret <= np.percentile(spy_ret, 5)].mean()]
risk_table_fmt = risk_table.copy()
for col in ["Total Return", "Annual Vol", "Max Drawdown", "Best Day", "Worst Day", "Daily VaR 95%", "Daily CVaR 95%"]:
    risk_table_fmt[col] = risk_table_fmt[col].apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "n/a")
risk_table_fmt["Sharpe"] = risk_table_fmt["Sharpe"].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "n/a")
risk_table_fmt["Beta vs SPY"] = risk_table_fmt["Beta vs SPY"].apply(lambda v: f"{v:+.2f}")
risk_table_fmt
""")

add_md(r"""
### Headline equity curve vs S&P 500

Since the rest of the dashboard is information-dense, I want to lead with the single chart that summarizes the whole semester: my portfolio's realized growth versus a buy-and-hold of SPY over the exact same window.
""")

add_py(r"""
fig, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(port_equity.index, 1, port_equity,
                where=port_equity >= 1, color=GAIN, alpha=0.10, linewidth=0)
ax.fill_between(port_equity.index, 1, port_equity,
                where=port_equity < 1, color=LOSS, alpha=0.10, linewidth=0)

ax.plot(spy_equity.index, spy_equity, color="#94a3b8", linewidth=2.0,
        linestyle=(0, (4, 3)), label=f"SPY (buy and hold) -> {spy_equity.iloc[-1]-1:+.2%}")
ax.plot(port_equity.index, port_equity, color=PORT_NAVY, linewidth=2.8,
        label=f"Portfolio (realized only) -> {port_equity.iloc[-1]-1:+.2%}")

ax.scatter(port_equity.index[-1], port_equity.iloc[-1],
           color=PORT_NAVY, s=70, zorder=5, edgecolor="white", linewidth=1.5)
ax.scatter(spy_equity.index[-1], spy_equity.iloc[-1],
           color="#94a3b8", s=60, zorder=5, edgecolor="white", linewidth=1.5)

ax.axhline(1.0, color="#cbd5e1", linewidth=1, linestyle=":")
ax.set_title(f"Portfolio vs S&P 500   |   {calendar[0]} to {calendar[-1]}   |   {(calendar[-1]-calendar[0]).days} days",
             fontsize=14)
ax.set_ylabel("Growth of $1")
ax.legend(loc="upper left", fontsize=11)
style_axes(ax, datex=True)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
ax.set_ylim(min(port_equity.min(), spy_equity.min())*0.998,
            max(port_equity.max(), spy_equity.max())*1.005)

plt.tight_layout()
plt.show()
""")

add_md(r"""
The portfolio line is conservative for the reason already stated: it tracks realized PnL only. Most of the discretionary book is still open, so any unrealized appreciation on those positions is invisible here. SPY's curve is exact (it is just price). With those caveats, the realized portfolio finished the window above SPY in percentage terms, but the underlying drivers (twelve weeks, three sleeves, lots of unrealized exposure) make the comparison directional rather than rigorous.

### Full tear sheet
""")

add_py(r"""
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 0.95, 1.5], width_ratios=[1.5, 1.0, 1.0],
                      hspace=0.70, wspace=0.45,
                      left=0.06, right=0.97, top=0.93, bottom=0.07)

ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(port_equity.index, 1, port_equity,
                 where=port_equity >= 1, color=GAIN, alpha=0.10, linewidth=0)
ax1.fill_between(port_equity.index, 1, port_equity,
                 where=port_equity < 1, color=LOSS, alpha=0.10, linewidth=0)
ax1.plot(spy_equity.index, spy_equity, color="#94a3b8", linewidth=1.8, linestyle=(0, (4, 3)), label="SPY")
ax1.plot(port_equity.index, port_equity, color=PORT_NAVY, linewidth=2.6, label="Portfolio (realized)")
ax1.axhline(1.0, color="#cbd5e1", linewidth=1, linestyle=":")
ax1.set_title("Equity curve vs SPY")
ax1.set_ylabel("Growth of $1")
ax1.legend(loc="upper left")
style_axes(ax1, datex=True)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))

ax2 = fig.add_subplot(gs[1, :])
dd = (port_equity / port_equity.cummax() - 1)
ax2.fill_between(dd.index, dd, 0, color=LOSS, alpha=0.50, linewidth=0)
ax2.plot(dd.index, dd, color="#7f1d1d", linewidth=1.4)
ax2.set_title(f"Drawdown   |   max {dd.min():+.1%}")
ax2.set_ylabel("Drawdown")
style_axes(ax2, ypct=True, datex=True)

buys = orders[orders["Side"].str.upper() == "BUY"].groupby("Bucket")["Cash Flow"].sum()
sells = orders[orders["Side"].str.upper() == "SELL"].groupby("Bucket")["Cash Flow"].sum()
net_deployed = (buys - sells.reindex(buys.index, fill_value=0)).clip(lower=0)
net_deployed = net_deployed.reindex([b for b in STRATEGY_ORDER if b in net_deployed.index]).dropna()

ax3 = fig.add_subplot(gs[2, 0])
wedges, _, autotexts = ax3.pie(
    net_deployed.values,
    labels=net_deployed.index,
    colors=[STRATEGY_COLORS[b] for b in net_deployed.index],
    autopct=lambda pct: f"{pct:.0f}%",
    startangle=90,
    pctdistance=0.78,
    wedgeprops={"edgecolor": "white", "linewidth": 2.5, "width": 0.42},
    textprops={"fontsize": 10, "color": TEXT_DK},
)
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("700")
ax3.text(0, 0, money(net_deployed.sum()),
         ha="center", va="center", fontsize=14, fontweight="700", color=TEXT_DK)
ax3.text(0, -0.18, "still deployed", ha="center", va="center", fontsize=9, color=TEXT_MID)
ax3.set_title("Capital still deployed")

ax4 = fig.add_subplot(gs[2, 1])
pnl_by_b = orders.groupby("Bucket")["PnL ($)"].sum().reindex(
    [b for b in STRATEGY_ORDER if b in orders["Bucket"].unique()]
)
colors_pnl = [GAIN if v >= 0 else LOSS for v in pnl_by_b]
bars = ax4.bar(range(len(pnl_by_b)), pnl_by_b.values, color=colors_pnl,
               edgecolor="white", linewidth=1.2, width=0.62)
vmax = max(abs(pnl_by_b.values)) if len(pnl_by_b) else 1.0
for i, v in enumerate(pnl_by_b.values):
    off = vmax * 0.06 * (1 if v >= 0 else -1)
    ax4.text(i, v + off, money(v), ha="center",
             va="bottom" if v >= 0 else "top",
             fontsize=10, fontweight="600", color=TEXT_DK)
ax4.set_xticks(range(len(pnl_by_b)))
ax4.set_xticklabels([b.replace(" ", "\n") for b in pnl_by_b.index], fontsize=9)
ax4.axhline(0, color=TEXT_DK, linewidth=0.8)
pos_max = max(pnl_by_b.max(), 0)
neg_min = min(pnl_by_b.min(), 0)
ax4.set_ylim(neg_min - vmax * 0.30, pos_max + vmax * 0.30)
ax4.set_title("Realized PnL by sleeve")
style_axes(ax4, ydollar=True)

ax5 = fig.add_subplot(gs[2, 2])
ax5.axis("off")
ps = port_stats
lines = [
    ("Total return",   f"{ps['Total Return']:+.2%}"),
    ("Annual vol",     f"{ps['Annual Vol']:.2%}"),
    ("Sharpe",         f"{ps['Sharpe']:+.2f}"),
    ("Max drawdown",   f"{ps['Max Drawdown']:+.2%}"),
    ("Beta vs SPY",    f"{beta:+.2f}"),
    ("Daily VaR 95%",  f"{var_95:+.2%}"),
    ("Daily CVaR 95%", f"{cvar_95:+.2%}"),
]
ax5.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0,
                            transform=ax5.transAxes, facecolor="#f8fafc", edgecolor=GRID, linewidth=1))
y0 = 0.92
for i, (k, v) in enumerate(lines):
    ax5.text(0.06, y0 - i*0.12, k, fontsize=10.5, color=TEXT_MID, transform=ax5.transAxes)
    ax5.text(0.94, y0 - i*0.12, v, fontsize=12, fontweight="700",
             color=TEXT_DK, ha="right", transform=ax5.transAxes)
ax5.set_title("Risk metrics")

fig.suptitle("Portfolio Tear Sheet", fontsize=18, fontweight="bold", y=0.97, color=TEXT_DK)
plt.show()
""")

add_md(r"""
The drawdown plot tells me the worst peak-to-trough realized loss came from two consecutive bad weeks in mid-February when Clenow exited a cluster of positions on rank-based stops. Those losses are small in dollar terms but visible in the percentage curve because of the deliberately conservative notional base.

The allocation pie is dominated by the discretionary sleeve. That is a finding, not a tear-sheet artifact. I should size the algorithmic sleeves up (or the discretionary book down) before claiming the algos drive the portfolio.

The Sharpe number on the realized series prints as plus two-point-five, which looks great in isolation but is misleading. The realized-only series has very low day-to-day volatility because most days no positions close, so the denominator is artificially small. Beta versus SPY is essentially zero for the same reason. VaR and CVaR at the 95% threshold are the size of normal daily realized-PnL noise on this account. None of these numbers is meaningful evidence for real-money deployment because the metric definitions themselves are biased by the realized-only convention. The fix is daily mark-to-market logging of `account.equity`, which is on the roadmap but did not happen this semester.
""")


# =============================================================================
# Per-strategy forensics
# =============================================================================
add_md(r"""
## Per-strategy forensics

The tear sheet collapsed the whole book to one number. Now I split it back out and ask: what did each sleeve actually do, with what capital, and at what cost?
""")

add_py(r"""
def summarize(group: pd.DataFrame) -> pd.Series:
    buys = group[group["Side"].str.upper() == "BUY"]
    sells = group[group["Side"].str.upper() == "SELL"]
    matched = sells[sells["PnL ($)"].notna()]
    bought = buys["Cash Flow"].sum()
    sold = sells["Cash Flow"].sum()
    realized = matched["PnL ($)"].sum()
    n_wins = (matched["PnL ($)"] > 0).sum()
    n_match = len(matched)
    return pd.Series({
        "# Orders": len(group),
        "# Buys": len(buys),
        "# Sells": len(sells),
        "# Symbols": group["Symbol"].nunique(),
        "Bought ($)": bought,
        "Sold ($)": sold,
        "Net Deployed ($)": bought - sold,
        "Realized PnL ($)": realized,
        "Realized Return on Bought (%)": (realized / bought * 100) if bought > 0 else np.nan,
        "Win Rate (matched)": (n_wins / n_match) if n_match else np.nan,
        "# Round Trips": n_match,
    })


summary = orders.groupby("Bucket").apply(summarize).reindex(
    [b for b in STRATEGY_ORDER if b in orders["Bucket"].unique()]
)
summary["Lifetime Budget ($)"] = summary.index.map(lambda b: STRATEGY_BUDGETS.get(b, np.nan))
summary["Budget Remaining ($)"] = summary["Lifetime Budget ($)"] - summary["Net Deployed ($)"]
summary["Budget Used (%)"] = summary["Net Deployed ($)"] / summary["Lifetime Budget ($)"] * 100

display = summary.copy()
for col in ["Bought ($)", "Sold ($)", "Net Deployed ($)", "Realized PnL ($)",
            "Lifetime Budget ($)", "Budget Remaining ($)"]:
    display[col] = display[col].apply(lambda v: money(v) if pd.notna(v) else "n/a")
for col in ["Realized Return on Bought (%)", "Budget Used (%)"]:
    display[col] = display[col].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a")
display["Win Rate (matched)"] = display["Win Rate (matched)"].apply(
    lambda v: f"{v:.0%}" if pd.notna(v) else "n/a"
)
for col in ["# Orders", "# Buys", "# Sells", "# Symbols", "# Round Trips"]:
    display[col] = display[col].astype(int)
display
""")

add_md(r"""
A few things jump out of this table. Discretionary moved roughly seven times more capital than Ranked Asset Allocation and an order of magnitude more than Clenow Trend. The discretionary win rate of 100% on 14 closed trades looks impressive until I realize what it actually means: I sold only the trades that worked. The losers are still in the book, unrealized. So 100% is a survivorship artifact, not evidence of discretionary skill.

Clenow's win rate of 36% is exactly what a momentum strategy with rank-based exits should produce: most trends fail before they pay, but the surviving trends are supposed to pay enough to compensate. With only 14 closed trades I cannot tell whether the surviving trends are big enough; the realised return on bought capital is negative four percent so far. Twelve weeks of paper is not enough to characterize a strategy whose edge lives in the right-tail of multi-month moves, so I am not yet ready to write Clenow off.

Ranked Asset Alloc has too few closed trades (3) to draw any conclusion. The two losses came from the live sleeve buying semiconductor names that then pulled back. The math says I need at least 30 closed rotations before I can speak about its hit rate with any confidence, and at one rotation per month that means a couple of years of live data. The strategy was designed for that timescale, not for the twelve weeks it has had so far.

High Pullback Reversion does not appear because it has zero fills. That is not a bug; the strategy is designed for pullback regimes and the window has been a steady uptrend. A long-only mean-reversion sleeve sitting in cash during a bull run is doing exactly what the prospectus says it will do.
""")

add_py(r"""
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"wspace": 0.32})
buckets = list(summary.index)
colors = [STRATEGY_COLORS[b] for b in buckets]

ax = axes[0]
bars = ax.bar(buckets, summary["Bought ($)"], color=colors,
              edgecolor="white", linewidth=1.5, alpha=0.95, width=0.6)
for bar, b in zip(bars, buckets):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + summary["Bought ($)"].max()*0.015,
            money(h), ha="center", va="bottom", fontsize=11, fontweight="600", color=TEXT_DK)
    bud = STRATEGY_BUDGETS.get(b)
    if bud is not None:
        ax.hlines(bud, bar.get_x(), bar.get_x()+bar.get_width(),
                  colors=TEXT_DK, linestyles="--", linewidth=1.5, alpha=0.85)
        ax.text(bar.get_x()+bar.get_width()/2, bud + summary["Bought ($)"].max()*0.005,
                f"budget {money(bud)}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT_MID, fontstyle="italic")
ax.set_title("Total dollars deployed (BUY fills)")
ax.set_ylabel("USD")
style_axes(ax, ydollar=True, xrot=15)
ax.set_ylim(0, summary["Bought ($)"].max()*1.18)

ax = axes[1]
pnl_vals = summary["Realized PnL ($)"].values
bar_colors = [GAIN if v >= 0 else LOSS for v in pnl_vals]
bars = ax.bar(buckets, pnl_vals, color=bar_colors,
              edgecolor="white", linewidth=1.5, alpha=0.95, width=0.6)
yspan = max(abs(pnl_vals).max() * 0.06, 100)
for bar, v in zip(bars, pnl_vals):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h + (yspan if h>=0 else -yspan),
            money(h), ha="center", va="bottom" if h>=0 else "top",
            fontsize=11, fontweight="600", color=TEXT_DK)
ax.axhline(0, color=TEXT_DK, linewidth=0.8)
ax.set_title("Realized PnL (closed round-trips only)")
ax.set_ylabel("USD")
style_axes(ax, ydollar=True, xrot=15)

plt.tight_layout()
plt.show()
""")

add_md(r"""
The left bar is dollars in. The dashed black tick on each algorithmic sleeve is its lifetime budget. The right bar is dollars realized. Discretionary is the only sleeve I have actually closed money on so far. Clenow's loss is the cost of running a momentum strategy through a quiet tape: the rank-based exits forced losses on trades that had not really broken trend yet. Ranked Alloc's loss is statistically meaningless given three closed trades.
""")

add_py(r"""
flow = orders.sort_values("Submitted At").copy()
flow["Net Deployed"] = flow.groupby("Bucket")["Signed Flow"].cumsum()

fig, ax = plt.subplots(figsize=(14.5, 6.2))
for b in [x for x in STRATEGY_ORDER if x in flow["Bucket"].unique()]:
    sub = flow[flow["Bucket"] == b]
    color = STRATEGY_COLORS[b]
    ax.plot(sub["Submitted At"], sub["Net Deployed"],
            label=b, color=color, linewidth=2.6, alpha=0.95)
    ax.fill_between(sub["Submitted At"], 0, sub["Net Deployed"],
                    color=color, alpha=0.08)
    last_x = sub["Submitted At"].iloc[-1]
    last_y = sub["Net Deployed"].iloc[-1]
    ax.annotate(money(last_y), xy=(last_x, last_y), xytext=(8, 0),
                textcoords="offset points", fontsize=10.5, fontweight="700",
                color=color, va="center")

ax.axhline(0, color=TEXT_DK, linewidth=0.8)
ax.set_title("Net capital deployed per sleeve  (cumulative BUY minus SELL)")
ax.set_ylabel("USD currently deployed")
ax.set_xlabel("")
ax.legend(loc="upper left")
style_axes(ax, ydollar=True, datex=True)
plt.tight_layout()
plt.show()
""")

add_md(r"""
The discretionary line shape tells the story. It steps up vertically on the day I bought my big basket (Mar 9), then steps down twice more recently as I closed UNH and a few other names. The Clenow sleeve breathes more steadily because it adds and removes small positions on its own schedule, but never strays far from its five-thousand-dollar budget. The Ranking sleeve sat flat after the Mar 11 entry until I exited a couple of names in late March; this is exactly the once-a-month cadence the strategy is designed to have.
""")

add_py(r"""
closed = orders[orders["PnL ($)"].notna()].sort_values("Submitted At").copy()
closed["Cum PnL"] = closed.groupby("Bucket")["PnL ($)"].cumsum()

fig, ax = plt.subplots(figsize=(14.5, 6.2))
for b in [x for x in STRATEGY_ORDER if x in closed["Bucket"].unique()]:
    sub = closed[closed["Bucket"] == b]
    color = STRATEGY_COLORS[b]
    ax.step(sub["Submitted At"], sub["Cum PnL"], where="post",
            label=b, color=color, linewidth=2.6)
    ax.scatter(sub["Submitted At"], sub["Cum PnL"], color=color,
               s=42, zorder=3, edgecolor="white", linewidth=1)
    last_x = sub["Submitted At"].iloc[-1]
    last_y = sub["Cum PnL"].iloc[-1]
    ax.annotate(f"  {money(last_y)}", xy=(last_x, last_y), xytext=(6, 0),
                textcoords="offset points", fontsize=11, fontweight="700",
                color=color, va="center")

ax.axhline(0, color=TEXT_DK, linewidth=0.8)
ax.set_title("Cumulative realized PnL by sleeve")
ax.set_ylabel("USD")
ax.legend(loc="lower left")
style_axes(ax, ydollar=True, datex=True)
plt.tight_layout()
plt.show()
""")

add_md(r"""
Three sleeves, three different stories. The discretionary book stair-steps upward as I closed winners. Clenow is a slow grind sideways with a lot of small chops. Ranking is essentially a flat line because so few rotations have closed.

I want to flag the obvious bias: discretionary closes were chosen by me, mostly when I felt like locking in gains. Algo closes are mechanical. So the cumulative PnL chart is comparing apples to oranges in terms of timing decisions. A fair comparison would mark every position to market every day, which my data does not yet support.
""")

add_py(r"""
order_in_data = [b for b in STRATEGY_ORDER if b in closed["Bucket"].unique()]
palette = {b: STRATEGY_COLORS[b] for b in order_in_data}

fig, ax = plt.subplots(figsize=(13, 6.2))
sns.violinplot(data=closed, x="Bucket", y="PnL ($)", order=order_in_data,
               palette=palette, inner=None, cut=0, linewidth=1.0,
               saturation=0.75, ax=ax)
sns.stripplot(data=closed, x="Bucket", y="PnL ($)", order=order_in_data,
              color=TEXT_DK, alpha=0.85, size=6, jitter=0.18, ax=ax)
medians = closed.groupby("Bucket")["PnL ($)"].median().reindex(order_in_data)
for i, m in enumerate(medians):
    ax.hlines(m, i-0.32, i+0.32, color="white", linewidth=2.4, zorder=4)
    ax.text(i+0.34, m, f"med {money(m)}", va="center", fontsize=9, color=TEXT_DK)
ax.axhline(0, color=TEXT_DK, linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_title("Per-trade realized PnL distribution")
ax.set_xlabel("")
ax.set_ylabel("Realized PnL on the trade ($)")
style_axes(ax, ydollar=True)
plt.tight_layout()
plt.show()
""")

add_md(r"""
Discretionary's distribution is right-skewed and entirely above zero in this sample (every closed trade was a winner; the losers are unrealized and not on this chart). Clenow is tight around zero with one bad tail (KALU and LCII early in February). Ranking is almost a delta function at three points; the violin shape is artifactual.

If I was actually a quant PM evaluating these sleeves, I would not yet trust any of these distributions to predict the next thirty trades. Twenty closed observations per sleeve is the absolute minimum I would want before forming a prior on the trade-level edge. The algo sleeves simply have not had enough wall-clock time yet.
""")


# =============================================================================
# Per-strategy risk metrics
# =============================================================================
add_md(r"""
## Per-strategy risk metrics

The per-trade view above is useful for understanding behavior but not for sizing decisions. For sizing I need risk metrics on the daily PnL series, sleeve by sleeve.

I treat each sleeve as a sub-account with `notional_capital` equal to either its lifetime budget (for the algos) or a proxy of net deployed capital (for discretionary). I then compute Sharpe, vol, and max drawdown on its daily realized-PnL return series. This is still understated because of unrealized PnL, but it lets me at least rank sleeves on a comparable basis. Note that the discretionary max drawdown comes out at zero because every closed discretionary trade was a winner; there is no realized loss to drag the cumulative curve down. Real drawdowns on the discretionary book are sitting in unrealized positions that this metric cannot see.
""")

add_py(r"""
notional = STRATEGY_BUDGETS.copy()
disc_buys = orders[(orders["Bucket"] == "Discretionary") & (orders["Side"].str.upper() == "BUY")]
notional["Discretionary"] = max(disc_buys["Cash Flow"].sum() * 0.5, 25_000)

per_strat_rets = pnl_per_day[[c for c in pnl_per_day.columns if c != "Total"]].copy()
for col in per_strat_rets.columns:
    per_strat_rets[col] = per_strat_rets[col] / notional.get(col, 1.0)

per_strat_stats = pd.DataFrame({
    col: perf_stats(per_strat_rets[col]) for col in per_strat_rets.columns
}).T

per_strat_stats["# Days w/ realized PnL"] = (per_strat_rets != 0).sum().values
per_strat_stats = per_strat_stats[["# Days w/ realized PnL", "Total Return", "Annual Vol",
                                   "Sharpe", "Max Drawdown", "Best Day", "Worst Day"]]
per_strat_stats_fmt = per_strat_stats.copy()
for col in ["Total Return", "Annual Vol", "Max Drawdown", "Best Day", "Worst Day"]:
    per_strat_stats_fmt[col] = per_strat_stats_fmt[col].apply(
        lambda v: f"{v:+.2%}" if pd.notna(v) else "n/a")
per_strat_stats_fmt["Sharpe"] = per_strat_stats_fmt["Sharpe"].apply(
    lambda v: f"{v:+.2f}" if pd.notna(v) else "n/a")
per_strat_stats_fmt["# Days w/ realized PnL"] = per_strat_stats_fmt["# Days w/ realized PnL"].astype(int)
per_strat_stats_fmt
""")

add_md(r"""
Three things to read out of this table.

First, the discretionary sleeve has the highest Sharpe and the largest total return because I happened to close winners in a benign tape. As discussed above, I should not interpret that as evidence I am a great trader; it is mostly a survivorship effect (I closed the winners; the losers, if any, are still open). The zero max drawdown reflects the same bias.

Second, Clenow's Sharpe is negative and modest. That is not unusual for a trend-following sleeve over a twelve-week window where the market trends gently rather than violently. Clenow's edge tends to live in big moves; a quiet uptrend with mean-reverting dips is exactly the regime where it bleeds slowly. The Clenow numbers in the live tape are consistent with this and not yet a reason to abandon the strategy.

Third, the Ranking sleeve has only one day of realized PnL because monthly rebalances close so few positions. The metric is essentially noise. I would not draw any conclusion about its risk profile from this until I have at least six rotations on the books, which means at least six months of live data.

No metric here adjusts for the fact that the algo sleeves run on capped budgets while the discretionary book swings around with my discretion. The Sharpe comparison is therefore not strictly apples-to-apples; it is the best I can do with realized data alone.
""")

add_py(r"""
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"wspace": 0.32})

valid = per_strat_stats.dropna(subset=["Sharpe"])
ax = axes[0]
sharpe_vals = valid["Sharpe"].values
sharpe_colors = [GAIN if v >= 0 else LOSS for v in sharpe_vals]
bars = ax.bar(valid.index, sharpe_vals, color=sharpe_colors, edgecolor="white",
              linewidth=1.2, alpha=0.92, width=0.6)
for bar, v in zip(bars, sharpe_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v + (0.15 if v >= 0 else -0.15),
            f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
            fontsize=12, fontweight="700", color=TEXT_DK)
ax.axhline(0, color=TEXT_DK, linewidth=0.8)
ax.set_title("Sharpe by sleeve (annualized, realized only)")
ax.set_ylabel("Sharpe")
style_axes(ax, xrot=15)

ax = axes[1]
mdd_vals = valid["Max Drawdown"].values
bars = ax.bar(valid.index, mdd_vals, color=LOSS, alpha=0.85,
              edgecolor="white", linewidth=1.2, width=0.6)
for bar, v in zip(bars, mdd_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v - 0.005,
            f"{v:+.1%}", ha="center", va="top", fontsize=12, fontweight="700", color="#7f1d1d")
ax.axhline(0, color=TEXT_DK, linewidth=0.8)
ax.set_title("Max drawdown by sleeve (realized only)")
ax.set_ylabel("Drawdown")
style_axes(ax, ypct=True, xrot=15)

plt.tight_layout()
plt.show()
""")

add_md(r"""
The Sharpe panel makes the relative ranking visually obvious. Discretionary is alone in positive territory. The drawdown panel inverts: discretionary also had the smallest drawdown, but for the same survivorship reason discussed above.

For a real investor I would refuse to size based on these numbers. The proper way to do this is to log Alpaca account equity daily, build a true mark-to-market sleeve equity curve, and recompute these metrics. That is on my roadmap but did not happen this semester.
""")


# =============================================================================
# Cross-strategy correlations
# =============================================================================
add_md(r"""
## Cross-strategy correlations

The headline reason to combine three sleeves is diversification. Diversification only helps if the sleeves do not all lose money at the same time. The standard tool is the correlation matrix of daily PnL.
""")

add_py(r"""
corr_data = per_strat_rets.copy()
corr_data = corr_data[(corr_data != 0).any(axis=1)]
corr = corr_data.corr()
print(f"Days where at least one sleeve had realized activity: {len(corr_data)}")
corr.round(3)
""")

add_py(r"""
fig, ax = plt.subplots(figsize=(7.5, 6))
mask = np.zeros_like(corr, dtype=bool)
sns.heatmap(corr, annot=True, fmt=".2f", mask=mask,
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            vmin=-1, vmax=1, center=0, square=True,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            linewidths=1.2, linecolor="white",
            annot_kws={"fontsize": 12, "fontweight": "600"}, ax=ax)
ax.set_title("Daily realized-PnL correlation between sleeves")
plt.tight_layout()
plt.show()
""")

add_md(r"""
Sleeve correlations from realized PnL alone are essentially uninformative on this sample. There are only a handful of days where two sleeves both had a non-zero realized PnL (Clenow trades almost daily, Ranking trades twice a month, Discretionary on irregular bursts). With so few overlapping observations the correlation matrix is statistical garbage. The two correlations near zero are zero because the series almost never co-vary; the small positive correlation between Clenow and Discretionary is one or two days of joint activity, not a real signal.

What I really want for a correlation analysis is sleeve equity curves at daily mark-to-market resolution. Without that I cannot say whether these strategies actually diversify each other in practice. From first principles I can argue:

- Clenow Trend is a long-only momentum sleeve in single-name equities. Its returns will correlate with the equity-market beta and with cross-sectional momentum factors (Fama-French UMD).
- Ranked Asset Allocation is sector ETF rotation with a hedge sleeve in bear regimes. In bull markets it correlates with whichever sector is leading (often XLK or XLY). In bear markets it rotates to TLT/GLD/UUP and decorrelates aggressively.
- High Pullback Reversion would be a contrarian sleeve buying deep pullbacks; structurally negative correlation with momentum. But it has no fills.
- Discretionary is whatever I happen to think is interesting; it has historically been growth and quality names so it correlates with the S&P 500.

So my a priori expectation is that Clenow, Discretionary, and Ranking-in-bull-mode are all positively correlated with the broad market. Mean reversion is the only structural diversifier and it has not traded. The portfolio is therefore much less diversified than the three-sleeve count suggests.
""")


# =============================================================================
# Trade activity heatmap and concentration
# =============================================================================
add_md(r"""
## Trade activity, winners and losers, concentration

Three diagnostic plots before I move to the deep dives.
""")

add_py(r"""
heat_src = orders.copy()
heat_src["Week"] = heat_src["Submitted At"].dt.tz_convert("US/Eastern").dt.to_period("W").apply(lambda p: p.start_time.date())
heat = heat_src.pivot_table(index="Bucket", columns="Week", values="Symbol",
                            aggfunc="count", fill_value=0)
heat = heat.reindex([b for b in STRATEGY_ORDER if b in heat.index])

fig, ax = plt.subplots(figsize=(max(11, 0.75*heat.shape[1]+4), 0.95*len(heat)+2.6))
sns.heatmap(heat, annot=True, fmt="d",
            cmap=sns.light_palette("#2563eb", as_cmap=True),
            cbar_kws={"label": "# orders", "shrink": 0.8},
            linewidths=1.0, linecolor="white",
            annot_kws={"fontsize": 11, "fontweight": "600"}, ax=ax)
ax.set_title("Orders per week by sleeve")
ax.set_xlabel("Week beginning")
ax.set_ylabel("")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
""")

add_md(r"""
The cadence is visible. Clenow trades almost every week (it is a weekly Wednesday signal). Ranking has its monthly rebalance signature. Discretionary is bursty; the giant Mar 9 spike is the one big basket day. The big white columns (no activity weeks) for Discretionary tell me how often I went hands-off, which is a useful self-discipline diagnostic.
""")

add_py(r"""
N = 8
ranked = closed.sort_values("PnL ($)")
losers = ranked.head(N).iloc[::-1]
winners = ranked.tail(N).iloc[::-1]


def label(row):
    return f"{row['Symbol']}  {row['Submitted At']:%b %d}"


fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), gridspec_kw={"wspace": 0.5})
for ax, df_, title in [
    (axes[0], winners, f"Top {N} winners"),
    (axes[1], losers, f"Top {N} losers"),
]:
    labels = [label(r) for _, r in df_.iterrows()]
    vals = df_["PnL ($)"].values
    bcolors = [STRATEGY_COLORS[b] for b in df_["Bucket"]]
    bars = ax.barh(labels, vals, color=bcolors, edgecolor="white", linewidth=1.2, alpha=0.92, height=0.7)
    for bar, v in zip(bars, vals):
        ax.text(v + (max(abs(vals))*0.02 * (1 if v >= 0 else -1)), bar.get_y()+bar.get_height()/2,
                money(v), va="center", ha="left" if v >= 0 else "right",
                fontsize=10, fontweight="600", color=TEXT_DK)
    ax.invert_yaxis()
    ax.axvline(0, color=TEXT_DK, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Realized PnL on the trade ($)")
    style_axes(ax, ydollar=False)
    ax.xaxis.set_major_formatter(DOLLAR_FMT)

handles = [Patch(facecolor=STRATEGY_COLORS[b], label=b)
           for b in STRATEGY_ORDER if b in closed["Bucket"].unique()]
fig.legend(handles=handles, loc="lower center", ncol=len(handles),
           frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=10)
fig.suptitle("Closed-trade extremes", fontsize=15, fontweight="bold", color=TEXT_DK)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
""")

add_md(r"""
The winners are dominated by purple (discretionary). The biggest winner is the discretionary sleeve closing winners during the late-March uptrend. The biggest loser is a Clenow trade where the rank-based exit fired on a name that had given back its momentum gains. Visually the dispersion of the winners tail is much larger than the dispersion of the losers tail; the realized PnL distribution is positively skewed, which is the right shape for a healthy book even when the mean is small.
""")

add_py(r"""
buys_only = orders[orders["Side"].str.upper() == "BUY"].copy()
sym_summary = (
    buys_only.groupby(["Symbol", "Bucket"], as_index=False)["Cash Flow"].sum()
    .sort_values("Cash Flow", ascending=False)
)
sym_top = (
    sym_summary.groupby("Symbol", as_index=False)
    .apply(lambda g: g.sort_values("Cash Flow", ascending=False).iloc[0])
    .sort_values("Cash Flow", ascending=False)
    .head(20).reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(15.5, 7.5))
y_pos = np.arange(len(sym_top))[::-1]
bars = ax.barh(y_pos, sym_top["Cash Flow"],
               color=[STRATEGY_COLORS[b] for b in sym_top["Bucket"]],
               edgecolor="white", linewidth=1.2, alpha=0.95, height=0.72)
ax.set_yticks(y_pos)
ax.set_yticklabels(sym_top["Symbol"], fontweight="600")
ax.invert_yaxis()
for bar, v, b in zip(bars, sym_top["Cash Flow"], sym_top["Bucket"]):
    ax.text(v + sym_top["Cash Flow"].max()*0.005, bar.get_y()+bar.get_height()/2,
            f"  {money(v)}  ({b})",
            va="center", fontsize=10, color=TEXT_DK)
ax.set_title("Top 20 symbols by total $ bought")
ax.set_xlabel("USD bought")
ax.xaxis.set_major_formatter(DOLLAR_FMT)
ax.set_xlim(0, sym_top["Cash Flow"].max() * 1.32)
handles = [Patch(facecolor=STRATEGY_COLORS[b], label=b)
           for b in STRATEGY_ORDER if b in sym_top["Bucket"].unique()]
ax.legend(handles=handles, loc="lower right")
style_axes(ax)
plt.tight_layout()
plt.show()
""")

add_md(r"""
Concentration is essentially a discretionary phenomenon. UNH is the largest position by deployed capital because I bought fifty shares as one ticket, which puts a single name above all of the algorithmic positions combined. That concentration is fine for a paper account; it would be unacceptable for a real one without a hard per-name cap.
""")


# =============================================================================
# Discretionary deep-dive
# =============================================================================
add_md(r"""
## Discretionary book audit

The discretionary sleeve is large enough that it deserves its own section. I look at the per-name book, the realized PnL per name, and a few of its risk numbers in isolation.
""")

add_py(r"""
disc = orders[orders["Bucket"] == "Discretionary"].copy()
disc_by_sym = (
    disc.groupby("Symbol")
    .apply(lambda g: pd.Series({
        "Bought": g.loc[g["Side"].str.upper()=="BUY", "Cash Flow"].sum(),
        "Sold":   g.loc[g["Side"].str.upper()=="SELL", "Cash Flow"].sum(),
        "Realized PnL": g["PnL ($)"].sum(skipna=True),
        "Trades": len(g),
    }))
)
disc_by_sym["Net Deployed"] = disc_by_sym["Bought"] - disc_by_sym["Sold"]
disc_by_sym = disc_by_sym.sort_values("Bought", ascending=False)
top_disc = disc_by_sym.head(15)

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"wspace": 0.5})

ax = axes[0]
y = np.arange(len(top_disc))[::-1]
ax.barh(y, top_disc["Bought"], color=STRATEGY_COLORS["Discretionary"], alpha=0.9,
        edgecolor="white", linewidth=1.2, label="Bought", height=0.72)
ax.barh(y, -top_disc["Sold"], color="#94a3b8", alpha=0.9,
        edgecolor="white", linewidth=1.2, label="Sold", height=0.72)
ax.set_yticks(y)
ax.set_yticklabels(top_disc.index, fontweight="600")
for i, (sym, row) in zip(y, top_disc.iterrows()):
    ax.text(row["Bought"]+top_disc["Bought"].max()*0.01, i,
            money(row["Bought"]), va="center", fontsize=9, color=TEXT_DK)
    if row["Sold"] > 0:
        ax.text(-row["Sold"]-top_disc["Bought"].max()*0.01, i,
                money(-row["Sold"]), va="center", ha="right",
                fontsize=9, color=TEXT_MID)
ax.axvline(0, color=TEXT_DK, linewidth=0.8)
ax.set_title("Top 15 discretionary names: bought vs sold")
ax.set_xlabel("USD")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: money(abs(x))))
ax.legend(loc="lower right")
style_axes(ax)

ax = axes[1]
disc_realized = disc_by_sym[disc_by_sym["Realized PnL"] != 0].copy()
disc_realized = disc_realized.sort_values("Realized PnL")
if len(disc_realized) == 0:
    ax.text(0.5, 0.5, "No realized PnL on discretionary trades yet.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color=TEXT_MID)
    ax.set_axis_off()
else:
    colors_pnl = [GAIN if v >= 0 else LOSS for v in disc_realized["Realized PnL"]]
    ax.barh(np.arange(len(disc_realized))[::-1], disc_realized["Realized PnL"],
            color=colors_pnl, edgecolor="white", linewidth=1.2, alpha=0.92, height=0.72)
    ax.set_yticks(np.arange(len(disc_realized))[::-1])
    ax.set_yticklabels(disc_realized.index, fontweight="600")
    for i, v in zip(np.arange(len(disc_realized))[::-1], disc_realized["Realized PnL"]):
        ax.text(v + (max(abs(disc_realized["Realized PnL"]))*0.02 * (1 if v>=0 else -1)), i,
                money(v), va="center", ha="left" if v>=0 else "right",
                fontsize=10, fontweight="600", color=TEXT_DK)
    ax.axvline(0, color=TEXT_DK, linewidth=0.8)
    ax.set_title("Discretionary trades: realized PnL by symbol")
    ax.set_xlabel("Realized PnL ($)")
    ax.xaxis.set_major_formatter(DOLLAR_FMT)
    style_axes(ax)

fig.suptitle("Discretionary book", fontsize=15, fontweight="bold", color=TEXT_DK)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

print(f"\nDiscretionary universe: {disc['Symbol'].nunique()} unique symbols across {len(disc)} fills")
print(f"Bought: {money(disc[disc['Side'].str.upper()=='BUY']['Cash Flow'].sum())}")
print(f"Sold:   {money(disc[disc['Side'].str.upper()=='SELL']['Cash Flow'].sum())}")
print(f"Net deployed: {money(disc['Signed Flow'].sum())}")
print(f"Realized PnL: {money(disc['PnL ($)'].sum(skipna=True))}")
""")

add_md(r"""
I made about $4k of realized PnL on a discretionary book that deployed about $80k of capital. That is roughly five percent on bought capital over twelve weeks. Annualized naively that is about twenty-two percent, which sounds great until I remember three things:

1. The window is short. Twelve weeks of a trending market is not a sample size.
2. I closed winners and held losers (or held everything that has not yet broken). The realized number is biased upward.
3. SPY itself returned roughly nine to ten percent in the same window. So a long-only buy-and-hold of the index would have made roughly the same dollar amount with one trade and zero process.

If I am brutally honest, the discretionary book did not earn alpha in this window. It earned beta. Most of what I bought were names that simply went up because the market went up. The Sharpe number from earlier reflects this: the realized series has low vol because I was rarely closing positions, not because I was producing risk-controlled returns. The zero max drawdown is the same artifact in another costume.

In a real portfolio the discretionary book would need to be either much more disciplined with explicit sizing rules and a stop-loss policy, or folded into the algorithmic stack as a fourth strategy with its own systematic edge.
""")


# =============================================================================
# Strategy Deep Dive #1 - Clenow Trend
# =============================================================================
add_md(r"""
## Deep dive: Clenow Trend (autotrading sleeve)

This is one of the two required strategy deep dives. Clenow Trend is the autotrading sleeve.

### Rationale

Clenow Trend is a cross-sectional time-series momentum strategy on US single-name equities. Long-only. The universe is the S&P 500 / 400 / 600 (mid and small cap), filtered down by `src/universe.py` to names trading above ten dollars with at least ten million dollars of average daily dollar volume, roughly 1400 names.

The signal is the classical Clenow score from *Stocks on the Move*. For each ticker, fit a linear regression on the last sixty trading days of log prices, take the regression slope and the R-squared, then form

\[
\text{score} = \big(e^{\text{slope}}\big)^{252} \times R^2
\]

The first factor is the annualized geometric drift implied by the regression slope. The second factor is the fit quality of the regression. Multiplying them rewards stocks that are trending strongly and steadily, and penalizes stocks that are noisy regardless of slope.

There are three entry filters that each candidate must pass: price above the 200-day SMA (a trend gate); a deceleration gate where the 30-day score must be at least half the 60-day score (filters out names that ran hard a month ago and are now stalling); and a 52-week high proximity gate where price must be within 25 percent of the rolling 52-week high (avoids buying broken stocks that have merely stopped falling). Above all of this sits a regime gate: only place new buys when at least two of three index ETFs (SPY, IJH, IJR) are above their 200-day SMA. Exits run regardless of regime.

Sizing uses inverse-volatility weighting on twenty-day realized vol, normalized to one and clamped to the range [2 percent, 10 percent] per name. With seven slots and a five-thousand-dollar cap, each name gets roughly six to ten percent of budget. Exits sell any held name whose rank dropped beyond 20, or whose price fell below the 50-day SMA, or that failed entry filters in the daily scan.

### Mathematical assumptions

The strategy implicitly bets on three things. Trend persistence: past 60-day momentum predicts forward 5 to 30-day returns with positive expected value, after accounting for transaction costs. Cross-sectional dispersion: the top decile of momentum-ranked names will outperform the universe average by enough to beat the cost of frequent exits. Regime stationarity: the relationships above hold across market regimes, or at least the regime gate is good enough to detect when they break down. The first two have been documented since Jegadeesh and Titman (1993). The third is the assumption the regime gate is supposed to defend.

### What I actually did before deploying

I want to be honest about this because it is the kind of thing the assignment is meant to surface. I did not do a formal backtest of Clenow Trend on my own data before going live. I read the strategy in the *Stocks on the Move* tradition, ported it, plugged in the universe scan, and let it run in paper. The notebook `analysis/strategies/momentum/clenow_trend.ipynb` exists, but I built it after the strategy had already been live for about four weeks, as a retrospective sanity check. The numbers in the triage memo (`reports/strategy_triage_decision_memo.md`) show a CAGR of around 56 percent over the last year and 109 percent over the last five years on a 1387-ticker cached universe with weekly rebalancing, but those backtests do not include transaction costs and they do not match the daily cadence at which the live strategy actually runs.

The honest defense is this: the entire purpose of paper trading is to get a strategy live cheaply enough that the operational and behavioral lessons are learned without real-money cost. With a one-semester window I prioritized building the live infrastructure (cron, regime checks, attribution tags, exits, reporting) over running an offline backtest first. That trade-off felt right at the time and I would defend it again, with the explicit understanding that I would never put real money behind a strategy this lightly characterized. The lesson goes on the next-semester roadmap as the first item.

### Execution evidence
""")

add_py(r"""
clenow = orders[orders["Bucket"] == "Clenow Trend"].copy().sort_values("Submitted At")
clenow["Cum Cash Flow"] = clenow["Signed Flow"].cumsum()
clenow_closed = clenow[clenow["PnL ($)"].notna()].copy()
clenow_closed["Cum PnL"] = clenow_closed["PnL ($)"].cumsum()

fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.4), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1], "hspace": 0.30})

ax = axes[0]
buys_c = clenow[clenow["Side"].str.upper() == "BUY"]
sells_c = clenow[clenow["Side"].str.upper() == "SELL"]
ax.scatter(buys_c["Submitted At"], buys_c["Cash Flow"], color=STRATEGY_COLORS["Clenow Trend"],
           s=55, alpha=0.85, edgecolor="white", linewidth=1, label="BUY")
ax.scatter(sells_c["Submitted At"], -sells_c["Cash Flow"], color=LOSS,
           s=55, alpha=0.85, edgecolor="white", linewidth=1, label="SELL")
ax.axhline(0, color=TEXT_DK, linewidth=0.8)
ax.set_title("Clenow Trend: every fill   (positive = BUY $, negative = SELL $)")
ax.set_ylabel("USD")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: money(abs(x))))
ax.legend(loc="upper right")
style_axes(ax)

ax = axes[1]
ax.step(clenow_closed["Submitted At"], clenow_closed["Cum PnL"], where="post",
        color=STRATEGY_COLORS["Clenow Trend"], linewidth=2.6)
ax.scatter(clenow_closed["Submitted At"], clenow_closed["Cum PnL"], color=STRATEGY_COLORS["Clenow Trend"],
           s=44, edgecolor="white", linewidth=1, zorder=3)
ax.axhline(0, color=TEXT_DK, linewidth=0.8)
ax.fill_between(clenow_closed["Submitted At"], 0, clenow_closed["Cum PnL"],
                where=clenow_closed["Cum PnL"] < 0, color=LOSS, alpha=0.10)
final = clenow_closed['Cum PnL'].iloc[-1] if len(clenow_closed) else 0
ax.set_title(f"Clenow Trend: cumulative realized PnL   |   final {money(final)}")
ax.set_ylabel("USD")
style_axes(ax, ydollar=True, datex=True)

plt.tight_layout()
plt.show()
""")

add_md(r"""
The top panel shows every Clenow fill. Buys cluster on Wednesdays because the daily run picks up new names when slots open up. Sells are scattered when names drop out of the top twenty or break their 50 SMA. Per-trade tickets are tiny, mostly under five hundred dollars; this is the inverse-vol sizing on a five-thousand-dollar budget split across seven names producing six to ten percent allocations of a tiny pie.

The bottom panel is the cumulative realized PnL. The first three weeks were rough (KALU, LCII, FTRE) and the curve never fully recovered. The line has been drifting sideways since.

### What worked, what did not, what surprised me

The infrastructure worked. Daily runs, regime checks, exit logic, attribution tags. None of these crashed in twelve weeks of live operation, which I am genuinely proud of given how much time GitHub Actions and I have spent fighting each other. The 52-week proximity gate kept the strategy out of value traps (broken stocks that have stopped falling), and I checked a dozen rejected candidates by hand and the gate was usually right. The exit logic was decisive; no position lingered with a broken trend hoping for recovery, which is the correct behavior for a momentum strategy even when individual exits are losses.

What did not work. Per-trade ticket size was too small to justify the operational overhead. With fees a real broker would have eaten the small wins. Rank-based exits triggered too often in a low-dispersion regime: when the entire market drifts upward, the cross-sectional rank can flip on noise even when nothing about the trade has actually broken. The regime gate is binary, which means it cannot react to a slow deterioration; by the time SPY is below its 200-SMA the damage is usually done.

What surprised me. The single biggest source of drag was not bad signals; it was the cost of frequent exits combined with small position sizes. That is consistent with the academic literature on real-world momentum (Lesmond, Schill, Zhou 2004 on the illusory nature of momentum profits), and it is the kind of thing a backtest with realistic frictions would have caught. I expected the regime gate to reduce drawdowns, but it never fired in this window because the market never broke its 200 SMA, so I have no live evidence that the regime logic actually works.

Assumptions held vs broken. Trend persistence on the survivors held: the names that did stay in the top twenty for several weeks and survived the exit checks generally did make money. Cross-sectional dispersion broke: the dispersion in a quiet uptrend is small, so the top-of-rank advantage was small, and exits ate the rest.

What I will do next time. Backtest before deploying. Even a one-month backtest with realistic transaction costs would have caught the per-trade size issue. Add a per-trade minimum size (skip the order if notional is below $250); that is a one-line change in `src/orders.py`. Make the rank-exit cutoff a function of dispersion rather than a constant. When cross-sectional dispersion is low, raise the cutoff so positions are not flipped on noise. Switch the regime gate from a step function to a continuous risk-on coefficient that scales position size linearly with breadth.

Even with the negative realized PnL I would not abandon Clenow yet. Twelve weeks of a smooth uptrend is a regime where pure trend strategies historically underperform; the right test is whether it survives a chop-to-trend transition with the operational fixes applied. I want at least another semester of live data before I revise my prior.
""")


# =============================================================================
# Strategy Deep Dive #2 - Ranked Asset Allocation V10
# =============================================================================
add_md(r"""
## Deep dive: Ranked Asset Allocation V10 (autotrading sleeve)

The second deep dive is also an autotrading sleeve, but unlike Clenow it has a real backtest behind it.

### Rationale

The strategy is a sector ETF rotation with a hedge sleeve in bear regimes. Long-only ETFs, monthly rebalance. The universe is eleven SPDR sector ETFs (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC), five hedge ETFs (TLT, GLD, UUP, FXY, FXF), SPY as benchmark and SHY as cash proxy.

The architecture is a 50/50 ensemble of two component strategies. V4-Best is a sector momentum rotation with conviction overweight. For each rebalance date it computes a composite rank for each sector ETF based on a weighted combination of momentum (84-day), EWMA volatility (84-day, lambda 0.94), pairwise correlation (84-day), and an ATR breakout trend signal, with weights (0.10, 0.20, 0.25, 0.45). It takes the top 6 sectors in bull regime, top 5 in neutral, top 2 in bear, equal-weights them, then overweights the #1 sector to 50 percent if its score is more than two standard deviations better than the group mean. V8-AW is the all-weather variant: in bull and neutral regimes it holds concentrated sectors (4 names), and in bear regime it rotates into the top 3 hedge ETFs by 42-day momentum. The portfolio for the day is `0.5 * V4_weights + 0.5 * V8_weights`, then leveraged 1x or 2x via a Dual-Filter Leverage rule (DAF) that applies 2x leverage when sector EWMA vol is in the bottom 35th percentile of its rolling 21-day distribution and the regime is bull, otherwise 1x.

Mathematically, let \(R_t = w_M M_t + w_V V_t + w_C C_t + w_T T_t\) be the composite rank with the weights above for V4 and (0.15, 0.15, 0.20, 0.50) for V8. The chosen ETFs at time t are the top-K of \(R_t\), where K depends on the regime. The leverage is

\[
\lambda_t = \begin{cases} 2 & \text{if } b_t = B \text{ and } \sigma_t < q_{0.35}(\sigma_{[t-21:t]}) \\ 1 & \text{otherwise} \end{cases}
\]

so the final portfolio weight is \(\lambda_t \cdot 0.5 \cdot (w^{V4}_t + w^{V8}_t)\) on the chosen ETFs, with cash fill in the remainder.

The key assumptions are that sector momentum has positive expected return at the monthly rebalance frequency; that volatility-targeting via the bottom-35th-percentile filter avoids leveraging into volatility spikes; that the ATR breakout trend signal is a useful tiebreaker when momentum and volatility disagree; and that the hedge sleeve actually decorrelates from equities in stress, which is empirically true on average but failed dramatically in 2022.

### Backtest evidence

This is the strategy I actually did backtest. The research lives in `analysis/strategies/momentum/ranked_asset_allocation.ipynb` and `analysis/strategies/momentum/ranked_2_asset_allocation.ipynb`. I have extracted the key metrics into `reports/strategy_comparison_metrics.csv`:
""")

add_py(r"""
metrics = pd.read_csv("../reports/strategy_comparison_metrics.csv")
v10_rows = metrics[metrics["strategy"].str.startswith("ranking_research_v10")].copy()
cols_to_show = ["strategy", "horizon_label", "sample_start", "sample_end",
                "cagr_pct", "sharpe", "max_drawdown_pct", "benchmark_cagr_pct",
                "benchmark_sharpe", "benchmark_max_drawdown_pct"]
v10_view = v10_rows[cols_to_show].rename(columns={
    "strategy": "Strategy",
    "horizon_label": "Sample",
    "sample_start": "Start",
    "sample_end": "End",
    "cagr_pct": "CAGR (%)",
    "sharpe": "Sharpe",
    "max_drawdown_pct": "MaxDD (%)",
    "benchmark_cagr_pct": "SPY CAGR (%)",
    "benchmark_sharpe": "SPY Sharpe",
    "benchmark_max_drawdown_pct": "SPY MaxDD (%)",
})
v10_view
""")

add_md(r"""
The research backtest tells me three things on the 2004 to 2026 window. V10 1x delivers about 10.8 percent CAGR with a Sharpe near 1.0 and a max drawdown around minus 16.5 percent, against SPY's 10.7 percent CAGR with a Sharpe of 0.56 and a max drawdown of minus 55 percent. V10 DAF (2x leverage in low-vol bull) delivers 14.1 percent CAGR with a Sharpe of 1.05 and a max drawdown of minus 18.5 percent. The out-of-sample test split (2016 to 2026) holds up: 1x gets Sharpe 1.02, DAF gets Sharpe 1.11.

So on backtest evidence this is a strategy I would defend in a meeting. It approximately matches SPY's CAGR while cutting max drawdown by two-thirds. That is the canonical all-weather payoff profile.

The bad news, which I documented in `reports/strategy_triage_decision_memo.md`: the live deployed code is `src/strategies/ranks/ranked_asset_alloc.py`, which is structurally similar to V10 but is not the same implementation as `src/strategies/ranks/v10_research_pipeline.py`. So the backtest above is research-only. The published Sharpe of 1.02 does not validate the live code that actually placed orders.

I made the conscious decision to deploy the simpler `ranked_asset_alloc.py` while leaving the research pipeline in place because I had not finished plumbing target-weight rebalancing into the runner. The simpler live code only sells names that drop out and buys new names that enter; it does not top-up or trim existing positions to a target weight. That means in practice the live portfolio drifts away from the intended V10 allocations as positions move. Closing that gap is on the roadmap.

### Live evidence
""")

add_py(r"""
ranking = orders[orders["Bucket"] == "Ranked Asset Alloc"].copy().sort_values("Submitted At")
ranking["Day"] = ranking["Submitted At"].dt.tz_convert("US/Eastern").dt.normalize()
ranking["Signed Cash"] = np.where(
    ranking["Side"].str.upper() == "BUY", ranking["Cash Flow"], -ranking["Cash Flow"]
)

fig, ax = plt.subplots(figsize=(14.5, 6.4))

day_groups = list(ranking.groupby("Day", sort=True))
n_groups = len(day_groups)
xs = np.arange(n_groups)

bar_w = 0.55
y_max_pos = 0.0
y_min_neg = 0.0

for xi, (day, day_df) in zip(xs, day_groups):
    buys = day_df[day_df["Side"].str.upper() == "BUY"].sort_values("Cash Flow", ascending=False)
    sells = day_df[day_df["Side"].str.upper() == "SELL"].sort_values("Cash Flow", ascending=False)

    bottom = 0.0
    for _, r in buys.iterrows():
        h = r["Cash Flow"]
        ax.bar(xi, h, bottom=bottom, width=bar_w,
               color=GAIN, alpha=0.85, edgecolor="white", linewidth=1.2)
        ax.text(xi, bottom + h/2, r["Symbol"],
                ha="center", va="center", fontsize=9.5, fontweight="700", color="white")
        bottom += h
    y_max_pos = max(y_max_pos, bottom)

    bottom = 0.0
    for _, r in sells.iterrows():
        h = -r["Cash Flow"]
        ax.bar(xi, h, bottom=bottom, width=bar_w,
               color=LOSS, alpha=0.85, edgecolor="white", linewidth=1.2)
        ax.text(xi, bottom + h/2, r["Symbol"],
                ha="center", va="center", fontsize=9.5, fontweight="700", color="white")
        bottom += h
    y_min_neg = min(y_min_neg, bottom)

ax.set_xticks(xs)
ax.set_xticklabels([d.strftime("%b %d") for d, _ in day_groups], fontsize=10)
ax.axhline(0, color=TEXT_DK, linewidth=1.0)

handles = [Patch(facecolor=GAIN, label="BUY"), Patch(facecolor=LOSS, label="SELL")]
ax.legend(handles=handles, loc="upper right", fontsize=10)

ax.set_title("Ranked Asset Alloc: every fill   (BUY stacks above zero, SELL stacks below)")
ax.set_ylabel("USD per fill")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: money(abs(x))))
ax.set_ylim(y_min_neg * 1.18 if y_min_neg < 0 else -200, y_max_pos * 1.18)
ax.set_xlim(-0.6, n_groups - 0.4)
style_axes(ax)
plt.tight_layout()
plt.show()
""")

add_md(r"""
The live trades in the chart above are a perfect illustration of the problem. The names UCTT, ICHR, MKSI, ACMR are all semiconductor capital equipment stocks. Those are not sector ETFs. They appeared because the live `ranked_asset_alloc.py` falls through to a fallback when the sector data download fails or when the regime detection misclassifies, and that fallback ranks the universe stocks the live runner happens to know about.

This is the central operational lesson of the semester. A backtested strategy is only as good as the live code path that actually produces trades. The `v10_research_pipeline.py` and the deployed `ranked_asset_alloc.py` are siblings, not the same thing, and only the latter trades. Until I unify them I cannot claim any of the V10 backtest metrics for the live sleeve. I knew this before I deployed; I deployed anyway because I wanted the operational path running in parallel while I worked on the unification. With more weeks I would have closed the gap.

### What worked, what did not, what surprised me

What worked. The infrastructure (cron, regime check, exit logic) is mechanically sound. No crashes. The few trades it placed were directionally consistent with momentum.

What did not work. The trades placed were not the trades the strategy was designed to place. That is a code-deployment failure, not a research failure.

What surprised me. Discovering that my live code and research code had drifted apart was the single most important finding of the whole semester. It changed how I think about deploying strategies: research is research, deployment is deployment, and the bridge between them needs explicit unit tests.

Assumptions held vs broken. The backtest assumptions (sector momentum, vol-percentile leverage, hedge decorrelation) were never tested live because the live code did not implement them. The operational assumption that "the live code matches the research code" was broken on day one.

What I will do next time. Refuse to deploy a strategy until the live code path passes a `pytest` that re-runs the backtest on the deployed code over a fixed window and asserts the metrics match. Build one shared backtest engine. Right now V10 lives in a notebook with its own data download, its own date conventions, and its own metric definitions; the live code has different data download, different conventions, different metrics. Until they share an engine they cannot share metrics. Implement target-weight rebalancing in `src/trade.py` so the live ranking sleeve can actually express the V10 weights instead of drifting. Add a daily diff between live positions and what V10 would say to hold; alert when the gap exceeds a threshold.
""")


# =============================================================================
# Strategies coded but not yet deployed
# =============================================================================
add_md(r"""
## Strategies coded but not yet deployed

Two strategies sit in `analysis/` as full notebooks but have not yet been deployed (or, in one case, are wired but quiet). I want to give them their own section because they are the next batch I would push live, after another round of optimization and stress-testing.

### Deep Momentum (LSTM forecaster)

File: `analysis/strategies/momentum/deep_momentum.ipynb`.

Deep Momentum is a neural-net momentum forecaster. I train a small LSTM on rolling windows of price, volume, and basic technicals to predict 5-day forward returns, then build a long-only portfolio of the top-decile predictions. The motivation is to see whether a sequence model can pick up non-linear momentum patterns that a linear regression Clenow score misses, particularly the "acceleration vs deceleration" structure that Clenow's deceleration filter only captures crudely.

I have not deployed it yet for two reasons. First, training data leakage: I am not yet confident that the rolling windows in my training pipeline truly avoid forward-looking bias, and until I have implemented strict purged k-fold cross-validation in the Lopez de Prado style I do not trust any of the test-set Sharpe numbers. Second, overfitting risk: the notebook reports a backtest Sharpe near 1.6 but on a relatively short test window, and neural nets overfit aggressively, so the burden of proof is high.

Before deploying I would implement combinatorial purged cross-validation, then walk-forward train across the entire 2010 to 2026 window with no peeking. If the median Sharpe across folds is above 0.8 and the worst-fold drawdown is below 25 percent, I would consider a tiny paper allocation for another semester before any real money. The reason to keep this on the roadmap is that if it works, it is the only sleeve in the lineup that genuinely uses a non-linear model, which would meaningfully diversify the existing trend-following stack.

### 52-week pullback mean reversion

Files: `analysis/strategies/mean_reversion/52W_mean_reversion.ipynb`, `analysis/strategies/mean_reversion/high_pullback_reversion.ipynb`. Production code is at `src/strategies/mean_reversion/52W_mean_reversion_strat.py` with an alias at `src/strategies/mean_reversion/high_pullback_reversion.py`.

The strategy buys stocks pulled back at least 40 percent from their 52-week high but still trading at least 50 percent of that high (filters out free-fallers) and at least 90 percent of their 200-day MA (filters out structurally broken names). It exits when the stock recovers to within 25 percent of its 52-week high, or hits a 25 percent stop from entry, or has been held for 63 trading days.

Backtest summary from `reports/strategy_comparison_metrics.csv`. On the 1-year window (Mar 2025 to Mar 2026): plus 63.3 percent return, Sharpe 1.83, max DD minus 18.5 percent, 24 trades, 71 percent win rate; SPY made 19.3 percent with Sharpe 1.03 over the same window. On the 5-year window (Mar 2021 to Mar 2026): plus 28.8 percent return, Sharpe 0.32, max DD minus 52.6 percent. The strategy ate a big drawdown in 2022, which is exactly when a long-only mean-reversion sleeve should suffer.

Technically the strategy is wired into `config.STRATEGY_ALLOCATIONS` and the GitHub Actions runs do call its `generate_signals()` on schedule. But it has zero fills in the window because the universe scan never produced a candidate that passed all three gates while the regime was risk-on. In that sense it is "deployed" as code but "not deployed" as a working position generator. I count it among the not-yet-deployed strategies because functionally the portfolio has never traded it.

Before sizing it up I would soften the pullback depth to 25 to 30 percent to admit more candidates while keeping the quality gates strict; add an explicit cooldown so that re-entries after stops are avoided (the existing `COOLDOWN_DAYS` constant is documented but never referenced in the live code, which is a real bug); and backtest the strategy through 2008, 2018 Q4, March 2020, and 2022 specifically. The 5-year drawdown of minus 52 percent suggests the current parameters are too aggressive in stress.

### Other research notebooks (less mature)

Three more notebooks exist for completeness. `analysis/strategies/momentum/triple_trigger_momentum.ipynb` is a Clenow variant that requires three independent triggers to fire together; it does not improve Sharpe over the simpler Clenow in my backtests and cuts trade count by half, so I shelved it. `analysis/strategies/momentum/ranked_2_asset_allocation.ipynb` is an early version of what evolved into the V10 strategy currently deployed. `analysis/universe_analysis.ipynb` analyzes the constructed S&P universe for sector composition, market-cap distribution, liquidity, and price filters; it is foundation work, not a strategy.
""")


# =============================================================================
# Final portfolio reflection
# =============================================================================
add_md(r"""
## Final portfolio reflection (forward-looking)

Now I answer the four prompted questions from the assignment description.

### How would I combine these strategies into a single portfolio?

I would not deploy equal-weighted sleeves. The triage memo (`reports/portfolio_combination_proposal.md`) and twelve weeks of live evidence both point to the same conclusion: I want a risk-budgeted allocation with regime caps, not nominal-dollar equality.

Baseline target weights (sleeve weights of total portfolio NAV, not nominal cash) are 35 percent for momentum (Clenow Trend), 15 percent for mean reversion (52W pullback), and 50 percent for ranking and rotation (V10). On top of that I apply regime caps that override the baseline when conditions warrant. Risk-on with normal vol uses the baseline (35 / 15 / 50). Risk-off, or a realized-vol spike (one-month annualized vol of SPY above 35 percent), caps momentum at 15 percent, caps mean reversion at 10 percent, allows ranking up to 75 percent, and leaves the rest in cash.

I would discontinue the discretionary book or move it into a separate tactical account sized to no more than 10 percent of total NAV with explicit per-name caps and a hard stop-loss policy.

### Relative sizing and diversification rationale

Three points drove the 35 / 15 / 50 baseline.

The ranking sleeve is the only one with built-in defensive structure. It has a hedge sleeve (TLT, GLD, UUP, FXY, FXF) it rotates into during bear regimes, plus a cash proxy. The other two sleeves are long-only equities with no built-in escape valve. Giving ranking 50 percent reflects its role as portfolio ballast, not just a third alpha sleeve.

Momentum should not dominate beta. A trend-following sleeve is great in trending tape and miserable in chop. With twelve weeks of live data showing slow bleed in a quiet uptrend, I want momentum sized aggressively enough to matter (35 percent) but not so much that a chop regime drags the whole portfolio (which would happen at 50 percent momentum).

Mean reversion should be a diversifier, not a co-equal alpha sleeve. It is structurally different from the other two (contrarian rather than trend-following). But the 5-year Sharpe of 0.32 in research suggests the strategy needs more validation before it gets risk-budget parity with momentum or ranking. 15 percent is enough to provide diversification value without risking a tail outcome dominating the portfolio.

### Expected correlations and sources of risk

I do not have clean live correlation data, but from first principles I expect the following. Momentum and Ranking are both trend-following at heart, so their correlation should be high in clean bull markets (both ride the leader sectors and leader stocks) and high in synchronized de-risking (both reduce exposure when SPY breaks 200 SMA). Estimated correlation 0.5 to 0.7 in normal regimes; can spike to 0.9 in crisis. Mean Reversion is structurally orthogonal to the other two. Should have negative or near-zero correlation in trending tape and modestly positive correlation in regime-change months when "buy the dip" actually works. Discretionary correlates with whatever beta my taste happens to express; in the past three months that was high-beta growth and quality, and it correlated 0.6 plus with QQQ.

Primary sources of risk, ranked by what I most worry about:

1. Concentration risk. All three algorithmic sleeves can independently buy the same name (most likely momentum and mean reversion if a recent winner pulls back 40 percent). The runner blocks duplicate symbols globally, so the second order is silently suppressed; the realized portfolio is therefore less diversified than the design suggests.
2. Regime-shift drag. Trend strategies bleed in chop. Mean-reversion blows up in trending crashes. Ranking is supposed to handle both but the live code is not the research code.
3. Operational risk. Cron timezone bugs, GitHub Actions runner outages, broker API failures, data download failures that fail open. Each one has bitten me at least once this semester.
4. Code-vs-research drift. The most insidious risk because it does not show up in any backtest. Every live deployment needs a parity test against research.

### Which strategies would I allocate capital to today?

For real capital, deploy now (with constraints): Ranked Asset Allocation V10, but only after I unify `ranked_asset_alloc.py` with `v10_research_pipeline.py` and have a parity test passing. Without that, paper-only.

For real capital, deploy after one more cycle of work: Clenow Trend, after I implement the per-trade minimum size check, the dispersion-aware rank-exit threshold, and at least one quarter of out-of-sample backtest evidence with realistic transaction costs.

Paper-only for now: High Pullback Reversion. Keep it running but do not size it up until I have backtest evidence through a major drawdown (2008, 2020, 2022) showing the entry/exit logic survives. The cooldown bug needs to be fixed first.

Exclude until materially redesigned: Discretionary book. It is not a strategy, it is my attention. As a sleeve in a real portfolio it would need explicit signal generation, sizing, and exit rules. Until then it is performance art. Deep Momentum, until purged-CV implementation makes me trust the Sharpe.

### What new data, signals, or risk controls would I prioritize before real money?

In rough priority order:

1. Daily account-level mark-to-market logging. Without this I cannot compute a true equity curve, true Sharpe, or true drawdown. This is one cron job appending Alpaca's `account.equity` to a CSV. Should have built it on day one.
2. A unified backtest engine. One adjusted-price source, one execution-lag convention, one fee/slippage model, one benchmark, one metric definition. Every sleeve must be evaluable on the same engine. Currently each sleeve has its own home-grown notebook.
3. Live-vs-research parity tests. Every deployed strategy must have a `pytest` that runs the same logic on the same data window in both code paths and asserts metric equivalence within a tolerance.
4. Portfolio-level allocator. A weekly job that recomputes sleeve target weights from rolling vol and correlation, applies regime caps, and emits target portfolio weights to the runner. The runner currently treats sleeve budgets as static lifetime caps; that needs to become dynamic NAV-fraction targets.
5. Concentration cap. A hard per-name cap (no name above 5 percent of portfolio NAV) and a hard sector cap (no sector above 30 percent of NAV) layered on top of all sleeves. Currently each sleeve has its own caps but there is no portfolio-level enforcement.
6. Cross-sleeve symbol overlap monitor. When two sleeves want the same name, log it explicitly rather than silently suppressing the second order. Either replace with the next-best candidate or net the orders intentionally.
7. A real risk model. Even something as simple as a daily Barra-style factor exposure report (market, value, size, momentum, quality, low-vol). The current portfolio almost certainly has factor tilts I am not measuring.
8. Better data. Yahoo Finance via yfinance has been good enough for paper trading but is not dependable enough for real money. I would migrate to Polygon or Alpaca's data API and add data-quality assertions (no missing bars, no zero volumes, no impossible price jumps).
9. A circuit breaker. If realized portfolio drawdown exceeds 5 percent in a week, halt new buys until manual review. Prevents algorithmic runaway.
10. Documentation discipline. The triage memo and this audit took multiple days each. They should be one-line cron outputs, not weekend writing exercises.

### Closing thought

The single most important thing I learned this semester is that deploying a strategy is a different discipline from researching one. Research lives in notebooks where assumptions are convenient and bugs are easy to spot. Deployment lives in cron jobs and broker APIs where the cost of one missed timezone is one bad trade and the cost of one fail-open data handler is a position you did not understand. I started the semester thinking that signal quality was the binding constraint on a quant portfolio. I end it thinking that operational discipline is.

I do not regret deploying Clenow without backtesting first. I learned more from twelve weeks of small live mistakes than I would have from a clean backtest. But I would never make that decision with real money, and I will never make it again.
""")


# =============================================================================
# Appendix
# =============================================================================
add_md(r"""
---

## Appendix: how to refresh this audit

The data behind every chart and metric is `Assignment_final/data/orders_latest.csv`, exported by the `Close Report` GitHub Action. To pull a fresh copy and re-run the notebook end-to-end:

```bash
gh run download $(gh run list -R Shazil10/alpaca-paper-trader --workflow "Close Report" --json databaseId --jq '.[0].databaseId') \
  -R Shazil10/alpaca-paper-trader \
  --dir /tmp/latest && \
cp /tmp/latest/trading-reports-close/orders_latest.csv Assignment_final/data/orders_latest.csv
```

Then either restart-and-run-all in Jupyter, or rerun the builder script:

```bash
venv/bin/python Assignment_final/_build_notebook.py
```

The narrative paragraphs are mine and stay fixed; every number and chart re-derives from the CSV.
""")


# ---------------------------------------------------------------------------
# Build the notebook
# ---------------------------------------------------------------------------

def build_nb() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.9.6"}
    cells = []
    for kind, src in CELLS:
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(src))
        else:
            cells.append(nbf.v4.new_code_cell(src))
    nb.cells = cells
    return nb


def main() -> int:
    nb = build_nb()
    print(f"Built notebook with {len(nb.cells)} cells")
    print(f"Executing in {HERE} ...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute()
    nbf.write(nb, NB_PATH)
    print(f"Wrote executed notebook to {NB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

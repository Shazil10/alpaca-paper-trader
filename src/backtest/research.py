"""One command that turns a strategy into a verdict.

The validation modules each answer one question well. Run individually they also
let a researcher stop as soon as the answer is flattering, which is the failure
mode the whole suite exists to prevent. This module runs all of them, in one
pass, and refuses to grade a strategy that skipped any of them.

What it produces, in order:

1. **Performance** -- the main backtest over the in-sample window.
2. **Out-of-sample** -- a held-back tail, plus anchored and rolling walk-forward
   with walk-forward efficiency against Pardo's 0.5 rule.
3. **Cost stress** -- the same strategy at progressively punitive frictions.
4. **Parameter stability** -- a grid sweep scored for plateaus, not peaks.
5. **Monte Carlo** -- three resamplers, because they answer different questions.
6. **Overfitting risk** -- Deflated Sharpe against the trial count, and PBO.
7. **Baselines** -- SPY, equal-weight, and a matched random-portfolio placebo.
8. **Drawdown risk** -- the loss an investor actually has to sit through.

Each section returns PASS, MODERATE, FAIL or NOT_RUN, and NOT_RUN never counts as
a pass. That rule is the point: the two bugs this suite has already produced --
PBO reading 0.0 when it could not run, and a one-metric card grading PROMISING --
were both cases of absent evidence scoring as good evidence.

The locked holdout
------------------
The final ``holdout_years`` of history are not touched unless
``--unlock-holdout`` is passed, and every unlock appends to
``runs/holdout_unlocks.log``. A holdout you consult freely is just more
in-sample data, and the log exists because the honest number is not "did we look"
but "how many times".
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest import baselines, metrics
from backtest.result import DEFAULT_RUNS_DIR
from backtest.types import BacktestConfig, BacktestResult, CostConfig
from backtest.validation import (
    clustering, montecarlo, overfitting, splits, stability, sweep, verdict,
)

logger = logging.getLogger(__name__)

PASS = "PASS"
MODERATE = "MODERATE"
FAIL = "FAIL"
NOT_RUN = "NOT RUN"

HOLDOUT_LOG = DEFAULT_RUNS_DIR / "holdout_unlocks.log"

#: Cost scenarios, in ascending severity. The base case comes from the config;
#: these are the stresses applied on top. The last is Alpaca-implausible on
#: purpose -- a strategy that only works at zero commission is a strategy whose
#: edge is the commission schedule.
COST_SCENARIOS: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ("base", {}),
    ("20bps", {"spread_bps": 20.0, "slippage_bps": 20.0}),
    ("50bps", {"spread_bps": 50.0, "slippage_bps": 50.0}),
    ("per_share", {"commission_per_share": 0.005}),
)

#: A drawdown past this is a failure regardless of return. Not a statistical
#: threshold -- a behavioural one: an investor who redeems at the bottom realises
#: it, and the Sharpe that survives it is irrelevant.
MAX_TOLERABLE_DRAWDOWN = -0.35
WARN_DRAWDOWN = -0.25


@dataclass
class Section:
    """One question, its answer, and the numbers behind it."""
    name: str
    status: str = NOT_RUN
    headline: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.status != NOT_RUN

    @property
    def passed(self) -> bool:
        return self.status == PASS


@dataclass
class ResearchReport:
    """Everything the research pass learned."""
    strategy_id: str
    window: Tuple[str, str]
    sections: Dict[str, Section] = field(default_factory=dict)
    main_result: Optional[BacktestResult] = None
    sweep_results: Optional[pd.DataFrame] = None
    verdict: str = NOT_RUN
    card: str = ""

    def section(self, name: str) -> Section:
        return self.sections.setdefault(name, Section(name=name))

    @property
    def scored_sections(self) -> List[Section]:
        return [s for s in self.sections.values() if s.scored]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "window": list(self.window),
            "verdict": self.verdict,
            "sections": {
                name: {
                    "status": s.status,
                    "headline": s.headline,
                    "detail": s.detail,
                    "notes": s.notes,
                }
                for name, s in self.sections.items()
            },
        }


# ---------------------------------------------------------------------------
# Holdout discipline
# ---------------------------------------------------------------------------

def holdout_window(
    start: str, end: str, holdout_years: int
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Split into (in-sample, holdout). The holdout is the most recent tail."""
    return splits.split_is_oos(start, end, holdout_years)


def record_holdout_unlock(strategy_id: str, window: Tuple[str, str], reason: str) -> None:
    """Append an unlock to the log. Append-only, deliberately.

    The number that matters is not whether the holdout was consulted but how
    many times, because each look spends a little of its out-of-sample-ness.
    """
    HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with HOLDOUT_LOG.open("a") as handle:
        handle.write(
            f"{stamp}\t{strategy_id}\t{window[0]}..{window[1]}\t{reason}\n"
        )
    logger.warning(
        "HOLDOUT UNLOCKED for %s (%s..%s). Recorded in %s",
        strategy_id, window[0], window[1], HOLDOUT_LOG,
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _performance_section(result: BacktestResult) -> Section:
    """Did it make risk-adjusted money at all, before any robustness question."""
    section = Section(name="Performance")
    if result is None or len(result.returns) < 30:
        section.headline = "too few sessions to measure"
        return section

    m = result.metrics or {}
    sharpe = float(m.get("sharpe", 0.0))
    alpha = m.get("alpha_jensen")
    bench_sharpe = float(m.get("benchmark_sharpe", 0.0))

    section.detail = {
        "cagr": m.get("cagr"),
        "sharpe": sharpe,
        "sortino": m.get("sortino"),
        "beta": m.get("beta"),
        "alpha_jensen": alpha,
        "benchmark_cagr": m.get("benchmark_cagr"),
        "benchmark_sharpe": bench_sharpe or None,
    }

    # Beating the benchmark's Sharpe is the bar, not beating zero. A long-only
    # sleeve in a bull market clears zero by holding anything.
    if sharpe <= 0:
        section.status = FAIL
        section.headline = f"Sharpe {sharpe:.2f} — no risk-adjusted return"
    elif bench_sharpe and sharpe < bench_sharpe:
        section.status = FAIL
        section.headline = (
            f"Sharpe {sharpe:.2f} below the benchmark's {bench_sharpe:.2f}"
        )
        section.notes.append(
            "Buying the benchmark would have been better risk-adjusted."
        )
    elif alpha is not None and float(alpha) < 0:
        section.status = MODERATE
        section.headline = (
            f"Sharpe {sharpe:.2f} but beta-adjusted alpha is {float(alpha):.1%}"
        )
        section.notes.append("The return looks like leveraged beta, not alpha.")
    else:
        section.status = PASS
        section.headline = f"Sharpe {sharpe:.2f}" + (
            f", alpha {float(alpha):.1%}" if alpha is not None else ""
        )
    return section


def _oos_section(
    oos_result: Optional[BacktestResult],
    walk_forward: Optional[splits.WalkForwardResult],
    locked: bool,
) -> Section:
    """Does the edge transfer to data it was not chosen on."""
    section = Section(name="Out-of-sample")

    if locked:
        section.headline = "holdout locked (pass --unlock-holdout to spend it)"
        section.notes.append(
            "A holdout consulted freely is just more in-sample data."
        )

    if walk_forward is None or not walk_forward.folds:
        if oos_result is None:
            return section
    else:
        section.detail["folds"] = len(walk_forward.folds)
        section.detail["median_wfe"] = walk_forward.median_wfe
        section.detail["aggregate_oos_sharpe"] = walk_forward.aggregate_oos_sharpe
        section.detail["fold_oos_sharpe"] = [
            round(float(f.test_sharpe), 3) for f in walk_forward.folds
        ]

    if oos_result is not None and len(oos_result.returns) > 30:
        section.detail["holdout_sharpe"] = float(
            (oos_result.metrics or {}).get("sharpe", 0.0)
        )

    wfe = walk_forward.median_wfe if walk_forward else None
    if wfe is None:
        return section

    # Fold-by-fold, not just the aggregate: three great folds and two
    # catastrophic ones is a different strategy from five mediocre ones, and the
    # average hides exactly that.
    fold_sharpes = [float(f.test_sharpe) for f in walk_forward.folds]
    negative = sum(1 for s in fold_sharpes if s < 0)
    if negative:
        section.notes.append(
            f"{negative} of {len(fold_sharpes)} out-of-sample folds lost money."
        )

    if wfe >= 0.5 and negative <= len(fold_sharpes) // 3:
        section.status = PASS
        section.headline = f"walk-forward efficiency {wfe:.2f} (Pardo ≥ 0.50)"
    elif wfe >= 0.5:
        section.status = MODERATE
        section.headline = (
            f"walk-forward efficiency {wfe:.2f} but {negative} losing fold(s)"
        )
    else:
        section.status = FAIL
        section.headline = (
            f"walk-forward efficiency {wfe:.2f} — optimization is not transferring"
        )
    return section


def _cost_section(scenarios: Dict[str, Dict[str, float]]) -> Section:
    """Does the edge survive friction, or is it a rounding error on spread."""
    section = Section(name="Cost stress")
    if not scenarios:
        return section

    # Nested under a key rather than spread into `detail`, so that adding a
    # summary field later cannot make it look like another cost scenario. It
    # could, and did: `base_sharpe` sitting alongside the scenarios was counted
    # as one and reported as "the edge disappears under base_sharpe".
    section.detail = {"scenarios": scenarios}

    base = scenarios.get("base", {}).get("sharpe")
    stressed = {name: vals for name, vals in scenarios.items() if name != "base"}
    survivors = {
        name: vals for name, vals in stressed.items() if vals.get("sharpe", 0) > 0
    }

    worst_name, worst = min(
        stressed.items(), key=lambda kv: kv[1].get("sharpe", 0), default=("", {}),
    )

    if base is None or not stressed:
        return section

    section.detail["base_sharpe"] = base
    if len(survivors) == len(stressed):
        decay = (
            (base - worst.get("sharpe", base)) / abs(base) if base else 0.0
        )
        if decay < 0.5:
            section.status = PASS
            section.headline = (
                f"Sharpe holds at {worst.get('sharpe', 0):.2f} under {worst_name}"
            )
        else:
            section.status = MODERATE
            section.headline = (
                f"survives {worst_name} but Sharpe decays {decay:.0%} "
                f"({base:.2f} → {worst.get('sharpe', 0):.2f})"
            )
    else:
        failed = [n for n in stressed if n not in survivors]
        section.status = FAIL
        section.headline = f"edge disappears under {', '.join(failed)}"
        section.notes.append(
            "A strategy that needs cheap execution is a bet on the fee schedule."
        )
    return section


def _stability_section(
    sweep_df: Optional[pd.DataFrame],
    param_x: Optional[str],
    param_y: Optional[str],
) -> Section:
    """Is the chosen parameter set on a plateau or on an isolated spike."""
    section = Section(name="Parameter stability")
    if sweep_df is None or sweep_df.empty:
        section.headline = "no parameter grid supplied"
        return section

    section.detail["n_parameter_sets"] = int(len(sweep_df))
    section.detail["spp_25th_percentile"] = clustering.spp_percentile(sweep_df)

    if not (param_x and param_y):
        section.status = MODERATE
        section.headline = (
            f"{len(sweep_df)} sets swept, SPP 25th pct Sharpe "
            f"{section.detail['spp_25th_percentile']:.2f}; "
            "two parameters needed for a plateau map"
        )
        return section

    try:
        heatmap = stability.stability_heatmap(sweep_df, param_x, param_y)
        scores = stability.plateau_score(heatmap)
    except (ValueError, KeyError) as exc:
        section.headline = f"heatmap unavailable: {exc}"
        return section

    values = heatmap.values.astype(float)
    if np.all(np.isnan(values)):
        section.headline = "every parameter set failed to produce a metric"
        return section

    flat = np.nanargmax(values)
    row, col = np.unravel_index(flat, values.shape)
    decay = stability.worst_step_decay(heatmap, int(row), int(col))
    best_plateau = float(np.nanmax(scores.values.astype(float)))

    section.detail.update({
        "best_cell_metric": float(values[row, col]),
        "worst_one_step_decay": None if np.isnan(decay) else float(decay),
        "best_plateau_score": best_plateau,
    })

    # A peak whose neighbours are far worse is a fitting artifact. Note the sign
    # convention: ``worst_step_decay`` returns ``center - min(neighbours)``, so a
    # *large positive* number is a cliff and a number near zero is a plateau. The
    # thresholds are in Sharpe because the metric is.
    if np.isnan(decay):
        section.status = MODERATE
        section.headline = "best cell sits on the grid edge; plateau unmeasurable"
    elif decay < 0.25:
        section.status = PASS
        section.headline = f"plateau: worst one-step Sharpe decay {decay:.2f}"
    elif decay < 0.50:
        section.status = MODERATE
        section.headline = f"soft peak: worst one-step Sharpe decay {decay:.2f}"
    else:
        section.status = FAIL
        section.headline = f"isolated peak: one grid step costs {decay:.2f} Sharpe"
        section.notes.append(
            "Performance depends on parameters no neighbouring set shares."
        )
    return section


def _monte_carlo_section(returns: pd.Series, n_simulations: int) -> Section:
    """Was the equity curve luck, and does it survive imperfect execution."""
    section = Section(name="Monte Carlo")
    if returns is None or len(returns) < 60:
        section.headline = "too few sessions to resample"
        return section

    bootstrap = montecarlo.block_bootstrap(
        returns, n_simulations=n_simulations
    )
    permutation = montecarlo.trade_permutation(
        returns, n_simulations=n_simulations
    )
    jitter = montecarlo.skip_trade_jitter(
        returns, n_simulations=max(n_simulations // 2, 50)
    )

    ci = bootstrap.sharpe_ci
    section.detail = {
        "bootstrap_sharpe_ci": ci,
        "permutation_p_loss": permutation.p_loss,
        "permutation_median_max_dd": (
            float(np.median(permutation.max_dds)) if permutation.max_dds else None
        ),
        "jitter_median_sharpe": (
            float(np.median(jitter.sharpes)) if jitter.sharpes else None
        ),
        "p_loss": bootstrap.p_loss,
    }

    p5 = ci.get("p5")
    if p5 is None:
        return section

    jitter_median = section.detail["jitter_median_sharpe"]
    if p5 > 0 and (jitter_median is None or jitter_median > 0):
        section.status = PASS
        section.headline = (
            f"5th-percentile Sharpe {p5:.2f} stays positive; "
            f"P(loss) {bootstrap.p_loss:.0%}"
        )
    elif p5 > -0.25:
        section.status = MODERATE
        section.headline = (
            f"5th-percentile Sharpe {p5:.2f}; P(loss) {bootstrap.p_loss:.0%}"
        )
    else:
        section.status = FAIL
        section.headline = (
            f"5th-percentile Sharpe {p5:.2f} — the result is within noise"
        )
    return section


def _overfitting_section(
    returns: pd.Series,
    trials: int,
    return_streams: Optional[pd.DataFrame],
) -> Section:
    """Corrected for how many times we looked."""
    section = Section(name="Overfitting risk")
    if returns is None or len(returns) < 60:
        section.headline = "too few sessions"
        return section

    dsr = metrics.deflated_sharpe(returns, num_trials=max(trials, 1))
    pbo = (
        overfitting.probability_of_backtest_overfitting(return_streams)
        if return_streams is not None and return_streams.shape[1] >= 2
        else float("nan")
    )

    section.detail = {
        "trials": int(max(trials, 1)),
        "deflated_sharpe": float(dsr),
        "pbo": None if np.isnan(pbo) else float(pbo),
        "psr": float(metrics.probabilistic_sharpe(returns)),
    }
    if np.isnan(pbo):
        section.notes.append(
            "PBO not run: it needs at least two return streams, so a sweep."
        )

    dsr_ok = dsr >= 0.5
    pbo_ok = (not np.isnan(pbo)) and pbo < 0.5

    if dsr_ok and pbo_ok:
        section.status = PASS
        section.headline = f"Deflated Sharpe {dsr:.2f}, PBO {pbo:.2f}"
    elif dsr_ok and np.isnan(pbo):
        # Deflated Sharpe alone is real evidence, but PBO is the test of the
        # *selection process* and it did not run. Not a clean pass.
        section.status = MODERATE
        section.headline = f"Deflated Sharpe {dsr:.2f}, PBO not run"
    elif dsr_ok:
        section.status = MODERATE
        section.headline = f"Deflated Sharpe {dsr:.2f} but PBO {pbo:.2f}"
    else:
        section.status = FAIL
        section.headline = (
            f"Deflated Sharpe {dsr:.2f} over {max(trials, 1)} trial(s)"
        )
        section.notes.append(
            "Adjusted for the number of parameter sets tried, the edge is not "
            "distinguishable from the best of that many coin flips."
        )
    return section


def _baseline_section(
    result: BacktestResult,
    close_matrix: pd.DataFrame,
    universe: List[str],
    config: BacktestConfig,
    n_simulations: int,
) -> Section:
    """Better than the benchmark, the basket, and a monkey with the same rules."""
    section = Section(name="Baselines")
    if result is None or len(result.returns) < 30 or close_matrix is None:
        section.headline = "no comparison available"
        return section

    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    strategy_sharpe = float((result.metrics or {}).get("sharpe", 0.0))

    # The placebo has to draw from the names the strategy could have chosen
    # among, which is not always the config's `universe_source`. TSMOM declares
    # its own 24-ETF book while its config says `etf_rotation`; drawing from the
    # rotation list instead compared a cross-asset trend follower against random
    # baskets containing SHY, whose near-zero volatility lifted every placebo
    # Sharpe and put the strategy at the 0th percentile of a distribution it was
    # never in. Unioning in what it actually held keeps the comparison matched.
    held_ever = sorted({
        symbol
        for snapshot in (result.snapshots or [])
        for symbol in snapshot.positions
    })
    universe = sorted(set(universe) | set(held_ever))
    universe = [s for s in universe if s in close_matrix.columns]
    if not universe:
        section.headline = "no universe to draw a placebo from"
        return section

    section.detail["strategy_sharpe"] = strategy_sharpe
    section.detail["placebo_universe_size"] = len(universe)

    ew = baselines.equal_weight(
        close_matrix, universe, start, end, config.initial_capital
    )
    if len(ew) > 2:
        ew_sharpe = metrics.sharpe_ratio(ew.pct_change().dropna())
        section.detail["equal_weight_sharpe"] = float(ew_sharpe)

    n_positions = max(
        int(np.median([s.position_count for s in result.snapshots]) or 1), 1
    ) if result.snapshots else 1

    placebo = baselines.random_portfolio_placebo(
        close_matrix, n_positions, universe, start, end,
        config.initial_capital, n_simulations=n_simulations,
        seed=config.random_seed,
    )
    percentile = baselines.percentile_vs_random(strategy_sharpe, placebo)
    section.detail.update({
        "random_portfolio_percentile": percentile,
        "random_portfolio_n": n_simulations,
        "matched_position_count": n_positions,
    })

    ew_sharpe = section.detail.get("equal_weight_sharpe")
    beats_basket = ew_sharpe is None or strategy_sharpe > ew_sharpe

    if percentile > 95 and beats_basket:
        section.status = PASS
        section.headline = f"{percentile:.0f}th percentile of matched random books"
    elif percentile > 75:
        section.status = MODERATE
        section.headline = f"{percentile:.0f}th percentile of matched random books"
        if not beats_basket:
            section.notes.append(
                f"Equal-weight over the same universe scored {ew_sharpe:.2f}; "
                "the ranking is not adding much."
            )
    else:
        section.status = FAIL
        section.headline = (
            f"{percentile:.0f}th percentile — a random book with the same "
            "position count does as well"
        )
    return section


def _drawdown_section(result: BacktestResult) -> Section:
    """The loss someone has to sit through, judged behaviourally."""
    section = Section(name="Drawdown risk")
    if result is None or len(result.equity_curve) < 30:
        section.headline = "no equity curve"
        return section

    dd = float(metrics.max_drawdown(result.equity_curve))
    series = metrics.drawdown_series(result.equity_curve)
    underwater = int((series < -0.05).sum())

    section.detail = {
        "max_drawdown": dd,
        "sessions_more_than_5pct_underwater": underwater,
        "pct_of_time_underwater": (
            underwater / len(series) if len(series) else None
        ),
        "calmar": float(metrics.calmar_ratio(result.equity_curve)),
    }

    if dd > WARN_DRAWDOWN:
        section.status = PASS
        section.headline = f"max drawdown {dd:.1%}"
    elif dd > MAX_TOLERABLE_DRAWDOWN:
        section.status = MODERATE
        section.headline = f"max drawdown {dd:.1%}"
        section.notes.append("Deep enough that position sizing is the real question.")
    else:
        section.status = FAIL
        section.headline = (
            f"max drawdown {dd:.1%} — past the point most capital redeems"
        )
    return section


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

#: A research pass must score at least this many sections before it earns a
#: grade. Mirrors ``verdict.MIN_SCORED_CHECKS``, for the same reason: a report
#: that ran two tests and passed both has not established anything.
MIN_SCORED_SECTIONS = 5


def grade(report: ResearchReport) -> str:
    """Overall verdict from the section statuses.

    Any FAIL caps the verdict. A single failed section is not averaged away,
    because these are not independent measurements of one quantity -- they are
    separate necessary conditions, and a strategy that cannot survive costs is
    not rescued by a good plateau map.
    """
    scored = report.scored_sections
    if len(scored) < MIN_SCORED_SECTIONS:
        return f"INCOMPLETE ({len(scored)}/{MIN_SCORED_SECTIONS} sections scored)"

    failures = [s for s in scored if s.status == FAIL]
    moderates = [s for s in scored if s.status == MODERATE]

    if failures:
        return "REJECT" if len(failures) > 1 else "WEAK"
    if len(moderates) > len(scored) // 2:
        return "WEAK"
    if moderates:
        return "CAUTIOUS"
    return "PROMISING"


def render_research_card(report: ResearchReport) -> str:
    """The summary a decision gets made from."""
    order = [
        "Performance", "Out-of-sample", "Cost stress", "Parameter stability",
        "Monte Carlo", "Overfitting risk", "Baselines", "Drawdown risk",
    ]
    width = max(len(name) for name in order)

    lines = [
        f"Strategy verdict: {report.verdict}",
        "",
        f"  {report.strategy_id}  |  {report.window[0]} .. {report.window[1]}",
        "",
    ]

    for name in order:
        section = report.sections.get(name)
        if section is None:
            continue
        lines.append(f"  {name:<{width}}  {section.status:<9}  {section.headline}")

    notes = [
        (s.name, note)
        for name in order
        if (s := report.sections.get(name)) is not None
        for note in s.notes
    ]
    if notes:
        lines.append("")
        lines.append("  NOTES")
        for name, note in notes:
            lines.append(f"    {name}: {note}")

    not_run = [
        s.name for name in order
        if (s := report.sections.get(name)) is not None and not s.scored
    ]
    if not_run:
        lines.append("")
        lines.append("  NOT SCORED (absence of evidence is not a pass)")
        for name in not_run:
            lines.append(f"    - {name}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _metrics_only(result: BacktestResult) -> Dict[str, Any]:
    """Reduce a result to the metrics the walk-forward helpers expect."""
    if result is None or len(result.returns) < 5:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
    m = result.metrics or {}
    return {
        "sharpe": float(m.get("sharpe", 0.0)),
        "cagr": float(m.get("cagr", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
    }


def run_research(
    config: BacktestConfig,
    *,
    run_backtest_fn: Callable[..., BacktestResult],
    price_panel: Optional[pd.DataFrame] = None,
    close_matrix: Optional[pd.DataFrame] = None,
    ohlc_adjusted: Optional[Dict] = None,
    universe_fn: Optional[Callable] = None,
    sector_map: Optional[Dict[str, str]] = None,
    holdout_years: int = 2,
    unlock_holdout: bool = False,
    param_grid: Optional[Dict[str, List]] = None,
    walk_forward_train_years: int = 4,
    walk_forward_test_years: int = 1,
    n_simulations: int = 500,
    save: bool = True,
) -> ResearchReport:
    """Run every validation section and produce a verdict.

    ``run_backtest_fn`` is injected rather than imported to keep this module free
    of the runner, which imports it -- and so a test can drive the whole pass
    with a stub.
    """
    full_start, full_end = config.start_date, config.end_date
    (is_start, is_end), (oos_start, oos_end) = holdout_window(
        full_start, full_end, holdout_years
    )

    report = ResearchReport(
        strategy_id=config.strategy_id, window=(is_start, is_end)
    )

    shared = dict(
        price_panel=price_panel,
        close_matrix=close_matrix,
        ohlc_adjusted=ohlc_adjusted,
        universe_fn=universe_fn,
        sector_map=sector_map,
    )

    def run_window(start: str, end: str, overrides: Optional[Dict] = None):
        cfg = dataclasses.replace(config, start_date=start, end_date=end)
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        return run_backtest_fn(cfg, save=False, **shared)

    # --- 1. Performance, in-sample only -----------------------------------
    logger.info("Research: main backtest %s..%s", is_start, is_end)
    main = run_window(is_start, is_end)
    report.main_result = main
    report.sections["Performance"] = _performance_section(main)

    # --- 2. Out-of-sample --------------------------------------------------
    oos_result = None
    if unlock_holdout:
        record_holdout_unlock(
            config.strategy_id, (oos_start, oos_end), "run_research"
        )
        logger.info("Research: holdout %s..%s", oos_start, oos_end)
        oos_result = run_window(oos_start, oos_end)

    logger.info("Research: walk-forward over %s..%s", is_start, is_end)
    wf = splits.anchored_walk_forward(
        lambda s, e, p: _metrics_only(run_window(s, e)),
        is_start, is_end,
        train_years=walk_forward_train_years,
        test_years=walk_forward_test_years,
    )
    report.sections["Out-of-sample"] = _oos_section(
        oos_result, wf, locked=not unlock_holdout
    )

    # --- 3. Cost stress ---------------------------------------------------
    logger.info("Research: cost stress")
    scenarios: Dict[str, Dict[str, float]] = {}
    for name, overrides in COST_SCENARIOS:
        cost = dataclasses.replace(config.cost, **overrides) if overrides else config.cost
        stressed = run_window(is_start, is_end, {"cost": cost})
        scenarios[name] = _metrics_only(stressed)
    report.sections["Cost stress"] = _cost_section(scenarios)

    # --- 4. Parameter stability + the sweep the later sections need --------
    sweep_df: Optional[pd.DataFrame] = None
    return_streams: Optional[pd.DataFrame] = None
    if param_grid:
        logger.info("Research: parameter sweep over %s", list(param_grid))
        streams: Dict[str, pd.Series] = {}

        def sweep_once(params: Dict[str, Any]) -> Dict[str, Any]:
            merged = {**config.params, **params}
            result = run_window(is_start, is_end, {"params": merged})
            label = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
            if len(result.returns) > 30:
                streams[label] = result.returns
            return _metrics_only(result)

        sweep_df = sweep.grid_sweep(sweep_once, param_grid)
        report.sweep_results = sweep_df
        if len(streams) >= 2:
            return_streams = pd.DataFrame(streams).dropna(how="all")

    keys = list(param_grid or {})
    report.sections["Parameter stability"] = _stability_section(
        sweep_df,
        keys[0] if len(keys) > 0 else None,
        keys[1] if len(keys) > 1 else None,
    )

    # --- 5. Monte Carlo ---------------------------------------------------
    logger.info("Research: Monte Carlo")
    report.sections["Monte Carlo"] = _monte_carlo_section(
        main.returns if main else pd.Series(dtype=float), n_simulations
    )

    # --- 6. Overfitting ---------------------------------------------------
    trials = max(
        config.trial_index,
        int(len(sweep_df)) if sweep_df is not None else 0,
        1,
    )
    report.sections["Overfitting risk"] = _overfitting_section(
        main.returns if main else pd.Series(dtype=float), trials, return_streams
    )

    # --- 7. Baselines -----------------------------------------------------
    logger.info("Research: baselines and placebo")
    universe: List[str] = []
    if close_matrix is not None:
        universe = (
            sorted(universe_fn(pd.Timestamp(is_end)))
            if universe_fn is not None
            else [str(c) for c in close_matrix.columns]
        )
        universe = [s for s in universe if s in close_matrix.columns]
    report.sections["Baselines"] = _baseline_section(
        main, close_matrix, universe,
        dataclasses.replace(config, start_date=is_start, end_date=is_end),
        min(n_simulations, 500),
    )

    # --- 8. Drawdown ------------------------------------------------------
    report.sections["Drawdown risk"] = _drawdown_section(main)

    # --- Verdict ----------------------------------------------------------
    report.verdict = grade(report)
    report.card = render_research_card(report)

    if save:
        save_research(report)

    return report


def save_research(
    report: ResearchReport, runs_dir: Optional[Path] = None
) -> Path:
    """Write the report next to the run artifacts."""
    base = Path(runs_dir or DEFAULT_RUNS_DIR)
    name = report.strategy_id.replace(".", "_").replace("/", "_")
    out = base / f"{datetime.now():%Y-%m-%d}_research_{name}"
    out.mkdir(parents=True, exist_ok=True)

    (out / "research.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str)
    )
    (out / "verdict.txt").write_text(report.card + "\n")

    if report.sweep_results is not None and not report.sweep_results.empty:
        report.sweep_results.to_csv(out / "sweep.csv", index=False)

    if report.main_result is not None and len(report.main_result.equity_curve):
        report.main_result.equity_curve.to_csv(out / "equity.csv", header=["equity"])

    logger.info("Saved research report to %s", out)
    return out

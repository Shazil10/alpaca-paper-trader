"""The verdict card — pass/fail summary of all validation tests."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: A card must score at least this many checks before it is allowed a grade
#: better than INCOMPLETE. Five of the seven scored checks is a deliberate
#: middle: strict enough that a one-metric card cannot read as a pass, loose
#: enough that a run without a parameter sweep (so no plateau, no Deflated
#: Sharpe, no PBO) can still be graded on Sharpe, walk-forward, costs and the
#: random-portfolio placebo.
MIN_SCORED_CHECKS = 5


def render_verdict(
    strategy_id: str,
    params: Dict[str, Any],
    data_desc: str,
    metrics: Dict[str, Any],
    oos_sharpe: Optional[float] = None,
    bootstrap_ci: Optional[Dict[str, float]] = None,
    wfe: Optional[float] = None,
    deflated_sharpe: Optional[float] = None,
    pbo: Optional[float] = None,
    random_pct: Optional[float] = None,
    survives_20bps: Optional[bool] = None,
    plateau_ok: Optional[bool] = None,
    worst_step_decay: Optional[float] = None,
    cluster_membership: Optional[str] = None,
    cluster_size: Optional[int] = None,
    cluster_total: Optional[int] = None,
    weak_regime: Optional[str] = None,
    limitations: Optional[list] = None,
) -> str:
    """Render a human-readable verdict card."""

    passes = 0
    total_tests = 0

    checks = []

    if oos_sharpe is not None:
        total_tests += 1
        ci_str = ""
        if bootstrap_ci:
            ci_str = f"  [{bootstrap_ci.get('p5', 0):.2f}, {bootstrap_ci.get('p95', 0):.2f}]  block bootstrap 95%"
        checks.append(f"  OOS Sharpe                 {oos_sharpe:.2f}{ci_str}")
        if oos_sharpe > 0:
            passes += 1

    if metrics.get("alpha_jensen") is not None:
        checks.append(f"  Alpha vs SPY (beta-adj)    {metrics['alpha_jensen']:.1%} annualized")

    if metrics.get("beta") is not None:
        checks.append(f"  Beta                       {metrics['beta']:.2f}")

    if metrics.get("max_drawdown") is not None:
        checks.append(f"  Max drawdown              {metrics['max_drawdown']:.1%}")

    if wfe is not None:
        total_tests += 1
        pass_str = "pass (>0.50)" if wfe >= 0.5 else "FAIL (<0.50)"
        checks.append(f"  Walk-forward efficiency    {wfe:.2f}   {pass_str}")
        if wfe >= 0.5:
            passes += 1

    if deflated_sharpe is not None:
        total_tests += 1
        n_trials = metrics.get("n_trials", "?")
        pass_str = "pass (>0.50)" if deflated_sharpe >= 0.5 else "FAIL (<0.50)"
        checks.append(f"  Deflated Sharpe            {deflated_sharpe:.2f}   {pass_str}, {n_trials} trials")
        if deflated_sharpe >= 0.5:
            passes += 1

    if pbo is not None:
        if math.isnan(pbo):
            # Not counted in total_tests: the test did not run, so scoring it
            # either way would misreport. Shown so it cannot pass unnoticed.
            checks.append("  PBO (CSCV)                  n/a   not run (need >=2 streams)")
        else:
            total_tests += 1
            pass_str = "pass (<0.50)" if pbo < 0.5 else "FAIL (>=0.50)"
            checks.append(f"  PBO (CSCV)                 {pbo:.2f}   {pass_str}")
            if pbo < 0.5:
                passes += 1

    if random_pct is not None:
        total_tests += 1
        pass_str = f"pass (>{95}th)" if random_pct > 95 else f"FAIL (<={95}th)"
        checks.append(f"  Random-portfolio pct       {random_pct:.0f}th   {pass_str}")
        if random_pct > 95:
            passes += 1

    if survives_20bps is not None:
        total_tests += 1
        checks.append(f"  Survives 20bps costs       {'yes' if survives_20bps else 'NO'}")
        if survives_20bps:
            passes += 1

    if plateau_ok is not None:
        total_tests += 1
        decay_str = (
            f"  worst 1-step Sharpe decay {worst_step_decay:.2f}"
            if worst_step_decay is not None
            else ""
        )
        checks.append(f"  Parameter plateau          {'yes' if plateau_ok else 'NO'}{decay_str}")
        if plateau_ok:
            passes += 1

    if cluster_membership:
        checks.append(f"  Return-cluster membership  {cluster_membership}   cluster n={cluster_size} of {cluster_total}")

    if weak_regime:
        checks.append(f"  Weak regime                {weak_regime}")

    # Overall verdict.
    #
    # The grade is a fraction of whatever checks were supplied, so a sparse card
    # scores as well as a thorough one: a single `oos_sharpe > 0` reads 1/1 and
    # renders PROMISING, even on a strategy with negative alpha and a 32%
    # drawdown. That is the same failure as scoring a test that never ran -- it
    # rewards absence of evidence. A positive grade therefore requires a quorum,
    # and a card below it says what is missing rather than guessing.
    not_supplied = [
        name for name, value in (
            ("OOS Sharpe", oos_sharpe),
            ("walk-forward efficiency", wfe),
            ("Deflated Sharpe", deflated_sharpe),
            ("PBO", None if pbo is None or math.isnan(pbo) else pbo),
            ("random-portfolio percentile", random_pct),
            ("cost survival", survives_20bps),
            ("parameter plateau", plateau_ok),
        )
        if value is None
    ]

    if total_tests == 0:
        verdict = "INCOMPLETE"
    elif total_tests < MIN_SCORED_CHECKS:
        verdict = f"INCOMPLETE ({passes}/{total_tests} scored, need {MIN_SCORED_CHECKS})"
    elif passes == total_tests:
        verdict = "PROMISING"
    elif passes >= total_tests * 0.7:
        verdict = "CAUTIOUS"
    elif passes >= total_tests * 0.5:
        verdict = "WEAK"
    else:
        verdict = "REJECT"

    params_str = ", ".join(f"{k}={v}" for k, v in params.items())

    lines = [
        f"Strategy: {strategy_id}  |  Params: {params_str}",
        f"Data: {data_desc}",
        "",
        f"VERDICT: {verdict}",
        "",
    ]
    lines.extend(checks)

    if not_supplied and total_tests < MIN_SCORED_CHECKS:
        lines.append("")
        lines.append("  NOT SUPPLIED (cannot be scored, so the grade is withheld)")
        for name in not_supplied:
            lines.append(f"    - {name}")

    if limitations:
        lines.append("")
        lines.append("  LIMITATIONS")
        for lim in limitations:
            lines.append(f"    - {lim}")

    return "\n".join(lines)

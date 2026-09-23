"""Tests for the research orchestrator.

The backtest function is stubbed throughout. These tests are about the *grading*
-- which sections score, what they score, and whether an unrun section can
flatter a strategy -- not about the engine, which has its own tests. Stubbing also
makes them deterministic and fast, which matters because a real research pass runs
fifty-odd backtests.

The rule under test everywhere: NOT RUN is never a pass. Both bugs this suite has
already produced were violations of it -- PBO reading 0.0 when it could not run,
and a one-metric card grading PROMISING.

Run with: ./venv/bin/python -m pytest tests/backtest/test_research.py -v
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from backtest import research
from backtest.research import FAIL, MODERATE, NOT_RUN, PASS
from backtest.types import (
    BacktestConfig, BacktestResult, CostConfig, PortfolioSnapshot, RiskConfig,
)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _returns(n=1500, mu=0.0006, sigma=0.01, seed=1):
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.Series(np.random.RandomState(seed).normal(mu, sigma, n), index=idx)


def _result(returns, initial=100_000.0, n_positions=10, benchmark_sharpe=0.4):
    """A BacktestResult carrying just enough for the sections to read."""
    equity = (1.0 + returns).cumprod() * initial
    from backtest import metrics

    snapshots = [
        PortfolioSnapshot(
            date=date, cash=0.0, equity=float(value),
            positions={f"S{i}": None for i in range(n_positions)},  # count only
        )
        for date, value in list(zip(equity.index, equity.values))[:5]
    ]

    return BacktestResult(
        config=_config(),
        equity_curve=equity,
        returns=returns,
        metrics={
            "cagr": metrics.cagr(equity),
            "sharpe": metrics.sharpe_ratio(returns),
            "sortino": metrics.sortino_ratio(returns),
            "max_drawdown": metrics.max_drawdown(equity),
            "beta": 0.5,
            "alpha_jensen": 0.04,
            "benchmark_sharpe": benchmark_sharpe,
            "benchmark_cagr": 0.08,
        },
        snapshots=snapshots,
    )


def _config(**overrides):
    base = dict(
        strategy_id="stub",
        strategy_module="stub.module",
        start_date="2010-01-04",
        end_date="2020-12-31",
        initial_capital=100_000.0,
        cost=CostConfig(),
        risk=RiskConfig(),
        params={"a": 1},
        trial_index=0,
    )
    base.update(overrides)
    return BacktestConfig(**base)


# ---------------------------------------------------------------------------
# Section grading
# ---------------------------------------------------------------------------

class TestPerformanceSection:
    def test_negative_sharpe_fails(self):
        # benchmark_sharpe=0 isolates this branch from the "lost to the index"
        # one, which would otherwise claim the failure first.
        result = _result(_returns(mu=-0.002), benchmark_sharpe=0.0)

        section = research._performance_section(result)

        assert section.status == FAIL
        assert "no risk-adjusted return" in section.headline

    def test_losing_to_the_benchmark_fails(self):
        """Beating zero is not the bar when you could have bought the index."""
        result = _result(_returns(mu=0.0002), benchmark_sharpe=1.5)

        section = research._performance_section(result)

        assert section.status == FAIL
        assert "below the benchmark" in section.headline

    def test_negative_alpha_is_moderate_not_pass(self):
        """A good Sharpe from leveraged beta is not alpha."""
        result = _result(_returns(), benchmark_sharpe=0.1)
        result.metrics["alpha_jensen"] = -0.03

        section = research._performance_section(result)

        assert section.status == MODERATE
        assert "leveraged beta" in section.notes[0]

    def test_a_real_edge_passes(self):
        section = research._performance_section(
            _result(_returns(mu=0.0008), benchmark_sharpe=0.3)
        )

        assert section.status == PASS

    def test_too_short_a_run_is_not_scored(self):
        section = research._performance_section(_result(_returns(n=10)))

        assert section.status == NOT_RUN


class TestDrawdownSection:
    def test_a_shallow_drawdown_passes(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        equity = pd.Series(np.linspace(100.0, 150.0, 300), index=idx)
        result = _result(_returns(n=300))
        result.equity_curve = equity

        assert research._drawdown_section(result).status == PASS

    def test_a_forty_percent_drawdown_fails_regardless_of_return(self):
        """Behavioural, not statistical: capital redeems at the bottom."""
        idx = pd.bdate_range("2020-01-01", periods=300)
        path = np.concatenate([
            np.linspace(100.0, 200.0, 100),
            np.linspace(200.0, 110.0, 100),   # -45%
            np.linspace(110.0, 320.0, 100),   # and a full recovery
        ])
        result = _result(_returns(n=300))
        result.equity_curve = pd.Series(path, index=idx)

        section = research._drawdown_section(result)

        assert section.status == FAIL
        assert "-45" in section.headline or "-4" in section.headline


class TestCostSection:
    def test_surviving_every_scenario_passes(self):
        section = research._cost_section({
            "base": {"sharpe": 1.0}, "20bps": {"sharpe": 0.9},
            "50bps": {"sharpe": 0.8}, "per_share": {"sharpe": 0.95},
        })

        assert section.status == PASS

    def test_heavy_decay_is_moderate(self):
        section = research._cost_section({
            "base": {"sharpe": 1.0}, "20bps": {"sharpe": 0.6},
            "50bps": {"sharpe": 0.2},
        })

        assert section.status == MODERATE
        assert "decays" in section.headline

    def test_an_edge_that_needs_cheap_execution_fails(self):
        section = research._cost_section({
            "base": {"sharpe": 1.0}, "20bps": {"sharpe": 0.4},
            "50bps": {"sharpe": -0.1},
        })

        assert section.status == FAIL
        assert "50bps" in section.headline

    def test_summary_fields_are_not_mistaken_for_scenarios(self):
        """Regression: base_sharpe once sat alongside the scenarios and was
        reported as one -- "the edge disappears under base_sharpe"."""
        section = research._cost_section({
            "base": {"sharpe": 1.0}, "20bps": {"sharpe": 0.9},
        })

        assert "base_sharpe" not in section.detail["scenarios"]
        assert section.status == PASS
        assert "base_sharpe" not in section.headline

    def test_no_scenarios_is_not_scored(self):
        assert research._cost_section({}).status == NOT_RUN


class TestOverfittingSection:
    def test_unrun_pbo_caps_the_section_at_moderate(self):
        """Deflated Sharpe alone is evidence; PBO tests the selection process."""
        section = research._overfitting_section(_returns(mu=0.0012), 1, None)

        assert section.status == MODERATE
        assert section.detail["pbo"] is None
        assert "PBO not run" in section.headline
        assert any("at least two return streams" in n for n in section.notes)

    def test_both_statistics_passing_is_a_pass(self):
        returns = _returns(mu=0.0012)
        streams = pd.DataFrame({
            "a": returns.values,
            "b": _returns(mu=0.0011, seed=2).values,
        })

        section = research._overfitting_section(returns, 4, streams)

        assert section.detail["pbo"] is not None
        assert section.status in (PASS, MODERATE)

    def test_many_trials_deflate_the_edge(self):
        """The same returns, claimed after more searching, are worth less."""
        returns = _returns(mu=0.0005)

        few = research._overfitting_section(returns, 1, None)
        many = research._overfitting_section(returns, 5000, None)

        assert many.detail["deflated_sharpe"] < few.detail["deflated_sharpe"]


class TestStabilitySection:
    def _sweep(self, values):
        rows = []
        for (x, y), sharpe in values.items():
            rows.append({"lookback": x, "halflife": y, "sharpe": sharpe})
        return pd.DataFrame(rows)

    def test_no_grid_is_not_scored(self):
        section = research._stability_section(None, None, None)

        assert section.status == NOT_RUN
        assert "no parameter grid" in section.headline

    def test_a_plateau_passes(self):
        """A flat field: stepping anywhere costs nothing."""
        grid = {
            (x, y): 1.0
            for x in (5, 10, 15) for y in (20, 40, 60)
        }
        section = research._stability_section(
            self._sweep(grid), "lookback", "halflife"
        )

        assert section.status == PASS
        assert "plateau" in section.headline
        assert section.detail["worst_one_step_decay"] == pytest.approx(0.0)

    def test_an_isolated_spike_fails(self):
        """One bright cell in a dark field is a fitting artifact.

        Also pins the sign convention, which is easy to invert:
        ``worst_step_decay`` is ``center - min(neighbours)``, so a large
        *positive* number is a cliff. Reading it as negative graded this exact
        grid as a plateau.
        """
        grid = {
            (x, y): 0.1
            for x in (5, 10, 15) for y in (20, 40, 60)
        }
        grid[(10, 40)] = 3.0

        section = research._stability_section(
            self._sweep(grid), "lookback", "halflife"
        )

        assert section.detail["worst_one_step_decay"] == pytest.approx(2.9)
        assert section.status == FAIL
        assert "isolated peak" in section.headline

    def test_one_parameter_still_reports_spp(self):
        sweep = pd.DataFrame({"lookback": [5, 10, 15], "sharpe": [0.5, 0.9, 0.6]})

        section = research._stability_section(sweep, "lookback", None)

        assert section.status == MODERATE
        assert "spp_25th_percentile" in section.detail


class TestMonteCarloSection:
    def test_a_robust_edge_passes(self):
        section = research._monte_carlo_section(_returns(mu=0.0015), 100)

        assert section.status == PASS
        assert section.detail["bootstrap_sharpe_ci"]["p5"] > 0

    def test_noise_fails(self):
        section = research._monte_carlo_section(_returns(mu=-0.0008), 100)

        assert section.status == FAIL

    def test_too_short_is_not_scored(self):
        assert research._monte_carlo_section(_returns(n=20), 100).status == NOT_RUN


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

class TestGrade:
    def _report(self, statuses):
        report = research.ResearchReport(strategy_id="x", window=("a", "b"))
        for name, status in statuses.items():
            report.sections[name] = research.Section(name=name, status=status)
        return report

    def test_a_sparse_report_is_incomplete(self):
        report = self._report({"Performance": PASS, "Monte Carlo": PASS})

        grade = research.grade(report)

        assert "INCOMPLETE" in grade
        assert "2/5" in grade

    def test_all_pass_is_promising(self):
        report = self._report({f"s{i}": PASS for i in range(6)})

        assert research.grade(report) == "PROMISING"

    def test_one_failure_caps_at_weak(self):
        """Necessary conditions, not measurements to average."""
        statuses = {f"s{i}": PASS for i in range(5)}
        statuses["bad"] = FAIL

        assert research.grade(self._report(statuses)) == "WEAK"

    def test_two_failures_is_a_reject(self):
        statuses = {f"s{i}": PASS for i in range(5)}
        statuses.update({"bad": FAIL, "worse": FAIL})

        assert research.grade(self._report(statuses)) == "REJECT"

    def test_mostly_moderate_is_weak(self):
        statuses = {f"m{i}": MODERATE for i in range(4)}
        statuses.update({"p1": PASS, "p2": PASS})

        assert research.grade(self._report(statuses)) == "WEAK"

    def test_some_moderate_is_cautious(self):
        statuses = {f"p{i}": PASS for i in range(5)}
        statuses["m"] = MODERATE

        assert research.grade(self._report(statuses)) == "CAUTIOUS"

    def test_unrun_sections_do_not_count_toward_the_quorum(self):
        statuses = {f"p{i}": PASS for i in range(3)}
        statuses.update({f"n{i}": NOT_RUN for i in range(5)})

        assert "INCOMPLETE" in research.grade(self._report(statuses))


class TestCard:
    def test_card_lists_unscored_sections_explicitly(self):
        report = research.ResearchReport(strategy_id="x", window=("2010", "2020"))
        report.sections["Performance"] = research.Section(
            name="Performance", status=PASS, headline="Sharpe 1.0"
        )
        report.sections["Cost stress"] = research.Section(name="Cost stress")
        report.verdict = "INCOMPLETE"

        card = research.render_research_card(report)

        assert "Strategy verdict: INCOMPLETE" in card
        assert "NOT SCORED" in card
        assert "Cost stress" in card

    def test_card_shows_every_section_status(self):
        report = research.ResearchReport(strategy_id="s", window=("a", "b"))
        for name in ("Performance", "Out-of-sample", "Cost stress",
                     "Parameter stability", "Monte Carlo", "Overfitting risk",
                     "Baselines", "Drawdown risk"):
            report.sections[name] = research.Section(
                name=name, status=PASS, headline="ok"
            )
        report.verdict = "PROMISING"

        card = research.render_research_card(report)

        for name in report.sections:
            assert name in card


# ---------------------------------------------------------------------------
# Holdout discipline
# ---------------------------------------------------------------------------

class TestHoldout:
    def test_split_puts_the_tail_out_of_sample(self):
        (is_start, is_end), (oos_start, oos_end) = research.holdout_window(
            "2010-01-01", "2020-12-31", holdout_years=2
        )

        assert is_start == "2010-01-01"
        assert oos_end == "2020-12-31"
        assert pd.Timestamp(oos_start) == pd.Timestamp("2018-12-31")
        assert pd.Timestamp(is_end) < pd.Timestamp(oos_start)

    def test_unlock_is_appended_to_the_log(self, tmp_path, monkeypatch):
        log = tmp_path / "holdout_unlocks.log"
        monkeypatch.setattr(research, "HOLDOUT_LOG", log)

        research.record_holdout_unlock("strat", ("2019-01-01", "2020-12-31"), "test")
        research.record_holdout_unlock("strat", ("2019-01-01", "2020-12-31"), "again")

        lines = log.read_text().strip().split("\n")
        assert len(lines) == 2, "the count of looks is the number that matters"
        assert all("strat" in line for line in lines)


# ---------------------------------------------------------------------------
# End to end with a stub engine
# ---------------------------------------------------------------------------

class TestRunResearch:
    def test_full_pass_scores_every_section_it_can(self, tmp_path, monkeypatch):
        monkeypatch.setattr(research, "DEFAULT_RUNS_DIR", tmp_path)

        calls = []

        def stub(config, save=False, **kwargs):
            calls.append((config.start_date, config.end_date))
            window = _returns(n=800, mu=0.0007, seed=len(calls))
            return _result(window)

        closes = pd.DataFrame(
            {f"S{i}": np.linspace(10, 20, 3000) for i in range(6)},
            index=pd.bdate_range("2010-01-04", periods=3000),
        )

        report = research.run_research(
            _config(),
            run_backtest_fn=stub,
            close_matrix=closes,
            holdout_years=2,
            param_grid={"a": [1, 2], "b": [10, 20]},
            n_simulations=60,
            save=False,
        )

        assert len(calls) > 10, "every section must actually run a backtest"
        assert report.verdict != NOT_RUN
        assert len(report.scored_sections) >= research.MIN_SCORED_SECTIONS
        assert report.sweep_results is not None
        assert len(report.sweep_results) == 4

    def test_holdout_is_not_touched_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(research, "DEFAULT_RUNS_DIR", tmp_path)
        monkeypatch.setattr(research, "HOLDOUT_LOG", tmp_path / "unlocks.log")

        windows = []

        def stub(config, save=False, **kwargs):
            windows.append((config.start_date, config.end_date))
            return _result(_returns(n=600))

        research.run_research(
            _config(), run_backtest_fn=stub, holdout_years=2,
            n_simulations=50, save=False,
        )

        # Nothing may reach into the final two years.
        holdout_start = pd.Timestamp("2018-12-31")
        assert all(
            pd.Timestamp(start) < holdout_start for start, _ in windows
        ), "a locked holdout was read"
        assert not (tmp_path / "unlocks.log").exists()

    def test_unlocking_runs_the_holdout_and_logs_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(research, "DEFAULT_RUNS_DIR", tmp_path)
        log = tmp_path / "unlocks.log"
        monkeypatch.setattr(research, "HOLDOUT_LOG", log)

        windows = []

        def stub(config, save=False, **kwargs):
            windows.append((config.start_date, config.end_date))
            return _result(_returns(n=600))

        research.run_research(
            _config(), run_backtest_fn=stub, holdout_years=2,
            unlock_holdout=True, n_simulations=50, save=False,
        )

        assert log.exists()
        assert any(
            pd.Timestamp(start) >= pd.Timestamp("2018-12-31")
            for start, _ in windows
        )

    def test_report_saves_a_card_and_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(research, "DEFAULT_RUNS_DIR", tmp_path)

        def stub(config, save=False, **kwargs):
            return _result(_returns(n=600))

        report = research.run_research(
            _config(), run_backtest_fn=stub, holdout_years=2,
            n_simulations=50, save=True,
        )

        saved = list(tmp_path.glob("*_research_stub"))
        assert len(saved) == 1
        assert (saved[0] / "verdict.txt").read_text().startswith("Strategy verdict:")
        assert (saved[0] / "research.json").exists()
        assert report.card in (saved[0] / "verdict.txt").read_text()

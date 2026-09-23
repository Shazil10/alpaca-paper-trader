"""Tests for the HTML comparison report.

The report is presentation, so these tests check the two things that would make it
*misleading* rather than ugly:

* every stream is rebased to the same start, or a $1M book and a $10k book appear
  to have wildly different performance when they are identical;
* a correlation computed over a handful of overlapping sessions is flagged, because
  it is a number that looks authoritative and is not.

Plus the loader, because it reads files written by older versions of the code, and
the version-diff section, whose whole job is to say when a metric change cannot be
attributed to the strategy.

Run with: ./venv/bin/python -m pytest tests/backtest/test_report.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _equity(n=500, start="2024-01-02", rate=0.0005, capital=100_000.0, seed=None):
    index = pd.bdate_range(start, periods=n)
    if seed is None:
        path = np.cumprod(np.full(n, 1.0 + rate))
    else:
        rng = np.random.RandomState(seed)
        path = np.cumprod(1.0 + rng.normal(rate, 0.01, n))
    return pd.Series(path * capital, index=index)


def _write_run(directory: Path, equity: pd.Series, config=None, metrics=None):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"date": equity.index, "equity": equity.to_numpy()}
    ).to_csv(directory / "equity.csv", index=False)
    if config is not None:
        (directory / "config.json").write_text(json.dumps(config))
    if metrics is not None:
        (directory / "metrics.json").write_text(json.dumps(metrics))
    return directory


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoadRun:
    def test_reads_equity_config_and_metrics(self, tmp_path):
        equity = _equity(120)
        _write_run(
            tmp_path / "run_a", equity,
            config={"strategy_id": "alpha", "start_date": "2024-01-02"},
            metrics={"sharpe": 9.9},
        )

        data = report.load_run(tmp_path / "run_a")

        assert len(data.equity) == 120
        assert data.strategy_id == "alpha"
        assert data.stats["sharpe"] == 9.9
        assert len(data.returns) == 119

    def test_metrics_are_recomputed_not_trusted(self, tmp_path):
        """A stored metrics.json may predate the current definition.

        Sortino changed definition once already in this repo, so a report that
        mixed stored and recomputed numbers would compare two different statistics
        under one heading.
        """
        _write_run(tmp_path / "run_a", _equity(300), metrics={"Sharpe": 9.9})

        data = report.load_run(tmp_path / "run_a")
        row = report.stream_metrics(data.equity)

        assert row["Sharpe"] != 9.9

    def test_missing_optional_files_are_tolerated(self, tmp_path):
        _write_run(tmp_path / "bare", _equity(50))

        data = report.load_run(tmp_path / "bare")

        assert data.config == {}
        assert data.stats == {}
        assert data.strategy_id == "bare"

    def test_unparseable_json_does_not_abort_the_load(self, tmp_path):
        directory = _write_run(tmp_path / "broken", _equity(50))
        (directory / "config.json").write_text("{not json")

        data = report.load_run(directory)

        assert len(data.equity) == 50
        assert data.config == {}

    def test_window_is_reported_from_the_curve(self, tmp_path):
        equity = _equity(10, start="2024-03-04")
        _write_run(tmp_path / "w", equity)

        assert report.load_run(tmp_path / "w").window.startswith("2024-03-04")


class TestDiscoverRuns:
    def test_research_directories_are_skipped(self, tmp_path):
        _write_run(tmp_path / "2026-01-01_alpha_aaa", _equity(50))
        _write_run(tmp_path / "2026-01-02_research_alpha", _equity(50))

        found = report.discover_runs(tmp_path)

        assert [p.name for p in found] == ["2026-01-01_alpha_aaa"]

    def test_directories_without_equity_are_skipped(self, tmp_path):
        (tmp_path / "2026-01-01_empty").mkdir()
        _write_run(tmp_path / "2026-01-02_real", _equity(50))

        assert [p.name for p in report.discover_runs(tmp_path)] == ["2026-01-02_real"]

    def test_ordering_is_chronological_by_name(self, tmp_path):
        """Directory names lead with the date, which is the only ordering there is:
        a run does not record which run preceded it."""
        for name in ("2026-03-01_a_1", "2026-01-01_a_1", "2026-02-01_a_1"):
            _write_run(tmp_path / name, _equity(20))

        names = [p.name for p in report.discover_runs(tmp_path)]

        assert names == sorted(names)

    def test_limit_keeps_the_newest(self, tmp_path):
        for name in ("2026-01-01_a_1", "2026-02-01_a_1", "2026-03-01_a_1"):
            _write_run(tmp_path / name, _equity(20))

        found = report.discover_runs(tmp_path, limit=2)

        assert [p.name for p in found] == ["2026-02-01_a_1", "2026-03-01_a_1"]

    def test_filter_by_strategy_id(self, tmp_path):
        _write_run(tmp_path / "2026-01-01_a", _equity(20), config={"strategy_id": "A"})
        _write_run(tmp_path / "2026-01-02_b", _equity(20), config={"strategy_id": "B"})

        found = report.discover_runs(tmp_path, strategy_id="B")

        assert [p.name for p in found] == ["2026-01-02_b"]


# ---------------------------------------------------------------------------
# Rebasing
# ---------------------------------------------------------------------------

class TestRebasing:
    def test_different_capital_bases_become_identical(self):
        """A $1M book and a $10k book following the same path must overlay."""
        big = _equity(200, capital=1_000_000.0)
        small = _equity(200, capital=10_000.0)

        a = report._normalize(big)
        b = report._normalize(small)

        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12)
        assert a.iloc[0] == pytest.approx(1.0)

    def test_rebasing_starts_at_the_common_date(self):
        """Otherwise a longer run gets credit for growth the others never saw."""
        early = _equity(400, start="2024-01-02")
        late = _equity(200, start="2024-06-03")
        common = late.index[0]

        rebased = report._normalize(early, common)

        assert rebased.index[0] == common
        assert rebased.iloc[0] == pytest.approx(1.0)

    def test_a_zero_start_does_not_divide_by_zero(self):
        broken = pd.Series([0.0, 1.0, 2.0], index=pd.bdate_range("2024-01-02", periods=3))

        result = report._normalize(broken)

        assert list(result) == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

class TestStreamMetrics:
    def test_benchmark_relative_metrics_appear_only_with_a_benchmark(self):
        equity = _equity(400, seed=1)
        bench = _equity(400, seed=2).pct_change().dropna()

        without = report.stream_metrics(equity)
        with_bench = report.stream_metrics(equity, bench)

        assert "Beta" not in without
        assert {"Beta", "Alpha", "Info ratio"} <= set(with_bench)

    def test_a_stub_curve_returns_nothing(self):
        assert report.stream_metrics(_equity(2)) == {}


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_identical_streams_correlate_at_one(self):
        streams = {"a": _equity(300, seed=5), "b": _equity(300, seed=5)}

        html = report.correlation_section(streams)

        assert "1.000" in html

    def test_thin_overlap_is_flagged(self):
        """A correlation over a handful of shared sessions looks authoritative
        and is not. It is shown, marked, and footnoted."""
        long_run = _equity(400, start="2024-01-02", seed=1)
        short_run = _equity(20, start="2025-06-02", seed=2)

        html = report.correlation_section({"long": long_run, "short": short_run})

        assert "class='warn'" in html
        assert "overlapping sessions" in html

    def test_ample_overlap_is_not_flagged(self):
        streams = {"a": _equity(400, seed=1), "b": _equity(400, seed=2)}

        html = report.correlation_section(streams)

        assert "class='warn'" not in html

    def test_one_stream_cannot_be_correlated(self):
        assert "need two streams" in report.correlation_section({"a": _equity(100)})


# ---------------------------------------------------------------------------
# Version diff
# ---------------------------------------------------------------------------

class TestVersionsSection:
    def test_config_differences_are_surfaced(self, tmp_path):
        """The trap: attributing a metric change to the strategy when the cost
        assumption moved as well."""
        old = report.load_run(_write_run(
            tmp_path / "2026-01-01_a", _equity(300, seed=1),
            config={"strategy_id": "A", "cost": {"spread_bps": 5}},
        ))
        new = report.load_run(_write_run(
            tmp_path / "2026-02-01_a", _equity(300, seed=2),
            config={"strategy_id": "A", "cost": {"spread_bps": 30}},
        ))

        html = report.versions_section([old, new])

        assert "Assumptions that also changed" in html
        assert "cost.spread_bps" in html
        assert "cannot be attributed to the strategy alone" in html

    def test_matching_configs_say_the_difference_is_the_strategy(self, tmp_path):
        config = {"strategy_id": "A", "cost": {"spread_bps": 5}, "params": {"n": 1}}
        old = report.load_run(_write_run(
            tmp_path / "2026-01-01_a", _equity(300, seed=1), config=config
        ))
        new = report.load_run(_write_run(
            tmp_path / "2026-02-01_a", _equity(300, seed=2), config=config
        ))

        html = report.versions_section([old, new])

        assert "Assumptions that also changed" not in html
        assert "the difference is the strategy" in html

    def test_a_single_run_has_nothing_to_compare(self, tmp_path):
        only = report.load_run(_write_run(tmp_path / "a", _equity(100)))

        assert "only one run" in report.versions_section([only])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRender:
    def _inputs(self, tmp_path, **overrides):
        runs = [
            report.load_run(_write_run(
                tmp_path / "2026-01-01_a", _equity(400, seed=1),
                config={"strategy_id": "A"},
            )),
            report.load_run(_write_run(
                tmp_path / "2026-01-02_b", _equity(400, seed=2),
                config={"strategy_id": "B"},
            )),
        ]
        kwargs = dict(title="T", runs=runs, benchmark=_equity(400, seed=3))
        kwargs.update(overrides)
        return report.ReportInputs(**kwargs)

    def test_page_is_self_contained(self, tmp_path):
        """One file, openable anywhere: no script tags, no external requests."""
        html = report.build_html(self._inputs(tmp_path))

        assert html.startswith("<!DOCTYPE html>")
        assert "<script" not in html
        assert "http://" not in html and "https://" not in html
        assert "<svg" in html

    def test_core_sections_are_present(self, tmp_path):
        html = report.build_html(self._inputs(tmp_path))

        for heading in (
            "Headline", "Equity curves", "Drawdowns", "Against the baselines",
            "Correlation of daily returns", "Calendar years",
        ):
            assert heading in html

    def test_split_adds_the_is_oos_section(self, tmp_path):
        html = report.build_html(
            self._inputs(tmp_path, is_oos_split=pd.Timestamp("2025-01-01"))
        )

        assert "In-sample versus out-of-sample" in html
        assert "in-sample" in html and "out-of-sample" in html

    def test_no_runs_renders_a_message_not_a_crash(self):
        html = report.build_html(report.ReportInputs(title="T", runs=[]))

        assert "No runs" in html

    def test_names_are_escaped(self, tmp_path):
        run = report.load_run(_write_run(tmp_path / "2026-01-01_x", _equity(300)))
        run.name = "<script>alert(1)</script>"

        html = report.build_html(report.ReportInputs(title="T", runs=[run]))

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_write_report_creates_parent_directories(self, tmp_path):
        inputs = self._inputs(tmp_path)
        out = tmp_path / "nested" / "deeper" / "report.html"

        written = report.write_report(inputs, out)

        assert written.exists()
        assert written.read_text().startswith("<!DOCTYPE html>")


class TestResearchSections:
    def test_verdict_cost_and_sweep_are_read(self, tmp_path):
        directory = tmp_path / "research"
        directory.mkdir()
        (directory / "research.json").write_text(json.dumps({
            "verdict": "CAUTIOUS",
            "sections": {
                "Performance": {"status": "PASS", "headline": "Sharpe 1.0"},
                "Cost stress": {
                    "status": "FAIL",
                    "headline": "dies at 50bps",
                    "detail": {"scenarios": {
                        "base": {"sharpe": 1.0},
                        "50bps": {"sharpe": -0.1},
                    }},
                },
            },
        }))
        pd.DataFrame({"n": [1, 2, 3], "sharpe": [0.2, 1.4, 0.3]}).to_csv(
            directory / "sweep.csv", index=False
        )

        sections = report.research_sections(directory)
        titles = [t for t, _, _ in sections]

        assert any("CAUTIOUS" in t for t in titles)
        assert "Cost scenarios" in titles
        assert "Parameter sweep" in titles
        # The best cell is highlighted, not just listed.
        sweep_html = next(b for t, b, _ in sections if t == "Parameter sweep")
        assert "class='mark'" in sweep_html

    def test_absent_research_directory_yields_nothing(self, tmp_path):
        assert report.research_sections(tmp_path / "nope") == []

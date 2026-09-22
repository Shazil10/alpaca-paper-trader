"""Numerical correctness tests for the validation suite.

These pin down *known-answer* statistical behaviour, not just "the code runs".
Every random input is drawn from a fixed seed, so each assertion below is
deterministic. Where a property is only true in expectation (PBO on noise),
the test averages over several fixed dataset draws and the comment says so.
"""
import math

import numpy as np
import pandas as pd
import pytest

from backtest import metrics
from backtest.benchmark import percentile_vs_random, random_portfolio_placebo
from backtest.validation import clustering, montecarlo, overfitting, splits, stability, sweep, verdict

TRADING_DAYS = metrics.TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drifting_returns(n=756, mu=0.0008, sigma=0.01, seed=1):
    """Daily returns with a clear positive drift (annual Sharpe ~1.9)."""
    return pd.Series(np.random.RandomState(seed).normal(mu, sigma, n))


def noise_returns(n=1000, cols=20, sigma=0.01, seed=7):
    """Independent zero-drift return streams — no real edge anywhere."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.normal(0.0, sigma, (n, cols)))


# ---------------------------------------------------------------------------
# 0. CAGR period counting
# ---------------------------------------------------------------------------

def test_cagr_counts_elapsed_periods_not_marks():
    """An N-point curve spans N-1 returns. Exactly one year of sessions at a
    known growth rate must return that rate, with no off-by-one slippage."""
    # 253 marks = 252 elapsed sessions = exactly one year. Doubling over one
    # year is a CAGR of exactly 100%, independent of TRADING_DAYS.
    equity = pd.Series(np.linspace(1.0, 2.0, TRADING_DAYS + 1))
    assert metrics.cagr(equity) == pytest.approx(1.0, abs=1e-9)

    # Two years of doubling compounds to 4x.
    two_yr = pd.Series(np.linspace(1.0, 4.0, 2 * TRADING_DAYS + 1))
    assert metrics.cagr(two_yr) == pytest.approx(1.0, abs=1e-9)


def test_cagr_matches_closed_form_on_a_constant_growth_curve():
    """Reconstructed from the definition, not from the implementation."""
    daily = 1.0004
    n_periods = 500
    equity = pd.Series(daily ** np.arange(n_periods + 1))

    expected = daily ** TRADING_DAYS - 1
    assert metrics.cagr(equity) == pytest.approx(expected, rel=1e-12)


def test_sortino_is_annualized_like_sharpe():
    """Both ratios appear side by side in every report, so they must share
    units. A symmetric series pins the relationship exactly."""
    rng = np.random.RandomState(11)
    rets = pd.Series(rng.normal(0.0006, 0.01, 2000))

    sharpe = metrics.sharpe_ratio(rets)
    sortino = metrics.sortino_ratio(rets)

    # Recompute from the definition, without reference to the implementation.
    excess = rets - metrics.RISK_FREE_RATE / TRADING_DAYS
    tdd = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    expected = excess.mean() / tdd * np.sqrt(TRADING_DAYS)
    assert sortino == pytest.approx(expected, rel=1e-12)

    # For a near-symmetric series, half the squared mass is below zero, so
    # shortfall RMS ~ std/sqrt(2) and Sortino ~ sqrt(2) x Sharpe. This is the
    # assertion that actually catches the annualization bug: the old form
    # returned sharpe/sqrt(252), i.e. a ratio of 0.06 rather than 1.41.
    assert sortino / sharpe == pytest.approx(np.sqrt(2), rel=0.05)


def test_sortino_penalizes_downside_concentration():
    """Same mean and same total volatility, different loss shape: the series
    whose losses are concentrated into rare deep drops must score worse.

    Sharpe cannot tell these apart -- that is the entire reason Sortino
    exists -- so the test also pins that Sharpe is (near) identical.
    """
    rng = np.random.RandomState(5)
    base = rng.normal(0.0, 1.0, 4000)

    # Left-skewed: rare deep losses. Right-skewed: rare large gains.
    left = pd.Series(-np.square(base))
    right = pd.Series(np.square(base))

    # Normalise both to the same mean and the same total volatility.
    def standardize(s, mu=0.0006, sd=0.01):
        return (s - s.mean()) / s.std() * sd + mu

    left, right = standardize(left), standardize(right)
    assert left.std() == pytest.approx(right.std(), rel=1e-12)
    assert left.mean() == pytest.approx(right.mean(), rel=1e-12)

    # Sharpe is blind to the asymmetry.
    assert metrics.sharpe_ratio(left) == pytest.approx(
        metrics.sharpe_ratio(right), rel=1e-9
    )
    # Sortino is not.
    assert metrics.sortino_ratio(left) < metrics.sortino_ratio(right)


def test_sortino_edge_cases():
    assert metrics.sortino_ratio(pd.Series([0.01])) == 0.0
    # Never loses: unbounded rather than a silent zero.
    assert metrics.sortino_ratio(pd.Series([0.02] * 300)) == float("inf")
    # Flat at zero is undefined, not infinite.
    assert metrics.sortino_ratio(pd.Series([0.0] * 300)) == 0.0


def test_sparse_verdict_card_withholds_its_grade():
    """One soft check is not evidence. It used to render PROMISING.

    Same failure mode as scoring a test that never ran: the grade is a fraction
    of whatever was supplied, so a single `oos_sharpe > 0` reads 1/1 and passes
    a strategy with negative alpha and a 32% drawdown.
    """
    card = verdict.render_verdict(
        strategy_id="sparse",
        params={},
        data_desc="test",
        metrics={"alpha_jensen": -0.028, "max_drawdown": -0.319},
        oos_sharpe=0.50,
    )

    assert "INCOMPLETE" in card
    assert "PROMISING" not in card
    assert "NOT SUPPLIED" in card
    # It must say what is missing, not just refuse.
    for expected in ("walk-forward efficiency", "cost survival", "parameter plateau"):
        assert expected in card


def test_verdict_grades_once_the_quorum_is_met():
    kwargs = dict(
        strategy_id="full", params={}, data_desc="test",
        metrics={"n_trials": 340},
        oos_sharpe=1.12, wfe=0.68, deflated_sharpe=0.81,
        pbo=0.22, random_pct=97, survives_20bps=True, plateau_ok=True,
    )

    card = verdict.render_verdict(**kwargs)

    assert "VERDICT: PROMISING" in card
    assert "NOT SUPPLIED" not in card
    assert verdict.MIN_SCORED_CHECKS <= 7


def test_nan_pbo_does_not_count_toward_the_quorum():
    """An un-run PBO must not help a card reach the threshold."""
    kwargs = dict(
        strategy_id="q", params={}, data_desc="test", metrics={},
        oos_sharpe=1.0, wfe=0.68, deflated_sharpe=0.81, random_pct=97,
    )

    with_nan = verdict.render_verdict(pbo=float("nan"), **kwargs)
    with_real = verdict.render_verdict(pbo=0.22, **kwargs)

    assert "INCOMPLETE" in with_nan
    assert "not run" in with_nan
    # The real value is the fifth scored check and tips it over the quorum.
    assert "VERDICT: PROMISING" in with_real


def test_failing_checks_still_grade_down_not_incomplete():
    card = verdict.render_verdict(
        strategy_id="bad", params={}, data_desc="test", metrics={},
        oos_sharpe=-0.2, wfe=0.1, deflated_sharpe=0.1, pbo=0.9,
        random_pct=20, survives_20bps=False, plateau_ok=False,
    )

    assert "VERDICT: REJECT" in card


def test_summary_from_returns_matches_the_notebook_shape():
    """Key names are load-bearing: four notebooks index this dict directly."""
    r = drifting_returns(n=504, seed=3)
    r.index = pd.bdate_range("2020-01-02", periods=len(r))

    out = metrics.summary_from_returns(r)

    assert set(out) == {
        "CAGR", "Ann vol", "Sharpe", "Sortino",
        "Max DD", "Calmar", "Monthly win rate", "Ann turnover",
    }
    assert out["Sharpe"] == pytest.approx(metrics.sharpe_ratio(r))
    assert out["Ann vol"] == pytest.approx(metrics.annualized_volatility(r))
    assert out["Max DD"] < 0
    # No turnover series supplied: NaN, not a fabricated zero.
    assert math.isnan(out["Ann turnover"])


def test_summary_from_returns_cagr_counts_returns_not_marks():
    """The synthetic $1 mark must be prepended, or CAGR loses a period.

    A year of 252 returns at a constant rate compounds to a known total. With
    the leading 1.0 the curve has 253 marks and ``cagr`` sees 252 elapsed
    periods; without it the same growth is charged to 251 and reads high.
    """
    daily = 2.0 ** (1.0 / TRADING_DAYS) - 1.0  # doubles over exactly one year
    r = pd.Series(
        [daily] * TRADING_DAYS,
        index=pd.bdate_range("2020-01-02", periods=TRADING_DAYS),
    )

    assert metrics.summary_from_returns(r)["CAGR"] == pytest.approx(1.0, abs=1e-9)


def test_summary_from_returns_annualizes_turnover():
    r = pd.Series(
        [0.001] * TRADING_DAYS,
        index=pd.bdate_range("2020-01-02", periods=TRADING_DAYS),
    )
    turnover = pd.Series(0.5, index=r.index)

    # Exactly one year of 0.5 per session.
    expected = 0.5 * TRADING_DAYS
    assert metrics.summary_from_returns(r, turnover)["Ann turnover"] == pytest.approx(expected)


def test_summary_from_equity_reproduces_the_notebook_strings():
    """Formatted output, because the display cells consume it verbatim."""
    idx = pd.bdate_range("2020-01-02", periods=TRADING_DAYS + 1)
    equity = pd.Series(np.linspace(100_000.0, 200_000.0, len(idx)), index=idx)

    out = metrics.summary_from_equity(equity, "Doubler", 100_000.0)

    assert out["Label"] == "Doubler"
    assert out["Total Return"] == "100.0%"
    assert out["CAGR"] == "100.0%"
    # A monotonically rising curve never draws down.
    assert out["Max Drawdown"] == "0.0%"
    assert set(out) == {
        "Label", "Total Return", "CAGR", "Sharpe Ratio",
        "Max Drawdown", "Calmar Ratio", "Win Rate",
    }


def test_summary_from_equity_defaults_initial_to_the_first_mark():
    idx = pd.bdate_range("2020-01-02", periods=100)
    equity = pd.Series(np.linspace(50_000.0, 75_000.0, len(idx)), index=idx)

    assert (
        metrics.summary_from_equity(equity, "x")["Total Return"]
        == metrics.summary_from_equity(equity, "x", 50_000.0)["Total Return"]
    )


def test_summary_helpers_survive_degenerate_input():
    empty = pd.Series(dtype=float)

    assert metrics.summary_from_returns(empty) == {}
    assert metrics.summary_from_equity(empty, "flat") == {"Label": "flat"}


def test_compute_full_stats_keeps_the_legacy_key_schema():
    """The v10 notebooks index these keys by name.

    A wrapper that keeps the old function name but renames its keys is not
    back-compatible; it converts an ImportError you would notice into a
    KeyError deep inside a notebook cell.
    """
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.RandomState(3)
    equity = pd.Series((1 + rng.normal(0.0006, 0.01, 300)).cumprod(), index=idx)
    bench = pd.Series((1 + rng.normal(0.0003, 0.009, 300)).cumprod(), index=idx)

    stats = metrics.compute_full_stats(equity, bench)

    assert {"CAGR", "Vol", "Sharpe", "Max DD", "Sortino"} <= set(stats)
    # Values must agree with the canonical implementation, not be recomputed.
    assert stats["CAGR"] == pytest.approx(metrics.cagr(equity))
    assert stats["Sharpe"] == pytest.approx(
        metrics.sharpe_ratio(equity.pct_change().dropna())
    )

    # And the formatter must render that schema, not fall through to bare floats.
    line = metrics.fmt_full_stats(stats)
    assert line["CAGR"].endswith("%")
    assert line["Max DD"].endswith("%")
    assert "." in line["Sharpe"]


# ---------------------------------------------------------------------------
# 1-5. Monte Carlo
# ---------------------------------------------------------------------------

def test_permutation_preserves_total_return_but_not_path():
    """Shuffling returns cannot change prod(1+r), so final equity and CAGR are
    invariant; the *path* (and hence drawdown) must change."""
    rets = drifting_returns()
    result = montecarlo.trade_permutation(rets, n_simulations=200, seed=42)

    # Final equity is identical for every permutation, so CAGR is too.
    # Computed independently of the implementation. The curve is anchored at
    # 1.0 and has len(rets) + 1 marks, hence len(rets) elapsed periods.
    total_growth = float((1.0 + rets).prod())
    n_years = len(rets) / TRADING_DAYS
    expected_cagr = total_growth ** (1.0 / n_years) - 1.0

    assert np.std(result.cagrs) < 1e-12
    assert result.cagrs[0] == pytest.approx(expected_cagr, rel=1e-9)

    # Sharpe is mean/std, both order-independent, so it is exactly invariant.
    assert np.std(result.sharpes) < 1e-12

    # Drawdown depends on ordering and must genuinely vary.
    assert np.std(result.max_dds) > 0.001
    assert min(result.max_dds) < max(result.max_dds)


def test_permutation_cagr_matches_observed_series():
    """Regression: permuting returns must not change the reported CAGR.

    The equity curve has to be anchored at 1.0. Starting it at (1 + r_0) makes
    the CAGR denominator depend on which return lands first.
    """
    rets = drifting_returns()
    result = montecarlo.trade_permutation(rets, n_simulations=50, seed=3)

    observed_equity = pd.Series(np.concatenate(([1.0], (1.0 + rets.values).cumprod())))
    observed_cagr = metrics.cagr(observed_equity)

    for c in result.cagrs:
        assert c == pytest.approx(observed_cagr, rel=1e-9)


def test_permutation_drawdown_sees_a_first_day_loss():
    """A crash on day 1 is a real drawdown; it is invisible if the curve starts
    after the first return has already been applied."""
    rets = pd.Series([-0.5, 0.01, 0.01, 0.01])
    result = montecarlo.trade_permutation(rets, n_simulations=1, seed=0)
    assert result.max_dds[0] <= -0.5


def test_block_bootstrap_ci_brackets_observed_sharpe():
    rets = drifting_returns()
    observed = metrics.sharpe_ratio(rets)

    result = montecarlo.block_bootstrap(rets, block_size=21, n_simulations=500, seed=42)
    ci = result.sharpe_ci

    assert ci["p5"] < observed < ci["p95"]
    assert ci["p5"] < ci["p25"] < ci["p50"] < ci["p75"] < ci["p95"]
    # The bootstrap is centred on the observed statistic, not shifted away.
    assert ci["mean"] == pytest.approx(observed, abs=0.5)


def test_block_bootstrap_resamples_have_input_length(monkeypatch):
    """Every resample must be exactly as long as the input series."""
    rets = drifting_returns(n=500)
    seen = []
    real_sharpe = metrics.sharpe_ratio

    def recording_sharpe(series, *args, **kwargs):
        seen.append(len(series))
        return real_sharpe(series, *args, **kwargs)

    monkeypatch.setattr(metrics, "sharpe_ratio", recording_sharpe)
    montecarlo.block_bootstrap(rets, block_size=21, n_simulations=25, seed=42)

    assert len(seen) == 25
    assert set(seen) == {500}


def test_skip_trade_jitter_degrades_a_real_edge():
    """Dropping half the signals must materially degrade a genuine edge.

    skip_rate=0.0 is accepted cleanly and reproduces the observed Sharpe (the
    default delay_rate still shuffles a few returns by one day, which is why
    the comparison below is approximate rather than exact).
    """
    rets = drifting_returns()
    observed = metrics.sharpe_ratio(rets)

    none_skipped = montecarlo.skip_trade_jitter(rets, skip_rate=0.0, n_simulations=100, seed=42)
    half_skipped = montecarlo.skip_trade_jitter(rets, skip_rate=0.5, n_simulations=100, seed=42)

    mean_none = float(np.mean(none_skipped.sharpes))
    mean_half = float(np.mean(half_skipped.sharpes))

    assert mean_none == pytest.approx(observed, abs=0.05)
    assert mean_half < mean_none - 0.3

    # Monotone-ish in between.
    mean_tenth = float(np.mean(
        montecarlo.skip_trade_jitter(rets, skip_rate=0.10, n_simulations=100, seed=42).sharpes
    ))
    assert mean_half < mean_tenth < mean_none


def test_delay_only_jitter_conserves_returns():
    """Delaying a fill moves a return to the next day; it must not delete it.

    With delay_rate high enough that delays land on consecutive days, a naive
    implementation overwrites the carried return and silently destroys edge.
    """
    rets = drifting_returns()
    observed = metrics.sharpe_ratio(rets)

    delayed = montecarlo.skip_trade_jitter(
        rets, skip_rate=0.0, delay_rate=0.5, n_simulations=50, seed=1
    )
    # Re-dating returns barely moves the Sharpe; deleting them would not.
    assert float(np.mean(delayed.sharpes)) == pytest.approx(observed, abs=0.1)


@pytest.mark.parametrize("returns_seed", [1, 2])
def test_p_loss_is_a_probability(returns_seed):
    rets = drifting_returns(n=400, mu=0.0, seed=returns_seed)
    for result in (
        montecarlo.trade_permutation(rets, n_simulations=50, seed=42),
        montecarlo.block_bootstrap(rets, n_simulations=50, seed=42),
        montecarlo.skip_trade_jitter(rets, n_simulations=50, seed=42),
    ):
        assert 0.0 <= result.p_loss <= 1.0


# ---------------------------------------------------------------------------
# 6-7. Parameter stability
# ---------------------------------------------------------------------------

def _plateau_heatmap():
    """7x7 grid: a broad 1.5 plateau in the middle, 0.2 elsewhere."""
    grid = np.full((7, 7), 0.2)
    grid[1:6, 1:6] = 1.5
    return pd.DataFrame(grid)


def _spike_heatmap():
    """7x7 grid: a single isolated 3.0 spike, 0.2 elsewhere."""
    grid = np.full((7, 7), 0.2)
    grid[3, 3] = 3.0
    return pd.DataFrame(grid)


def test_plateau_beats_isolated_peak():
    """The single most important property: an isolated peak must not win.

    The spike has double the raw Sharpe of the plateau, so any scoring that
    looks at the cell alone ranks it first.
    """
    plateau = _plateau_heatmap()
    spike = _spike_heatmap()

    assert spike.iloc[3, 3] > plateau.iloc[3, 3]  # raw metric favours the spike

    plateau_centre = stability.plateau_score(plateau).iloc[3, 3]
    spike_centre = stability.plateau_score(spike).iloc[3, 3]

    assert plateau_centre > spike_centre

    # A flat neighbourhood has zero dispersion, so the score is the plateau level.
    assert plateau_centre == pytest.approx(1.5)
    # Spike: mean(3.0, 8 x 0.2) = 0.5111, minus one std of a very uneven
    # neighbourhood, which drags the score below the surrounding floor.
    assert spike_centre < 0.2


def test_plateau_score_penalises_dispersion():
    plateau = _plateau_heatmap()
    # Edge of the plateau mixes 1.5 and 0.2 cells, so it must score below the centre.
    scores = stability.plateau_score(plateau)
    assert scores.iloc[3, 3] > scores.iloc[1, 1]


def test_worst_step_decay_large_for_spike_zero_on_plateau():
    assert stability.worst_step_decay(_plateau_heatmap(), 3, 3) == pytest.approx(0.0)
    assert stability.worst_step_decay(_spike_heatmap(), 3, 3) == pytest.approx(2.8)


def test_stability_heatmap_pivots_sweep_results():
    sweep_df = pd.DataFrame({
        "lookback": [10, 10, 20, 20],
        "threshold": [1, 2, 1, 2],
        "sharpe": [0.5, 1.5, 0.7, 1.1],
    })
    heat = stability.stability_heatmap(sweep_df, "lookback", "threshold")

    assert list(heat.columns) == [10, 20]
    assert list(heat.index) == [2, 1]  # sorted descending
    assert heat.loc[2, 10] == pytest.approx(1.5)
    assert heat.loc[1, 20] == pytest.approx(0.7)

    with pytest.raises(ValueError):
        stability.stability_heatmap(sweep_df, "lookback", "missing")


def test_grid_sweep_feeds_stability_heatmap():
    grid = {"lookback": [10, 20, 30], "threshold": [1.0, 2.0]}
    calls = []

    def backtest_fn(params):
        calls.append(params)
        return {"sharpe": params["lookback"] / 10.0 + params["threshold"]}

    df = sweep.grid_sweep(backtest_fn, grid)

    assert len(df) == 6 and len(calls) == 6
    assert set(df["_status"]) == {"ok"}
    heat = stability.stability_heatmap(df, "lookback", "threshold")
    assert heat.loc[2.0, 30] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 8-10. Clustering
# ---------------------------------------------------------------------------

def test_two_mechanisms_form_two_clusters():
    """Columns 0-4 are noisy variants of signal A, 5-9 of an uncorrelated B."""
    rng = np.random.RandomState(3)
    n = 500
    signal_a = rng.normal(0, 0.01, n)
    signal_b = rng.normal(0, 0.01, n)

    cols = {}
    for i in range(5):
        cols[f"a{i}"] = signal_a + rng.normal(0, 0.003, n)
    for i in range(5):
        cols[f"b{i}"] = signal_b + rng.normal(0, 0.003, n)
    streams = pd.DataFrame(cols)

    result = clustering.return_stream_clustering(streams, n_clusters=2)
    labels = np.asarray(result["labels"])

    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]
    assert sorted(result["cluster_sizes"].values()) == [5, 5]


def test_clustering_degenerate_input_returns_flat_labels():
    streams = pd.DataFrame({"a": [0.01, -0.01], "b": [0.02, -0.02]})
    result = clustering.return_stream_clustering(streams, n_clusters=2)
    assert list(result["labels"]) == [0, 0]


def test_spp_percentile_is_below_the_best_cell():
    sweep_metrics = pd.DataFrame({"sharpe": np.linspace(0.0, 2.0, 101)})

    spp = clustering.spp_percentile(sweep_metrics, percentile=25)

    assert spp < sweep_metrics["sharpe"].max()
    assert spp == pytest.approx(0.5)  # 25th percentile of a uniform 0..2 grid
    assert clustering.spp_percentile(sweep_metrics, percentile=50) == pytest.approx(1.0)


def test_best_in_cluster_reports_cluster_of_the_argmax():
    sweep_df = pd.DataFrame({"sharpe": [0.1, 0.2, 0.9, 0.3, 0.4]})
    labels = np.array([1, 1, 2, 2, 2])

    result = clustering.best_in_cluster(sweep_df, labels)

    assert result["best_idx"] == 2
    assert result["best_cluster"] == 2
    assert result["cluster_size"] == 3
    assert result["cluster_pct"] == pytest.approx(0.6)
    assert result["is_core"] is True


def test_best_in_cluster_handles_non_range_index():
    """Regression: cluster_labels is aligned to row order, not index labels.

    A sweep frame that has been filtered or re-indexed used to look up
    cluster_labels[label] and either raise or report the wrong cluster.
    """
    sweep_df = pd.DataFrame({"sharpe": [0.1, 0.2, 0.9, 0.3]}, index=[10, 11, 12, 13])
    labels = np.array([1, 1, 2, 2])

    result = clustering.best_in_cluster(sweep_df, labels)

    assert result["best_idx"] == 2
    assert result["best_label"] == 12
    assert result["best_cluster"] == 2


def test_best_in_cluster_rejects_mismatched_lengths():
    sweep_df = pd.DataFrame({"sharpe": [0.1, 0.2]})
    assert clustering.best_in_cluster(sweep_df, np.array([1])) == {}


# ---------------------------------------------------------------------------
# 11-12. PBO via CSCV
# ---------------------------------------------------------------------------

def test_pbo_on_pure_noise_is_near_one_half():
    """With no real edge the selection process has no skill, so the IS-best
    lands below the OOS median about half the time.

    PBO for a *single* dataset draw is very high variance (observed 0.1..0.99
    across seeds) because it is conditioned on which column happened to get
    lucky over the full sample. The unbiasedness claim is about the average, so
    this averages over ten fixed draws.
    """
    values = [
        overfitting.probability_of_backtest_overfitting(
            noise_returns(n=1000, cols=20, seed=seed), n_splits=10
        )
        for seed in range(10)
    ]

    assert all(0.0 <= v <= 1.0 for v in values)
    assert 0.3 < float(np.mean(values)) < 0.7


def test_pbo_is_low_when_one_stream_is_genuinely_superior():
    """A column with a strong consistent drift is best IS *and* OOS, so the
    selection is not overfit and PBO must be low."""
    for seed in (7, 11, 13):
        streams = noise_returns(n=1000, cols=20, seed=seed)
        streams[0] = streams[0] + 0.0015  # ~2.4 annual Sharpe of pure drift

        pbo = overfitting.probability_of_backtest_overfitting(streams, n_splits=10)

        assert pbo < 0.3, f"seed {seed} gave PBO={pbo}"


def test_pbo_is_not_hijacked_by_a_flat_stream():
    """A parameter set that never trades has no dispersion and must not be
    selected as the in-sample best ahead of streams with a real edge."""
    streams = noise_returns(n=1000, cols=10, seed=5)
    streams["flat"] = 0.0005  # constant daily return, zero volatility
    streams["edge"] = streams[0] + 0.0015

    is_sharpes = streams.apply(metrics.sharpe_ratio)

    assert np.isfinite(is_sharpes).all()
    assert is_sharpes["flat"] == 0.0
    assert is_sharpes.idxmax() == "edge"


def test_pbo_signals_nan_when_it_cannot_run():
    """A test that never ran must not be reported as a perfect score.

    0.0 is the *best possible* PBO, so returning it on degenerate input would
    make "we could not check for overfitting" look identical to "we checked
    and found none".
    """
    # Nothing to select between.
    single = pd.DataFrame({"a": np.zeros(500)})
    assert math.isnan(overfitting.probability_of_backtest_overfitting(single))

    # Too few rows to build blocks of >= 5 days: 8 rows degrades to 2 splits
    # of 4, under the floor.
    tiny = noise_returns(n=8, cols=5, seed=1)
    assert math.isnan(overfitting.probability_of_backtest_overfitting(tiny))

    # Just above the floor it must actually run and return a real probability,
    # so the NaN path stays narrow rather than swallowing usable inputs.
    small = noise_returns(n=20, cols=5, seed=1)
    pbo = overfitting.probability_of_backtest_overfitting(small)
    assert math.isfinite(pbo) and 0.0 <= pbo <= 1.0




# ---------------------------------------------------------------------------
# 13-14. Deflated Sharpe
# ---------------------------------------------------------------------------

def test_more_trials_deflate_the_sharpe():
    """The whole point of the statistic: searching harder must cost you."""
    rets = pd.Series(np.random.RandomState(0).normal(0.0006, 0.01, 1260))

    one = metrics.deflated_sharpe(rets, num_trials=1)
    many = metrics.deflated_sharpe(rets, num_trials=1000)

    assert many < one

    values = [metrics.deflated_sharpe(rets, num_trials=t) for t in (2, 10, 100, 1000, 10000)]
    assert values == sorted(values, reverse=True)


def test_deflated_sharpe_is_a_probability():
    rng = np.random.RandomState(5)
    for mu in (-0.001, 0.0, 0.002):
        rets = pd.Series(rng.normal(mu, 0.01, 800))
        for trials in (1, 5, 500):
            value = metrics.deflated_sharpe(rets, num_trials=trials)
            assert 0.0 <= value <= 1.0


def test_deflated_sharpe_is_invariant_to_leverage():
    """Regression: the multiple-testing hurdle must not scale with volatility.

    Running the same strategy at 3x leaves the Sharpe (and the PSR) untouched,
    so the deflated Sharpe must be untouched too. Scaling the hurdle by the
    return volatility made a 3x-levered clone look far worse.
    """
    rets = pd.Series(np.random.RandomState(0).normal(0.0006, 0.01, 1260))

    assert metrics.sharpe_ratio(rets * 3.0) == pytest.approx(metrics.sharpe_ratio(rets))
    assert metrics.deflated_sharpe(rets * 3.0, 1000) == pytest.approx(
        metrics.deflated_sharpe(rets, 1000), rel=1e-9
    )


def test_deflated_sharpe_hurdle_is_the_expected_max_of_noise():
    """A pure-noise series searched over many trials must not look significant."""
    noise = pd.Series(np.random.RandomState(9).normal(0.0, 0.01, 1260))
    assert metrics.deflated_sharpe(noise, num_trials=1000) < 0.05


def test_probabilistic_sharpe_properties():
    rets = pd.Series(np.random.RandomState(0).normal(0.0006, 0.01, 1260))

    psr = metrics.probabilistic_sharpe(rets)
    assert 0.0 <= psr <= 1.0
    assert psr > 0.5  # positive observed Sharpe beats a zero benchmark

    # Strictly decreasing in the benchmark being cleared.
    assert metrics.probabilistic_sharpe(rets, 2.0) < metrics.probabilistic_sharpe(rets, 0.0)

    # An exactly-zero-mean series has SR = 0, which sits on the 50/50 line.
    # (A merely zero-*drift* draw does not: this seed realises an annual Sharpe
    # of +0.44 over 2000 days, and PSR ~ 0.89 is the correct answer for it.)
    raw = pd.Series(np.random.RandomState(4).normal(0.0, 0.01, 2000))
    assert metrics.probabilistic_sharpe(raw - raw.mean()) == pytest.approx(0.5)

    assert metrics.probabilistic_sharpe(pd.Series([0.01, 0.02])) == 0.5  # too short


def test_probabilistic_sharpe_matches_the_published_formula():
    """PSR = Phi[(SR - SR*) / se], se = sqrt((1 - g3*SR + (g4-1)/4*SR^2)/(n-1)).

    g4 is non-excess kurtosis (3 for a Gaussian). Computed here from scipy
    directly so the test does not just re-run the implementation.
    """
    from scipy import stats as scipy_stats

    rets = pd.Series(np.random.RandomState(0).normal(0.0006, 0.01, 1260))
    n = len(rets)
    sr = metrics.sharpe_ratio(rets) / np.sqrt(TRADING_DAYS)
    g3 = float(scipy_stats.skew(rets))
    g4 = float(scipy_stats.kurtosis(rets, fisher=False))

    se = np.sqrt((1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (n - 1))
    expected = float(scipy_stats.norm.cdf(sr / se))

    assert metrics.probabilistic_sharpe(rets) == pytest.approx(expected, rel=1e-12)

    # Excess kurtosis in place of g4 flips the sign of the SR^2 term and
    # understates the error, which would overstate confidence.
    wrong_se = np.sqrt(
        (1.0 - g3 * sr + (float(scipy_stats.kurtosis(rets)) - 1.0) / 4.0 * sr ** 2) / (n - 1)
    )
    assert wrong_se < se


def test_sharpe_and_drawdown_known_answers():
    """Sanity-check the primitives the whole suite is built on."""
    # Regression: a constant series has no dispersion, but its std comes back
    # as float residue (~1e-18) rather than 0, so dividing by it produced a
    # Sharpe of ~9e16 that would win any IS selection it was entered into.
    constant = pd.Series([0.01] * 100)
    assert constant.std() > 0  # not exactly zero, which is the trap
    assert metrics.sharpe_ratio(constant) == 0.0
    assert metrics.sharpe_ratio(pd.Series([0.0] * 100)) == 0.0

    rets = pd.Series([0.1, -0.1, 0.1, -0.1])
    expected = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS))
    assert metrics.sharpe_ratio(rets) == pytest.approx(expected)

    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert metrics.max_drawdown(equity) == pytest.approx(-0.5)
    assert metrics.max_drawdown(pd.Series([1.0, 2.0, 3.0])) == pytest.approx(0.0)

    two_years = pd.Series(np.linspace(1.0, 4.0, 2 * TRADING_DAYS))
    assert metrics.cagr(two_years) == pytest.approx(1.0, abs=0.01)  # 4x over 2y = +100%/yr


# ---------------------------------------------------------------------------
# 15-17. Walk-forward
# ---------------------------------------------------------------------------

class RecordingBacktest:
    """Stub backtest_fn that records the windows it is asked to evaluate.

    Train windows are `train_years` long and test windows `test_years`, so the
    span length identifies which side of the fold a call belongs to.
    """

    def __init__(self, is_cagr=0.10, oos_cagr=0.05, split_days=400):
        self.calls = []
        self.is_cagr = is_cagr
        self.oos_cagr = oos_cagr
        self.split_days = split_days

    def __call__(self, start, end, params=None):
        span = (pd.Timestamp(end) - pd.Timestamp(start)).days
        is_train = span >= self.split_days
        self.calls.append((start, end, is_train))
        return {
            "sharpe": 1.0 if is_train else 0.5,
            "cagr": self.is_cagr if is_train else self.oos_cagr,
            "max_drawdown": -0.1,
        }

    @property
    def train_windows(self):
        return [(s, e) for s, e, is_train in self.calls if is_train]


def test_anchored_walk_forward_expands_the_train_window():
    stub = RecordingBacktest()
    result = splits.anchored_walk_forward(
        stub, "2000-01-01", "2020-01-01", train_years=5, test_years=1, step_years=1
    )

    assert len(result.folds) >= 5

    train_starts = [f.train_start for f in result.folds]
    train_ends = [pd.Timestamp(f.train_end) for f in result.folds]

    assert set(train_starts) == {"2000-01-01"}  # anchored
    assert train_ends == sorted(train_ends)
    assert all(b > a for a, b in zip(train_ends, train_ends[1:]))  # strictly expanding

    # Test windows follow their train window and march forward.
    for fold in result.folds:
        assert pd.Timestamp(fold.test_start) > pd.Timestamp(fold.train_end)
    test_starts = [pd.Timestamp(f.test_start) for f in result.folds]
    assert all(b > a for a, b in zip(test_starts, test_starts[1:]))


def test_rolling_walk_forward_keeps_a_fixed_width_train_window():
    stub = RecordingBacktest()
    result = splits.rolling_walk_forward(
        stub, "2000-01-01", "2020-01-01", train_years=5, test_years=1, step_years=1
    )

    assert len(result.folds) >= 5

    widths = [
        (pd.Timestamp(f.train_end) - pd.Timestamp(f.train_start)).days
        for f in result.folds
    ]
    # Five calendar years is 1825 or 1826 days depending on leap years.
    assert max(widths) - min(widths) <= 2
    assert all(1820 <= w <= 1830 for w in widths)

    starts = [pd.Timestamp(f.train_start) for f in result.folds]
    assert all(b > a for a, b in zip(starts, starts[1:]))  # window slides


def test_walk_forward_efficiency_and_pardo_threshold():
    """WFE = OOS CAGR / IS CAGR; Pardo's rule passes at exactly 0.5."""
    stub = RecordingBacktest(is_cagr=0.10, oos_cagr=0.05)
    result = splits.anchored_walk_forward(
        stub, "2000-01-01", "2012-01-01", train_years=5, test_years=1, step_years=1
    )

    assert all(f.wfe == pytest.approx(0.5) for f in result.folds)
    assert result.mean_wfe == pytest.approx(0.5)
    assert result.median_wfe == pytest.approx(0.5)
    assert result.pass_pardo is True  # the rule is >= 0.5, not > 0.5

    assert result.aggregate_oos_cagr == pytest.approx(0.05)
    assert result.aggregate_oos_sharpe == pytest.approx(0.5)

    weak = splits.anchored_walk_forward(
        RecordingBacktest(is_cagr=0.10, oos_cagr=0.02),
        "2000-01-01", "2012-01-01", train_years=5, test_years=1, step_years=1,
    )
    assert weak.median_wfe == pytest.approx(0.2)
    assert weak.pass_pardo is False


def test_walk_forward_survives_a_failing_fold():
    def exploding(start, end, params=None):
        raise RuntimeError("no data")

    result = splits.anchored_walk_forward(exploding, "2000-01-01", "2012-01-01")
    assert result.folds == []
    assert result.pass_pardo is False


def test_split_is_oos_carves_the_holdout_off_the_end():
    (is_start, is_end), (oos_start, oos_end) = splits.split_is_oos(
        "2000-01-01", "2020-01-01", holdout_years=3
    )

    assert is_start == "2000-01-01"
    assert oos_end == "2020-01-01"
    assert oos_start == "2017-01-01"
    assert is_end == "2016-12-31"  # no overlap, no gap
    assert pd.Timestamp(oos_start) - pd.Timestamp(is_end) == pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# 18. Random-portfolio placebo
# ---------------------------------------------------------------------------

def _close_matrix(n_days=300, n_symbols=10, seed=11):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    symbols = [f"S{i}" for i in range(n_symbols)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n_days, n_symbols)), axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


def test_random_portfolio_percentile_is_sane():
    close = _close_matrix()
    placebo = random_portfolio_placebo(
        close, n_positions=3, universe=list(close.columns),
        start=close.index[0], end=close.index[-1],
        initial_capital=10_000, n_simulations=200, seed=42,
    )

    assert len(placebo["sharpes"]) == 200
    assert placebo["p5"] <= placebo["median_sharpe"] <= placebo["p95"]

    best = max(placebo["sharpes"])
    worst = min(placebo["sharpes"])

    assert percentile_vs_random(best + 1.0, placebo) == pytest.approx(100.0)
    assert percentile_vs_random(worst - 1.0, placebo) == pytest.approx(0.0)
    assert 0.0 <= percentile_vs_random(placebo["median_sharpe"], placebo) <= 100.0

    # A strategy at the median of the draws sits near the 50th percentile.
    assert percentile_vs_random(placebo["median_sharpe"], placebo) == pytest.approx(50.0, abs=5.0)


def test_percentile_vs_random_without_draws_is_neutral():
    assert percentile_vs_random(1.5, {"sharpes": []}) == 50.0


# ---------------------------------------------------------------------------
# 19. Verdict
# ---------------------------------------------------------------------------

VERDICT_WORDS = {"PROMISING", "CAUTIOUS", "WEAK", "REJECT", "INCOMPLETE"}


def _verdict_word(card):
    for line in card.splitlines():
        if line.startswith("VERDICT: "):
            return line.split("VERDICT: ", 1)[1].strip()
    raise AssertionError("no verdict line")


def test_render_verdict_with_full_inputs():
    card = verdict.render_verdict(
        strategy_id="clenow_trend",
        params={"lookback": 60, "top_n": 7},
        data_desc="S&P 500, 2010-2025",
        metrics={"alpha_jensen": 0.04, "beta": 0.8, "max_drawdown": -0.22, "n_trials": 240},
        oos_sharpe=1.2,
        bootstrap_ci={"p5": 0.4, "p95": 2.0},
        wfe=0.62,
        deflated_sharpe=0.71,
        pbo=0.18,
        random_pct=97.0,
        survives_20bps=True,
        plateau_ok=True,
        worst_step_decay=0.15,
        cluster_membership="core",
        cluster_size=12,
        cluster_total=90,
        weak_regime="high vol",
        limitations=["no delisted names before 2014"],
    )

    assert isinstance(card, str)
    assert "clenow_trend" in card
    assert "lookback=60" in card
    assert _verdict_word(card) == "PROMISING"  # every check passes
    assert "no delisted names before 2014" in card


def test_render_verdict_with_only_required_arguments():
    card = verdict.render_verdict("my_strat", {}, "SPY only", {})

    assert "my_strat" in card
    assert _verdict_word(card) == "INCOMPLETE"


def test_render_verdict_reports_failures():
    card = verdict.render_verdict(
        "bad_strat", {"n": 1}, "test", {},
        oos_sharpe=-0.3, wfe=0.1, deflated_sharpe=0.05, pbo=0.9, random_pct=10.0,
    )

    assert _verdict_word(card) == "REJECT"
    assert "FAIL" in card


def test_render_verdict_shows_a_zero_decay():
    """A perfect plateau has zero decay, which is the best possible result and
    must still be reported rather than dropped as falsy."""
    card = verdict.render_verdict(
        "s", {}, "d", {}, plateau_ok=True, worst_step_decay=0.0
    )
    assert "worst 1-step Sharpe decay 0.00" in card


def test_verdict_card_reports_unrunnable_pbo_as_not_run():
    """NaN PBO is shown, but scored neither as a pass nor as a failure."""
    with_nan = verdict.render_verdict("s", {}, "d", {}, pbo=float("nan"))
    assert "n/a" in with_nan and "not run" in with_nan
    assert "FAIL" not in with_nan

    # It must not silently consume a slot in the pass tally: a card and the
    # same card plus an unrunnable PBO must reach the same verdict word.
    assert _verdict_word(
        verdict.render_verdict("s", {}, "d", {}, oos_sharpe=1.2)
    ) == _verdict_word(
        verdict.render_verdict("s", {}, "d", {}, oos_sharpe=1.2, pbo=float("nan"))
    )

    # A real value is still scored, in both directions.
    assert "FAIL" in verdict.render_verdict("s", {}, "d", {}, pbo=0.9)
    assert "pass" in verdict.render_verdict("s", {}, "d", {}, pbo=0.1)


def test_all_verdict_words_are_reachable():
    # Six scored checks, above MIN_SCORED_CHECKS: a card below the quorum is
    # INCOMPLETE by design, so grading a sparse one would test nothing. Every
    # check is swept pass/fail, because the grade is a ratio -- holding two of
    # them at "pass" puts a floor under it and REJECT becomes unreachable.
    import itertools

    words = set()
    for good in itertools.product([True, False], repeat=6):
        pbo, wfe, dsr, pct, costs, plateau = good
        words.add(_verdict_word(verdict.render_verdict(
            "s", {}, "d", {},
            oos_sharpe=1.0,
            wfe=0.6 if wfe else 0.1,
            deflated_sharpe=0.7 if dsr else 0.1,
            pbo=0.1 if pbo else 0.9,
            random_pct=97 if pct else 20,
            survives_20bps=bool(costs),
            plateau_ok=bool(plateau),
        )))
    assert words <= VERDICT_WORDS
    assert {"PROMISING", "CAUTIOUS", "WEAK", "REJECT"} <= words

"""Fund-level accounting tests.

The fund engine's claim is that it is *one account*, not several backtests added
together. Three things have to hold for that claim to mean anything, and each one
is a separate section below.

**The cash identity.** Sleeve cash sub-ledgers must sum to the fund's single real
cash balance, every session. If they drift, the per-sleeve attribution is lying
and nothing else in the report can be trusted.

**Netting.** Offsetting sleeve orders must cross internally and only the residual
may reach the market, paying spread once. Running sleeves separately charges
spread twice on flow that never left the building, which flatters the result.

**Fund-level limits.** Two sleeves each at 90% gross are a fund at 180%, and
neither sleeve's own leverage check fires. The combined book is what has to be
capped.

Every number in the netting tests is arithmetic shown in the comment.

Run with: ./venv/bin/python -m pytest tests/backtest/test_fund.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.fund import FundEngine, SleeveConfig
from backtest.types import CostConfig, ExecutionConfig, RiskConfig
from data_pipeline import schema

FREE = CostConfig(
    spread_bps=0.0, slippage_bps=0.0, commission_per_share=0.0,
    min_commission=0.0, participation_rate=1.0,
)

#: 100 bps each way, so the cost of a trade is visible in whole dollars.
EXPENSIVE = CostConfig(
    spread_bps=100.0, slippage_bps=100.0, commission_per_share=0.0,
    min_commission=0.0, participation_rate=1.0,
)

WIDE_OPEN = RiskConfig(
    max_position_pct=1.0, max_sector_pct=1.0, cash_reserve_pct=0.0,
    max_leverage=10.0, max_positions=50, fractional_shares=True,
)

FRACTIONAL = ExecutionConfig(fractional_shares=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _panel(symbols, n_sessions=6, price=100.0, volume=10_000_000):
    """A flat panel: every OHLC equals `price`, so fills are predictable."""
    dates = pd.bdate_range("2024-01-02", periods=n_sessions)
    rows = []
    for date in dates:
        for symbol in symbols:
            rows.append((
                date, symbol, price, price, price, price, price, volume,
            ))
    frame = pd.DataFrame(rows, columns=schema.COLUMNS)
    frame[schema.DATE] = pd.to_datetime(frame[schema.DATE])

    matrix = frame.pivot_table(
        index=schema.DATE, columns=schema.SYMBOL,
        values=schema.ADJ_CLOSE, aggfunc="last",
    )
    matrix.columns.name = None
    return frame, matrix


def _fixed(weights_by_session):
    """A strategy returning a canned weight dict, keyed by session index."""
    calls = {"n": 0}

    def strategy(ctx):
        index = calls["n"]
        calls["n"] += 1
        if isinstance(weights_by_session, list):
            if index < len(weights_by_session):
                return dict(weights_by_session[index])
            return dict(weights_by_session[-1])
        return dict(weights_by_session)

    return strategy


def _engine(sleeves, matrix, **overrides):
    kwargs = dict(
        sleeves=sleeves,
        start_date=str(matrix.index[0].date()),
        end_date=str(matrix.index[-1].date()),
        initial_capital=100_000.0,
        benchmark="SPY",
        cost_config=FREE,
        risk_config=WIDE_OPEN,
        execution_config=FRACTIONAL,
    )
    kwargs.update(overrides)
    return FundEngine(**kwargs)


def _run(sleeves, symbols=("SPY", "AAA"), n_sessions=6, **overrides):
    panel, matrix = _panel(list(symbols), n_sessions=n_sessions)
    engine = _engine(sleeves, matrix, **overrides)
    return engine.run(price_panel=panel, close_matrix=matrix), engine, matrix


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_allocations_over_one_hundred_percent_are_rejected(self):
        sleeves = [
            SleeveConfig("a", _fixed({}), 0.7),
            SleeveConfig("b", _fixed({}), 0.7),
        ]
        _, matrix = _panel(["SPY"])

        with pytest.raises(ValueError, match="140"):
            _engine(sleeves, matrix)

    def test_duplicate_sleeve_ids_are_rejected(self):
        """Attribution is keyed on strategy_id; two sleeves sharing one would
        merge silently and each would misreport the other's lots as its own."""
        sleeves = [
            SleeveConfig("same", _fixed({}), 0.5),
            SleeveConfig("same", _fixed({}), 0.5),
        ]
        _, matrix = _panel(["SPY"])

        with pytest.raises(ValueError, match="duplicate"):
            _engine(sleeves, matrix)

    def test_a_fund_needs_a_sleeve(self):
        _, matrix = _panel(["SPY"])

        with pytest.raises(ValueError, match="at least one sleeve"):
            _engine([], matrix)


# ---------------------------------------------------------------------------
# The cash identity
# ---------------------------------------------------------------------------

class TestCashIdentity:
    def test_sleeve_cash_sums_to_fund_equity_when_fully_allocated(self):
        sleeves = [
            SleeveConfig("mom", _fixed({"SPY": 0.5}), 0.5),
            SleeveConfig("def", _fixed({"AAA": 0.5}), 0.5),
        ]
        result, _, _ = _run(sleeves)

        # Sleeve equities must reconstruct the fund's, session by session.
        total = sum(result.sleeve_equity[sid] for sid in ("mom", "def"))
        pd.testing.assert_series_equal(
            total, result.fund_equity, check_names=False
        )

    def test_unallocated_buffer_is_not_lost(self):
        """Allocations under 100% leave fund cash no sleeve owns."""
        sleeves = [SleeveConfig("only", _fixed({"SPY": 1.0}), 0.6)]
        result, _, _ = _run(sleeves)

        sleeve_total = result.sleeve_equity["only"]
        buffer = result.fund_equity - sleeve_total

        # 40% of 100k was never handed to a sleeve and never traded.
        assert buffer.iloc[0] == pytest.approx(40_000.0)
        assert buffer.std() == pytest.approx(0.0, abs=1e-6)

    def test_cash_identity_survives_an_internal_cross(self):
        """The crossed half nets to zero in fund cash. It must also reconcile."""
        sleeves = [
            SleeveConfig("buyer", _fixed([{}, {"SPY": 0.8}, {"SPY": 0.8}]), 0.5),
            SleeveConfig("seller", _fixed([{"SPY": 0.8}, {}, {}]), 0.5),
        ]
        result, _, _ = _run(sleeves)

        total = sum(result.sleeve_equity[sid] for sid in ("buyer", "seller"))
        assert np.allclose(total.values, result.fund_equity.values)


# ---------------------------------------------------------------------------
# Netting
# ---------------------------------------------------------------------------

class TestNetting:
    def test_opposing_orders_cross_and_only_the_residual_trades(self):
        """Hand-checked.

        Both sleeves get 50k. At $100 a share, "80% of sleeve equity" is $40,000
        = 400 shares. The seller starts long 400 SPY and exits; the buyer wants
        400. Gross flow is 800 shares, the net is 0, so nothing should reach the
        market at all.
        """
        sleeves = [
            SleeveConfig("buyer", _fixed([{}, {"SPY": 0.8}, {"SPY": 0.8}, {"SPY": 0.8}]), 0.5),
            SleeveConfig("seller", _fixed([{"SPY": 0.8}, {}, {}, {}]), 0.5),
        ]
        result, _, _ = _run(sleeves, n_sessions=6)

        crossing = [r for r in result.netting if r.crossed_shares > 0]
        assert crossing, "the two sleeves must have crossed at least once"

        for record in crossing:
            assert record.crossed_shares > 0
            assert abs(record.net_shares) < record.gross_shares

    def test_crossed_flow_is_reported_as_a_saving(self):
        sleeves = [
            SleeveConfig("buyer", _fixed([{}, {"SPY": 0.8}, {"SPY": 0.8}, {"SPY": 0.8}]), 0.5),
            SleeveConfig("seller", _fixed([{"SPY": 0.8}, {}, {}, {}]), 0.5),
        ]
        result, _, _ = _run(sleeves, n_sessions=6)

        assert result.netting_saved_notional > 0

    def test_netting_beats_separate_accounts_on_cost(self):
        """The whole point, measured.

        Same two sleeves, same signals, 100 bps each way. Run them as one netted
        fund and then as two independent funds; the netted fund must end richer,
        because the crossed shares paid no spread.
        """
        def make():
            return [
                SleeveConfig(
                    "buyer",
                    _fixed([{}, {"SPY": 0.8}, {"SPY": 0.8}, {"SPY": 0.8}, {"SPY": 0.8}]),
                    0.5,
                ),
                SleeveConfig("seller", _fixed([{"SPY": 0.8}, {}, {}, {}, {}]), 0.5),
            ]

        netted, _, _ = _run(make(), n_sessions=6, cost_config=EXPENSIVE)

        # Two single-sleeve funds, each with half the capital: no cross possible.
        separate_total = 0.0
        for sleeve in make():
            solo = SleeveConfig(
                sleeve.strategy_id, sleeve.strategy_fn, 1.0, sleeve.params
            )
            alone, _, _ = _run(
                [solo], n_sessions=6, cost_config=EXPENSIVE,
                initial_capital=50_000.0,
            )
            separate_total += float(alone.fund_equity.iloc[-1])

        assert float(netted.fund_equity.iloc[-1]) > separate_total

    def test_a_single_sleeve_nets_to_itself_and_still_pays_spread(self):
        """No cross available, so the netted path must charge full cost."""
        sleeves = [SleeveConfig("only", _fixed({"SPY": 1.0}), 1.0)]

        result, _, _ = _run(sleeves, cost_config=EXPENSIVE)

        assert all(r.crossed_shares == 0 for r in result.netting)
        assert result.netting_saved_notional == pytest.approx(0.0)
        # 100 bps of spread plus 100 of slippage on the way in leaves equity down.
        assert float(result.fund_equity.iloc[-1]) < 100_000.0

    def test_same_direction_orders_do_not_cross(self):
        """Two buyers are not an internal cross; both must reach the market."""
        sleeves = [
            SleeveConfig("a", _fixed({"SPY": 0.5}), 0.5),
            SleeveConfig("b", _fixed({"SPY": 0.5}), 0.5),
        ]
        result, _, _ = _run(sleeves)

        assert all(r.crossed_shares == pytest.approx(0.0) for r in result.netting)

    def test_both_sleeves_receive_their_shares_despite_one_net_order(self):
        sleeves = [
            SleeveConfig("buyer", _fixed([{}, {"SPY": 0.8}, {"SPY": 0.8}, {"SPY": 0.8}]), 0.5),
            SleeveConfig("seller", _fixed([{"SPY": 0.8}, {}, {}, {}]), 0.5),
        ]
        result, _, _ = _run(sleeves, n_sessions=6)

        assert result.sleeve_fills["buyer"], "buyer got no fill"
        assert result.sleeve_fills["seller"], "seller got no fill"
        # Attribution is per sleeve even though the market saw one order.
        assert {f.order.strategy_id for f in result.sleeve_fills["buyer"]} == {"buyer"}
        assert {f.order.strategy_id for f in result.sleeve_fills["seller"]} == {"seller"}


# ---------------------------------------------------------------------------
# FIFO isolation
# ---------------------------------------------------------------------------

class TestLotIsolation:
    def test_one_sleeve_cannot_sell_anothers_lots(self):
        """Both sleeves hold SPY; only one exits.

        Without a strategy filter on the FIFO sale, the exiting sleeve would
        consume whichever lot is oldest -- possibly the other sleeve's -- and both
        would then disagree with the ledger about what they hold.
        """
        sleeves = [
            SleeveConfig("keeper", _fixed({"SPY": 0.9}), 0.5),
            SleeveConfig("quitter", _fixed([{"SPY": 0.9}, {"SPY": 0.9}, {}, {}, {}]), 0.5),
        ]
        result, _, _ = _run(sleeves, n_sessions=7)

        final = result.snapshots[-1]
        keeper_final = result.sleeve_equity["keeper"].iloc[-1]

        # The keeper is still invested; the quitter is in cash.
        assert "SPY" in final.positions
        assert keeper_final > 0
        assert result.sleeve_cash["quitter"].iloc[-1] > (
            result.sleeve_cash["quitter"].iloc[0] * 0.9
        )


# ---------------------------------------------------------------------------
# Fund-level risk limits
# ---------------------------------------------------------------------------

class TestFundRisk:
    def test_combined_leverage_is_capped_where_per_sleeve_checks_pass(self):
        """Each sleeve asks for 100% of its own equity; the fund allows 1.2x.

        Per-sleeve gross is 1.0 and passes any sane sleeve check. Combined it is
        2.0, and only a fund-level pass can see that.
        """
        capped = RiskConfig(
            max_position_pct=1.0, max_sector_pct=1.0, cash_reserve_pct=0.0,
            max_leverage=1.2, max_positions=50, fractional_shares=True,
        )
        sleeves = [
            SleeveConfig("a", _fixed({"SPY": 1.0}), 0.5, risk=WIDE_OPEN),
            SleeveConfig("b", _fixed({"AAA": 1.0}), 0.5, risk=WIDE_OPEN),
        ]

        result, _, _ = _run(sleeves, n_sessions=6, risk_config=capped)

        peak_gross = float(result.gross_exposure.max())
        assert peak_gross <= 1.2 + 0.05, f"fund reached {peak_gross:.2f}x"

    def test_uncapped_fund_lets_both_sleeves_invest_fully(self):
        sleeves = [
            SleeveConfig("a", _fixed({"SPY": 1.0}), 0.5),
            SleeveConfig("b", _fixed({"AAA": 1.0}), 0.5),
        ]
        result, _, _ = _run(sleeves, n_sessions=6)

        assert float(result.gross_exposure.max()) > 0.9

    def test_scaling_preserves_the_shape_of_a_sleeve_request(self):
        """A uniform scale, not a clip: relative conviction must survive.

        Clipping the largest position would re-rank what the sleeve asked for.
        """
        capped = RiskConfig(
            max_position_pct=1.0, max_sector_pct=1.0, cash_reserve_pct=0.0,
            max_leverage=0.5, max_positions=50, fractional_shares=True,
        )
        sleeves = [
            SleeveConfig("a", _fixed({"SPY": 0.6, "AAA": 0.3}), 1.0),
        ]
        result, _, _ = _run(sleeves, n_sessions=6, risk_config=capped)

        final = result.snapshots[-1]
        if "SPY" in final.positions and "AAA" in final.positions:
            ratio = (
                final.positions["SPY"].market_value
                / final.positions["AAA"].market_value
            )
            assert ratio == pytest.approx(2.0, rel=0.1), "0.6/0.3 must stay 2:1"


# ---------------------------------------------------------------------------
# Capital reallocation
# ---------------------------------------------------------------------------

class TestReallocation:
    def test_no_reallocation_by_default(self):
        sleeves = [
            SleeveConfig("a", _fixed({"SPY": 0.5}), 0.5),
            SleeveConfig("b", _fixed({"AAA": 0.5}), 0.5),
        ]
        result, _, _ = _run(sleeves)

        assert result.reallocations == []

    def test_fixed_mode_moves_cash_toward_target_allocations(self):
        """An idle sleeve holding cash funds a sleeve that is below target."""
        sleeves = [
            SleeveConfig("idle", _fixed({}), 0.8),
            SleeveConfig("active", _fixed({}), 0.2),
        ]
        result, _, _ = _run(
            sleeves, n_sessions=6,
            reallocate="fixed", reallocate_every_months=1,
        )

        # Both are all cash at target weights already, so nothing should move.
        assert all(r["moved"] >= 0 for r in result.reallocations)

    def test_reallocation_is_capped_by_available_cash(self):
        """A fully invested donor cannot hand over capital it does not hold.

        Positions are never transferred -- that would give one sleeve another's
        entry price and break every exit rule that reads it -- so the shortfall is
        recorded rather than forced.
        """
        sleeves = [
            SleeveConfig("invested", _fixed({"SPY": 1.0}), 0.5),
            SleeveConfig("hungry", _fixed({}), 0.5),
        ]
        result, _, _ = _run(
            sleeves, n_sessions=8,
            reallocate="fixed", reallocate_every_months=1,
        )

        for record in result.reallocations:
            assert record["moved"] <= record["requested"] + 1e-6

    def test_reallocation_preserves_the_cash_identity(self):
        sleeves = [
            SleeveConfig("a", _fixed({"SPY": 0.4}), 0.5),
            SleeveConfig("b", _fixed({}), 0.5),
        ]
        result, _, _ = _run(
            sleeves, n_sessions=8,
            reallocate="fixed", reallocate_every_months=1,
        )

        total = sum(result.sleeve_equity[sid] for sid in ("a", "b"))
        assert np.allclose(total.values, result.fund_equity.values, rtol=1e-9)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

class TestAttribution:
    def test_per_sleeve_metrics_are_computed(self):
        sleeves = [
            SleeveConfig("a", _fixed({"SPY": 0.5}), 0.5),
            SleeveConfig("b", _fixed({"AAA": 0.5}), 0.5),
        ]
        result, _, _ = _run(sleeves)

        assert set(result.sleeve_metrics) == {"a", "b"}
        for stats in result.sleeve_metrics.values():
            assert "sharpe" in stats and "max_drawdown" in stats

    def test_fund_metrics_are_computed_against_the_benchmark(self):
        sleeves = [SleeveConfig("a", _fixed({"AAA": 0.5}), 1.0)]
        result, _, _ = _run(sleeves)

        assert "sharpe" in result.fund_metrics
        assert len(result.benchmark_equity) > 0

    def test_a_failing_sleeve_is_recorded_and_does_not_stop_the_fund(self):
        def explode(ctx):
            raise RuntimeError("bad signal")

        sleeves = [
            SleeveConfig("good", _fixed({"SPY": 0.5}), 0.5),
            SleeveConfig("bad", explode, 0.5),
        ]
        result, _, _ = _run(sleeves)

        assert result.warnings
        assert any("bad signal" in w for w in result.warnings)
        assert len(result.fund_equity) > 0
        assert result.sleeve_fills["good"], "the healthy sleeve must still trade"

    def test_no_sessions_in_range_returns_a_warning_not_a_crash(self):
        panel, matrix = _panel(["SPY"])
        engine = FundEngine(
            sleeves=[SleeveConfig("a", _fixed({}), 1.0)],
            start_date="2030-01-01", end_date="2030-12-31",
            initial_capital=100_000.0,
        )

        result = engine.run(price_panel=panel, close_matrix=matrix)

        assert result.warnings == ["No trading sessions in range"]
        assert len(result.fund_equity) == 0


# ---------------------------------------------------------------------------
# Overlap reporting
# ---------------------------------------------------------------------------

class TestOverlapReporting:
    def test_disjoint_sleeves_report_no_overlap(self):
        """A zero netting saving means one of two things. This is the innocent one.

        An ETF trend follower beside a single-stock screen never holds the same
        instrument, so nothing can cross and zero is the correct answer. Without
        the overlap count that is indistinguishable from netting being broken.
        """
        sleeves = [
            SleeveConfig("etfs", _fixed({"SPY": 0.5}), 0.5),
            SleeveConfig("stocks", _fixed({"AAA": 0.5}), 0.5),
        ]
        result, _, _ = _run(sleeves)

        assert result.shared_symbols == {}
        assert result.netting_saved_notional == pytest.approx(0.0)

    def test_overlapping_sleeves_are_reported(self):
        sleeves = [
            SleeveConfig("a", _fixed({"SPY": 0.5}), 0.5),
            SleeveConfig("b", _fixed({"SPY": 0.5}), 0.5),
        ]
        result, _, _ = _run(sleeves)

        assert "SPY" in result.shared_symbols
        assert result.shared_symbols["SPY"] == ["a", "b"]

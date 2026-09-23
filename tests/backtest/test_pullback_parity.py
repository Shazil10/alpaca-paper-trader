"""Parity gate for the 52-week pullback mean-reversion migration.

Both paths are run over one identical panel and their decisions compared: which
symbols clear the entry gates, the pullback ranking, the regime verdict, and the
exit reason for each held name.

The exit rules are where this sleeve differs from the others, because two of the
three need to know what *we* paid and when. Live reads ``avg_entry_price`` off the
Alpaca position and reconstructs entry dates from order history; the backtest
reads both off ``ctx.portfolio``. ``exit_reason`` takes them as arguments so the
arithmetic is shared and only the source differs -- so these tests drive
``exit_reason`` directly with hand-computed numbers, then check the adapter
reaches the same verdict through the context.

The cooldown constants get their own section. The live sleeve declares
``COOLDOWN_DAYS = 20`` and ``STOP_ENTRY_COOLDOWN = 30`` and enforces neither, so
the default here is the live behaviour and the documented behaviour is opt-in via
``params``. Both are tested, because the point is to make the difference
measurable rather than to pick one quietly.

Run with: ./venv/bin/python -m pytest tests/backtest/test_pullback_parity.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.context import StrategyContext
from backtest.types import Position, PortfolioSnapshot
from data_pipeline import store
from strategies.mean_reversion import pullback_target_weights as pbw

pb = pbw.pb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lake_panel():
    calendar = store.trading_calendar()
    if len(calendar) == 0:
        pytest.skip("price lake is empty")

    as_of = pd.Timestamp(calendar[-1])
    start = as_of - pd.Timedelta(days=pb.LOOKBACK_DAYS * 2)

    panel = store.load_prices(start=start, end=as_of)
    if len(panel) == 0:
        pytest.skip("no bars in the lake window")

    return as_of, panel, store.load_close_matrix(start=start, end=as_of)


@pytest.fixture(scope="module")
def candidates(lake_panel):
    """Symbols with the 252 sessions the entry gates require."""
    as_of, _, matrix = lake_panel
    window = matrix.loc[matrix.index >= as_of - pd.Timedelta(days=pb.LOOKBACK_DAYS)]

    deep = [
        str(c) for c in window.columns
        if int(window[c].notna().sum()) >= pbw.MIN_SESSIONS
        and str(c) not in set(pb.REGIME_ETFS)
    ]
    if len(deep) < 5:
        pytest.skip(f"only {len(deep)} symbol(s) carry enough history")

    return sorted(deep)[:80]


def _ctx(lake_panel, universe, held=None, as_of=None, exit_dates=None, params=None):
    lake_as_of, panel, matrix = lake_panel
    as_of = as_of or lake_as_of

    portfolio = PortfolioSnapshot(date=as_of, cash=0.0, equity=0.0)
    if held:
        equity = float(sum(v["value"] for v in held.values()))
        positions = {
            sym: Position(
                symbol=sym,
                shares=1.0,
                avg_entry_price=float(spec["entry_price"]),
                market_price=float(spec["value"]),
                market_value=float(spec["value"]),
                unrealized_pnl=0.0,
                entry_date=pd.Timestamp(spec["entry_date"]),
            )
            for sym, spec in held.items()
        }
        portfolio = PortfolioSnapshot(
            date=as_of, cash=0.0, equity=equity, positions=positions
        )

    return StrategyContext(
        as_of=as_of,
        price_panel=panel,
        close_matrix=matrix,
        universe_fn=lambda _d: set(universe) | set(pb.REGIME_ETFS),
        portfolio=portfolio,
        params=params or {},
        exit_dates=exit_dates,
    )


# ---------------------------------------------------------------------------
# Entry gate parity
# ---------------------------------------------------------------------------

class TestEntryParity:
    def test_same_candidates_and_same_order(self, lake_panel, candidates):
        """The adapter must call _score_series, not reimplement the gates."""
        ctx = _ctx(lake_panel, candidates)
        panel = pbw.panel_from_ctx(ctx, candidates, pb.LOOKBACK_DAYS)

        direct = [
            pb._score_series(str(c), panel[c]) for c in panel.columns
        ]
        expected = pd.DataFrame([r for r in direct if r is not None])
        if not expected.empty:
            expected = expected.sort_values(
                "pullback", ascending=False
            ).reset_index(drop=True)

        new = pbw.decide(ctx)

        assert new is not None
        assert new["scored"] == len(expected)
        if not expected.empty:
            head = expected["symbol"].tolist()[: len(new["candidates"])]
            assert new["candidates"] == head

    def test_gates_reject_a_shallow_pullback(self):
        """A name 10% off its high is not a 40% pullback candidate."""
        idx = pd.bdate_range("2023-01-02", periods=300)
        # Rises to 100 then eases to 90: a 10% pullback.
        series = pd.Series(
            list(range(1, 251)) + [250 - i * 0.1 for i in range(50)], index=idx
        ).astype(float)

        assert pb._score_series("SHALLOW", series) is None

    def test_gates_accept_a_deep_decline_that_has_based_out(self):
        """The three gates together only admit an *old* decline, not a fresh one.

        F3 (price >= 90% of the 200-day average) is what forces this. To be 40%
        below the 52-week high and still within 10% of the 200-day average, the
        average must already have fallen to meet the price -- which means the
        drop happened months ago and the price has stabilised since. A name that
        slid 45% over the last fifty sessions fails F3 outright, because its
        200-day average is still up near the old level.

        That interaction is the strategy: buy repaired damage, not falling knives.
        """
        idx = pd.bdate_range("2023-01-02", periods=300)
        series = pd.Series(
            [100.0] * 50                                # the old high
            + [100.0 - 0.45 * i for i in range(100)]    # a long slide to ~55
            + [55.0] * 150,                             # then a flat base
            index=idx,
        )

        record = pb._score_series("BASED", series)

        assert record is not None
        assert record["pullback"] >= pb.ENTRY_DEPTH
        assert record["price"] == pytest.approx(55.0)

    def test_gates_reject_a_fresh_slide_of_the_same_depth(self):
        """Same 45% pullback, but recent: F3 rejects it."""
        idx = pd.bdate_range("2023-01-02", periods=300)
        series = pd.Series(
            [100.0] * 250 + [100.0 - i * 0.9 for i in range(50)], index=idx
        )

        assert pb._score_series("FALLING", series) is None

    def test_regime_verdict_matches_the_shared_helper(self, lake_panel, candidates):
        ctx = _ctx(lake_panel, candidates)
        regime_panel = pbw.panel_from_ctx(
            ctx, list(pb.REGIME_ETFS), pb.REGIME_LOOKBACK_DAYS
        )

        assert pbw.decide(ctx)["risk_on"] == pb.regime_from_panel(regime_panel)


# ---------------------------------------------------------------------------
# Exit rule parity -- hand-computed
# ---------------------------------------------------------------------------

class TestExitRules:
    def test_target_recovery_wins_over_everything(self):
        """Rule order is the decision: the first match is the reason recorded."""
        reason = pb.exit_reason(
            pullback=0.20,          # inside the 25% exit band
            price=50.0,
            entry_price=100.0,      # also 50% underwater, which would stop out
            entry_date=pd.Timestamp("2020-01-01").to_pydatetime(),
            now=pd.Timestamp("2024-01-01").to_pydatetime(),
        )

        assert reason.startswith("exit:target_recovery")

    def test_stop_fires_at_exactly_minus_25_percent(self):
        """75 from an entry of 100 is -25%, which is the boundary, inclusive."""
        assert pb.exit_reason(
            pullback=0.50, price=75.0, entry_price=100.0
        ) == "exit:stop_entry_25.0%_below_entry"

        # A hair above the threshold must hold.
        assert pb.exit_reason(pullback=0.50, price=75.01, entry_price=100.0) is None

    def test_timeout_uses_the_live_252_over_365_conversion(self):
        """63 trading days at 252/365 is 92 calendar days, not 63 and not 90."""
        entry = pd.Timestamp("2024-01-01").to_pydatetime()

        # 91 calendar days -> int(91 * 252/365) = 62 trading days: holds.
        assert pb.exit_reason(
            pullback=0.50, price=100.0, entry_price=100.0,
            entry_date=entry, now=pd.Timestamp("2024-04-01").to_pydatetime(),
        ) is None

        # 92 calendar days -> 63: exits.
        reason = pb.exit_reason(
            pullback=0.50, price=100.0, entry_price=100.0,
            entry_date=entry, now=pd.Timestamp("2024-04-02").to_pydatetime(),
        )
        assert reason == "exit:timeout_63d"

    def test_no_rule_means_hold(self):
        assert pb.exit_reason(pullback=0.50, price=100.0, entry_price=100.0) is None

    def test_adapter_reaches_the_same_verdict_through_the_context(
        self, lake_panel, candidates
    ):
        """Entry price and date come off ctx.portfolio, and must be used."""
        as_of, _, matrix = lake_panel
        symbol = candidates[0]
        price = float(matrix[symbol].dropna().iloc[-1])

        # An entry price four times the current one is a 75% loss: stop territory.
        ctx = _ctx(
            lake_panel, candidates,
            held={symbol: {
                "value": 1000.0,
                "entry_price": price * 4.0,
                "entry_date": as_of - pd.Timedelta(days=5),
            }},
        )

        decision = pbw.decide(ctx)
        reason = decision["exit_reasons"].get(symbol, "")

        assert symbol in decision["exiting"]
        assert reason.startswith(("exit:stop_entry", "exit:target_recovery"))

    def test_missing_price_data_defers_rather_than_selling(
        self, lake_panel, candidates
    ):
        """An infrastructure fault must not realise a loss. Live holds; so do we."""
        ctx = _ctx(
            lake_panel, candidates,
            held={"ZZZZ_NOT_A_TICKER": {
                "value": 1000.0, "entry_price": 100.0,
                "entry_date": pd.Timestamp("2020-01-01"),
            }},
        )

        decision = pbw.decide(ctx)

        assert "ZZZZ_NOT_A_TICKER" in decision["missing_prices"]
        assert "ZZZZ_NOT_A_TICKER" not in decision["exiting"]
        # And it survives into the target book rather than being flattened.
        assert "ZZZZ_NOT_A_TICKER" in pbw.target_weights(ctx)


# ---------------------------------------------------------------------------
# The cooldown the live sleeve declares but never enforces
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_constants_exist_and_are_unused_live(self):
        """Pin the premise: they are declared, and nothing reads them."""
        import inspect

        assert pbw.COOLDOWN_DAYS == 20
        assert pbw.STOP_ENTRY_COOLDOWN == 30

        source = inspect.getsource(pb)
        # One definition each, and no second reference anywhere in the module.
        assert source.count("COOLDOWN_DAYS") == 1
        assert source.count("STOP_ENTRY_COOLDOWN") == 1

    def test_default_is_live_behaviour_immediate_reentry(
        self, lake_panel, candidates
    ):
        """No cooldown parameter means no cooldown, matching production."""
        as_of, _, _ = lake_panel
        recent = {s: as_of - pd.Timedelta(days=1) for s in candidates}

        decision = pbw.decide(_ctx(lake_panel, candidates, exit_dates=recent))

        assert decision["cooldown_blocked"] == []

    def test_cooldown_blocks_a_recent_exit_when_enabled(
        self, lake_panel, candidates
    ):
        as_of, _, _ = lake_panel
        recent = {s: as_of - pd.Timedelta(days=1) for s in candidates}

        decision = pbw.decide(
            _ctx(lake_panel, candidates, exit_dates=recent),
            cooldown_days=pbw.COOLDOWN_DAYS,
        )

        if not decision["candidates"]:
            pytest.skip("no candidates to block")
        assert decision["cooldown_blocked"], "a 1-day-old exit must be blocked"
        assert decision["new_weights"] == {}

    def test_cooldown_expires(self, lake_panel, candidates):
        as_of, _, _ = lake_panel
        old = {s: as_of - pd.Timedelta(days=pbw.COOLDOWN_DAYS + 1) for s in candidates}

        decision = pbw.decide(
            _ctx(lake_panel, candidates, exit_dates=old),
            cooldown_days=pbw.COOLDOWN_DAYS,
        )

        assert decision["cooldown_blocked"] == []

    def test_cooldown_reads_from_params(self, lake_panel, candidates):
        as_of, _, _ = lake_panel
        recent = {s: as_of - pd.Timedelta(days=1) for s in candidates}
        ctx = _ctx(
            lake_panel, candidates,
            exit_dates=recent, params={"cooldown_days": 20},
        )

        # Reached through target_weights, which is what the runner calls.
        assert pbw.target_weights(ctx) == {}


# ---------------------------------------------------------------------------
# Context exit history
# ---------------------------------------------------------------------------

class TestExitHistory:
    def test_unknown_symbol_has_no_exit(self, lake_panel, candidates):
        ctx = _ctx(lake_panel, candidates)

        assert ctx.last_exit("AAPL") is None
        assert ctx.days_since_exit("AAPL") is None

    def test_days_since_exit_counts_calendar_days(self, lake_panel, candidates):
        as_of, _, _ = lake_panel
        ctx = _ctx(
            lake_panel, candidates,
            exit_dates={"AAPL": as_of - pd.Timedelta(days=7)},
        )

        assert ctx.days_since_exit("AAPL") == 7
        assert ctx.last_exit("aapl") == as_of - pd.Timedelta(days=7)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

class TestSizing:
    def test_deeper_pullback_gets_more_weight(self, lake_panel, candidates):
        decision = pbw.decide(_ctx(lake_panel, candidates))
        weights = decision["new_weights"]

        if len(weights) < 2:
            pytest.skip("need two picks to compare")

        depths = {
            sym: pb._score_series(
                sym,
                pbw.panel_from_ctx(
                    _ctx(lake_panel, candidates), [sym], pb.LOOKBACK_DAYS
                )[sym],
            )["pullback"]
            for sym in weights
        }
        deepest = max(depths, key=depths.get)

        assert weights[deepest] == pytest.approx(max(weights.values()))

    def test_a_full_book_sums_to_one(self, lake_panel, candidates):
        """slots_filled / top_n scaling: five of five slots is the whole book."""
        decision = pbw.decide(_ctx(lake_panel, candidates))

        if len(decision["new_weights"]) != pbw.TOP_N:
            pytest.skip("fewer candidates than slots")
        assert sum(decision["new_weights"].values()) == pytest.approx(1.0)

    def test_a_partial_fill_asks_for_its_share_only(self, lake_panel, candidates):
        """Four held of five means one free slot, worth 1/5 of equity -- not all."""
        as_of, _, matrix = lake_panel
        # Hold four names that are not exit candidates, at a flat entry price.
        holders = {}
        for sym in candidates[:4]:
            price = float(matrix[sym].dropna().iloc[-1])
            holders[sym] = {
                "value": 1000.0, "entry_price": price,
                "entry_date": as_of - pd.Timedelta(days=5),
            }

        decision = pbw.decide(_ctx(lake_panel, candidates, held=holders))

        assert decision["slots_available"] == 1
        if not decision["new_weights"]:
            pytest.skip("nothing eligible to buy")
        assert sum(decision["new_weights"].values()) == pytest.approx(
            1.0 / pbw.TOP_N
        )

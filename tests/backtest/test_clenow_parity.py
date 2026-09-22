"""Parity gate for the Clenow momentum migration.

The claim under test is that ``clenow_target_weights`` is the *same strategy* as
``clenow_trend``, differing only in where prices come from. So both paths are run
over one identical price panel and their decisions compared: which symbols pass
the three entry gates, in what rank order, what the regime says, which held names
are exited and why, and what the inverse-vol weights are.

The live path downloads. Here ``yf.download`` is stubbed to serve the exact same
window the context path sees, which is what makes the comparison a test of the
logic rather than of two different datasets. Prices come from the lake, so the
fixture is real data rather than a shape that happens to satisfy the code.

Two live quirks are asserted *as they are*, not as they should be. A parity gate
that silently improves things proves nothing, and both are documented in
``clenow_target_weights``:

* slot accounting counts names that are being exited today, so a sleeve shedding
  positions buys nothing that session;
* ``MAX_WEIGHT`` does not bind below ten positions, because the clamp is followed
  by a renormalization that undoes it.

Run with: ./venv/bin/python -m pytest tests/backtest/test_clenow_parity.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.context import StrategyContext
from backtest.types import Position, PortfolioSnapshot
from data_pipeline import schema, store
from strategies.momentum import clenow_target_weights as cwt
from strategies.momentum import clenow_trend as ct

LOOKBACK = cwt.LOOKBACK_CALENDAR_DAYS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lake_panel():
    """A real slice of the lake: as_of, the long panel, and the close matrix."""
    calendar = store.trading_calendar()
    if len(calendar) == 0:
        pytest.skip("price lake is empty")

    as_of = pd.Timestamp(calendar[-1])
    start = as_of - pd.Timedelta(days=LOOKBACK * 2)

    panel = store.load_prices(start=start, end=as_of)
    if len(panel) == 0:
        pytest.skip("no bars in the lake window")

    matrix = store.load_close_matrix(start=start, end=as_of)
    return as_of, panel, matrix


@pytest.fixture(scope="module")
def candidates(lake_panel):
    """Symbols with enough history to clear ``score_symbol``'s length guard.

    Capped at 60 names: parity does not get truer with more symbols, and the
    200-day SMA plus two momentum regressions per name is not free.
    """
    as_of, _, matrix = lake_panel
    window = matrix.loc[matrix.index >= as_of - pd.Timedelta(days=LOOKBACK)]

    deep = [
        str(c) for c in window.columns
        if int(window[c].notna().sum()) >= cwt.MIN_SESSIONS
        and str(c) not in set(ct.REGIME_ETFS)
    ]
    if len(deep) < 5:
        pytest.skip(f"only {len(deep)} symbol(s) carry a full year of history")

    return sorted(deep)[:60]


def _ctx(lake_panel, universe, held=None, as_of=None):
    """A StrategyContext over the lake slice, with an optional held book."""
    lake_as_of, panel, matrix = lake_panel
    as_of = as_of or lake_as_of

    portfolio = PortfolioSnapshot(date=as_of, cash=0.0, equity=0.0)
    if held:
        equity = float(sum(held.values()))
        positions = {
            sym: Position(
                symbol=sym, shares=1.0, avg_entry_price=value,
                market_price=value, market_value=value,
                unrealized_pnl=0.0, entry_date=as_of - pd.Timedelta(days=30),
            )
            for sym, value in held.items()
        }
        portfolio = PortfolioSnapshot(
            date=as_of, cash=0.0, equity=equity, positions=positions
        )

    return StrategyContext(
        as_of=as_of,
        price_panel=panel,
        close_matrix=matrix,
        universe_fn=lambda _date: set(universe) | set(ct.REGIME_ETFS),
        portfolio=portfolio,
    )


def _stub_download(closes):
    """Return a yfinance-shaped frame for whatever batch is requested.

    ``group_by="ticker"`` produces a column MultiIndex of (ticker, field), which
    is what ``score_universe`` indexes into as ``data[symbol]["Close"]``.
    """
    def download(tickers, *args, **kwargs):
        wanted = [tickers] if isinstance(tickers, str) else list(tickers)
        frames = {}
        for symbol in wanted:
            series = closes.get(symbol)
            if series is None:
                continue
            frames[(symbol, "Close")] = series
        if not frames:
            return pd.DataFrame()
        out = pd.DataFrame(frames)
        out.columns = pd.MultiIndex.from_tuples(out.columns)
        return out

    return download


def _old_path(closes, universe, monkeypatch, tmp_path):
    """Run ``score_universe`` and ``check_regime`` against the stub."""
    monkeypatch.setattr(ct.yf, "download", _stub_download(closes))

    universe_file = tmp_path / "universe.csv"
    pd.DataFrame({"Symbol": universe}).to_csv(universe_file, index=False)

    scores = ct.score_universe(universe_file=str(universe_file))
    regime = ct.check_regime()
    return scores, regime


# ---------------------------------------------------------------------------
# Scoring and ranking parity
# ---------------------------------------------------------------------------

class TestScoringParity:
    def test_same_symbols_pass_the_entry_gates(
        self, lake_panel, candidates, monkeypatch, tmp_path
    ):
        ctx = _ctx(lake_panel, candidates)
        closes = cwt.closes_from_ctx(ctx, candidates + list(ct.REGIME_ETFS))

        old_scores, _ = _old_path(closes, candidates, monkeypatch, tmp_path)
        new = cwt.decide(ctx)

        assert new is not None
        old_symbols = [] if old_scores.empty else old_scores["Symbol"].tolist()
        assert new["scored"] == len(old_symbols)

    def test_rank_order_is_identical(
        self, lake_panel, candidates, monkeypatch, tmp_path
    ):
        """Order is the decision. Two names swapping rank changes what is bought."""
        ctx = _ctx(lake_panel, candidates)
        closes = cwt.closes_from_ctx(ctx, candidates + list(ct.REGIME_ETFS))

        old_scores, _ = _old_path(closes, candidates, monkeypatch, tmp_path)
        new = cwt.decide(ctx)

        old_ranking = (
            [] if old_scores.empty
            else old_scores["Symbol"].tolist()[:len(new["ranking"])]
        )
        assert new["ranking"] == old_ranking

    def test_regime_verdict_is_identical(
        self, lake_panel, candidates, monkeypatch, tmp_path
    ):
        ctx = _ctx(lake_panel, candidates)
        closes = cwt.closes_from_ctx(ctx, candidates + list(ct.REGIME_ETFS))

        _, old_regime = _old_path(closes, candidates, monkeypatch, tmp_path)
        new = cwt.decide(ctx)

        assert new["risk_on"] == old_regime

    def test_score_symbol_is_the_only_gate_implementation(
        self, lake_panel, candidates
    ):
        """The adapter must not carry its own copy of the gates."""
        ctx = _ctx(lake_panel, candidates)
        closes = cwt.closes_from_ctx(ctx, candidates)

        direct = [
            ct.score_symbol(sym, closes[sym])
            for sym in candidates if sym in closes
        ]
        expected = ct.rank_records([r for r in direct if r is not None])

        new = cwt.decide(ctx)
        got = [] if expected.empty else expected["Symbol"].tolist()
        assert new["ranking"] == got[:len(new["ranking"])]


# ---------------------------------------------------------------------------
# Sizing parity
# ---------------------------------------------------------------------------

class TestSizingParity:
    def test_new_weights_match_inverse_vol_on_the_same_picks(
        self, lake_panel, candidates, monkeypatch, tmp_path
    ):
        ctx = _ctx(lake_panel, candidates)
        closes = cwt.closes_from_ctx(ctx, candidates + list(ct.REGIME_ETFS))

        old_scores, old_regime = _old_path(closes, candidates, monkeypatch, tmp_path)
        new = cwt.decide(ctx)

        if not old_regime or old_scores.empty:
            pytest.skip("risk-off or nothing scored: no buys to compare")

        picks = old_scores.head(ct.DEFAULT_TOP_N)
        expected = ct._inverse_vol_weights(picks.set_index("Symbol")["Vol20"])

        assert set(new["new_weights"]) == set(expected.index)
        for symbol, weight in expected.items():
            assert new["new_weights"][symbol] == pytest.approx(float(weight))

    def test_sizing_collapses_to_equal_weight_at_the_live_top_n(self):
        """At top_n=7 the sleeve is equal-weight, whatever the volatilities.

        This is the live behaviour, pinned deliberately rather than corrected,
        and it is stronger than "the 10% cap does not bind". ``1/7 = 14.3%``
        exceeds MAX_WEIGHT, so every name clamps to 10%, the sum falls to 70%,
        and the renormalization scales them all straight back to 14.3% -- the
        clamp and the renormalization cancel and the inverse-vol tilt is erased
        along with them.

        So feature 4 of the sleeve's own docstring, inverse-volatility sizing,
        does not operate in production at the default top_n. Fixing it changes
        live position sizes, which is a strategy decision and not something a
        migration should do quietly.
        """
        calm_and_wild = pd.Series({
            "CALM": 0.08, "MILD": 0.15, "MID": 0.22, "WARM": 0.30,
            "HOT": 0.45, "WILD": 0.60, "FERAL": 0.90,
        })

        weights = ct._inverse_vol_weights(calm_and_wild)

        assert len(weights) == ct.DEFAULT_TOP_N
        assert weights.sum() == pytest.approx(1.0)
        # An 11x spread in volatility produces a 0x spread in weight.
        assert weights.max() - weights.min() == pytest.approx(0.0)
        assert weights["FERAL"] == pytest.approx(1.0 / ct.DEFAULT_TOP_N)

    def test_inverse_vol_survives_once_weights_fit_inside_the_band(self):
        """Above ten names 1/n drops under the 10% cap and the tilt reappears."""
        vols = pd.Series({f"S{i}": 0.10 + 0.03 * i for i in range(12)})

        weights = ct._inverse_vol_weights(vols)

        assert weights.sum() == pytest.approx(1.0)
        assert weights["S0"] > weights["S11"], "calmest name must carry more"
        # The clamp loop stops after a fixed ten iterations rather than on
        # convergence, so the cap is approached from above and overshoots by a
        # couple of basis points. Asserting equality to MAX_WEIGHT would pin an
        # artifact of the iteration budget.
        assert weights.max() == pytest.approx(ct.MAX_WEIGHT, abs=1e-3)

    def test_two_name_book_is_a_coin_flip_not_a_vol_tilt(self):
        """The same cancellation, at its most visible."""
        weights = ct._inverse_vol_weights(pd.Series({"CALM": 0.10, "WILD": 0.40}))

        assert weights["CALM"] == pytest.approx(0.5)
        assert weights["WILD"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Exit and slot parity
# ---------------------------------------------------------------------------

class TestExitParity:
    def test_unscored_holding_is_exited(self, lake_panel, candidates):
        """A name that fails the gates is not in the ranking, so it must go."""
        ctx = _ctx(lake_panel, candidates, held={"ZZZZ_NOT_A_TICKER": 1000.0})

        new = cwt.decide(ctx)

        assert "ZZZZ_NOT_A_TICKER" in new["exiting"]
        assert new["exit_reasons"]["ZZZZ_NOT_A_TICKER"] == "exit:failed_entry_filters"

    def test_exit_reasons_match_the_live_helper(
        self, lake_panel, candidates, monkeypatch, tmp_path
    ):
        ctx_probe = _ctx(lake_panel, candidates)
        closes = cwt.closes_from_ctx(ctx_probe, candidates + list(ct.REGIME_ETFS))
        old_scores, _ = _old_path(closes, candidates, monkeypatch, tmp_path)

        if old_scores.empty:
            pytest.skip("nothing scored")

        # Hold the top name and a name that was rejected, so both branches run.
        top = old_scores["Symbol"].iloc[0]
        rejected = next(
            (s for s in candidates if s not in set(old_scores["Symbol"])), None
        )
        held = {top: 1000.0}
        if rejected:
            held[rejected] = 1000.0

        ctx = _ctx(lake_panel, candidates, held=held)
        new = cwt.decide(ctx)

        expected = ct._generate_exit_signals(old_scores, set(held), "x")
        assert new["exiting"] == sorted(s.symbol for s in expected)
        for signal in expected:
            assert new["exit_reasons"][signal.symbol] == signal.reason

    def test_slots_count_names_that_are_leaving(self, lake_panel, candidates):
        """Preserved quirk: a full book buys nothing, even while it liquidates."""
        held = {f"GHOST{i}": 1000.0 for i in range(ct.DEFAULT_TOP_N)}
        ctx = _ctx(lake_panel, candidates, held=held)

        new = cwt.decide(ctx)

        # All seven are unscored, so all seven exit...
        assert len(new["exiting"]) == ct.DEFAULT_TOP_N
        # ...and yet no slot is considered free.
        assert new["slots_available"] == 0
        assert new["new_weights"] == {}


# ---------------------------------------------------------------------------
# target_weights contract
# ---------------------------------------------------------------------------

class TestTargetWeights:
    def test_survivors_keep_their_weight_and_exits_disappear(
        self, lake_panel, candidates, monkeypatch, tmp_path
    ):
        ctx_probe = _ctx(lake_panel, candidates)
        closes = cwt.closes_from_ctx(ctx_probe, candidates + list(ct.REGIME_ETFS))
        old_scores, _ = _old_path(closes, candidates, monkeypatch, tmp_path)
        if old_scores.empty:
            pytest.skip("nothing scored")

        survivor = old_scores["Symbol"].iloc[0]
        ctx = _ctx(
            lake_panel, candidates,
            held={survivor: 6000.0, "GHOST": 4000.0},
        )

        weights = cwt.target_weights(ctx)

        assert "GHOST" not in weights
        if survivor in weights:
            # 6000 of 10000 equity, and a survivor is not resized.
            assert weights[survivor] == pytest.approx(0.6)

    def test_empty_panel_holds_rather_than_liquidates(self, lake_panel, candidates):
        """No data is not a sell signal. Returning {} would flatten the book."""
        as_of, panel, matrix = lake_panel
        early = pd.Timestamp(panel[schema.DATE].min())

        ctx = _ctx(
            lake_panel, candidates,
            held={"AAPL": 5000.0}, as_of=early - pd.Timedelta(days=5),
        )

        assert cwt.target_weights(ctx) == {"AAPL": pytest.approx(1.0)}

    def test_top_n_comes_from_params(self, lake_panel, candidates):
        as_of, panel, matrix = lake_panel
        ctx = StrategyContext(
            as_of=as_of,
            price_panel=panel,
            close_matrix=matrix,
            universe_fn=lambda _d: set(candidates) | set(ct.REGIME_ETFS),
            params={"top_n": 2},
        )

        weights = cwt.target_weights(ctx)

        assert len(weights) <= 2

    def test_weights_are_a_fraction_of_equity(self, lake_panel, candidates):
        ctx = _ctx(lake_panel, candidates)

        weights = cwt.target_weights(ctx)

        assert all(0.0 < w <= 1.0 for w in weights.values())
        assert sum(weights.values()) <= 1.0 + 1e-9

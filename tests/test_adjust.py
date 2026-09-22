"""Tests for as-of-date price adjustment (data_pipeline.adjust).

No network. Every test builds its own lake in a temp directory.

The property under test: ``adj_close`` in the lake is back-adjusted using every
split and dividend up to the *download* date. Replaying 2015 with today's
``adj_close`` means the 2015 price *levels* already know about a 2020 split.
Returns survive that; levels do not. Anything comparing price to an absolute
dollar figure -- a ``MIN_PRICE`` screen, whole-share rounding, a notional cap --
is then reasoning about a price that never printed.

``adjust`` re-anchors the adjustment to a chosen date so levels are honest
again, and these tests pin the two things that must both hold:

* on the anchor date the series returns the genuine raw print, and
* returns are bit-for-bit unchanged, because the anchor cancels in a ratio.

Run with: ./venv/bin/python -m pytest tests/test_adjust.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_pipeline import adjust, schema, store  # noqa: E402
from data_pipeline.schema import (  # noqa: E402
    ADJ_CLOSE,
    CLOSE,
    COLUMNS,
    DATE,
    SYMBOL,
)


def bars(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def row(date: str, symbol: str, close: float, adj: float, vol: int = 1_000_000):
    """One bar where OHLC are all `close` and adj_close is stated separately."""
    return (date, symbol, close, close, close, close, adj, vol)


class SplitFixtureMixin:
    """A lake holding one 2-for-1 split, written to a temp directory.

    ACME trades at a true $100 for three sessions, splits 2-for-1, then trades
    at a true $50. Post-split the vendor restates the pre-split bars, so the
    lake stores raw ``close`` of 100 alongside an ``adj_close`` of 50.

    Economically nothing happened on the split date: one $100 share became two
    $50 shares. Any measure that reports a -50% move there is wrong.
    """

    SPLIT_DATE = "2024-01-04"

    def build_lake(self, root: Path) -> None:
        frame = bars(
            [
                # Pre-split: raw print $100, restated to $50 by the 2:1 split.
                row("2024-01-02", "ACME", 100.0, 50.0),
                row("2024-01-03", "ACME", 102.0, 51.0),
                # Split effective. Raw print halves; adjusted series continues.
                row("2024-01-04", "ACME", 52.0, 52.0),
                row("2024-01-05", "ACME", 53.0, 53.0),
                # A symbol with no corporate actions: adj_close == close always.
                row("2024-01-02", "PLAIN", 20.0, 20.0),
                row("2024-01-03", "PLAIN", 21.0, 21.0),
                row("2024-01-04", "PLAIN", 22.0, 22.0),
                row("2024-01-05", "PLAIN", 23.0, 23.0),
            ]
        )
        schema.write_year(frame, 2024, root, hot=True)


class AnchorDateTests(SplitFixtureMixin, unittest.TestCase):
    """On the anchor date the adjusted series must equal the raw print."""

    def test_anchor_date_returns_raw_print(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            # Anchor before the split: the $100 print must come back as $100,
            # not as the $50 that today's adj_close reports.
            asof = pd.Timestamp("2024-01-03")
            out = adjust.load_prices_asof(["ACME"], as_of=asof, root=root)

            on_anchor = out[out[DATE] == asof].iloc[0]
            self.assertAlmostEqual(float(on_anchor[CLOSE]), 102.0, places=6)
            self.assertAlmostEqual(float(on_anchor[ADJ_CLOSE]), 102.0, places=6)

    def test_pre_split_levels_are_not_retroactively_halved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            asof = pd.Timestamp("2024-01-03")
            out = adjust.load_prices_asof(["ACME"], as_of=asof, root=root)
            by_date = out.set_index(DATE)[ADJ_CLOSE]

            # Standing on 2024-01-03, the split has not happened yet. Both
            # sessions must show the prints that actually traded.
            self.assertAlmostEqual(float(by_date.loc["2024-01-02"]), 100.0, places=6)
            self.assertAlmostEqual(float(by_date.loc["2024-01-03"]), 102.0, places=6)

            # The unanchored lake is what we are protecting against: it reports
            # the restated $50/$51, which is a price that never traded.
            raw = store.load_prices(["ACME"], root=root).set_index(DATE)[ADJ_CLOSE]
            self.assertAlmostEqual(float(raw.loc["2024-01-02"]), 50.0, places=6)

    def test_anchor_after_split_matches_lake(self):
        """Anchoring at the newest session reproduces the stored adjustment."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            asof = pd.Timestamp("2024-01-05")
            out = adjust.load_prices_asof(["ACME"], as_of=asof, root=root)
            anchored = out.set_index(DATE)[ADJ_CLOSE]

            raw = store.load_prices(["ACME"], root=root).set_index(DATE)[ADJ_CLOSE]

            for date in ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"):
                self.assertAlmostEqual(
                    float(anchored.loc[date]), float(raw.loc[date]), places=6,
                    msg=f"anchoring at the last session should be a no-op on {date}",
                )


class ReturnInvarianceTests(SplitFixtureMixin, unittest.TestCase):
    """Re-anchoring must not disturb returns -- the anchor cancels in a ratio."""

    def test_returns_identical_across_anchor_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            early = adjust.load_close_matrix_asof(
                ["ACME"], as_of=pd.Timestamp("2024-01-03"), root=root
            )["ACME"].pct_change().dropna()

            late = adjust.load_close_matrix_asof(
                ["ACME"], as_of=pd.Timestamp("2024-01-05"), root=root
            )["ACME"].pct_change().dropna()

            common = early.index.intersection(late.index)
            self.assertGreater(len(common), 0, "fixture should share sessions")

            for date in common:
                self.assertAlmostEqual(
                    float(early.loc[date]), float(late.loc[date]), places=10,
                    msg=f"anchor date changed the return on {date}",
                )

    def test_split_is_not_a_price_move(self):
        """The split date must not register as a -50% return."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            closes = adjust.load_close_matrix_asof(
                ["ACME"], as_of=pd.Timestamp("2024-01-05"), root=root
            )["ACME"]
            returns = closes.pct_change().dropna()

            split_return = float(returns.loc[pd.Timestamp(self.SPLIT_DATE)])
            self.assertLess(
                abs(split_return), 0.05,
                msg=f"2-for-1 split surfaced as a {split_return:.1%} return",
            )


class PassthroughTests(SplitFixtureMixin, unittest.TestCase):
    def test_no_asof_matches_store(self):
        """as_of=None must be exactly store.load_prices -- no silent rescaling."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            plain = store.load_prices(["ACME"], root=root).reset_index(drop=True)
            passthrough = adjust.load_prices_asof(["ACME"], root=root).reset_index(drop=True)

            pd.testing.assert_frame_equal(plain, passthrough)

    def test_symbol_without_corporate_actions_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            out = adjust.load_prices_asof(
                ["PLAIN"], as_of=pd.Timestamp("2024-01-03"), root=root
            ).set_index(DATE)

            # adj_close == close throughout, so every factor is 1.0.
            self.assertAlmostEqual(float(out.loc["2024-01-02", ADJ_CLOSE]), 20.0, places=6)
            self.assertAlmostEqual(float(out.loc["2024-01-03", ADJ_CLOSE]), 21.0, places=6)

    def test_asof_truncates_future_sessions(self):
        """Nothing after as_of may come back -- this is the firewall's data side."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            asof = pd.Timestamp("2024-01-03")
            out = adjust.load_prices_asof(["ACME"], as_of=asof, root=root)

            self.assertTrue(
                (out[DATE] <= asof).all(),
                msg="load_prices_asof leaked sessions after as_of",
            )


class OhlcConsistencyTests(SplitFixtureMixin, unittest.TestCase):
    def test_ohlc_scaled_on_the_same_factor_as_close(self):
        """Open/high/low must move with close, or intraday logic silently breaks."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_lake(root)

            out = adjust.load_prices_asof(
                ["ACME"], as_of=pd.Timestamp("2024-01-03"), root=root
            )
            pre_split = out[out[DATE] == pd.Timestamp("2024-01-02")].iloc[0]

            # The fixture sets OHLC all equal, so they must stay equal after
            # re-anchoring; a per-column factor bug would break this.
            for col in (schema.OPEN, schema.HIGH, schema.LOW, schema.CLOSE):
                self.assertAlmostEqual(
                    float(pre_split[col]), 100.0, places=6,
                    msg=f"{col} did not track close through re-anchoring",
                )


if __name__ == "__main__":
    unittest.main()

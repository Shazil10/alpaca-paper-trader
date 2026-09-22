"""Strategy context — the point-in-time data firewall.

Nothing inside a strategy may touch store, yfinance, or datetime.today()
directly. All data arrives through this context object, already truncated
at as_of. That is the difference between a backtester and a plausible lie.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, Optional, Sequence, Set

import pandas as pd
import numpy as np

from backtest.types import Position, PortfolioSnapshot

logger = logging.getLogger(__name__)


class LookaheadError(Exception):
    """Raised when a strategy attempts to access future data."""
    pass


class StrategyContext:
    """Immutable view of the world as of a specific date.

    The engine creates a new context each simulation day. The strategy
    receives it and returns target weights. The strategy CANNOT:
    - See prices after as_of
    - See future universe membership
    - Access the real clock
    - Modify the context

    The full price panel is loaded once at engine start (a 20-year,
    1500-symbol panel is ~7.5M cells — fits in RAM). Each day, the context
    presents a truncated *view* of that panel, not a re-read.
    """

    def __init__(
        self,
        as_of: pd.Timestamp,
        price_panel: pd.DataFrame,          # long format, full history
        close_matrix: pd.DataFrame,          # date x symbol, full history
        ohlc_adjusted: Optional[Dict] = None,
        universe_fn=None,                    # callable(date) -> Set[str]
        portfolio: Optional[PortfolioSnapshot] = None,
        params: Optional[Dict[str, Any]] = None,
        trading_sessions: Optional[pd.DatetimeIndex] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ):
        self._as_of = pd.Timestamp(as_of).normalize()
        self._price_panel = price_panel
        self._close_matrix = close_matrix
        self._ohlc_adjusted = ohlc_adjusted or {}
        self._universe_fn = universe_fn
        self._portfolio = portfolio or PortfolioSnapshot(
            date=self._as_of, cash=0.0, equity=0.0
        )
        self._params = dict(params or {})
        self._trading_sessions = trading_sessions
        self._sector_map = sector_map or {}

    @property
    def as_of(self) -> pd.Timestamp:
        """Current simulation date."""
        return self._as_of

    @property
    def portfolio(self) -> PortfolioSnapshot:
        """Current portfolio state (positions, cash, equity)."""
        return self._portfolio

    @property
    def equity(self) -> float:
        return self._portfolio.equity

    @property
    def cash(self) -> float:
        return self._portfolio.cash

    @property
    def positions(self) -> Dict[str, Position]:
        return self._portfolio.positions

    @property
    def held_symbols(self) -> FrozenSet[str]:
        return self._portfolio.symbols

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    def _enforce_no_lookahead(self, end: Optional[pd.Timestamp], label: str):
        """Hard gate: no data past as_of."""
        if end is not None and end > self._as_of:
            raise LookaheadError(
                f"{label}: requested end={end:%Y-%m-%d} is after "
                f"as_of={self._as_of:%Y-%m-%d}. This is look-ahead bias."
            )

    def prices(
        self,
        symbols: Optional[Sequence[str]] = None,
        lookback: Optional[int] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Date x symbol matrix of adj_close, truncated at as_of.

        Args:
            symbols: restrict to these tickers; None = all available.
            lookback: number of trading sessions to return (from as_of backward).
            start: inclusive start date.
            end: inclusive end date (must be <= as_of).

        Returns:
            DataFrame with DatetimeIndex rows and symbol columns.

        Raises:
            LookaheadError if end > as_of.
        """
        effective_end = end or self._as_of
        self._enforce_no_lookahead(effective_end, "prices()")

        matrix = self._close_matrix

        # Truncate at as_of
        matrix = matrix.loc[matrix.index <= self._as_of]

        if start is not None:
            matrix = matrix.loc[matrix.index >= start]

        if end is not None:
            matrix = matrix.loc[matrix.index <= end]

        if symbols is not None:
            available = [s for s in symbols if s in matrix.columns]
            matrix = matrix[available]

        if lookback is not None and lookback > 0:
            matrix = matrix.iloc[-lookback:]

        return matrix

    def price_panel(
        self,
        symbols: Optional[Sequence[str]] = None,
        start: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Long-format price data truncated at as_of.

        Returns DataFrame with columns: date, symbol, open, high, low, close,
        adj_close, volume.
        """
        panel = self._price_panel
        panel = panel[panel["date"] <= self._as_of]

        if start is not None:
            panel = panel[panel["date"] >= start]

        if symbols is not None:
            syms = {s.upper() for s in symbols}
            panel = panel[panel["symbol"].isin(syms)]

        return panel

    def ohlc(
        self,
        symbol: str,
        lookback: Optional[int] = None,
    ) -> pd.DataFrame:
        """Adjusted OHLC for a single symbol, truncated at as_of.

        Returns DataFrame with columns Open, High, Low, Close and DatetimeIndex.
        """
        if symbol not in self._ohlc_adjusted:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

        df = self._ohlc_adjusted[symbol]
        df = df.loc[df.index <= self._as_of]

        if lookback is not None and lookback > 0:
            df = df.iloc[-lookback:]

        return df

    def universe(self) -> Set[str]:
        """PIT index members eligible on as_of.

        Falls back to all symbols in the price panel if no PIT membership
        is available.
        """
        if self._universe_fn is not None:
            return self._universe_fn(self._as_of)

        # Fallback: symbols with data on as_of
        panel = self._close_matrix
        row = panel.loc[panel.index <= self._as_of]
        if len(row) == 0:
            return set()
        last = row.iloc[-1]
        return set(last.dropna().index)

    def sessions(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DatetimeIndex:
        """Trading sessions up to as_of."""
        effective_end = end or self._as_of
        self._enforce_no_lookahead(effective_end, "sessions()")

        if self._trading_sessions is not None:
            idx = self._trading_sessions[self._trading_sessions <= self._as_of]
            if start is not None:
                idx = idx[idx >= start]
            if end is not None:
                idx = idx[idx <= end]
            return idx

        # Infer from close matrix
        dates = self._close_matrix.index[self._close_matrix.index <= self._as_of]
        if start is not None:
            dates = dates[dates >= start]
        return pd.DatetimeIndex(dates)

    def sector(self, symbol: str) -> str:
        """Return sector for a symbol, or '' if unknown."""
        return self._sector_map.get(symbol.upper(), "")

    def has_lookback(self, symbols: Sequence[str], sessions: int) -> bool:
        """True when every symbol has at least `sessions` bars through as_of."""
        matrix = self.prices(symbols)
        for sym in symbols:
            s = sym.upper()
            if s not in matrix.columns:
                return False
            if int(matrix[s].notna().sum()) < sessions:
                return False
        return True

    def today_bar(self, symbol: str) -> Optional[pd.Series]:
        """Return the bar for symbol on as_of, or None."""
        panel = self._price_panel
        mask = (panel["date"] == self._as_of) & (panel["symbol"] == symbol.upper())
        rows = panel[mask]
        if len(rows) == 0:
            return None
        return rows.iloc[0]

    def latest_close(self, symbol: str) -> Optional[float]:
        """Latest adj_close for symbol on or before as_of."""
        if symbol not in self._close_matrix.columns:
            return None
        col = self._close_matrix[symbol]
        col = col.loc[col.index <= self._as_of].dropna()
        if len(col) == 0:
            return None
        return float(col.iloc[-1])

    def raw_close(self, symbol: str) -> Optional[float]:
        """Latest *unadjusted* close on or before as_of — the price as printed.

        Use this, not ``latest_close``, for any comparison against an absolute
        dollar amount: minimum-price screens, round-lot sizing, dollar-volume
        filters. ``adj_close`` is back-adjusted using every split and dividend
        up to the day the data was downloaded, so on 2010-06-30 adjusted AAPL
        reads about $8 while the actual print was $251. A ``price >= 10`` screen
        on adjusted levels silently excludes names that were never cheap.

        Returns are unaffected by the adjustment and should keep using
        ``prices()``; only levels are corrupted.
        """
        panel = self._price_panel
        mask = (panel["symbol"] == symbol.upper()) & (panel["date"] <= self._as_of)
        rows = panel.loc[mask]
        if len(rows) == 0:
            return None
        rows = rows.sort_values("date")
        value = rows.iloc[-1].get("close")
        if value is None or pd.isna(value):
            return None
        return float(value)

    def raw_closes(self, symbols: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """{symbol: unadjusted close} as of as_of. See ``raw_close``."""
        panel = self._price_panel
        panel = panel[panel["date"] <= self._as_of]
        if symbols is not None:
            wanted = {s.upper() for s in symbols}
            panel = panel[panel["symbol"].isin(wanted)]
        if len(panel) == 0:
            return {}
        latest = (
            panel.sort_values("date")
            .groupby("symbol")["close"]
            .last()
            .dropna()
        )
        return {str(k): float(v) for k, v in latest.items()}

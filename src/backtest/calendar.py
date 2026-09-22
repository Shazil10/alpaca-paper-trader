"""Trading session calendar.

Wraps store.trading_calendar for the backtester, caching the session index
so it is computed once rather than per-day.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from data_pipeline import store


class TradingCalendar:
    """Cached trading session index derived from the price lake."""

    def __init__(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        *,
        root=None,
    ):
        self._sessions = store.trading_calendar(start, end, root=root)
        self._start = pd.Timestamp(start) if start else None
        self._end = pd.Timestamp(end) if end else None

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def __iter__(self):
        return iter(self._sessions)

    def __contains__(self, date) -> bool:
        return pd.Timestamp(date) in self._sessions

    def slice(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DatetimeIndex:
        """Return sessions in [start, end]."""
        idx = self._sessions
        if start is not None:
            idx = idx[idx >= start]
        if end is not None:
            idx = idx[idx <= end]
        return idx

    def next_session(self, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Return the first session strictly after date, or None."""
        later = self._sessions[self._sessions > date]
        if len(later) == 0:
            return None
        return later[0]

    def prev_session(self, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Return the last session strictly before date, or None."""
        earlier = self._sessions[self._sessions < date]
        if len(earlier) == 0:
            return None
        return earlier[-1]

    def session_index(self, date: pd.Timestamp) -> int:
        """Return the 0-based index of date in the session list. -1 if missing."""
        try:
            return int(self._sessions.get_loc(date))
        except KeyError:
            return -1

    def is_first_of_month(self, date: pd.Timestamp) -> bool:
        """True if date is the first trading session of its calendar month."""
        month_sessions = self._sessions[
            (self._sessions.year == date.year)
            & (self._sessions.month == date.month)
        ]
        return len(month_sessions) > 0 and month_sessions[0] == date

    def is_last_of_month(self, date: pd.Timestamp) -> bool:
        """True if date is the last trading session of its calendar month."""
        month_sessions = self._sessions[
            (self._sessions.year == date.year)
            & (self._sessions.month == date.month)
        ]
        return len(month_sessions) > 0 and month_sessions[-1] == date

    def sessions_between(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Count of trading sessions in [start, end] inclusive."""
        return int(((self._sessions >= start) & (self._sessions <= end)).sum())

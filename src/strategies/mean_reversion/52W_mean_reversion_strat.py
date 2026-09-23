"""
STRATEGY: 52-Week High Pullback Mean Reversion (R2 v2)
=======================================================

Buys stocks that have pulled back >= 40% from their 52-week high,
then sells once they recover to within 25% of that high.

R2 v2 parameters (finalised in 52W_mean_reversion.ipynb):
  Entry depth         : >= 40% below 52-week high
  Quality gate F2     : price >= 50% of 52-week high (not in freefall)
  Quality gate F3     : price >= 90% of 200-day MA (not structurally broken)
  Regime gate R2      : SPY + IJH + IJR, >= 2 of 3 above 200-day SMA
  Exit target         : within 25% of 52-week high (cur_pullback <= 0.25)
  Stop-loss           : -25% from entry price (uses Alpaca avg_entry_price)
  Max hold            : 63 trading days (~3 months)
  Max positions       : 5 concurrent
  Cooldown            : 20 days after any exit, 30 days after stop-loss
  Sizing              : pullback-weighted (deeper pullback → more allocation)
  Budget              : $10,000 (set in config.STRATEGY_ALLOCATIONS)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yfinance as yf

from data_pipeline import source as price_source
from data_pipeline import store
from trade_models import Signal, Side


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (R2 v2 final)
# ---------------------------------------------------------------------------
HIGH_LOOKBACK         = 252    # 52-week high lookback (trading days)
ENTRY_DEPTH           = 0.40   # stock must be >= 40% below 52w high
EXIT_PULLBACK         = 0.25   # exit when within 25% of 52w high
STOP_FROM_ENTRY       = 0.25   # exit if down 25% from our entry price
MAX_HOLD_DAYS         = 63     # exit after ~3 months (calendar day equivalent: 90)
TOP_N                 = 5      # max concurrent positions

COOLDOWN_DAYS         = 20     # days before re-entering same ticker after exit
STOP_ENTRY_COOLDOWN   = 30     # days before re-entering after a stop-loss

MA_FREEFALL_RATIO     = 0.50   # F2: price must be >= 50% of 52w high
MA_BROKEN_RATIO       = 0.90   # F3: price must be >= 90% of 200d MA
MA_LOOKBACK           = 200    # 200-day moving average

REGIME_ETFS           = ("SPY", "IJH", "IJR")
REGIME_SMA_PERIOD     = 200
REGIME_MIN_ABOVE      = 2      # at least 2 of 3 ETFs above 200-SMA → risk-on

DATA_PERIOD           = "2y"   # enough for 252-day high + 200-day MA

#: Explicit calendar-day equivalents of DATA_PERIOD / the regime's period="1y".
#: Pinned so the download path and the lake path request the *identical* window;
#: every window here is a trailing slice (``iloc[-n:]``, ``min(252, len(s))``), so
#: unlike the rotation sleeve there is no expanding() to protect -- but keeping
#: the windows equal is what makes the canary a real comparison.
LOOKBACK_DAYS         = 730
REGIME_LOOKBACK_DAYS  = 365


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window(days: int, as_of: Optional[pd.Timestamp] = None) -> tuple:
    """Return an inclusive (start, end) window ``days`` wide."""
    end = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now().normalize()
    return end - pd.Timedelta(days=days), end


def _close_panel(
    symbols: List[str],
    days: int,
    as_of: Optional[pd.Timestamp] = None,
    *,
    context: str,
) -> pd.DataFrame:
    """Load a date x symbol panel of adjusted closes from the active source.

    Returns the same shape either way, so callers need no source-specific code.
    ``adj_close`` is what the download path produced via ``auto_adjust=True``'s
    ``Close``; there is no fallback between sources.
    """
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not symbols:
        return pd.DataFrame(dtype="float64")

    start, end = _window(days, as_of)

    if price_source.using_lake():
        panel = store.load_close_matrix(symbols, start=start, end=end)
        last = None if panel.empty else pd.Timestamp(panel.index[-1])
        price_source.log_staleness(last, context=context)
        return panel

    data = yf.download(
        symbols,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame(dtype="float64")

    out: Dict[str, pd.Series] = {}
    multi = isinstance(data.columns, pd.MultiIndex)
    for sym in symbols:
        try:
            sub = data[sym] if multi else data
            series = sub["Close"]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.dropna()
            if len(series) > 0:
                out[sym] = series
        except (KeyError, IndexError):
            continue

    if not out:
        return pd.DataFrame(dtype="float64")
    return pd.DataFrame(out).sort_index()


def _sma(series: pd.Series, period: int) -> float:
    """Return simple moving average of the last *period* values."""
    if len(series) < period:
        return float("nan")
    return float(series.iloc[-period:].mean())


def _high_52w(series: pd.Series) -> float:
    """Return the 252-day rolling high."""
    n = min(HIGH_LOOKBACK, len(series))
    if n < 10:
        return float("nan")
    return float(series.iloc[-n:].max())


# ---------------------------------------------------------------------------
# 1. Regime filter (R2 Trend Stack)
# ---------------------------------------------------------------------------

def regime_from_panel(panel: pd.DataFrame) -> bool:
    """Risk-on when >= 2 of SPY/IJH/IJR close above their 200-day SMA.

    The decision, separated from where the panel came from, so the live path and
    the backtest adapter cannot drift apart. An empty panel defaults to risk-ON:
    missing data is an infrastructure condition, not a market signal, and the
    sleeve's whole asymmetry is that failing to act is recoverable while acting
    wrongly is not. An ETF short of ``REGIME_SMA_PERIOD`` bars abstains rather
    than voting against.
    """
    if panel is None or panel.empty:
        logger.warning("Regime check got no data. Defaulting to risk-ON.")
        return True

    above = 0
    for etf in REGIME_ETFS:
        if etf not in panel.columns:
            continue
        s = panel[etf].dropna()
        if len(s) < REGIME_SMA_PERIOD:
            continue
        if float(s.iloc[-1]) > _sma(s, REGIME_SMA_PERIOD):
            above += 1

    risk_on = above >= REGIME_MIN_ABOVE
    logger.info(
        "R2 regime: %d/%d ETFs above %d-SMA → %s",
        above, len(REGIME_ETFS), REGIME_SMA_PERIOD,
        "RISK-ON" if risk_on else "RISK-OFF",
    )
    return risk_on


def check_regime(as_of: Optional[pd.Timestamp] = None) -> bool:
    """Return True (risk-on) if >= 2 of SPY/IJH/IJR are above their 200-day SMA."""
    try:
        panel = _close_panel(
            list(REGIME_ETFS), REGIME_LOOKBACK_DAYS, as_of, context="pullback_regime"
        )
    except Exception:
        logger.warning("Regime check failed (data error). Defaulting to risk-ON.")
        return True

    return regime_from_panel(panel)


# ---------------------------------------------------------------------------
# 2. Score universe — find candidates
# ---------------------------------------------------------------------------

def _score_series(symbol: str, series: pd.Series) -> Optional[Dict]:
    """Apply the entry gates to one price series. Returns a record or None.

    Extracted so both the lake and download paths run byte-identical gate logic
    -- the gates are the decision, and they should live in exactly one place.
    """
    s = series.dropna()
    if len(s) < max(HIGH_LOOKBACK, MA_LOOKBACK):
        return None

    cur_price = float(s.iloc[-1])
    high52 = _high_52w(s)
    ma200 = _sma(s, MA_LOOKBACK)

    if cur_price <= 0 or np.isnan(high52) or np.isnan(ma200):
        return None

    pullback = (high52 - cur_price) / high52

    # Entry depth gate
    if pullback < ENTRY_DEPTH:
        return None

    # F2: not in freefall
    if cur_price < MA_FREEFALL_RATIO * high52:
        return None

    # F3: not structurally broken
    if cur_price < MA_BROKEN_RATIO * ma200:
        return None

    return {
        "symbol": symbol,
        "pullback": round(pullback, 4),
        "price": round(cur_price, 4),
        "high52": round(high52, 4),
        "ma200": round(ma200, 4),
    }


def _score_universe(
    universe_file: str, as_of: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Return DataFrame of stocks meeting entry criteria, ranked by pullback depth."""
    if not os.path.exists(universe_file):
        universe_file = os.path.join(os.path.dirname(__file__), "../../../universe.csv")
    if not os.path.exists(universe_file):
        logger.error("universe.csv not found")
        return pd.DataFrame()

    tickers = [
        str(t).strip().upper()
        for t in pd.read_csv(universe_file)["Symbol"].tolist()
        if str(t).strip()
    ]
    records: List[Dict] = []

    if price_source.using_lake():
        # One read serves the whole universe; no chunking needed off local disk.
        try:
            panel = _close_panel(
                tickers, LOOKBACK_DAYS, as_of, context="pullback_universe"
            )
        except Exception:
            logger.exception("Lake read failed for universe scoring")
            return pd.DataFrame()

        for sym in panel.columns:
            record = _score_series(str(sym), panel[sym])
            if record is not None:
                records.append(record)

        logger.info(
            "Scored %d symbol(s) from the lake (%d requested, %d candidates)",
            len(panel.columns), len(tickers), len(records),
        )
    else:
        chunk_size = 300
        for i in range(0, len(tickers), chunk_size):
            batch = tickers[i: i + chunk_size]
            try:
                panel = _close_panel(
                    batch, LOOKBACK_DAYS, as_of, context="pullback_universe"
                )
            except Exception:
                logger.exception("Download failed for batch %d:%d", i, i + chunk_size)
                continue

            for sym in panel.columns:
                record = _score_series(str(sym), panel[sym])
                if record is not None:
                    records.append(record)

        logger.info(
            "Scored %d requested symbol(s) via yfinance (%d candidates)",
            len(tickers), len(records),
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values("pullback", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Entry date lookup (for timeout check)
# ---------------------------------------------------------------------------

def _get_entry_dates(
    client,
    strategy_id: str,
    held_symbols: Set[str],
) -> Dict[str, datetime]:
    """Return {symbol: earliest_buy_fill_datetime} from order history."""
    entry_dates: Dict[str, datetime] = {}
    if not held_symbols:
        return entry_dates

    prefix = f"{strategy_id}:"
    try:
        from trade_models import fetch_all_orders
        orders = fetch_all_orders(client, status="all")
    except Exception:
        logger.exception("Failed to fetch order history for entry-date lookup")
        return entry_dates

    for o in orders:
        cid    = str(getattr(o, "client_order_id", "") or "")
        if not cid.startswith(prefix):
            continue
        side   = str(getattr(o, "side", "")).upper()
        status = str(getattr(o, "status", "")).upper()
        sym    = str(getattr(o, "symbol", "")).strip().upper()
        if "BUY" not in side or "FILLED" not in status or sym not in held_symbols:
            continue
        filled_at = getattr(o, "filled_at", None) or getattr(o, "submitted_at", None)
        if filled_at is None:
            continue
        try:
            ts = pd.Timestamp(str(filled_at))
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            dt_raw = ts.to_pydatetime()
            if dt_raw is pd.NaT or dt_raw is None:
                continue
            dt = dt_raw.replace(tzinfo=None)
        except Exception:
            continue
        if sym not in entry_dates or dt < entry_dates[sym]:
            entry_dates[sym] = dt

    return entry_dates


# ---------------------------------------------------------------------------
# 4. Exit signals for held positions
# ---------------------------------------------------------------------------

def exit_reason(
    *,
    pullback: float,
    price: float,
    entry_price: Optional[float] = None,
    entry_date: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Return the exit reason for one held position, or None to keep holding.

    The three rules, in the order they are checked -- and the order is the
    decision, because the first match wins and the reason string is what gets
    recorded:

    1. Target: recovered to within ``EXIT_PULLBACK`` of the 52-week high.
    2. Stop: down ``STOP_FROM_ENTRY`` from our own entry price.
    3. Timeout: held ``MAX_HOLD_DAYS`` trading days.

    Only rule 1 needs market data. Rules 2 and 3 need to know what *we* paid and
    when, which live reads from the Alpaca position and the order history, and a
    backtest reads from ``ctx.portfolio``. Same arithmetic, different source --
    which is the point of taking them as arguments here.

    The timeout converts calendar days to approximate trading days with the
    252/365 ratio the live sleeve uses. That is an approximation, and it is kept
    rather than corrected so the two paths agree.
    """
    if pullback <= EXIT_PULLBACK:
        return f"exit:target_recovery_pb{pullback:.2f}"

    if entry_price is not None and float(entry_price) > 0:
        loss_pct = (float(price) - float(entry_price)) / float(entry_price)
        if loss_pct <= -STOP_FROM_ENTRY:
            return f"exit:stop_entry_{-loss_pct:.1%}_below_entry"

    if entry_date is not None and now is not None:
        calendar_days = (now - entry_date).days
        approx_trading_days = int(calendar_days * 252 / 365)
        if approx_trading_days >= MAX_HOLD_DAYS:
            return f"exit:timeout_{approx_trading_days}d"

    return None


def _generate_exit_signals(
    scores_df: pd.DataFrame,
    held_symbols: Set[str],
    strategy_id: str,
    universe_file: str,
    as_of: Optional[pd.Timestamp] = None,
) -> List[Signal]:
    """Generate SELL signals for positions that hit target, stop, or timeout."""
    if not held_symbols:
        return []

    # Get Alpaca positions for entry price (stop-loss check)
    alpaca_positions: Dict[str, object] = {}
    entry_dates: Dict[str, datetime] = {}
    try:
        import config as _cfg
        client = _cfg.get_client()
        for p in client.get_all_positions():
            sym = str(getattr(p, "symbol", "")).strip().upper()
            if sym:
                alpaca_positions[sym] = p
        entry_dates = _get_entry_dates(client, strategy_id, held_symbols)
    except Exception:
        logger.warning("Could not fetch Alpaca positions/orders — stop/timeout checks may be skipped")

    # Current prices + 52w high for held stocks
    held_list = sorted(held_symbols)
    current_data: Dict[str, Dict] = {}
    try:
        panel = _close_panel(
            held_list, LOOKBACK_DAYS, as_of, context="pullback_exits"
        )
        for sym in held_list:
            if sym not in panel.columns:
                continue
            s = panel[sym].dropna()
            if len(s) < 10:
                continue
            cur_price = float(s.iloc[-1])
            high52 = _high_52w(s)
            if cur_price > 0 and not np.isnan(high52):
                pullback = (high52 - cur_price) / high52
                current_data[sym] = {"price": cur_price, "high52": high52, "pullback": pullback}
    except Exception:
        logger.exception("Failed to load prices for exit checks")

    now = datetime.now(tz=None)
    signals: List[Signal] = []

    for sym in held_symbols:
        reason: Optional[str] = None
        info = current_data.get(sym)

        if info is None:
            # Deliberately does NOT liquidate. Absent price data is an
            # infrastructure condition, not a market signal, and it became far
            # more likely once the source moved to a local lake (a failed sync,
            # a name dropped from the universe). Selling on it would realise a
            # loss and pay spread because of a plumbing fault.
            #
            # Asymmetry: failing to exit is recoverable -- the next run exits.
            # Selling by mistake is not. Stop-loss and timeout below are also
            # skipped for this symbol, since both need a current price.
            logger.error(
                "PRICE_DATA_MISSING: no usable bars for held %s — holding "
                "position, exit checks deferred (source=%s)",
                sym, price_source.active_source(),
            )
            continue
        else:
            entry_price: Optional[float] = None
            if sym in alpaca_positions:
                try:
                    entry_price = float(
                        getattr(alpaca_positions[sym], "avg_entry_price", 0) or 0
                    )
                except Exception:
                    entry_price = None

            reason = exit_reason(
                pullback=info["pullback"],
                price=info["price"],
                entry_price=entry_price,
                entry_date=entry_dates.get(sym),
                now=now,
            )

        if reason:
            signals.append(
                Signal(
                    symbol=sym,
                    side=Side.SELL,
                    reason=reason,
                    notional=None,
                    strategy_id=strategy_id,
                )
            )
            logger.info("EXIT %s (%s)", sym, reason)

    return signals


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def generate_signals(
    *,
    universe_file: str = "universe.csv",
    top_n: int = TOP_N,
    budget: Optional[float] = None,
    strategy_id: str = "strategies.mean_reversion.high_pullback_reversion",
    held_symbols: Optional[Set[str]] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> List[Signal]:
    """Main entry point called by trade.py.

    Returns SELL signals (always) + BUY signals (only when R2 regime gate is ON).
    Sizing is pullback-weighted: deeper pullback → larger allocation.

    Args:
        universe_file: CSV with a ``Symbol`` column.
        top_n:         Max simultaneous positions.
        budget:        Remaining budget for new buys.
        strategy_id:   Attribution prefix for client_order_id.
        held_symbols:  Symbols currently held by this strategy.
    """
    if held_symbols is None:
        held_symbols = set()

    if not os.path.exists(universe_file):
        universe_file = os.path.join(os.path.dirname(__file__), "../../../universe.csv")

    # ── Step 1: Score universe (entry filters built-in) ──
    scores_df = _score_universe(universe_file, as_of)

    # ── Step 2: EXIT signals (always, even in risk-off) ──
    exit_signals = _generate_exit_signals(
        scores_df, held_symbols, strategy_id, universe_file, as_of
    )

    # ── Step 3: Regime gate ──
    if not check_regime(as_of):
        logger.info("R2 regime RISK-OFF → no new buys. Returning %d exit signals.", len(exit_signals))
        return exit_signals

    # ── Step 4: Pick top N candidates not already held ──
    if scores_df.empty:
        return exit_signals

    available = scores_df[~scores_df["symbol"].isin(list(held_symbols))].head(top_n)
    if available.empty:
        return exit_signals

    # ── Step 5: Pullback-weighted sizing ──
    pb_vals = available.set_index("symbol")["pullback"].clip(lower=1e-6)
    weights = pb_vals / pb_vals.sum()

    slots_available = max(top_n - len(held_symbols), 0)
    if slots_available == 0:
        logger.info("All %d slots filled — no new buys.", top_n)
        return exit_signals

    picks = available.head(slots_available)
    pb_picks = picks.set_index("symbol")["pullback"].clip(lower=1e-6)
    w_picks  = pb_picks / pb_picks.sum()

    buy_signals: List[Signal] = []
    for sym in w_picks.index:
        w       = float(w_picks[sym])
        notional: Optional[float] = None
        if budget is not None and float(budget) > 0:
            notional = float(budget) * w

        buy_signals.append(
            Signal(
                symbol=sym,
                side=Side.BUY,
                reason=f"52w_pullback_{float(pb_picks[sym]):.0%}",
                notional=notional,
                strategy_id=strategy_id,
            )
        )

    logger.info(
        "Generated %d BUY signals and %d EXIT signals (budget=%.2f)",
        len(buy_signals),
        len(exit_signals),
        float(budget or 0),
    )

    return exit_signals + buy_signals

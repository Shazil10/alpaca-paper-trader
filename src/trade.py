"""trade.py

Runs all enabled strategies and places orders.

Contract:
- `config.STRATEGY_ALLOCATIONS` maps strategy module path -> max dollars to deploy.
- Each strategy exposes `generate_signals(budget=..., strategy_id=...)` and returns `Signal`s.
- Strategies own sizing by setting `Signal.notional`. This runner executes signals safely.
"""

import importlib
import logging
from uuid import uuid4

import config
import orders
from signals import Side, Signal
from budget import committed_dollars_from_orders


logger = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _existing_symbols(client) -> set[str]:
    """Return symbols we already hold or have pending buy orders for."""
    existing: set[str] = set()

    # 1) Current positions
    try:
        for p in client.get_all_positions():
            existing.add(str(p.symbol).strip().upper())
    except Exception:
        logger.exception("Failed to fetch positions")

    # 2) Open orders (prevents double-buys when a previous run already submitted orders)
    try:
        for o in client.get_orders():
            side = str(getattr(o, "side", "")).upper()
            status = str(getattr(o, "status", "")).lower()
            sym = str(getattr(o, "symbol", "")).strip().upper()
            if not sym:
                continue
            if side == "BUY" and status in {"new", "accepted", "pending_new", "held", "partially_filled"}:
                existing.add(sym)
    except Exception:
        # If this fails (API hiccup), we still protect with positions and per-run de-dupe.
        logger.exception("Failed to fetch open orders")

    return existing


def _strategy_committed_dollars(client, *, strategy_id: str) -> float:
    """Return lifetime dollars committed for a strategy.

    "Committed" means BUY orders that are either filled or still open/pending.
    We attribute orders to strategies by `client_order_id` prefix:
        f"{strategy_id}:..."

    Notes:
    - We ignore canceled/rejected orders.
    - Alpaca orders may expose different fields depending on type; we try multiple.
    """

    total = 0.0

    # Fetch in pages. Alpaca caps `limit` at 500.
    before_order_id: str | None = None
    while True:
        try:
            kwargs = {"status": "all", "limit": 500, "direction": "desc", "nested": True}
            if before_order_id:
                kwargs["before_order_id"] = before_order_id
            batch = client.get_orders(**kwargs)
        except Exception:
            logger.exception("Failed to fetch orders for strategy accounting (%s)", strategy_id)
            break

        if not batch:
            break

        total += committed_dollars_from_orders(batch, strategy_id=strategy_id)

        # Paginate by oldest order in this batch.
        last = batch[-1]
        before_order_id = str(getattr(last, "id", "") or "")
        if not before_order_id:
            break

    return total


def execute_daily_trades() -> None:
    """Run all configured strategies and place the resulting trades."""
    client = config.get_client()

    # Risk guard: keep some cash untouched.
    try:
        account = client.get_account()
        cash = _safe_float(getattr(account, "cash", 0.0))
    except Exception:
        logger.exception("Failed to fetch account")
        return

    cash_reserve = cash * 0.10  # keep 10% as a safety buffer

    existing = _existing_symbols(client)
    submitted_this_run: set[str] = set()

    for strategy_path, budget in config.STRATEGY_ALLOCATIONS.items():
        if budget <= 0:
            logger.info("Skipping %s (budget=%s)", strategy_path, budget)
            continue

        committed = _strategy_committed_dollars(client, strategy_id=strategy_path)
        remaining_budget = float(budget) - committed

        logger.info(
            "Strategy=%s budget=%.2f committed=%.2f remaining=%.2f",
            strategy_path,
            float(budget),
            committed,
            remaining_budget,
        )
        if remaining_budget <= 0:
            logger.info(
                "Skipping %s (lifetime cap reached: budget=%.2f committed=%.2f)",
                strategy_path,
                float(budget),
                committed,
            )
            continue

        try:
            module = importlib.import_module(strategy_path)
        except ModuleNotFoundError:
            logger.exception("Strategy module not found: %s", strategy_path)
            continue

        generate_signals = getattr(module, "generate_signals", None)
        if not callable(generate_signals):
            logger.error("Strategy %s has no callable generate_signals()", strategy_path)
            continue

        try:
            # Give the strategy its *remaining lifetime* budget so it can size signals itself.
            signals = list(generate_signals(budget=remaining_budget, strategy_id=strategy_path))
        except Exception:
            logger.exception("Strategy %s failed while generating signals", strategy_path)
            continue

        # For now we only act on BUY signals.
        buy_signals: list[Signal] = [
            s
            for s in signals
            if isinstance(s, Signal) and s.side == Side.BUY and s.normalized_symbol()
        ]
        if not buy_signals:
            logger.info("Strategy %s returned no BUY signals", strategy_path)
            continue

        for s in buy_signals:
            symbol = s.normalized_symbol()
            try:
                if symbol in existing or symbol in submitted_this_run:
                    logger.info("Already have/pending %s (skipping)", symbol)
                    continue

                dollars = _safe_float(s.notional)
                if dollars <= 0:
                    logger.info("No sizing for %s (strategy=%s), skipping", symbol, strategy_path)
                    continue

                # Don't spend into the cash reserve.
                if cash - cash_reserve < dollars:
                    logger.warning(
                        "Not enough cash left for %s (need %.2f, cash %.2f, reserve %.2f)",
                        symbol,
                        dollars,
                        cash,
                        cash_reserve,
                    )
                    continue

                # Strategy attribution: use Alpaca's client_order_id.
                client_order_id = f"{strategy_path}:{uuid4().hex[:16]}"

                orders.buy_market_notional(client, symbol, dollars, client_order_id=client_order_id)
                submitted_this_run.add(symbol)
                cash -= dollars
                logger.info(
                    "Submitted BUY %s notional=%.2f (strategy=%s, client_order_id=%s)",
                    symbol,
                    dollars,
                    strategy_path,
                    client_order_id,
                )
            except Exception:
                logger.exception(
                    "Order failed for %s (strategy=%s, notional=%.2f)",
                    symbol,
                    strategy_path,
                    _safe_float(getattr(s, "notional", 0.0)),
                )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    execute_daily_trades()
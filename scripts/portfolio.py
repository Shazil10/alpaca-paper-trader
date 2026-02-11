#!/usr/bin/env python
"""portfolio.py — Quick portfolio snapshot (diagnostic / manual use).

Usage:
    PYTHONPATH=src python scripts/portfolio.py
"""

import os
import sys

# Allow running from repo root.
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config


def main():
    try:
        client = config.get_client()
        account = client.get_account()
        positions = client.get_all_positions()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Summary
    print(f"Cash:      ${float(account.cash):.2f}")
    print(f"Portfolio: ${float(account.portfolio_value):.2f}")
    print("-" * 35)

    # Positions
    if not positions:
        print("No active positions.")
    else:
        print(f"{'Symbol':<10} {'Qty':<10} {'P/L ($)':<15}")
        for p in positions:
            print(f"{p.symbol:<10} {p.qty:<10} ${float(p.unrealized_pl):.2f}")


if __name__ == "__main__":
    main()

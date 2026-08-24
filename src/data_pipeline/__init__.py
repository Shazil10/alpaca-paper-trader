"""Shared daily price lake for the trading system.

``store``  read side: the only thing strategies should import.
``schema`` column/dtype contract and on-disk serialization rules.
``sync_prices``    incremental daily fetch (writes the hot year).
``rebuild_prices`` full re-download (corrects split/dividend drift).
"""

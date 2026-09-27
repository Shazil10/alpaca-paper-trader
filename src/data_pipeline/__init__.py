"""Shared daily price lake for the trading system.

``store``  read side: the only thing strategies should import.
``schema`` column/dtype contract and on-disk serialization rules.
``providers``      market-data sources behind one ``fetch_batch`` contract;
                   ``MARKET_DATA_PROVIDER`` picks the canonical one (Alpaca SIP).
``manifest``       per-year provenance: one provider per partition.
``anchor``         keeps ``adj_close`` on one anchor per symbol between rebuilds.
``quarantine``     secondary-source bars, held apart until reconciled.
``membership``     point-in-time S&P 500/400/600 membership and the SP1500 union.
``sync_prices``    incremental daily fetch (writes the hot year).
``rebuild_prices`` staged whole-partition replacement (provider change, drift).
"""

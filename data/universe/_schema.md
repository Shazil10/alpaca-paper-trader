# Universe data — schema and contracts

Three files, three different jobs. The price lake says what something traded at;
these say what was tradable, when, and whether the symbol meant the same company
throughout.

| File | Written by | Read by |
|---|---|---|
| `master_tickers.csv` | `src/universe.py` (daily) | `sync_prices`, `scripts/build_securities.py` |
| `membership.parquet` | `scripts/build_membership.py` (manual) | `data_pipeline.membership`, `backtest.runner` |
| `securities.parquet` | `scripts/build_securities.py` (manual) | `data_pipeline.securities`, `backtest.runner` |

Both parquet files are **derived and manual**. Nothing on the daily trading path
writes or needs them; they exist for the backtester. Rebuild with:

```bash
PYTHONPATH=src ./venv/bin/python scripts/build_membership.py
PYTHONPATH=src ./venv/bin/python scripts/build_securities.py   # reads membership
```

`scripts/check_lake_readiness.py --backtest` asserts both exist and are
plausible.

---

## `membership.parquet` — point-in-time index membership

Interval form, one row per continuous stretch of membership:

| Column | Type | Meaning |
|---|---|---|
| `symbol` | string, upper case | Ticker as the source records it |
| `index` | string | `SP500` today; the schema allows more |
| `start_date` | `datetime64[ns]` | First date in the index |
| `end_date` | `datetime64[ns]` or `NaT` | Last date; `NaT` means still a member |

Current build: 1,262 records, 1,209 unique symbols, 503 current members,
1996-01-02 → 2026-08-18, roughly 500 members on any historical date.

### Contract 1: this is what removes survivorship bias, and only partly

Using today's constituents for a 2010 backtest deletes every company that failed
between then and now, which inflates returns by roughly 1-3% annually. Membership
fixes the universe side of that. It does **not** fix the price side: a name in the
1998 index is only usable if the lake can price it, and the lake is populated
from a survivor-biased symbol list. Coverage is measured by
`check_lake_readiness.py --backtest` under "membership joins the lake".

### Contract 2: symbols are final tickers, not point-in-time tickers

This is the sharpest limitation and the easiest to miss.

The source ([fja05680/sp500](https://github.com/fja05680/sp500), itself derived
from the dataset behind Clenow's *Trading Evolved*) records many companies under
the symbol they ended with, not the symbol they traded under at the time. Lehman
Brothers appears as `LEHMQ` — the post-bankruptcy pink-sheet symbol — and never
as `LEH`, which is what it actually traded as for the whole period a 2007
backtest cares about.

Two consequences:

- A join against the price lake misses those names entirely. yfinance has no
  history under `LEHMQ`. They show up as fetch failures in
  `scripts/backfill_history.py`, not as errors.
- A lookup for a ticker that *did* exist historically can return nothing. Asking
  `members_asof("2007-06-30")` for `LEH` is a false negative.

Resolving this needs a ticker-change table keyed on a permanent identifier —
Tiingo's `permaTicker`, FIGI, or CUSIP. Until then, treat missing historical
names as expected rather than as a bug to chase.

### Contract 3: dates are index-change dates, not listing dates

`start_date` is when the company entered the index and `end_date` is when it
left. Neither is an IPO or a delisting. A company can leave the index and trade
for another decade (most do), and `end_date` says nothing about why it left.

### Contract 4: snapshots, interpolated

The source is a series of ~2,720 dated snapshots, not a continuous change log, so
a membership change is dated to the first snapshot that shows it. Snapshot density
runs about 90-130 per year, so a start or end date can be off by days. Immaterial
for a monthly-rebalanced sleeve; material if a strategy trades index adds on the
effective date.

---

## `securities.parquet` — security master

| Column | Type | Meaning |
|---|---|---|
| `security_id` | string | Permanent id, `TICKER_n` per listing span |
| `ticker` | string | Trading symbol |
| `name` | string | Currently always empty — no free source wired |
| `sector` `industry` | string | GICS labels from `master_tickers.csv` |
| `start_date` `end_date` | `datetime64[ns]` | Span validity; `NaT` = open |
| `delist_reason` | string | `index_removal` or empty. See contract 6 |
| `recycle_candidate` | bool | True when the ticker has more than one span |
| `gap_years` | float | Years between this span and the previous one |
| `source` | string | `membership+registry` |

Current build: 1,254 spans over 1,209 tickers, 503 currently in-index, 90 spans
on 45 recycle-candidate tickers, sector known for 663 spans.

### Contract 5: a split span is a hypothesis, not a finding

Ticker recycling is real: a symbol freed by a delisting gets reassigned, and
naive concatenation splices two unrelated companies into one price series. But
membership intervals cannot tell recycling apart from a company that simply left
the index for a while, so the builder flags rather than decides. Spans separated
by less than `RECYCLE_GAP_YEARS` (2.0) are merged; longer gaps start a new span
and set `recycle_candidate`.

Both error directions are present in the current output and both are visible by
hand:

- `AMP_1` / `AMP_2` — genuine reassignment. AMP Incorporated (connectors,
  acquired 1999), then Ameriprise Financial from 2005.
- `AMD_1` / `AMD_2` — false positive. One company that left the index from 2013
  to 2017 and came back.

`gap_years` exists so triage does not require re-deriving the gap. Nothing in
`src/backtest/` currently keys on `security_id`, so a false split costs nothing
today; it would matter the moment price history is joined on it.

### Contract 6: `delist_reason` is not a delisting reason

It records `index_removal`, because that is the only thing the source knows. No
free feed here distinguishes merged from acquired from bankrupt. The backtester
therefore applies a flat configurable haircut on forced liquidation
(`ExecutionConfig.delisting_return`, default -30%, Shumway's conservative
assumption) rather than a real delisting return, and
`BacktestResult.limitations` says so on every run.

### Contract 7: sectors are current, not point-in-time

`sector` comes from today's `master_tickers.csv`, so a company that changed GICS
sector mid-history carries today's label throughout, and names that left the
index before the registry existed carry no label at all (591 of 1,254 spans).

Good enough for a sector *cap* — `RiskConfig.max_sector_pct` does nothing at all
without a map, which is worse — and not good enough for sector attribution.
`data_pipeline.securities.sector_map()` says the same in its docstring.

---

## `master_tickers.csv` — observation registry

Covered by contract 3 of `data/prices/_schema.md`: `first_seen` / `last_seen` are
the dates this pipeline observed a symbol, not index membership. Append-only, so
a delisted name keeps its history. It is the sector source for the security
master and the symbol source for `sync_prices`.

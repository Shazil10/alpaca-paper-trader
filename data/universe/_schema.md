# Universe data — schema and contracts

Three files, three different jobs. The price lake says what something traded at;
these say what was tradable, when, and whether the symbol meant the same company
throughout.

| File | Written by | Read by |
|---|---|---|
| `master_tickers.csv` | `sync_prices` (after close) | `sync_prices`, `scripts/build_securities.py` |
| `membership.parquet` | `scripts/build_membership.py` (manual) | `data_pipeline.membership`, `backtest.runner` |
| `membership_baseline.csv` | `src/universe.py --capture-only` (first run only) | `data_pipeline.membership` |
| `membership_events.csv` | `src/universe.py --capture-only` (daily, append-only) | `data_pipeline.membership` |
| `current_membership.csv` | `src/universe.py --capture-only` (daily) | `sync_prices`, `universe.py --filter-only`, audit |
| `symbol_aliases.csv` | `scripts/refresh_aliases.py` (daily) | `data_pipeline.aliases`, audit, gap classifier |
| `securities.parquet` | `scripts/build_securities.py` (manual) | `data_pipeline.securities`, `backtest.runner` |
| `../../universe.csv` | `src/universe.py --filter-only` (daily) | live strategies (tradable list only) |

The parquet files are **derived and manual**; they exist for the backtester.
The capture files are written every weekday evening by
`.github/workflows/sync_data.yml`. Rebuild the parquet files with:

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

---

## The daily capture — unfiltered S&P 500 / 400 / 600

Nothing free reconstructs S&P 400/600 history, so it is recorded from now on.
`data_pipeline.membership_capture` scrapes the three Wikipedia constituent
tables every weekday evening, **unfiltered** — the liquidity screen is applied
afterwards, to `universe.csv` only, so a stock dipping under $10 never reads as
an index removal.

| File | Columns |
|---|---|
| `membership_baseline.csv` | `observed_date, index, symbol, source` |
| `membership_events.csv` | `observed_date, effective_date, index, symbol, event, source` (`event` is `add` / `remove`) |
| `current_membership.csv` | `observed_date, index, symbol, security, sector, industry, source` |

### Contract 8: the first capture is a baseline, not additions

Baseline members joined before anyone looked, so their intervals are
**left-censored** at the baseline date and `coverage_start()` reports that date.
`members_asof` before it answers "nothing recorded" for the S&P 400/600, and
the S&P 500 part only for the S&P 1500 — with a warning, because a backtest
there is not survivorship-free.

### Contract 9: `effective_date` is an upper bound

A change is recorded the first evening the page shows it, so `effective_date`
equals `observed_date` and the true date is on or before it. A removal observed
on day D closes the interval on D − 1.

### Contract 10: a failed scrape records nothing

Before anything is written the snapshot must look like the index: each index in
its normal count range (S&P 500 480-520, 400 380-420, 600 560-640), the union
1,440-1,560, sectors present, and no index losing more than 10% of its members
in one capture. Otherwise `SuspiciousScrapeError`, exit code 2, all three files
untouched, yesterday's membership stays in force, and the workflow fails loudly
after committing prices. A missed day costs a day of precision; a phantom purge
would corrupt the history permanently.

### Contract 11: S&P 1500 is derived, and moves are continuous

`SP1500` is never stored: it is the union of the three components, with gaps of
up to 7 days between leaving one and joining another bridged. An S&P 600 → 400
promotion is two events (`remove` SP600, `add` SP400), and when the two pages are
edited on different days the company is still continuously in the S&P 1500.

For the S&P 500 the reconstructed history (`membership.parquet`) is used up to
the day before the baseline and the capture from then on; an interval spanning
the join is one interval.

### Contract 12: renames are explicit

Alpaca's `asof` mapping stores a renamed company's whole history under its
current ticker (FB's 2018 bars under `META`). Membership keeps the ticker as it
was. `symbol_aliases.csv`, from Alpaca's corporate-actions `name_change`
records, is what joins the two; the audit and the gap classifier count a member
as priced when its successor is.

---

## Audit: what this data currently supports

Measured by `scripts/audit_data.py`. Re-run it after any backfill; these numbers
are a snapshot, not a contract.

| Question | Answer |
|---|---|
| Symbols complete over their own span | 1,399 of 1,404 |
| Symbols with 15+ years | 34 (the ETFs) |
| Median span | 3.6 years |
| Removed-index intervals priceable *during* membership | **42 of 759 (5.5%)** |
| Membership symbols the lake can price | 619 of 1,209 |
| Unfetchable post-bankruptcy tickers | 29 |
| Index coverage on 2024-06-28 | 96% |
| Index coverage on 2008-06-30 … 2020-03-31 | **0%** |
| Membership date precision | median 2 days, 90th pct 10, worst 91 |
| Spans with a sector label | 663 of 1,254, all current-vintage |

### The finding that governs everything else

Index member coverage is **zero on every historical probe date before 2024**,
because equities in the lake start 2023-01-03. The point-in-time membership
machinery works and is wired into the engine, but for equities it has nothing to
bite on yet: a stock backtest today is a backtest of names that survived into
today's symbol list, regardless of what `members_asof` returns.

Only 5.5% of removed-index intervals can be priced during the window they were
actually members. 592 of 759 have no bars at all. So the residual survivorship
bias is large, and results from the stock sleeves should be described accordingly.

The ETF sleeves are unaffected — all 24 TSMOM tickers and all 20 rotation ETFs
reach inception — which is why TSMOM is the only strategy here with a sample worth
drawing conclusions from.

### What would move each number

- **Equity depth and delisted names**: the staged Alpaca SIP rebuild
  (`rebuild_prices.py`, see `data/prices/_schema.md` "Migration") over every
  S&P 1500 member since 2016. Equity history before 2016 is outside the
  canonical provider's reach; `scripts/classify_gaps.py` classifies what
  remains (A-F) and decides whether a Tiingo trial is worth it.
- **The 29 unfetchable tickers**: nothing free fixes these. They need a
  ticker-change table keyed on a permanent id (Tiingo `permaTicker`, FIGI, CUSIP).
- **Sector vintage**: needs a historical GICS source; no free one is wired.
- **Membership precision**: already better than it needs to be for monthly
  rebalancing. Only worth improving for a strategy trading index adds on the
  effective date.

---

## `master_tickers.csv` — observation registry

Covered by contract 3 of `data/prices/_schema.md`: `first_seen` / `last_seen` are
the dates this pipeline observed a symbol, not index membership. Append-only, so
a delisted name keeps its history. It is the sector source for the security
master and the symbol source for `sync_prices`.

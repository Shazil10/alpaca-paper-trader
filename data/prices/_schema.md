# Daily price lake — schema and contracts

One shared store of daily equity/ETF bars. Live strategies read it through
`src/data_pipeline/store.py`; `sync_prices.py` is the only routine writer and
runs **after the close** (`.github/workflows/sync_data.yml`). It replaces the
per-strategy `yf.download` calls that each invented their own window, and is
intended to eventually replace the ad-hoc `.cache/*.pkl` files the research
notebooks maintain.

The canonical provider is **Alpaca, SIP feed** (`MARKET_DATA_PROVIDER`, default
`alpaca`). Yahoo, Stooq and Tiingo are cross-check or gap-repair sources that can
never write the lake directly (Contract 8).

## Columns

Fixed order, enforced by `schema.COLUMNS`:

| Column | Type | Meaning |
|---|---|---|
| `date` | `datetime64[ns]`, naive, midnight | Session date (New York) |
| `symbol` | string, upper case | Lake form (`BRK-B`; Alpaca's wire form `BRK.B` is converted at the boundary) |
| `open` `high` `low` `close` | `float64` | The print. **Raw** on Alpaca years; **split-adjusted** on Yahoo-era years (see below) |
| `adj_close` | `float64` | Split + dividend (+ spin-off on Alpaca) adjusted close |
| `volume` | `Int64` (nullable) | Shares traded. Raw on Alpaca years; split-adjusted on Yahoo-era years |

Primary key is `(date, symbol)` — one row per symbol per session.

**What `close` means depends on the year's provider**, and the manifest says
which (Contract 6). Alpaca's `adjustment=raw` is the actual print: NVDA on
2024-06-07 closed near $1,200. Yahoo's `Close` with `auto_adjust=False` is
adjusted for splits but not dividends, so the same bar reads near $120. Every
consumer in `src/` uses `close` only as the same-row ratio `adj_close / close`
or as a single-date level, both of which are correct — and more correct — on
raw prints. Nothing computes returns on `close`.

## Contract 1: strategies use `adj_close`

Live momentum, rotation and mean-reversion logic must read **`adj_close`** for
returns, moving averages and 52-week highs. Use raw `close` only when you
specifically mean the unadjusted print.

This preserves pre-lake behaviour exactly. The old code called
`yf.download(..., auto_adjust=True)` and read `["Close"]`, which *is* adjusted
close. Verified empirically on `yfinance==1.1.0` against `auto_adjust=False`'s
`Adj Close` for KO, AAPL and SPY: maximum relative difference `2e-7` (float32
rounding inside yfinance), while raw `close` diverges from `adj_close` by
dollars on dividend payers (KO $1.95, SPY $7.26 over one year). The two
adjusted series are equivalent; raw close is not a substitute.

Because the agreement is to float32 precision and not bit-exact, tests and
canaries assert on **decisions** (which symbols, what weights, which regime),
never on exact float equality of prices.

## Contract 2: completed sessions only

The lake contains **no bar for a session that has not finished**. Each provider
enforces its own cutoff, computed in America/New_York rather than the runner's
zone (a GitHub runner is on UTC; at 21:00 ET its date is already tomorrow):

- Alpaca (`completed_session_cutoff`): today's bar is admitted only once the
  extended-hours session has ended at 20:00 ET. The after-close workflow runs at
  01:30 UTC (21:30 EDT / 20:30 EST), past that and past Alpaca's 15-minute SIP
  embargo in both seasons, so it stores the session that just closed.
- Yahoo (`strict_cutoff`): today is always excluded.

The 09:30 trading run no longer syncs at all; it reads bars completed and
committed the evening before. Share **sizing** is unaffected — `src/orders.py`
fetches a live quote separately from signal data.

## Contract 3: `master_tickers.csv` is not an index-membership tape

`data/universe/master_tickers.csv` records `first_seen` / `last_seen` — the
dates this pipeline observed a symbol. That is **not** point-in-time membership.
Point-in-time membership lives in `data/universe/` (see its `_schema.md`): the
reconstructed S&P 500 history, and the daily unfiltered capture of the S&P
500/400/600 that begins at its first run.

The registry is append-only: symbols are never deleted, so a delisted name keeps
its history and stays queryable. Which names are *synced* is decided by the
unfiltered membership plus the registry, never by the liquidity-screened
`universe.csv`.

## Contract 4: `volume` may be missing, never zero-filled

`volume` is nullable `Int64`. A vendor may omit volume on some sessions; those
rows carry `NA`. They are not filled with `0`, because zero volume asserts "no
shares traded", which is a different and false claim.

## Contract 5: completeness is enforced, because gaps change decisions

A missing bar is not cosmetic. One hole inside a 50-day window makes
`rolling(50).mean()` return `NaN`, so `price > sma50` evaluates False and the
symbol silently drops out of any threshold comparison downstream — no exception,
no warning, different trade.

This was observed, not theorised. During the rotation-sleeve migration, five
sector ETFs were each missing 2-3 bars. The sleeve's breadth count fell from
10/11 to 5/11, flipping the regime from bull to neutral and changing the target
allocation. The strategy code was untouched; only data completeness differed.

**Cause (Yahoo era).** yfinance intermittently omits individual ticker/date
pairs inside large multi-ticker requests. A `NaN` close cannot be stored, so the
row is dropped, and the daily overlap never reaches back far enough to heal it.
On Alpaca the equivalent is a row present in only one of the raw / adjusted
responses; it is dropped and counted (`FetchReport.missing_adjusted` /
`missing_raw`), never filled.

**Detection** — `store.trading_calendar` marks a date as a session once
`MIN_SYMBOLS_FOR_SESSION` (10) distinct symbols report a bar. An *absolute*
floor, and both alternatives were tried and rejected:

- *A proportional quorum is blind to the worst case.* One bad fetch window
  dropped the same three dates for ~75% of the universe (2026-07-21/22/31 held
  only 319-442 of 1,380 symbols). Those dates fell below a 50% quorum, so the
  calendar concluded they were never sessions — nothing looked incomplete and a
  lake-wide hole passed review.
- *A single reference instrument can itself be short.* Keying the calendar to
  SPY failed once SPY was missing 2 sessions: those dates left the calendar, and
  every symbol missing them looked complete.

A genuine session keeps hundreds of reporters even in a bad window, while a
phantom date would need ten symbols to independently invent it.

**Repair** — `sync_prices.repair_gaps` re-fetches affected symbols in batches
over the affected window, then retries stragglers individually. It runs
automatically after each full sync.

It is **frontier-bounded**: repairs never fetch past the last session the lake
already holds. Advancing a subset of symbols beyond the rest leaves the newest
row mostly `NaN`, breaks rolling windows for everyone left behind, and is
invisible to `find_gaps` — the new dates fall outside the lagging symbols' own
spans. Advancing the frontier is sync's job, because sync covers every symbol.

Gaps are only reported *between* a symbol's own first and last bar: a late
listing or an early delisting is incomplete by nature, not by error. Some
residue is expected and will not heal — halts and corporate actions (a renamed
or taken-private ticker) look identical to a gap.

## Contract 6: a year holds one provider (`manifest.json`)

`data/prices/manifest.json` records, per year partition: `provider`, `feed`,
`adjustment`, `downloaded_at`, `start_date`, `end_date`, `row_count`,
`symbol_count`, `content_hash`, `notes`, `exceptions`.

- `writer.persist` checks every year a write touches **before writing any of
  them** and refuses (`MixedProviderError`) if a year would end up with two
  providers. A write that names no provider is `unspecified`, which is a
  provider like any other.
- `content_hash` is SHA-256 over the rows in canonical CSV form, so it is
  independent of Parquet/CSV and library versions.
  `scripts/bootstrap_manifest.py --verify` compares every partition with it.
- A year changes provider only by **whole-partition replacement**
  (`rebuild_prices.py`), never by upsert.
- Partitions written before the manifest existed were recorded by
  `scripts/bootstrap_manifest.py` as `yahoo`, with `downloaded_at` set to the
  file's last commit (an approximation, and noted as one).

The manifest is metadata about the canonical lake. A lake directory not named
`daily` (every test lake) keeps its own manifest in `_meta/` inside itself, so
two lakes can never share one (`schema.sidecar_dir`).

## Contract 7: one `adj_close` anchor per symbol (`anchor_factors.csv`)

An adjusted close is back-anchored: when a stock splits or pays a dividend the
vendor rescales every earlier bar. An incremental lake fetches at different
times, so without correction a 10:1 split leaves `1,200 → 121` at the seam
between old rows and the freshly fetched overlap — a fake −90% day inside every
moving average that spans it.

`writer.persist` measures the factor on the earliest overlapping row
(`fresh adj / stored adj`) and applies it to every older stored row of that
symbol:

- in place, for any partition the write rewrites anyway (the hot CSV);
- as a **pending factor** in `data/prices/anchor_factors.csv` for untouched
  cold Parquet years, which `store.load_prices` applies on read. Rewriting ten
  committed Parquet years because one stock paid a dividend would add tens of
  megabytes of undeltable blobs to git every day.

Whenever a partition *is* rewritten its pending factors are folded into the
bytes and cleared, so each factor applies exactly once. A rebuild leaves the file
empty for the years it replaced. What this cannot see: a vendor back-correction
dated before the overlap window — that is what the periodic rebuild and the
audit's section 11 are for.

## Contract 8: secondary sources go through quarantine

`data/prices/quarantine/<provider>/` holds bars from anything but the canonical
provider, with a metadata file and a hash. `quarantine.promote` lets a symbol's
held bars fill **only** (date, symbol) keys the canonical lake lacks, only after
they reconcile against the canonical series around the gap (enough overlap,
agreeing returns, a stable price-scale ratio), rescaled onto the canonical scale,
and records an `exceptions` entry in that year's manifest record. Nothing calls
it automatically; `scripts/classify_gaps.py` decides whether it is worth trying.

## Storage layout

```
data/prices/manifest.json         per-year provenance (Contract 6)
data/prices/anchor_factors.csv    pending adj_close factors (Contract 7), often absent
data/prices/quarantine/           secondary-source bars (Contract 8)
data/prices/staging/              rebuild working area — gitignored
data/prices/daily/2005.parquet    cold — legacy Yahoo ETF backfill, preserved
...
data/prices/daily/2015.parquet    cold — legacy Yahoo ETF backfill, preserved
data/prices/daily/2016.parquet    cold — canonical (Alpaca SIP after the rebuild)
...
data/prices/daily/2025.parquet    cold
data/prices/daily/2026.csv        hot  — appended and committed every weekday evening
```

Closed years are Parquet (compact, ~5-8 MB/year for ~1,500 tickers). The
current year is **CSV**, and this is deliberate: git deltas append-only text
cheaply, whereas appending a row to a snappy-compressed Parquet file shifts the
compressed blocks and git ends up storing a near-complete new copy — roughly a
gigabyte of objects per year for one daily-committed file.

For that saving to hold, unchanged rows must serialize to identical bytes.
Every CSV write goes through `schema.write_csv`, which pins float format
(`%.6f`), date format (`%Y-%m-%d`), column order, row order (`date`, `symbol`)
and line terminator. A test asserts writing the same frame twice yields
byte-identical output. **Never call `DataFrame.to_csv` on lake data directly.**

New sessions sort to the end of the file, so a daily append leaves every
preceding byte untouched. The 7-day correction overlap rewrites only the tail.

## January rollover (manual, once a year)

1. `python src/data_pipeline/rebuild_prices.py --finalize-year <closing year>`
   converts that year's CSV to Parquet and removes the CSV (the manifest entry
   moves with it; provenance and hash are unchanged).
2. Add `!data/prices/daily/<closing year>.parquet` to `.gitignore`.
3. Commit the new Parquet, the manifest and the `.gitignore` change.
4. The next `sync_prices.py` run creates the new hot CSV on its own.

`.gitignore` cannot know the current year, so the exception list is maintained
by hand. `schema.resolve_year_path` prefers Parquet when both exist, so the lake
stays readable mid-rollover.

## Coverage

- **Canonical history starts `2016-01-01`** (`sync_prices.LOOKBACK_START`),
  where Alpaca's SIP history begins. A new constituent is backfilled that far
  and no further: older bars would have to come from a second provider, which
  would mix sources inside a year.
- **2005-2015: legacy ETF-only history** from the Yahoo-era backfill, kept as it
  is and labelled `yahoo` in the manifest. Its `close` is split-adjusted, not the
  print. Its `adj_close` is spliced onto the canonical anchor at 2016 by one
  measured factor per ETF (pending factors in `anchor_factors.csv`, see
  `rebuild_prices.measure_seam`), so returns across the seam equal Yahoo's own.
  Equity backtests before 2016 are not supported by canonical data.

Depth is not uniform even among the ETFs, and the shortfalls are inceptions
rather than gaps: `XLC` listed 2018-06-19, `XLRE` 2015-10-08, `UUP` 2007-03-01,
`FXY` 2007-02-13, `FXF` 2006-06-26. Anything reading these must check for data,
not just for a column — a symbol before its inception is present as an all-`NaN`
column, which passes a naive membership test and then sorts `NaN`s.

Symbols synced:

- today's **unfiltered** S&P 500/400/600 constituents (never retired for
  failing — a constituent that fails is a fetch problem);
- every S&P 1500 member since 2016, every symbol in `master_tickers.csv`, and
  today's `universe.csv`;
- the fixed ETFs: the rotation sleeve (`XLK XLF XLV XLE XLI XLY XLP XLU XLB XLRE
  XLC`), its hedges (`TLT GLD UUP FXY FXF`), `SPY`, `SHY`, `IJH`, `IJR`.

A known symbol with no bar within 30 days of the lake frontier is **dormant**
(delisted, acquired): its history stays and it is not re-requested.

## Migration to Alpaca SIP (one time, local)

1. `scripts/bootstrap_manifest.py` — done; every existing year recorded as `yahoo`.
2. `scripts/pilot_alpaca.py` — Alpaca vs Yahoo on the hard cases; writes only
   quarantine and `data/audits/pilot_alpaca_<date>.*`.
3. `rebuild_prices.py --stage-only` — fetch 2016+ into staging, validate,
   compare; read `data/audits/rebuild_<run>.json`.
4. `rebuild_prices.py --resume <run>` — swap 2016+ in, splice the 2015/2016
   seam, verify; any failure restores lake, manifest and factors.
5. `scripts/refresh_aliases.py --since 2016-01-01`, then
   `scripts/classify_gaps.py` for the Tiingo decision.

Until step 4, an Alpaca sync into the lake is refused: 2016-2026 belong to
`yahoo`, and a year holds one provider.

## Failure policy

`sync_prices.py` exits 0 on partial failure: per-symbol misses leave the
committed bars in place, and the next run's overlap re-fetches the gap, so the
lake self-heals.

It exits 1, and the workflow fails loudly, on errors no retry fixes: rejected
credentials (401), a missing SIP entitlement (403 — there is deliberately no
retry on IEX), a write that would mix providers inside a year, or an unreadable
lake. Counting those as per-symbol failures would retire the whole registry one
run at a time. There is no fallback to another vendor anywhere in the write path.

Strategies trade on whatever the lake holds and log staleness loudly. A strategy
skips only when its required lookback is *not covered* (`store.has_lookback`) —
missing today is not that. `PRICE_SOURCE` controls what live strategies *read*
(lake or a live download) and is independent of `MARKET_DATA_PROVIDER`, which
controls what the pipeline *writes*.

# Daily price lake — schema and contracts

One shared store of daily equity/ETF bars. Live strategies read it through
`src/data_pipeline/store.py`; `sync_prices.py` is the only writer on the daily
path. It replaces the per-strategy `yf.download` calls that each invented their
own window, and is intended to eventually replace the ad-hoc `.cache/*.pkl`
files the research notebooks maintain.

## Columns

Fixed order, enforced by `schema.COLUMNS`:

| Column | Type | Meaning |
|---|---|---|
| `date` | `datetime64[ns]`, naive, midnight | Session date |
| `symbol` | string, upper case | Yahoo/Alpaca form (`BRK-B`, not `BRK.B`) |
| `open` `high` `low` `close` | `float64` | **Unadjusted** prints |
| `adj_close` | `float64` | Split/dividend adjusted close |
| `volume` | `Int64` (nullable) | Shares traded |

Primary key is `(date, symbol)` — one row per symbol per session.

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

The lake contains **no bar for the current date**. `sync_prices.py` drops it.

The trading workflow runs at 09:30 ET — market open — so "today" has no close
yet and yfinance may return a partial intraday bar. Storing that as a daily
OHLCV row would persist a 09:35 quote as if it were a session close.

Consequence, accepted deliberately: signals compute off the last *completed*
session rather than a live intraday price. On a 60-day slope or a 252-day high
this is immaterial; for a mean-reversion gate sitting exactly on its threshold a
candidate can flip. Using completed bars is the more defensible of the two.

Share **sizing** is unaffected — `src/orders.py` fetches a genuine live quote via
`yf.Ticker().fast_info["last_price"]`, separately from signal data.

## Contract 3: `master_tickers.csv` is not an index-membership tape

`data/universe/master_tickers.csv` records `first_seen` / `last_seen` — the
dates this pipeline observed a symbol. That is **not** point-in-time S&P 1500
membership. `universe.py` scrapes today's Wikipedia constituent tables, so the
symbol set reflects the index as it stands now.

Backtests over this lake therefore still carry survivorship bias. `first_seen`
tells you when we started recording a name, not when it entered an index. Do not
present results from this data as survivorship-free.

The file is append-only: symbols are never deleted, so a delisted name keeps its
history and stays queryable.

## Contract 4: `volume` may be missing, never zero-filled

`volume` is nullable `Int64`. yfinance omits volume on some sessions; those rows
carry `NA`. They are not filled with `0`, because zero volume asserts "no shares
traded", which is a different and false claim.

## Storage layout

```
data/prices/daily/2023.parquet   cold — committed, only rebuild rewrites it
data/prices/daily/2024.parquet   cold
data/prices/daily/2025.parquet   cold
data/prices/daily/2026.csv       hot  — appended and committed every weekday
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
preceding byte untouched. The 5-day correction overlap rewrites only the tail.

## January rollover (manual, once a year)

1. `python src/data_pipeline/rebuild_prices.py --finalize-year <closing year>`
   converts that year's CSV to Parquet and removes the CSV.
2. Add `!data/prices/daily/<closing year>.parquet` to `.gitignore`.
3. Commit the new Parquet and the `.gitignore` change.
4. The next `sync_prices.py` run creates the new hot CSV on its own.

`.gitignore` cannot know the current year, so the exception list is maintained
by hand. `schema.resolve_year_path` prefers Parquet when both exist, so the lake
stays readable mid-rollover.

## Coverage

- Start: `2023-01-01` (whole calendar years only — no partial-year edge cases,
  and comfortably above the ~2 years live strategies need).
- Equities: union of today's `universe.csv` and every symbol already in
  `master_tickers.csv`, so a name that fails today's liquidity screen does not
  develop a hole in its history.
- ETFs, always included regardless of the screen: the rotation sleeve
  (`XLK XLF XLV XLE XLI XLY XLP XLU XLB XLRE XLC`), its hedges
  (`TLT GLD UUP FXY FXF`), `SPY`, `SHY`, and the regime ETFs `IJH` / `IJR`.

`IJH` and `IJR` are present so the Clenow sleeve's eventual migration is
data-ready. That module is **not** wired to the lake.

## Contract 5: completeness is enforced, because gaps change decisions

A missing bar is not cosmetic. One hole inside a 50-day window makes
`rolling(50).mean()` return `NaN`, so `price > sma50` evaluates False and the
symbol silently drops out of any threshold comparison downstream — no exception,
no warning, different trade.

This was observed, not theorised. During the rotation-sleeve migration, five
sector ETFs were each missing 2-3 bars. The sleeve's breadth count fell from
10/11 to 5/11, flipping the regime from bull to neutral and changing the target
allocation. The strategy code was untouched; only data completeness differed.

**Cause.** yfinance intermittently omits individual ticker/date pairs inside
large multi-ticker requests. A `NaN` close cannot be stored, so the row is
dropped, and the daily 5-day overlap never reaches back far enough to heal it.
Single-ticker requests do not show this behaviour.

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

## Failure policy

`sync_prices.py` is non-fatal: it exits 0 on partial failure and the trading run
continues. A failed fetch leaves yesterday's committed bars in place, and the
next run's 5-day overlap re-fetches the gap, so the lake self-heals.

Strategies trade on whatever the lake holds and log staleness loudly. A strategy
skips only when its required lookback is *not covered* (`store.has_lookback`) —
missing today is not that. After cutover there is no silent yfinance fallback:
lake or skip. `PRICE_SOURCE=yfinance` exists as a local development escape
hatch, not a production path.

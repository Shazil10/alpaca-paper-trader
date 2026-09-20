# content.md — Paper Trading Microsite Copy & Spec

Give this file to the website LLM together with `DESIGN_SYSTEM.md` and `PROMPT_FOR_WEB_LLM.md`.

**Product URL (target):** `paper-trading.shazilfarukh.com`  
**Main portfolio:** https://shazilfarukh.com  
**Code repo:** https://github.com/Shazil10/alpaca-paper-trader  
**Owner:** Muhammad Shazil Farukh  
**Audience:** Quant finance / trading systems / fintech recruiters (not retail traders)

---

## Positioning (read this first)

This is a **systems + research showcase**, not a “I’m a profitable trader” page.

- Paper trading on Alpaca (not live money)
- Multi-strategy book with shared orchestration
- Show **process, architecture, attribution, risk constraints**
- Do **not** lead with equity curves, win rates, or uncaveated Sharpes
- Tone: institutional, sparse, precise, slightly understated
- One honest line about results is fine; no marketing alpha language

**Tagline spirit:** research → production loop on a scheduled paper book.

---

## Page sections (in order)

### 0. Nav
- Logo / mark: `SF` → links to https://shazilfarukh.com
- Links: `01. Architecture` · `02. Strategies` · `03. Book` · `04. Contribution`
- CTA button: `GitHub` → https://github.com/Shazil10/alpaca-paper-trader

### 1. Hero

**Eyebrow (mono):**  
`Alpaca paper · research → production`

**Title:**  
Multi-strategy  
**paper trading system** ← second line can be muted slate

**Body:**  
Strategies own selection and sizing. A single orchestrator enforces capital caps, attribution, and execution safety. The whole loop runs unattended on GitHub Actions — built to show process, not a flashy equity curve.

**Chips:**  
`Python` · `Alpaca paper` · `GitHub Actions` · `Lifetime budgets` · `Order attribution`

**CTAs:**  
- Primary: `See the system` → `#architecture`  
- Secondary: `View code` → GitHub repo

### 2. Architecture (PRIMARY WOW SECTION — scroll-synced)

**Section label:** `01. Architecture`  
**Title:** Research to a scheduled book  
**Lede:**  
Scroll through the loop. A sticky story card tracks the active stage while the spine lights up — research, strategies, orchestration, ops.

**Interaction (required):**  
Kanvas / ClearSlate–style scroll storytelling:
- Left: sticky glass detail card that updates with the active step
- Right: vertical timeline / spine with glowing nodes + small glass chips
- As user scrolls, active node advances; card content crossfades; a soft “packet” glow moves down the spine
- Mobile: stack gracefully; respect `prefers-reduced-motion` (clickable steps, no scrub)

**Four scroll beats:**

#### Beat 01 — RESEARCH
- Badge: `01 / RESEARCH`
- Headline: Ideas earn a budget first.
- Body: Hypotheses are tested in notebooks. Only rules that survive research get a live adapter and dollars — live metrics stay separate from backtest Sharpes.
- Chip title: `Notebooks first`
- Chip sub: `parameters · walk-forward`
- Bullets:
  - Parameter sets locked from studies (e.g. R2 v2)
  - Research sleeves stay capital-free until ready
  - No confusing paper PnL with research Sharpe

#### Beat 02 — STRATEGIES
- Badge: `02 / STRATEGIES`
- Headline: One contract, three engines.
- Body: Every live sleeve exposes `generate_signals(budget, strategy_id, held_symbols)`. Inside: momentum, regime rotation, or pullback mean reversion.
- Chip title: `Shared interface`
- Chip sub: `momentum · rotation · MR`
- Bullets:
  - Clenow: cross-sectional trend + inverse-vol sizing
  - Ranked: monthly V4/V8 blend + optional DAF leverage
  - Pullback MR: deep 52-week drawdowns with quality gates

#### Beat 03 — ORCHESTRATOR
- Badge: `03 / ORCHESTRATOR`
- Headline: Capital and ownership, not just signals.
- Body: `trade.py` turns intents into safe broker actions — lifetime caps, attribution, sell-before-buy, and a cash reserve.
- Chip title: `trade.py`
- Chip sub: `budgets · attribution`
- Bullets:
  - Lifetime budget caps; sells recycle capacity
  - Fills tagged `strategies.*:uuid`
  - Per-strategy holdings from history ∩ positions
  - SELL first, refresh, then BUY · 10% cash reserve

#### Beat 04 — OPS
- Badge: `04 / OPS`
- Headline: Unattended runs with an audit trail.
- Body: Weekday GitHub Actions: universe → trade → report. DST-aware gates, daily artifacts, and a manual liquidation path when orders stick.
- Chip title: `GitHub Actions`
- Chip sub: `cron · reports · liquidate`
- Bullets:
  - Dual cron + ET time gate
  - CSV / MD / HTML order tape with FIFO PnL
  - Emergency close workflow for paper positions

**System truth (for the designer’s mental model):**

```text
Universe refresh → Strategy sleeves → Orchestrator → Broker / Report
S&P 500/400/600     Clenow / Ranked / MR    SELL first, budgets    Alpaca paper
liquidity filters   generate_signals()      attribution tags       daily artifacts
```

### 3. Strategies

**Section label:** `02. Strategies`  
**Title:** Running, research, retired  
**Lede:**  
Live sleeves are capital-constrained and monitored. Research sleeves stay in notebooks until they clear ops and validation constraints.

**UI:** Filter tabs — All / Running / Research / Retired  
**Cards:** glass panels; status pill; thesis; 3 rules visible + expand for more; budget; note; links to Code / Notebook

#### Running

**Clenow Trend** — budget $10,000  
- Module: `strategies.momentum.clenow_trend`  
- Code: https://github.com/Shazil10/alpaca-paper-trader/tree/main/src/strategies/momentum  
- Thesis: Cross-sectional equity momentum with trend and regime filters; size by inverse volatility.  
- Rules:
  - Score = 60-day log-price slope × R²
  - Entry: price > 200-SMA, near 52-week highs, momentum persistence
  - Regime: ≥2 of SPY / IJH / IJR above 200-SMA (fail closed on bad data)
  - Sizing: inverse volatility, concentrated top names
  - Exits owned by the strategy; runner tags every fill
- Note: Live sample short. Modest positive sleeve PnL; budget-capped.

**Ranked Asset Alloc** — budget $15,000  
- Module: `strategies.ranks.ranked_asset_alloc`  
- Code: https://github.com/Shazil10/alpaca-paper-trader/blob/main/src/strategies/ranks/ranked_asset_alloc.py  
- Notebook: https://github.com/Shazil10/alpaca-paper-trader/blob/main/analysis/strategies/momentum/ranked_asset_allocation.ipynb  
- Thesis: Monthly sector rotation: 50/50 blend of momentum rotation and all-weather hedges, with optional low-vol leverage.  
- Rules:
  - Rebalance on the first trading day of each month
  - V4 sleeve: multi-factor ranks on sector ETFs by regime
  - V8 sleeve: bull → sectors; bear → hedge ETFs (TLT / GLD / UUP…)
  - DAF: up to 2× when sector vol is in a low percentile and regime is bull
  - Lifetime budget cap; sell proceeds recycle
- Note: Live adapter ≠ full research harness. Flat-to-slightly-red so far — treated as a learning sleeve.

**High Pullback Reversion** — budget $15,000  
- Module: `strategies.mean_reversion.high_pullback_reversion`  
- Code: https://github.com/Shazil10/alpaca-paper-trader/blob/main/src/strategies/mean_reversion/52W_mean_reversion_strat.py  
- Notebook: https://github.com/Shazil10/alpaca-paper-trader/blob/main/analysis/strategies/mean_reversion/52W_mean_reversion.ipynb  
- Thesis: Buy deep pullbacks from the 52-week high that still pass quality and regime gates; exit on recovery, stop, or time.  
- Rules:
  - Entry: ≥40% below 52-week high
  - Quality: not freefall vs high; not broken vs 200-day MA
  - Regime: ≥2 of SPY / IJH / IJR above 200-SMA
  - Exit: within 25% of high, −25% stop, or ~63 trading days
  - Cooldown after exits; pullback-weighted sizing; max 5 names
- Note: Primary live contributor so far. Parameters locked from R2 v2 research.

#### Research (no live budget)

**Deep Momentum**  
- Notebook: https://github.com/Shazil10/alpaca-paper-trader/blob/main/analysis/strategies/momentum/deep_momentum.ipynb  
- Thesis: Learned expected-return ranking from cross-sectional features — research only until walk-forward discipline is solid.  
- Rules: Featured in research notebooks, not live capital; requires purged / walk-forward validation before any budget; not wired into STRATEGY_ALLOCATIONS  
- Note: Interesting research track. Not deployed.

**TSMOM ETF**  
- Notebook: https://github.com/Shazil10/alpaca-paper-trader/blob/main/analysis/strategies/momentum/tsmom_etf.ipynb  
- Thesis: Classic time-series momentum on liquid ETFs.  
- Rules: Long-only and long/short variants studied in-notebook; no live adapter yet  
- Note: Research sleeve.

**Betting Against Beta**  
- Notebook: https://github.com/Shazil10/alpaca-paper-trader/blob/main/analysis/strategies/defensive/betting_against_beta.ipynb  
- Thesis: Long low-beta / defensive tilt studied as a research strategy.  
- Rules: Notebook stress tests only; no production capital  
- Note: Research sleeve.

#### Retired
- Empty for now — keep the filter/tab ready for future items.

### 4. Book snapshot (anti-vague concrete artifact)

**Section label:** `03. Book snapshot`  
**Title:** Positions by strategy  
**As-of line (mono):** `As of YYYY-MM-DD (EOD snapshot · illustrative)`  
**Disclaimer:**  
Alpaca paper account. Not investment advice. Short live sample; research metrics are not live performance.

**Table columns:** Strategy · Symbol · Qty · Notional · Notes  

**Rules:**
- Group / label by strategy sleeve
- Discretionary / manual trades must be visually de-emphasized and labeled separately
- Optional: thin budget utilization bars per sleeve (deployed vs lifetime cap) — muted mint, not loud green PnL

**Placeholder snapshot** (illustrative — will be replaced with live JSON later):

| Strategy | Symbol | Qty | Notional | Notes |
|---|---|---:|---:|---|
| Clenow Trend | SEZL | 24 | ~$4,100 | illustrative EOD |
| Ranked Asset Alloc | XLE | 25 | ~$2,200 | monthly sleeve |
| Ranked Asset Alloc | XLF | 28 | ~$1,400 | monthly sleeve |
| High Pullback Reversion | FOUR | 120 | ~$10,400 | largest MR name (approx) |
| High Pullback Reversion | CRVL | 12 | ~$3,400 | |
| Discretionary | NFLX | 30 | ~$2,200 | manual — labeled separately |
| Discretionary | PLTR | 10 | ~$1,590 | manual — labeled separately |
| Discretionary | SPY | 1 | ~$770 | manual — labeled separately |

**Lifetime budgets:**
- Clenow Trend: $10,000
- Ranked Asset Alloc: $15,000
- High Pullback Reversion: $15,000

### 5. Sleeve contribution

**Section label:** `04. Sleeve contribution`  
**Title:** Where live PnL came from  
**Lede:**  
Muted view of realized contribution by algorithmic sleeve. Short paper sample — process evidence, not a performance pitch.

**Chart:** Horizontal muted bars (not a flashy equity curve)  
**Caption (mono):** `Paper account · discretionary excluded · approximate`

**Illustrative values (algorithmic only):**
- High Pullback Reversion: +$5,000
- Clenow Trend: +$600
- Ranked Asset Alloc: −$50

Use mint for positive, restrained red (`#e06c75` or similar) for negative — no neon.

### 6. Footer
- Links: shazilfarukh.com · alpaca-paper-trader
- Line: `Built for quant interviews · paper trading only`

---

## Data contract (for later wiring)

Prefer a public JSON feed so content can update without redesign:

```json
{
  "as_of": "2026-08-05",
  "disclaimer": "...",
  "repo": "https://github.com/Shazil10/alpaca-paper-trader",
  "portfolio_home": "https://shazilfarukh.com",
  "budgets": { "...": 10000 },
  "strategies": [ /* status, thesis, rules, links */ ],
  "positions": [ /* strategy, symbol, qty, notional, note */ ],
  "sleeve_realized_pnl": [ /* label, pnl */ ]
}
```

Hardcode content from this file for v1 if needed; structure components so JSON can replace placeholders later.

---

## Visual references (interaction, not palette)

Match the *interaction pattern* of modern SaaS product pages (ClearSlate / Kanvas Chrome extension landings):
- Sticky explanatory card + vertical glowing timeline
- Glass panels, thin borders, generous whitespace
- Soft bloom / glow on the **active** node only
- Scroll-scrubbed progress along a spine

**Do not copy their purple/cyan palette.** Use only the shazilfarukh.com design system (navy + mint `#64ffda`). Mint glow instead of purple glow.

---

## Explicit do-nots

- No big green equity curve as the hero
- No “beat the market” / alpha marketing copy
- No uncaveated Sharpe / Calmar presented as live performance
- No purple gradient SaaS defaults, warm cream, terracotta, Inter-as-hero-branding if Calibre is available
- No dense notebook embeds on the page (link out)
- No live flashing tickers
- Don’t make discretionary trades look like strategy alpha

---

## Success criteria (recruiter in 60 seconds)

1. “This person built an **operating system** for strategies, not a Yahoo Finance script.”
2. “They know research Sharpe ≠ live paper.”
3. “I can jump to **code + notebooks** in one click.”
4. “Positions/budgets make it **concrete**, not a vibe.”

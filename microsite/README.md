# Paper Trading System Microsite

The single-page frontend for `paper-trading.shazilfarukh.com`. It presents Muhammad Shazil Farukh's multi-strategy Alpaca paper-trading system as a research-to-production operating system — architecture, strategy sleeves, capital constraints, and realized attribution — without performance theatre.

## Project Overview

| Area | Implementation |
|---|---|
| Framework | React 19, TypeScript, Vite |
| Motion | Framer Motion, with a full `prefers-reduced-motion` fallback |
| Styling | Tailwind 4 foundation plus a bespoke CSS design system |
| Data | Replaceable static contract at `client/public/data/portfolio.json` |
| Primary interaction | Scroll-synced architecture story with a side-alternating sticky card and a payload-carrying marker |
| Fonts | Calibre and SF Mono head the stacks; Inter and Fira Code are served locally as the working faces |

## Local Development

```bash
pnpm install
pnpm dev
```

Run both checks before publishing:

```bash
pnpm check   # tsc --noEmit
pnpm build
```

The production frontend is emitted to `dist/public`.

## Content and Data Updates

All portfolio evidence lives in `client/public/data/portfolio.json`, fetched at runtime, so budgets, strategies, positions, and contribution can be updated without touching components.

| JSON field | Purpose |
|---|---|
| `as_of` | End-of-day snapshot date shown in the book section |
| `strategies[].budget` | Lifetime cap for that sleeve; every total on the page is summed from these |
| `strategies[].short` | Short label used for capital attribution; must match `positions` and `sleeve_realized_pnl` |
| `strategies[].plain` | One jargon-free clause, reused in the architecture narrative |
| `strategies[].thesis` | Plain-language explanation shown on the card |
| `strategies[].chips` | Technical terminology, shown as secondary texture only |
| `strategies[].rules` | Longer rule list, rendered on the featured sleeve |
| `strategies[].evidence` | Optional `{ label, url }` citation; omit where there isn't one yet |
| `strategies[].mechanism` | Optional schematic: `inverse-vol`, `sector-tilt`, or `pullback-gauge` |
| `positions` | Sleeve ownership, symbols, quantities, and notional |
| `sleeve_realized_pnl` | Approximate realized contribution per sleeve |

There is deliberately no top-level budget map: caps live only on the strategies, and `client/src/lib/sleeves.ts` derives every count, cap, and total from that list at render time. No component contains a number tied to how many sleeves exist, so promoting or retiring a sleeve is a data edit and nothing else.

Budgets must stay in step with `STRATEGY_ALLOCATIONS` in the bot's `src/config.py`; that file is the source of truth for every capital figure on the page.

### Unwritten copy

The hero, each live sleeve card, and the footer carry visibly tagged first-person placeholders, each paired with a `PERSONAL NOTE` comment in the source. Replace the bracketed text with real sentences; nothing invented ships as fact in the meantime.

## Architecture Section

On desktop, the detail card is sticky while tall stages pass the viewport, and it alternates sides as the payload advances — the spine and its marker mirror to match. A single scroll-progress value drives the active stage, the illuminated spine, and the traveling marker, so the three can never disagree; it is also read once on mount, so arriving part-way down the section shows the correct stage rather than the first one. The marker label names what each stage hands to the next: candidate shares, requested trades, approved orders, then filled orders. The signal-generation stage builds its own bullets from the live strategy list. Clicking a stage scrolls it into the reading position.

Below `901px`, and whenever reduced motion is requested, the stages become a directly selectable single-column list down a short spine with the marker hidden, and the section's lede changes to describe that instead of scrolling. Nothing depends on the animation running — the stage content is visible by default and the fallback needs no horizontal scrolling.

## Brand and Font Notes

The CSS variables in `client/src/index.css` reproduce the portfolio palette, type scale, radius, and motion tokens. Mint `#64ffda` is rationed to live system state, the primary call to action, focus rings, the architecture section, and positive contribution; routine labels, chips, and capital-deployment bars stay slate. Negative contribution uses the restrained semantic red `#e06c75`.

Calibre and SF Mono are not licensed for self-hosting here, so `client/public/fonts/` carries the latin subsets of Inter and Fira Code (OFL, vendored from Google Fonts) and `client/public/fonts/fonts.css` declares them. If licensed Calibre and SF Mono files become available, add their `@font-face` declarations alongside; the family names already lead each stack.

## Hosting

Static output, no server required for the page itself. On Vercel:

| Setting | Value |
|---|---|
| Install command | `pnpm install` |
| Build command | `pnpm build` |
| Output directory | `dist/public` |
| Node runtime | Node 22 or a compatible supported release |

Add `paper-trading.shazilfarukh.com` in the domain settings and create the DNS record the host requests. All images and fonts are local to the build, so there are no external asset dependencies to re-host.

## Important Files

| Path | Responsibility |
|---|---|
| `client/src/components/ArchitectureScroll.tsx` | Scroll-synced system story and its static fallback |
| `client/src/components/Hero.tsx` | Headline, computed system stats, and the sleeve-to-gate instrument panel |
| `client/src/components/Strategies.tsx` | Asymmetric sleeve board and the research queue |
| `client/src/components/SleeveMechanism.tsx` | Illustrative decision schematics; never results |
| `client/src/lib/sleeves.ts` | Single source for every derived count, cap, and total |
| `client/src/components/Attribution.tsx` | Capital-in-use bars and realized contribution, drawn in on first view |
| `client/src/index.css` | Complete visual system and responsive behavior |
| `client/public/data/portfolio.json` | Replaceable data contract |
| `scripts/make_og_image.py` | Regenerates `client/public/og.png` for link shares |
| `DESIGN_DECISIONS.md` | Interaction and brand rationale |
| `ideas.md` | Full design direction and review amendments |

## Content Caveat

The displayed account is an **Alpaca paper account**. Snapshot and contribution values are illustrative, the live sample is short, and none of it is live performance or investment advice.

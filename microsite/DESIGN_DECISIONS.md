# Design Decisions

The site uses the navy-and-mint portfolio system as a technical instrument rather than a generic fintech dashboard. Dark navy provides continuity with `shazilfarukh.com`; near-white slate establishes editorial hierarchy. Mint is rationed: it marks live status, the primary call to action, the realized-gain direction, and the architecture chapter. Routine section labels, chips, and capital bars use slate so mint stays a signal rather than decoration.

## Language

Primary copy is written for a reader who knows what a stock is and nothing more: headlines and opening sentences carry no jargon, and technical vocabulary appears only in secondary positions — term chips, stage sub-labels, and the citation on each card. "Risk overlay" was replaced with "capital control," which gestures at the function before the explanation arrives. The system's autonomy is claimed once, prominently, in the status line; later references reword it ("no manual step," "no button to press") rather than repeating the same term.

Each strategy carries one quiet citation naming the paper, book, or write-up behind the approach. These are worded as sources rather than as code — the only raw-source links on the site remain the two "View Source on GitHub" links in the nav and footer.

## Nothing counts the sleeves

Every count, cap, and total is derived in `client/src/lib/sleeves.ts` from the strategy list, at render time. No component contains a digit or a spelled-out number tied to how many sleeves exist, and the top-level budget map was removed from the data file so a cap cannot be stored in two places. The hero's ring diagram generates marker positions for however many live sleeves there are, and the architecture section's signal-generation stage builds its own bullets from them.

The acceptance test is to add a strategy to the data file and reload: the capital total, the live/queued counts, the ring diagram, the stage narrative, the board, and both book panels all follow, with no text edit anywhere.

## Motion and shape

The architecture chapter is the signature interaction, and the only section with scroll-driven motion. Viewport-height stages drive a sticky detail card, a continuous spine, an active node, and a traveling marker whose label names what is being handed on — candidate shares, requested trades, approved orders, filled orders. The card alternates sides as the marker advances, and the spine mirrors with it, so the section has a left-right rhythm instead of one fixed column. Active stage, spine fill, and marker position all derive from a single scroll progress value, seeded once on mount so arriving part-way down the section shows the right stage. On mobile and under reduced-motion preferences the same content collapses to a directly selectable list down a short spine, and the section's own lede changes to describe that behaviour instead.

The other sections vary by skeleton rather than by effect, so the page does not read as the same card grid four times:

- **Hero** pairs the copy column with an instrument panel: sleeve markers converge by hairline links on a single mint capital-control gate — the system's actual shape, stated once, before any prose explains it. Below 1080px the panel becomes low-opacity texture behind the copy and drops its labels.
- **Sleeves** is an asymmetric board: the featured sleeve runs full width and splits internally between its rules and its mechanism diagram, with the remaining sleeves as compact cards beneath it. Weighting the featured card by width rather than by row span means no card has to stretch to fill a void.
- **Book** is two stacked instrument panels rather than cards — neutral bars for capital in use against each lifetime cap, then realized contribution plotted around a zero line with each dollar figure paired to its percentage of that sleeve's allowance. The bars replay from zero the first time they are seen; the resting state is already the real width, so a missing in-view signal costs the animation and not the data.

## Diagrams show mechanism, not performance

Each live sleeve carries a small schematic of how it decides — an inverse-volatility weighting split, a sector-tilt ranking, a distance-from-the-year's-low threshold. Every one is a fixed illustration, labelled "Illustrative," with no backtested or live figures in it. There are deliberately no equity curves, Sharpe ratios, candlesticks, or other performance claims: the page presents process, and every figure on it is a paper-trading result labelled as such.

Ambient texture in the empty zones is drawn in CSS — a faint measured field, masked to fade out before it reaches any content. No photography, and nothing representational.

## Unwritten copy is marked

First-person notes in the hero, on each live sleeve, and in the footer are placeholders, visibly tagged and paired with a `PERSONAL NOTE` comment in the source. They exist so the site owner's judgement calls have a home without an agent inventing personal claims to fill it.

## Type

Typography follows the portfolio stacks, with Calibre and SF Mono at the head. Neither is licensed for self-hosting here, so Inter and Fira Code are served locally as the working faces instead of falling through to whatever the visitor happens to have installed.

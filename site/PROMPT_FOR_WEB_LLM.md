# PROMPT_FOR_WEB_LLM.md

Copy everything below the line into the website-specialist LLM. Attach these files in the same chat:

1. `DESIGN_SYSTEM.md` (colors, type, components — source of visual truth)
2. `CONTENT.md` (all copy, section structure, strategy facts, do-nots)

---

## PROMPT (paste from here)

You are an elite product/marketing website designer-engineer. Your job is to design and build a **slick, premium, scroll-driven single-page microsite** for my quant paper-trading system.

### Who I am / what this is
I am Muhammad Shazil Farukh. My main portfolio is https://shazilfarukh.com (Brittany Chiang–style dark navy + mint). I need a sister microsite for:

**paper-trading.shazilfarukh.com**

It showcases a multi-strategy **Alpaca paper** trading system (Python + GitHub Actions) aimed at **quant finance recruiters**. It must feel like a serious research→production systems project — **not** a retail trader flexing PnL.

Repo (link out, don’t rebuild trading logic): https://github.com/Shazil10/alpaca-paper-trader

### Inputs I am giving you
1. **DESIGN_SYSTEM.md** — mandatory visual system (navy, `#64ffda` mint, Calibre, SF Mono, buttons, cards, motion tokens). Follow it strictly.
2. **CONTENT.md** — mandatory copy, section order, strategy details, architecture beats, placeholders for positions/PnL, do-nots.

If anything conflicts: **DESIGN_SYSTEM wins for look; CONTENT wins for words and structure.**

### Aesthetic goal
Make it feel as polished as modern Chrome-extension / SaaS product pages (ClearSlate, Kanvas): glass panels, generous whitespace, thin borders, soft glow on active elements, buttery scroll storytelling.

BUT:
- Keep **my** palette and fonts from DESIGN_SYSTEM.md
- Replace any purple/cyan reference accents with **mint `#64ffda`**
- Soft mint bloom is OK; neon purple glow is not
- Calm technical luxury > loud fintech dashboard

### Must-have interaction: Architecture section
This is the centerpiece. Build a **scroll-synced architecture story**:

- Desktop: **sticky left glass detail card** + **right vertical timeline/spine**
- As the user scrolls, the active step advances (4 beats: Research → Strategies → Orchestrator → Ops)
- Left card content crossfades/updates with badge, headline, body, bullets
- Right spine fills; active node glows; inactive nodes dim; a small “packet” indicator travels down the line
- Mobile: elegant stack; still readable
- Honor `prefers-reduced-motion`: no scrub pinning required; steps selectable

Use whatever stack you need for quality (React + Vite + Framer Motion + GSAP ScrollTrigger, or Next.js, etc.). Prefer maintainable React. Selective React Bits–style accents are fine if recolored to mint and not gimmicky.

### Full page structure (from CONTENT.md)
1. Nav (SF → shazilfarukh.com, section anchors, GitHub CTA)
2. Hero (process-first, chips, two CTAs — no equity curve)
3. Architecture (scroll story above)
4. Strategies (tabs: All / Running / Research / Retired + glass cards)
5. Book snapshot (positions-by-strategy table + optional budget bars)
6. Sleeve contribution (muted horizontal bars; discretionary excluded)
7. Footer

### Content rules
- Use the **exact positioning and copy** in CONTENT.md (you may tighten line breaks for layout, don’t invent alpha claims)
- Strategy theses/rules/notes/budgets/links come from CONTENT.md
- Positions and sleeve PnL in CONTENT.md are **illustrative placeholders** — structure them so they can later be fed from a JSON file (`portfolio.json`)
- Always label paper / short sample / not investment advice where CONTENT.md says so

### Explicitly forbid
- Hero equity curves, big green “+X%” hero stats
- Uncaveated Sharpe as live performance
- Purple SaaS defaults, cream/terracotta templates, generic Inter-only AI landing look
- Dense dashboard chrome, pill-stat spam, emoji
- Making discretionary trades look like strategy alpha
- Embedding whole Jupyter notebooks (link out only)

### Deliverables
1. A complete runnable frontend project (prefer Vite + React + TypeScript)
2. Global CSS variables matching DESIGN_SYSTEM.md
3. Self-hosted Calibre + SF Mono if I provide font files; otherwise Inter / Fira Code fallbacks exactly as DESIGN_SYSTEM allows, noted in README
4. Components split cleanly (Nav, Hero, ArchitectureScroll, Strategies, Book, Contribution, Footer)
5. `public/data/portfolio.json` (or equivalent) seeded from CONTENT.md so later we can swap real data without redesign
6. README: local run, build, notes for deploying to Vercel on subdomain `paper-trading.shazilfarukh.com`
7. Short “design decisions” note: how the scroll architecture works and how brand tokens were applied

### Quality bar
A recruiter should feel in 60 seconds:
1. This is an **operating system for strategies**, not a script
2. Author separates **research vs live paper**
3. Code/notebooks are one click away
4. Book/budgets make it **concrete**

Build the full site now. Start with the Architecture scroll experience, then flesh the rest to the same polish level. Do not leave placeholder lorem ipsum — use CONTENT.md.

## END PROMPT

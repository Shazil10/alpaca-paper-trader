# Paper Trading Microsite — Design Direction

The supplied `DESIGN_SYSTEM.md` and `CONTENT.md` are the ground-truth specification. Visual fidelity to Muhammad Shazil Farukh’s existing portfolio language takes priority over introducing an unrelated aesthetic.

## Chosen Approach: Calm Technical Instrument

**Design Movement:** Brittany Chiang–style developer portfolio refined through contemporary technical-product storytelling: dark navy canvas, mint signal accents, restrained glass surfaces, and systems-diagram precision.

**Core Principles:**

1. **Process over performance:** architecture, controls, and attribution lead; PnL appears only as a muted supporting artifact.
2. **Signal through restraint:** mint is reserved for active states, links, and system flow rather than applied as broad decoration.
3. **Operational credibility:** visible data contracts, budgets, modules, and audit language make the project feel like an operating system for strategies.
4. **Quiet depth:** layered navy surfaces, hairline borders, subtle bloom, and faint technical textures create polish without fintech spectacle.

**Color Philosophy:** `#0a192f` is the stable institutional canvas. Near-white slate creates hierarchy without harsh contrast; muted blue-slate carries explanation. `#64ffda` acts like an electrical signal—scarce, precise, and brightest only where the system is active. `#e06c75` is restricted to negative contribution semantics.

**Layout Paradigm:** An asymmetric editorial flow with a wide text-led hero, a cinematic two-column architecture chapter, and denser evidence sections below. The architecture centerpiece uses a sticky detail instrument on the left and a scroll-height timeline on the right. Supporting sections alternate between spacious editorial headers and precise technical panels rather than a repetitive centered card grid.

**Signature Elements:**

- A hairline mint system spine with one traveling packet and one illuminated active node.
- Mono “instrument labels” with indexed section numbers, module paths, and status annotations.
- Corner-cut glass panels with subtle internal grid texture and a restrained offset mint interaction shadow.

**Interaction Philosophy:** Every interaction should clarify state. Scroll reveals advance the architecture story, tabs isolate strategy lifecycle status, and expandable rules expose depth on demand. Hover states use short positional shifts and line growth; no gimmicks or live-ticker motion.

**Animation:** Use the supplied 250ms brand easing for controls. Architecture content crossfades with an 8–12px vertical settle while the packet moves continuously down the spine. Section entrances use low-amplitude opacity/translate transitions with 40–60ms stagger. Under `prefers-reduced-motion`, all reveal transforms and scroll scrubbing are removed; architecture steps remain directly selectable.

**Typography System:** Calibre-style sans hierarchy via the provided fallback stack, with SF Mono/Fira Code for labels, badges, tabs, modules, data, and buttons. Hero title uses 600 weight at `clamp(42px, 7vw, 78px)`. Section titles use `clamp(30px, 5vw, 48px)`. Body copy stays generous and readable at 18–20px; mono metadata stays 12–14px.

**Brand Essence:** A research-to-production operating system for quant recruiters who value implementation discipline over performance theatre. **Precise, understated, operational.**

**Brand Voice:** Headlines are declarative and systems-oriented. CTAs are direct; microcopy is candid about paper status and sample length. Examples: “Ideas earn a budget first.” and “Capital and ownership, not just signals.”

**Wordmark & Logo:** A custom angular `SF` monogram built as a compact circuit glyph: two interlocking strokes connected by a single mint routing node. The visible header mark must read as a designed symbol rather than default-font initials.

**Signature Brand Color:** Signal Mint — `#64ffda`.

## Implementation Decisions

The site will use React, TypeScript, Framer Motion, CSS variables from the supplied design system, and semantic static JSON at `client/public/data/portfolio.json`. The architecture story will rely on intersection/scroll progress rather than heavy imperative animation, preserving maintainability and a clean reduced-motion fallback.

Generated imagery will remain abstract, low-key, and subordinate to the interface: one hero system visual and a small set of technical textures for prominent storytelling areas. The data table and contribution chart will remain deterministic HTML/CSS so all values stay accurate and accessible.
@@
 Generated imagery will remain abstract, low-key, and subordinate to the interface: one hero system visual and a small set of technical textures for prominent storytelling areas. The data table and contribution chart will remain deterministic HTML/CSS so all values stay accurate and accessible.
+
+## Style Decisions
+
+The `SF` monogram is presented as a visible circuit-glyph lockup with an explicit Shazil Systems signature, so ownership reads immediately in the header rather than as a generic square icon.
+
+The architecture chapter’s mint spine is the primary signature motif. It uses one continuous illuminated rail, one active node, and one packet-like signal; the surrounding stage panels remain quiet enough that the flow is legible at first glance.
+
+Strategy cards are intentionally asymmetric and preceded by a lifecycle control board. Status, module path, shared interface, and capital constraints drive the composition so the section reads as an operational system rather than an even SaaS card grid.

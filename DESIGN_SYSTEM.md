ger# shazilfarukh.com — Design System

Use this to make **paper-trading.shazilfarukh.com** (or any subdomain) match the main portfolio visually. Source of truth in this repo: `src/styles/variables.js`, `src/styles/fonts.js`, `src/styles/mixins.js`, `src/styles/GlobalStyle.js`, `src/config.js`.

---

## Aesthetic summary

| Trait | Value |
|--------|--------|
| Mood | Dark navy, calm, technical |
| Accent | Mint / aqua green (`#64ffda`) |
| Body text | Muted blue-slate |
| Headings | Near-white blue-slate |
| Sans | **Calibre** (with Inter / system fallbacks) |
| Mono | **SF Mono** (with Fira Code / Roboto Mono fallbacks) |
| Radius | Soft square — `4px` (not pill / not zero) |
| Motion | `0.25s` with `cubic-bezier(0.645, 0.045, 0.355, 1)` |

This is the Brittani / Brittany Chiang–style dark portfolio look: navy canvas, green accents, mono labels, Calibre UI type.

---

## Colors

### Core palette (CSS variables)

```css
:root {
  --dark-navy: #020c1b;
  --navy: #0a192f;
  --light-navy: #112240;
  --lightest-navy: #233554;
  --navy-shadow: rgba(2, 12, 27, 0.7);

  --dark-slate: #495670;
  --slate: #8892b0;
  --light-slate: #a8b2d1;
  --lightest-slate: #ccd6f6;
  --white: #e6f1ff;

  --green: #64ffda;
  --green-tint: rgba(100, 255, 218, 0.1);

  --pink: #f57dff;   /* rare accent */
  --blue: #57cbff;   /* rare accent */
}
```

### How to use them

| Token | Hex / value | Typical use |
|--------|-------------|-------------|
| `--navy` | `#0a192f` | Page background (`body`) |
| `--dark-navy` | `#020c1b` | Deeper panels, footer, loader |
| `--light-navy` | `#112240` | Cards, job panels, elevated surfaces |
| `--lightest-navy` | `#233554` | Borders, section dividers, hover borders |
| `--navy-shadow` | `rgba(2,12,27,0.7)` | Card / image shadows |
| `--slate` | `#8892b0` | Default body text |
| `--light-slate` | `#a8b2d1` | Secondary text |
| `--lightest-slate` | `#ccd6f6` | Headings, important text |
| `--white` | `#e6f1ff` | Brightest text (sparingly) |
| `--dark-slate` | `#495670` | Scrollbar thumb, muted UI |
| `--green` | `#64ffda` | Links, buttons, accents, mono labels, focus |
| `--green-tint` | `rgba(100,255,218,0.1)` | Soft green backgrounds / highlights |

### Semantic mapping for a trading dashboard

| UI element | Use |
|------------|-----|
| Page bg | `--navy` |
| Sidebar / cards | `--light-navy` |
| Borders | `--lightest-navy` |
| Primary text | `--slate` |
| Titles / PnL labels | `--lightest-slate` |
| Links / CTAs / active tab | `--green` |
| Positive PnL (optional) | `--green` |
| Negative PnL (optional) | keep a restrained red, e.g. `#ff6b6b` or `#e06c75` — **not in main site palette**; add only for trading semantics |
| Charts | Prefer greens + light-slate lines on navy; avoid purple/glow defaults |

---

## Typography

### Font stacks

```css
:root {
  --font-sans: 'Calibre', 'Inter', 'San Francisco', 'SF Pro Text', -apple-system,
    system-ui, sans-serif;
  --font-mono: 'SF Mono', 'Fira Code', 'Fira Mono', 'Roboto Mono', monospace;
}
```

### Font files in this repo

Located under `src/fonts/`:

**Calibre** (loaded weights): `400`, `500`, `600` (+ italics)  
Formats: `.woff2` + `.woff`

**SF Mono** (loaded weights): `400`, `600` (+ italics)  
Formats: `.woff2` + `.woff`

To match exactly on the paper-trading site, **copy the same font files** (or self-host the same families). If licensing blocks Calibre elsewhere, closest free stand-ins:

- Sans: **Inter** or **Cal Sans** — won’t be identical; Inter is already in the fallback stack  
- Mono: **JetBrains Mono** / **Fira Code** / **IBM Plex Mono**

For a seamless handoff from `shazilfarukh.com` → `paper-trading.shazilfarukh.com`, **copy Calibre + SF Mono** if you have the right to use them.

### Type roles

| Role | Font | Weight | Color | Size notes |
|------|------|--------|-------|------------|
| Body | sans | 400 | `--slate` | Base `20px` (`--fz-xl`); `18px` on ≤480px |
| Headings | sans | 600 | `--lightest-slate` | line-height ~1.1 |
| Hero / big title | sans | 600 | `--lightest-slate` | `clamp(40px, 8vw, 80px)` (`.big-heading`) |
| Section title | sans | 600 | `--lightest-slate` | `clamp(26px, 5vw, 32px)` |
| Mono labels / nav / buttons / code | mono | 400 | often `--green` | `12–14px` typical |
| Numbered section prefix (`01.`, `02.`) | mono | 400 | `--green` | `16–20px` |

### Font-size scale

```css
:root {
  --fz-xxs: 12px;
  --fz-xs: 13px;
  --fz-sm: 14px;
  --fz-md: 16px;
  --fz-lg: 18px;
  --fz-xl: 20px;
  --fz-xxl: 22px;
  --fz-heading: 32px;
}
```

### Body defaults

```css
body {
  background-color: var(--navy);
  color: var(--slate);
  font-family: var(--font-sans);
  font-size: var(--fz-xl); /* 20px */
  line-height: 1.3;
  -webkit-font-smoothing: antialiased;
}
```

---

## Layout & spacing

```css
:root {
  --border-radius: 4px;
  --nav-height: 100px;
  --nav-scroll-height: 70px;
  --tab-height: 42px;
  --tab-width: 120px;
}
```

| Rule | Value |
|------|--------|
| Main max width | `1600px` |
| Section max width | `~1000px` |
| Main padding (desktop) | `200px 150px` (homepage fill: `0 150px`) |
| Main padding (≤1080) | `100px` horizontal |
| Main padding (≤768) | `50px` horizontal |
| Main padding (≤480) | `25px` horizontal |
| Section vertical padding | `100px` → `80px` (tablet) → `60px` (mobile) |

### Breakpoints (from `theme.js`)

| Name | Query |
|------|--------|
| mobileS | `max-width: 330px` |
| mobileM | `max-width: 400px` |
| mobileL | `max-width: 480px` |
| tabletS | `max-width: 600px` |
| tabletL | `max-width: 768px` |
| desktopXS | `max-width: 900px` |
| desktopS | `max-width: 1080px` |
| desktopM | `max-width: 1200px` |
| desktopL | `max-width: 1400px` |

---

## Motion

```css
:root {
  --easing: cubic-bezier(0.645, 0.045, 0.355, 1);
  --transition: all 0.25s cubic-bezier(0.645, 0.045, 0.355, 1);
}
```

- Hover transitions: use `--transition`
- Focus outline: `2px dashed var(--green)` with `outline-offset: 3px`
- Prefer `prefers-reduced-motion` respect for underline animations

---

## Components (match these patterns)

### Links

- Default: inherit color  
- Inline accent links: `color: var(--green)` with thin green underline that grows on hover  
- Hover text → `--green`

### Buttons (primary CTA)

- Transparent background  
- `1px solid var(--green)`  
- Text: `--green`  
- Font: mono, `--fz-xs` or `--fz-sm`  
- Radius: `--border-radius` (`4px`)  
- Hover: green box-shadow offset + slight translate up-left  
  - e.g. `box-shadow: 4px 4px 0 0 var(--green); transform: translate(-5px, -5px);`

### Cards / panels

- Background: `--light-navy`  
- Shadow: `0 10px 30px -15px var(--navy-shadow)`  
- Hover: slightly deeper shadow  
- Border (if any): `--lightest-navy`

### Lists (experience-style)

- No bullets  
- Green `▹` prefix (`color: var(--green)`)

### Scrollbar

- Track: `--navy`  
- Thumb: `--dark-slate`  
- Thin scrollbar

### Selection

- Background: `--lightest-navy`  
- Color: `--lightest-slate`

---

## Ready-to-paste starter CSS

Drop this into the paper-trading app global CSS, then load Calibre + SF Mono the same way (or via `@font-face` pointing at copied files):

```css
:root {
  --dark-navy: #020c1b;
  --navy: #0a192f;
  --light-navy: #112240;
  --lightest-navy: #233554;
  --navy-shadow: rgba(2, 12, 27, 0.7);
  --dark-slate: #495670;
  --slate: #8892b0;
  --light-slate: #a8b2d1;
  --lightest-slate: #ccd6f6;
  --white: #e6f1ff;
  --green: #64ffda;
  --green-tint: rgba(100, 255, 218, 0.1);
  --pink: #f57dff;
  --blue: #57cbff;

  --font-sans: 'Calibre', 'Inter', 'San Francisco', 'SF Pro Text', -apple-system,
    system-ui, sans-serif;
  --font-mono: 'SF Mono', 'Fira Code', 'Fira Mono', 'Roboto Mono', monospace;

  --fz-xxs: 12px;
  --fz-xs: 13px;
  --fz-sm: 14px;
  --fz-md: 16px;
  --fz-lg: 18px;
  --fz-xl: 20px;
  --fz-xxl: 22px;
  --fz-heading: 32px;

  --border-radius: 4px;
  --easing: cubic-bezier(0.645, 0.045, 0.355, 1);
  --transition: all 0.25s cubic-bezier(0.645, 0.045, 0.355, 1);
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
  scrollbar-width: thin;
  scrollbar-color: var(--dark-slate) var(--navy);
}

body {
  margin: 0;
  min-height: 100%;
  background-color: var(--navy);
  color: var(--slate);
  font-family: var(--font-sans);
  font-size: var(--fz-xl);
  line-height: 1.3;
  -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4, h5, h6 {
  margin: 0 0 10px;
  font-weight: 600;
  color: var(--lightest-slate);
  line-height: 1.1;
}

a {
  color: var(--green);
  text-decoration: none;
  transition: var(--transition);
}

a:hover {
  color: var(--green);
}

button,
.btn {
  color: var(--green);
  background: transparent;
  border: 1px solid var(--green);
  border-radius: var(--border-radius);
  font-family: var(--font-mono);
  font-size: var(--fz-xs);
  padding: 1.25rem 1.75rem;
  cursor: pointer;
  transition: var(--transition);
}

button:hover,
.btn:hover {
  box-shadow: 4px 4px 0 0 var(--green);
  transform: translate(-5px, -5px);
}

:focus-visible {
  outline: 2px dashed var(--green);
  outline-offset: 3px;
}

::selection {
  background-color: var(--lightest-navy);
  color: var(--lightest-slate);
}

.card {
  background: var(--light-navy);
  border-radius: var(--border-radius);
  box-shadow: 0 10px 30px -15px var(--navy-shadow);
}

.mono {
  font-family: var(--font-mono);
}
```

---

## Checklist for paper-trading.shazilfarukh.com

- [ ] Same CSS variables (colors + type scale + radius + easing)
- [ ] Same Calibre + SF Mono files (or intentional Inter / Fira fallbacks)
- [ ] Body bg `--navy`, text `--slate`, headings `--lightest-slate`
- [ ] Accent / links / primary buttons `--green`
- [ ] Cards `--light-navy`, borders `--lightest-navy`
- [ ] Buttons: outlined green + offset shadow hover (not filled purple pills)
- [ ] Mono for labels, tabs, tickers, timestamps, buttons
- [ ] Max content width ~1000–1600px with similar side padding
- [ ] No warm cream / terracotta / purple-gradient defaults
- [ ] Link from main portfolio Featured / Work opens this subdomain in same visual language

---

## Files to copy from this repo

| What | Path |
|------|------|
| Tokens | `src/styles/variables.js` |
| Font faces | `src/styles/fonts.js` |
| Font binaries | `src/fonts/Calibre/*`, `src/fonts/SFMono/*` |
| Buttons / links | `src/styles/mixins.js` |
| Global base | `src/styles/GlobalStyle.js` |
| Brand greens (config) | `src/config.js` → `colors.green / navy / darkNavy` |

---

## Note on charts & trading UI

The main site has almost no data viz. For the paper-trading app:

- Keep navy background and green accents  
- Use `--light-slate` / `--lightest-slate` for axes and labels  
- Grid lines: low-opacity `--lightest-navy`  
- One restrained red for losses only  
- Avoid neon glow, purple themes, and heavy card chrome so it still feels like the same product family as the portfolio

# Paper trading microsite

Vite + React app for **paper-trading.shazilfarukh.com**.

Matches [DESIGN_SYSTEM.md](../DESIGN_SYSTEM.md) (Calibre + SF Mono, navy / mint) with a scroll-synced Architecture story.

## Local

```bash
cd site
npm install
npm run dev
# open http://localhost:5173
```

## Build

```bash
cd site
npm run build
npm run preview
```

## Deploy (Vercel)

Root [`vercel.json`](../vercel.json) builds `site/` and serves `site/dist`.

1. Import the GitHub repo into Vercel.
2. Add domain `paper-trading.shazilfarukh.com`.
3. Namecheap: `CNAME` host `paper-trading` → `cname.vercel-dns.com`.

## Data

Edit [`public/data/portfolio.json`](public/data/portfolio.json) for strategies, positions, and sleeve contribution.

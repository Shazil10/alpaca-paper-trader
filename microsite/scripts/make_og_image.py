"""Render the Open Graph share card in the navy/mint system palette.

Figures are read from client/public/data/portfolio.json so the card cannot
drift from the page. Re-run after changing budgets:

    python3 microsite/scripts/make_og_image.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "client" / "public" / "data" / "portfolio.json"
OUT = ROOT / "client" / "public" / "og.png"

W, H = 1200, 630
BG = (10, 25, 47)
GRID = (25, 46, 79)
BORDER = (35, 53, 84)
HEADING = (204, 214, 246)
MUTED = (168, 178, 209)
DIM = (73, 86, 112)
EMERALD = (100, 255, 218)

SANS = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(path: str, size: int, weight: str | None = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size)
    if weight:
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
    return f


def main() -> None:
    data = json.loads(DATA.read_text())
    budgets = data["budgets"]
    capital = sum(budgets.values())
    live = sum(1 for s in data["strategies"] if s["status"] == "Running")

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    for x in range(0, W, 48):
        d.line([(x, 0), (x, H)], fill=GRID, width=1)
    for y in range(0, H, 48):
        d.line([(0, y), (W, y)], fill=GRID, width=1)

    # Vignette the grid away from the lower half so text stays clean.
    veil = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    vd = ImageDraw.Draw(veil)
    for i in range(H // 2, H):
        alpha = int(235 * (i - H // 2) / (H // 2))
        vd.line([(0, i), (W, i)], fill=(10, 25, 47, min(alpha, 235)))
    img = Image.alpha_composite(img.convert("RGBA"), veil).convert("RGB")
    d = ImageDraw.Draw(img)

    d.line([(0, 0), (W, 0)], fill=EMERALD, width=4)

    pad = 84
    y = 96

    d.ellipse([pad, y + 5, pad + 12, y + 17], fill=EMERALD)
    d.text((pad + 26, y), "LIVE PAPER TRADING", font=font(MONO, 20, "Medium"), fill=EMERALD)

    y += 74
    title = font(SANS, 66, "Bold")
    d.text((pad, y), "Automated Multi-Strategy", font=title, fill=HEADING)
    d.text((pad, y + 80), "Trading Engine", font=title, fill=HEADING)

    y += 200
    body = font(SANS, 28)
    d.text(
        (pad, y),
        "Autonomous research-to-production pipeline with strict risk",
        font=body,
        fill=MUTED,
    )
    d.text((pad, y + 42), "orchestration and unattended scheduled execution.", font=body, fill=MUTED)

    rule_y = H - 132
    d.line([(pad, rule_y), (W - pad, rule_y)], fill=BORDER, width=1)

    stats = [
        (f"${capital:,}", "CAPITAL CAPPED"),
        (f"{live}", "LIVE SLEEVES"),
        ("DAILY", "EOD EXECUTION"),
    ]
    x = pad
    for value, label in stats:
        d.text((x, rule_y + 30), value, font=font(MONO, 34, "Medium"), fill=HEADING)
        d.text((x, rule_y + 76), label, font=font(MONO, 18), fill=DIM)
        x += 268

    d.text(
        (W - pad - 250, rule_y + 52),
        "shazilfarukh.com",
        font=font(MONO, 22),
        fill=DIM,
    )

    img.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()

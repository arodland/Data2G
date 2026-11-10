#!/usr/bin/env python3
"""Generate the Data2G registered-user ID card.

    python3 docs/id-card/make_card.py

Writes, next to this script:

    data2g-id-card.svg   one card, CR80 size (85.6 x 54 mm)
    sheet.html           US Letter page of 8 cards with cut marks

The PNG and PDF are rendered from those with Chromium (see README.md).
The ornaments (guilloche border, rosette, seal) are computed here so the
curves close exactly; edit the text or palette below and re-run.
"""

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# CR80 card, 1 unit = 1/300 inch-ish (1012 x 638 units = 85.6 x 54 mm).
W, H = 1012, 638
R_OUT = 34          # outer corner radius

# Palette lifted from native/packaging/data2g.svg.
NAVY = "#0b2545"
BLUE = "#1b4f8a"
SKY = "#8ecae6"
TEAL = "#2a9d8f"
ORANGE = "#f4a259"
PAPER = "#fbf6e9"
PAPER2 = "#f1e7cc"

SERIF = "Georgia, 'Times New Roman', 'Liberation Serif', 'DejaVu Serif', serif"

LINES = [
    "This card certifies that",
    None,  # name line
    "is a registered Data2G user",
    "entitled to the fastest bandwidth and most advanced features",
    "so please treat them with the respect they deserve.",
]
FINE_PRINT = ("Everyone on earth is a registered Data2G user. "
              "Distribute these cards freely. 73, KC2G.")


def f(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- geometry

def roundrect_point(s, x0, y0, w, h, r):
    """Point and outward normal at arc length s along a rounded rect,
    clockwise from the start of the top edge."""
    straight = [w - 2 * r, h - 2 * r, w - 2 * r, h - 2 * r]
    arc = math.pi * r / 2
    total = sum(straight) + 4 * arc
    s %= total
    # corner centres, in the order reached after each straight edge
    centres = [(x0 + w - r, y0 + r), (x0 + w - r, y0 + h - r),
               (x0 + r, y0 + h - r), (x0 + r, y0 + r)]
    starts = [(x0 + r, y0), (x0 + w, y0 + r), (x0 + w - r, y0 + h), (x0, y0 + h - r)]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    normals = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for i in range(4):
        if s <= straight[i]:
            sx, sy = starts[i]
            dx, dy = dirs[i]
            return sx + dx * s, sy + dy * s, normals[i]
        s -= straight[i]
        if s <= arc:
            a0 = -math.pi / 2 + i * math.pi / 2
            a = a0 + s / r
            cx, cy = centres[i]
            nx, ny = math.cos(a), math.sin(a)
            return cx + r * nx, cy + r * ny, (nx, ny)
        s -= arc
    raise AssertionError


def roundrect_len(w, h, r):
    return 2 * (w - 2 * r) + 2 * (h - 2 * r) + 2 * math.pi * r


def path_from(points):
    head = f"M{f(points[0][0])} {f(points[0][1])}"
    return head + "".join(f"L{f(x)} {f(y)}" for x, y in points[1:]) + "Z"


def guilloche_band(x0, y0, w, h, r, amp, cycles, phases, steps=2400):
    """Interlaced sine waves riding along the centre line of a border band."""
    total = roundrect_len(w, h, r)
    paths = []
    for ph in phases:
        pts = []
        for k in range(steps):
            s = total * k / steps
            x, y, (nx, ny) = roundrect_point(s, x0, y0, w, h, r)
            off = amp * math.sin(2 * math.pi * cycles * k / steps + ph)
            pts.append((x + nx * off, y + ny * off))
        paths.append(path_from(pts))
    return paths


def rosette(cx, cy, R, r, d, turns, steps=3000):
    """Hypotrochoid (spirograph) curve."""
    pts = []
    for k in range(steps):
        t = 2 * math.pi * turns * k / steps
        x = (R - r) * math.cos(t) + d * math.cos((R - r) / r * t)
        y = (R - r) * math.sin(t) - d * math.sin((R - r) / r * t)
        pts.append((cx + x, cy + y))
    return path_from(pts)


def rose_ring(cx, cy, r0, amp, petals, phase=0.0, steps=720):
    pts = []
    for k in range(steps):
        t = 2 * math.pi * k / steps
        rr = r0 + amp * math.sin(petals * t + phase)
        pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
    return path_from(pts)


def starburst(cx, cy, r_in, r_out, n):
    pts = []
    for k in range(2 * n):
        rr = r_out if k % 2 == 0 else r_in
        a = math.pi * k / n - math.pi / 2
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return path_from(pts)


# ---------------------------------------------------------------- pieces

def logo(x, y, size):
    """The app icon, inlined and scaled (no text, so it renders anywhere)."""
    s = size / 256
    return f"""
  <g transform="translate({f(x)} {f(y)}) scale({f(s)})">
    <rect x="8" y="8" width="240" height="240" rx="52" fill="url(#logobg)"/>
    <path d="M24 92 Q128 20 232 92" fill="none" stroke="{SKY}" stroke-width="16" stroke-linecap="round"/>
    <path d="M0 214 Q128 166 256 214 V256 H0 Z" fill="{TEAL}" clip-path="url(#logotile)"/>
    <path d="M56 192 L128 66 L200 192" fill="none" stroke="{ORANGE}" stroke-width="20" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="56" cy="192" r="15" fill="#e8f1f8"/>
    <circle cx="200" cy="192" r="15" fill="#e8f1f8"/>
  </g>"""


def seal(cx, cy):
    rings = "".join(
        f'<path d="{rose_ring(cx, cy, 54 - i * 3, 3.2, 24, i * 0.9)}" fill="none" '
        f'stroke="{TEAL}" stroke-width="0.9" opacity="0.85"/>'
        for i in range(4))
    return f"""
  <g id="seal">
    <path d="{starburst(cx, cy, 62, 70, 36)}" fill="{ORANGE}"/>
    <circle cx="{cx}" cy="{cy}" r="60" fill="{PAPER}" stroke="{NAVY}" stroke-width="2"/>
    {rings}
    <circle cx="{cx}" cy="{cy}" r="40" fill="{PAPER}" stroke="{NAVY}" stroke-width="1.2"/>
    <path id="sealtext" d="M{cx - 49} {cy} a49 49 0 1 1 98 0 a49 49 0 1 1 -98 0" fill="none"/>
    <circle cx="{cx}" cy="{cy}" r="56" fill="none" stroke="{NAVY}" stroke-width="0.8"/>
    <text font-family="{SERIF}" font-size="10.5" font-weight="bold" letter-spacing="1.6" fill="{NAVY}">
      <textPath href="#sealtext">REGISTERED ✦ DATA2G ✦ USER ✦ VERIFIED ✦ </textPath>
    </text>
    <text x="{cx}" y="{cy - 6}" text-anchor="middle" font-family="{SERIF}" font-size="11" font-weight="bold" letter-spacing="1" fill="{BLUE}">GENUINE</text>
    <text x="{cx}" y="{cy + 18}" text-anchor="middle" font-family="{SERIF}" font-size="25" font-weight="bold" fill="{NAVY}">73</text>
  </g>"""


def flourish(cx, y, half):
    """Horizontal rule with tapering ends and a diamond in the middle."""
    d = 7
    return f"""
  <g fill="{NAVY}">
    <path d="M{cx - half} {y} Q{cx - half / 2} {y - 2.2} {cx - 20} {y} Q{cx - half / 2} {y + 2.2} {cx - half} {y}Z"/>
    <path d="M{cx + half} {y} Q{cx + half / 2} {y - 2.2} {cx + 20} {y} Q{cx + half / 2} {y + 2.2} {cx + half} {y}Z"/>
    <path d="M{cx} {y - d} L{cx + d} {y} L{cx} {y + d} L{cx - d} {y}Z" fill="{ORANGE}"/>
    <circle cx="{cx - 14}" cy="{y}" r="2.4"/>
    <circle cx="{cx + 14}" cy="{y}" r="2.4"/>
  </g>"""


def corner_medallion(cx, cy):
    return f"""
  <g>
    <circle cx="{f(cx)}" cy="{f(cy)}" r="17" fill="{NAVY}"/>
    <path d="{rose_ring(cx, cy, 11, 3.5, 8, 0, 240)}" fill="none" stroke="{SKY}" stroke-width="1.3"/>
    <path d="{rose_ring(cx, cy, 11, 3.5, 8, math.pi, 240)}" fill="none" stroke="{ORANGE}" stroke-width="1.3"/>
    <circle cx="{f(cx)}" cy="{f(cy)}" r="3" fill="{ORANGE}"/>
  </g>"""


# ---------------------------------------------------------------- card

def card_svg():
    band_in = 14          # outer edge of the guilloche band
    band_w = 26           # band thickness
    mid = band_in + band_w / 2
    gx, gy, gw, gh, gr = mid, mid, W - 2 * mid, H - 2 * mid, R_OUT - mid / 2

    waves_a = guilloche_band(gx, gy, gw, gh, gr, band_w / 2 - 2, 140,
                             [i * math.pi / 3 for i in range(6)])
    waves_b = guilloche_band(gx, gy, gw, gh, gr, band_w / 2 - 6, 61,
                             [0, math.pi])
    guil = "".join(f'<path d="{p}" fill="none" stroke="{BLUE}" stroke-width="0.8" opacity="0.8"/>'
                   for p in waves_a)
    guil += "".join(f'<path d="{p}" fill="none" stroke="{ORANGE}" stroke-width="1.1"/>'
                    for p in waves_b)

    inner = band_in + band_w + 6
    micro_r = 14
    micro_path = (f"M{inner + micro_r} {inner} H{W - inner - micro_r} "
                  f"A{micro_r} {micro_r} 0 0 1 {W - inner} {inner + micro_r} "
                  f"V{H - inner - micro_r} A{micro_r} {micro_r} 0 0 1 {W - inner - micro_r} {H - inner} "
                  f"H{inner + micro_r} A{micro_r} {micro_r} 0 0 1 {inner} {H - inner - micro_r} "
                  f"V{inner + micro_r} A{micro_r} {micro_r} 0 0 1 {inner + micro_r} {inner}Z")
    micro_text = "DATA2G·REGISTERED·USER·" * 40

    bg_rosettes = "".join(
        f'<path d="{rosette(W / 2, H / 2 + 20, 150, 150 * fr, 150 * fr * 1.35, tn)}" '
        f'fill="none" stroke="{col}" stroke-width="0.6" opacity="0.2"/>'
        for fr, tn, col in [(7 / 25, 7, TEAL), (6 / 23, 6, BLUE), (8 / 29, 8, ORANGE)])

    cx = W / 2
    corners = "".join(corner_medallion(x, y) for x, y in
                      [(mid + 6, mid + 6), (W - mid - 6, mid + 6),
                       (mid + 6, H - mid - 6), (W - mid - 6, H - mid - 6)])

    body = f"""
  <text x="{cx}" y="208" text-anchor="middle" font-family="{SERIF}" font-style="italic" font-size="30" fill="{NAVY}">{LINES[0]}</text>
  <line x1="190" y1="290" x2="{W - 190}" y2="290" stroke="{NAVY}" stroke-width="1.6"/>
  <path d="M174 290 l8 -4 v8z M{W - 174} 290 l-8 -4 v8z" fill="{NAVY}"/>
  <text x="{cx}" y="308" text-anchor="middle" font-family="{SERIF}" font-size="11" letter-spacing="3" fill="{BLUE}" opacity="0.8">NAME OF BEARER</text>
  <text x="{cx}" y="360" text-anchor="middle" font-family="{SERIF}" font-weight="bold" font-size="34" fill="{NAVY}">{LINES[2]}</text>
  <text x="{cx}" y="404" text-anchor="middle" font-family="{SERIF}" font-style="italic" font-size="24.5" fill="{NAVY}">{LINES[3]}</text>
  <text x="{cx}" y="437" text-anchor="middle" font-family="{SERIF}" font-style="italic" font-size="24.5" fill="{NAVY}">{LINES[4]}</text>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Data2G registered-user ID card. Generated by make_card.py; edit that, not this. -->
<svg xmlns="http://www.w3.org/2000/svg" width="85.6mm" height="54mm" viewBox="0 0 {W} {H}">
  <defs>
    <linearGradient id="paper" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{PAPER}"/>
      <stop offset="0.55" stop-color="#fdfaf1"/>
      <stop offset="1" stop-color="{PAPER2}"/>
    </linearGradient>
    <linearGradient id="logobg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{NAVY}"/>
      <stop offset="1" stop-color="{BLUE}"/>
    </linearGradient>
    <linearGradient id="banner" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{NAVY}"/>
      <stop offset="1" stop-color="{BLUE}"/>
    </linearGradient>
    <clipPath id="logotile"><rect x="8" y="8" width="240" height="240" rx="52"/></clipPath>
    <clipPath id="cardclip"><rect width="{W}" height="{H}" rx="{R_OUT}"/></clipPath>
  </defs>

  <g clip-path="url(#cardclip)">
  <rect width="{W}" height="{H}" fill="url(#paper)"/>

  <!-- border: solid rules framing a guilloche band -->
  <rect x="{band_in / 2}" y="{band_in / 2}" width="{W - band_in}" height="{H - band_in}" rx="{R_OUT - band_in / 2}" fill="none" stroke="{NAVY}" stroke-width="{band_in}"/>
  <rect x="{band_in}" y="{band_in}" width="{W - 2 * band_in}" height="{H - 2 * band_in}" rx="{R_OUT - band_in}" fill="#eef4f8" stroke="none"/>
  {guil}
  <rect x="{band_in}" y="{band_in}" width="{W - 2 * band_in}" height="{H - 2 * band_in}" rx="{R_OUT - band_in}" fill="none" stroke="{NAVY}" stroke-width="1.5"/>
  <rect x="{band_in + band_w}" y="{band_in + band_w}" width="{W - 2 * (band_in + band_w)}" height="{H - 2 * (band_in + band_w)}" rx="10" fill="url(#paper)" stroke="{NAVY}" stroke-width="2"/>
  {bg_rosettes}
  {corners}

  <!-- microtext security line -->
  <path id="micro" d="{micro_path}" fill="none"/>
  <text font-family="Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif" font-size="6.2" letter-spacing="0.6" fill="{TEAL}" opacity="0.9">
    <textPath href="#micro">{micro_text}</textPath>
  </text>
  </g>

  <!-- header -->
  {logo(78, 62, 92)}
  <text x="{cx - 40}" y="112" text-anchor="middle" font-family="{SERIF}" font-weight="bold" font-size="52" letter-spacing="6" fill="{NAVY}">DATA2G</text>
  <rect x="{cx - 245}" y="124" width="410" height="26" rx="3" fill="url(#banner)"/>
  <text x="{cx - 40}" y="142.5" text-anchor="middle" font-family="{SERIF}" font-size="14.5" letter-spacing="4.5" fill="#ffffff">OFFICIAL USER IDENTIFICATION</text>
  {seal(W - 150, 128)}
{body}
  {flourish(cx, 492, 300)}
  <text x="{cx}" y="{H - 94}" text-anchor="middle" font-family="{SERIF}" font-size="14" fill="{NAVY}" opacity="0.85">{FINE_PRINT}</text>
</svg>
"""


def sheet_html():
    cards = "\n".join('    <div class="cell"><img src="data2g-id-card.svg" alt=""></div>'
                      for _ in range(8))
    return f"""<!doctype html>
<!-- Printable sheet: 8 Data2G ID cards on US Letter. Generated by make_card.py. -->
<html><head><meta charset="utf-8"><title>Data2G ID cards</title>
<style>
  @page {{ size: 8.5in 11in; margin: 0; }}
  html, body {{ margin: 0; background: #fff; }}
  .page {{ width: 8.5in; height: 11in; display: flex; align-items: center; justify-content: center; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, 85.6mm); grid-auto-rows: 54mm; gap: 6mm; }}
  .cell {{ position: relative; }}
  .cell img {{ width: 85.6mm; height: 54mm; display: block; }}
  /* cut marks */
  .cell::before, .cell::after {{ content: ""; position: absolute; inset: -2.5mm; pointer-events: none;
    background:
      linear-gradient(#999,#999) 0 2.5mm / 1.8mm 0.15mm no-repeat,
      linear-gradient(#999,#999) 2.5mm 0 / 0.15mm 1.8mm no-repeat,
      linear-gradient(#999,#999) 100% 2.5mm / 1.8mm 0.15mm no-repeat,
      linear-gradient(#999,#999) calc(100% - 2.5mm) 0 / 0.15mm 1.8mm no-repeat,
      linear-gradient(#999,#999) 0 calc(100% - 2.5mm) / 1.8mm 0.15mm no-repeat,
      linear-gradient(#999,#999) 2.5mm 100% / 0.15mm 1.8mm no-repeat,
      linear-gradient(#999,#999) 100% calc(100% - 2.5mm) / 1.8mm 0.15mm no-repeat,
      linear-gradient(#999,#999) calc(100% - 2.5mm) 100% / 0.15mm 1.8mm no-repeat; }}
  .cell::after {{ display: none; }}
</style></head>
<body><div class="page"><div class="grid">
{cards}
</div></div></body></html>
"""


def main():
    (HERE / "data2g-id-card.svg").write_text(card_svg())
    (HERE / "sheet.html").write_text(sheet_html())
    print(f"wrote {HERE / 'data2g-id-card.svg'} and {HERE / 'sheet.html'}")


if __name__ == "__main__":
    main()

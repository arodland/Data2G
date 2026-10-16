"""The submode map (HTML): every mode on one raw-bps axis, by band, then modulation.

    uv run python scripts/submode_map.py --out <page.html> [--compact] [--csv runs/ladder_dd_10pct.csv]

Rows: the four bands (the FSK modes count as their nearest: fsk8r50 and fsk16r25 at 500,
fsk32r62 at 2400). Within a band, one lane per modulation: a bar over its bps
range and a dot per code rate, labeled rate / bps / AWGN 10% point. Modes
outside a code-rate family (polar acks and short-block polar) share a
"specialty" lane per band, drawn as diamonds and labeled by name.

A ring marks a mode that ranks at least FADE_LEAD places higher (among all
modes, by 10% threshold) on MPG, MPP or MPD than on AWGN.

--compact: one screen high on a wide display. A lane per row, the band
name beside its lanes, each dot labeled only with its rate (above) and AWGN
point (below); bps and the fading channel are on hover.
"""

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).parent))
import ladder_page as L

from data2g.arq.modes import MODES

CH = L.CH
FADE_LEAD = 5
FADED = ' class="faded"'
BANDS = (200, 500, 1200, 2400)
# lane order inside a band, top to bottom
MODS = ("Specialty", "16-FSK", "8-FSK", "32-FSK", "QPSK", "QAM16", "QAM64", "QAM256")
MOD_VAR = {"QPSK": "qpsk", "QAM16": "q16", "QAM64": "q64", "QAM256": "q256",
           "8-FSK": "f8", "16-FSK": "f16", "32-FSK": "f32", "Specialty": "spec"}
MOD_NOTE = {"QAM64": "learned", "QAM256": "learned", "Specialty": "polar acks, short blocks"}

# geometry (SVG user units)
LO, HI = 20, 10000
W, X0, X1 = 1500, 196, 1466           # full
CW, CX0, CX1 = 1700, 180, 1672        # compact
LABEL_W = 34          # a label block's width, centered on its dot
LINE = 12             # label line height
LANE_PAD = 10
BAND_GAP = 22
BAND_HEAD = 30


def band(s) -> int:
    w = L.width(s)
    return min(BANDS, key=lambda b: abs(b - w))


def mod(m, s) -> str:
    if s.code == "polar":
        return "Specialty"
    return L.const_name(s).split(" ")[0]


def x_of(bps: float, x0=X0, x1=X1) -> float:
    return x0 + (math.log10(bps) - math.log10(LO)) / (math.log10(HI) - math.log10(LO)) * (x1 - x0)


def snr_txt(v) -> str:
    if v is None or math.isnan(v):
        return "—"
    return f"{v:.1f}".replace("-", "−")


def load(path) -> dict:
    with open(path) as f:
        return {r["name"]: {c: (float(r[c]) if r.get(c) else math.nan) for c in CH} for r in csv.DictReader(f)}


def ranks(th: dict) -> dict:
    """1 = most robust; ties share the better rank; never-reaches ranks last."""
    out = {}
    for c in CH:
        key = {m: (math.inf if math.isnan(th[m][c]) else th[m][c]) for m in th}
        for m in th:
            out[(m, c)] = 1 + sum(1 for o in th if key[o] < key[m])
    return out


def short(m: str) -> str:
    return m.replace("n4-", "").replace("n10-", "").replace("polar-", "")


def place(items):
    """Greedy label sides: above unless that collides with the last label
    above; then below; else whichever side needs the smaller nudge right."""
    last = {"up": -1e9, "dn": -1e9}
    for it in items:
        x = it["x"]
        want = x - LABEL_W / 2
        free = [s for s in ("up", "dn") if want >= last[s] + 4]
        if free:
            side, lx = free[0], x
        else:
            side = min(last, key=last.get)
            lx = last[side] + 4 + LABEL_W / 2
        it["side"], it["lx"] = side, lx
        last[side] = lx + LABEL_W / 2


def spread(items, gap=3.0):
    """Label centers (it["lx"]) as near their dots as they go without
    overlapping: overlapping runs merge and center on their dots' mean."""
    runs = []  # [items, left edge]
    for it in items:
        runs.append([[it], it["x"] - it["w"] / 2])
        while len(runs) > 1:
            (a, la), (b, lb) = runs[-2], runs[-1]
            if la + sum(i["w"] for i in a) + gap * (len(a) - 1) + gap <= lb:
                break
            m = a + b
            offs, o = [], 0.0
            for i in m:
                offs.append(o + i["w"] / 2)
                o += i["w"] + gap
            runs[-2:] = [[m, sum(i["x"] - f for i, f in zip(m, offs)) / len(m)]]
    for its, left in runs:
        for it in its:
            it["lx"] = left + it["w"] / 2
            left += it["w"] + gap


def lanes_of(modes, b):
    lanes = [(md, sorted((d for d in modes if d["band"] == b and d["mod"] == md), key=lambda d: d["bps"]))
             for md in MODS]
    return [(md, ds) for md, ds in lanes if ds]


def marker(d, md, x, cy, r=6.5, ring=11.5):
    v = MOD_VAR[md]
    g = [f'<rect class="hit" x="{x - 16:.1f}" y="{cy - 16}" width="32" height="32"/>']
    if d["fade"]:
        g.append(f'<circle class="fadering" cx="{x:.1f}" cy="{cy}" r="{ring}"/>')
    if md == "Specialty":
        k = r + 1
        g.append(f'<path class="dot" style="fill:var(--{v})" d="M{x:.1f} {cy - k}l{k} {k}l-{k} {k}l-{k} -{k}z"/>')
    else:
        g.append(f'<circle class="dot" style="fill:var(--{v})" cx="{x:.1f}" cy="{cy}" r="{r}"/>')
    return g


def compact_svg(modes):
    """One row per lane; returns (svg parts, height)."""
    lane_h, pad, gap, top0 = 40, 4, 6, 26
    svg, y = [], top0
    for b in BANDS:
        lanes = lanes_of(modes, b)
        top = y
        y += pad
        rows = []
        for md, ds in lanes:
            rows.append((md, ds, y + lane_h / 2))
            y += lane_h
        y += pad
        svg.append(f'<rect class="bandbg" x="6" y="{top}" width="{CW - 12}" height="{y - top}" rx="7"/>')
        svg.append(f'<text class="bandname c" x="18" y="{rows[0][2] + 5}">{b} Hz</text>'
                   f'<text class="bandsub" x="18" y="{rows[0][2] + 19}">{sum(len(ds) for _, ds in lanes)} modes</text>')
        for i, (md, ds, cy) in enumerate(rows):
            v = MOD_VAR[md]
            if i:
                svg.append(f'<line class="lanesep" x1="88" x2="{CW - 18}" y1="{cy - lane_h / 2}" y2="{cy - lane_h / 2}"/>')
            svg.append(f'<rect x="88" y="{cy - 5}" width="10" height="10" rx="{5 if md != "Specialty" else 2}" '
                       f'style="fill:var(--{v})"/><text class="modname c" x="104" y="{cy + 4}">{md}</text>')
            xs = [d["x"] for d in ds]
            if md == "Specialty":
                svg.append(f'<line class="specline" x1="{min(xs) - 9:.1f}" x2="{max(xs) + 9:.1f}" y1="{cy}" y2="{cy}"/>')
            else:
                svg.append(f'<line class="bar c" style="stroke:var(--{v})" x1="{min(xs):.1f}" x2="{max(xs):.1f}" y1="{cy}" y2="{cy}"/>')
            tops = [{"x": d["x"], "t": short(d["name"]) if md == "Specialty" else d["rate"], "d": d} for d in ds]
            bots = [{"x": d["x"], "t": snr_txt(d["th"]["awgn"]), "d": d} for d in ds]
            for row in (tops, bots):
                for it in row:
                    it["w"] = 6.1 * len(it["t"])
                spread(row, 4.0)
            for d, tp, bt in zip(ds, tops, bots):
                x = d["x"]
                g = [f'<g class="mk" tabindex="0" data-m="{html.escape(d["name"])}">'] + marker(d, md, x, cy, 5, 8.5)
                for it, ty, cls, ly0, ly1 in ((tp, cy - 11, "l-rate", cy - 8, cy - 9), (bt, cy + 19, "l-snr", cy + 8, cy + 10)):
                    if abs(it["lx"] - x) > 2:
                        g.append(f'<path class="leader" d="M{x:.1f} {ly0}L{it["lx"]:.1f} {ly1}"/>')
                    g.append(f'<text class="lbl c {cls}" x="{it["lx"]:.1f}" y="{ty}">{html.escape(it["t"])}</text>')
                g.append("</g>")
                svg.append("".join(g))
        y += gap
    return svg, y - gap + 6


def build(th: dict, compact: bool = False) -> str:
    w, x0, x1 = (CW, CX0, CX1) if compact else (W, X0, X1)
    rk = ranks(th)
    n = len(th)
    modes = []
    for m, s in MODES.items():
        if m not in th:
            continue
        lead = {c: rk[(m, "awgn")] - rk[(m, c)] for c in CH[1:]}
        best = max(lead, key=lead.get)
        nom = re.search(r"-r(\d+/\d+)$", m)
        rtxt = nom.group(1) if nom and s.code != "polar" else L.rate(s)[0]
        modes.append({
            "name": m, "band": band(s), "mod": mod(m, s), "bps": L.bps(s), "x": x_of(L.bps(s), x0, x1), "rate": rtxt,
            "hz": L.width(s), "th": {c: th[m][c] for c in CH}, "rk": {c: rk[(m, c)] for c in CH},
            "fade": best if lead[best] >= FADE_LEAD else None, "lead": lead[best]})

    if compact:
        svg, H = compact_svg(modes)
        return finish(modes, n, svg, H, w, x0, x1, compact)
    svg, y = [], 46
    for b in BANDS:
        lanes = lanes_of(modes, b)
        top = y
        y += BAND_HEAD
        rows = []
        for md, ds in lanes:
            place(ds)
            nl = max(4 if d["fade"] else 3 for d in ds)
            up = any(d["side"] == "up" for d in ds) * nl
            dn = any(d["side"] == "dn" for d in ds) * nl
            cy = y + LANE_PAD + up * LINE + 12
            rows.append((md, ds, cy, nl))
            y = cy + 12 + dn * LINE + LANE_PAD
        bottom = y
        svg.append(f'<rect class="bandbg" x="8" y="{top}" width="{W - 16}" height="{bottom - top}" rx="8"/>')
        cnt = sum(len(ds) for _, ds in lanes)
        svg.append(f'<text class="bandname" x="24" y="{top + 24}">{b} Hz</text>'
                   f'<text class="bandsub" x="{24 + 14 * len(str(b)) + 44}" y="{top + 24}">{cnt} modes</text>')
        for i, (md, ds, cy, nl) in enumerate(rows):
            v = MOD_VAR[md]
            if i:
                sep = cy - 12 - (nl * LINE if any(d["side"] == "up" for d in ds) else 0) - LANE_PAD
                svg.append(f'<line class="lanesep" x1="24" x2="{W - 24}" y1="{sep}" y2="{sep}"/>')
            note = MOD_NOTE.get(md, "")
            svg.append(f'<g class="lanelabel"><rect x="24" y="{cy - 6}" width="12" height="12" rx="{6 if md != "Specialty" else 2}" '
                       f'style="fill:var(--{v})"/><text class="modname" x="44" y="{cy + 4.5}">{md}</text>'
                       + (f'<text class="modnote" x="44" y="{cy + 17}">{note}</text>' if note else "") + "</g>")
            xs = [d["x"] for d in ds]
            if md == "Specialty":
                svg.append(f'<line class="specline" x1="{min(xs) - 10}" x2="{max(xs) + 10}" y1="{cy}" y2="{cy}"/>')
            else:
                svg.append(f'<line class="bar" style="stroke:var(--{v})" x1="{min(xs)}" x2="{max(xs)}" y1="{cy}" y2="{cy}"/>')
            for d in ds:
                x = d["x"]
                g = [f'<g class="mk" tabindex="0" data-m="{html.escape(d["name"])}">'] + marker(d, md, x, cy)
                lines = [(short(d["name"]) if md == "Specialty" else d["rate"], "l-rate"),
                         (f'{d["bps"]:.0f}', "l-bps"), (snr_txt(d["th"]["awgn"]), "l-snr")]
                gap = 4 if d["fade"] else 0
                if d["fade"]:
                    # the tag sits at the far end of the block, away from the ring
                    tag = (f'{d["fade"].upper()}+{d["lead"]}', "l-fade")
                    lines = [tag] + lines if d["side"] == "up" else lines + [tag]
                lx = d["lx"]
                if d["side"] == "up":
                    y0 = cy - 13 - gap - (len(lines) - 1) * LINE
                else:
                    y0 = cy + 22 + gap
                if abs(lx - x) > 1:
                    ly = cy - 9 if d["side"] == "up" else cy + 9
                    ty = y0 + (len(lines) - 1) * LINE + 3 if d["side"] == "up" else y0 - 10
                    g.append(f'<path class="leader" d="M{x:.1f} {ly}L{lx:.1f} {ty}"/>')
                for j, (t, c) in enumerate(lines):
                    g.append(f'<text class="lbl {c}" x="{lx:.1f}" y="{y0 + j * LINE}">{html.escape(t)}</text>')
                g.append("</g>")
                svg.append("".join(g))
        y = bottom + BAND_GAP
    return finish(modes, n, svg, y + 24, w, x0, x1, compact)


def finish(modes, n, svg, H, w, x0, x1, compact) -> str:
    """Axis, the page around the chart, the hover data and the table."""
    # gridlines behind everything; ticks on top, and at the bottom unless compact
    grid, ticks = [], []
    y1 = H - 6 if compact else H - 30
    for dec in (10, 100, 1000, 10000):
        for k in range(1, 10):
            v = dec * k
            if not LO <= v <= HI:
                continue
            x = x_of(v, x0, x1)
            major = k in (1, 2, 5)
            grid.append(f'<line class="{"grid" if major else "gridminor"}" x1="{x:.1f}" x2="{x:.1f}" y1="{24 if compact else 34}" y2="{y1}"/>')
            if major:
                t = f"{v // 1000}k" if v >= 1000 else str(v)
                ticks.append(f'<text class="tick" x="{x:.1f}" y="{16 if compact else 24}">{t}</text>')
                if not compact:
                    ticks.append(f'<text class="tick" x="{x:.1f}" y="{H - 12}">{t}</text>')
    axis_t = f'<text class="axtitle" x="{x0 - 14}" y="{16 if compact else 24}" text-anchor="end">bps</text>'
    if not compact:
        axis_t += f'<text class="axtitle" x="{x0 - 14}" y="{H - 12}" text-anchor="end">bps</text>'
    body = "".join(grid) + "".join(svg) + "".join(ticks) + axis_t
    svgdoc = (f'<svg viewBox="0 0 {w} {H}" width="{w}" height="{H}" role="img" '
              f'aria-label="Data2G submodes by band and modulation against raw bits per second">{body}</svg>')

    data = {d["name"]: {k: d[k] for k in ("band", "mod", "bps", "rate", "hz", "th", "rk", "fade", "lead")}
            for d in modes}
    for d in data.values():
        d["th"] = {c: (None if math.isnan(v) else v) for c, v in d["th"].items()}

    table = []
    for d in sorted(modes, key=lambda d: (d["band"], MODS.index(d["mod"]), d["bps"])):
        cells = "".join(f'<td>{snr_txt(d["th"][c])}<span class="rk">{d["rk"][c]}</span></td>' for c in CH)
        table.append(f'<tr{FADED if d["fade"] else ""}><td class="nm">{html.escape(d["name"])}</td>'
                     f'<td>{d["band"]}</td><td class="tl">{d["mod"]}</td><td>{d["rate"]}</td>'
                     f'<td>{d["bps"]:.0f}</td>{cells}</tr>')
    nf = sum(1 for d in modes if d["fade"])
    page = TEMPLATE
    for k, v in (COMPACT if compact else FULL).items():
        page = page.replace(k, v)
    return (page.replace("{SVG}", svgdoc).replace("{DATA}", json.dumps(data))
            .replace("{ROWS}", "\n".join(table)).replace("{N}", str(n)).replace("{NF}", str(nf))
            .replace("{LEAD}", str(FADE_LEAD)))


FULL = {
    "{TITLE}": "Data2G Submode Map",
    "{MAXW}": "1540px",
    "{MAINCLS}": "",
    "{FADENOTE}": "The ring tag names the fading channel with the biggest gain over the mode's AWGN rank.",
    "{LEDE}": "All {N} submodes on one speed axis. Each band is a panel and each modulation a lane: the bar spans the "
              "lane's speeds and every dot is one code rate. Labels give the code rate, raw bps, and the AWGN SNR "
              "where 10% of bursts fail.",
    "{STACK}": "<b>3/4</b><span>bps</span><span>AWGN dB</span>",
    "{SPEC}": "Specialty (polar), labeled by name",
}
COMPACT = {
    "{TITLE}": "Data2G Submode Map, Compact",
    "{MAXW}": "1760px",
    "{MAINCLS}": ' class="c"',
    "{FADENOTE}": "A ring marks a gain of {LEAD} or more places over the mode's AWGN rank; the tooltip names the channel.",
    "{LEDE}": "All {N} submodes by band and modulation on one raw-bps axis. Each dot is a code rate: rate above, AWGN "
              "SNR for 10% burst failure below. Hover a dot for its bps and all four channels.",
    "{STACK}": "<b>3/4</b><span>&#9679;</span><span>AWGN dB</span>",
    "{SPEC}": "Specialty (polar), labeled by name; QAM64 and QAM256 are learned",
}

TEMPLATE = """<title>{TITLE}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
/* layout: one wide strip chart, four band panels stacked, a lane per modulation; scrolls sideways on narrow screens */
:root{--ground:#F4F6F5;--paper:#FCFCFB;--band:#F1F4F3;--ink:#1C2428;--muted:#5D6A70;--faint:#9AA6AA;--rule:#DCE2E1;--grid:#E4E8E7;--gridminor:#EEF1F0;
--qpsk:#2a78d6;--q16:#1baf7a;--q64:#6250d6;--q256:#e87ba4;--f8:#eb6834;--f16:#eda100;--f32:#e34948;--spec:#7d8a8f;--fade:#008300;--fadeink:#006300;
--sans:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--ground:#111618;--paper:#171D20;--band:#1B2225;--ink:#E2E8EA;--muted:#93A1A7;--faint:#5E6B70;--rule:#283236;--grid:#263034;--gridminor:#1F282B;
--qpsk:#3987e5;--q16:#199e70;--q64:#9085e9;--q256:#d55181;--f8:#d95926;--f16:#c98500;--f32:#e66767;--spec:#8f9ca1;--fade:#0ca30c;--fadeink:#3fc23f}}
:root[data-theme="dark"]{color-scheme:dark;--ground:#111618;--paper:#171D20;--band:#1B2225;--ink:#E2E8EA;--muted:#93A1A7;--faint:#5E6B70;--rule:#283236;--grid:#263034;--gridminor:#1F282B;
--qpsk:#3987e5;--q16:#199e70;--q64:#9085e9;--q256:#d55181;--f8:#d95926;--f16:#c98500;--f32:#e66767;--spec:#8f9ca1;--fade:#0ca30c;--fadeink:#3fc23f}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5}
main{max-width:{MAXW};margin:0 auto;padding-block:32px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font-size:26px;font-weight:600;margin:0 0 6px;text-wrap:balance;letter-spacing:-0.01em}
.lede{color:var(--muted);margin:0 0 18px;max-width:80ch}
.key{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;color:var(--muted);font-size:13px;margin:0 0 14px}
.key>span{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:50%;display:inline-block}
.sw.dia{border-radius:2px;transform:rotate(45deg);width:9px;height:9px}
.ring{width:15px;height:15px;border-radius:50%;border:2px dashed var(--fade);display:inline-block;box-sizing:border-box}
.stack{display:inline-grid;font-family:var(--mono);font-size:10.5px;line-height:1.15;text-align:center;border-left:1px solid var(--rule);padding-left:8px}
.stack b{font-weight:600;color:var(--ink)}
.wrap{overflow-x:auto;background:var(--paper);border:1px solid var(--rule);border-radius:8px;position:relative}
svg{display:block}
svg text{font-family:var(--mono)}
.grid{stroke:var(--grid);stroke-width:1}.gridminor{stroke:var(--gridminor);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;text-anchor:middle}
.axtitle{fill:var(--faint);font-size:11px;font-family:var(--sans)!important;letter-spacing:.06em;text-transform:uppercase}
.bandbg{fill:var(--band);fill-opacity:.72}
.bandname{fill:var(--ink);font-family:var(--sans)!important;font-size:19px;font-weight:700;letter-spacing:-.01em}
.bandsub{fill:var(--faint);font-size:11px}
.lanesep{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 4}
.modname{fill:var(--ink);font-family:var(--sans)!important;font-size:13.5px;font-weight:600}
.modnote{fill:var(--faint);font-family:var(--sans)!important;font-size:11px}
.bar{stroke-width:9;stroke-linecap:round;stroke-opacity:.28}
.specline{stroke:var(--spec);stroke-width:1.5;stroke-dasharray:1 5;stroke-linecap:round}
.dot{stroke:var(--paper);stroke-width:2}
.fadering{fill:none;stroke:var(--fade);stroke-width:2;stroke-dasharray:3.2 2.4}
.hit{fill:transparent}
.lbl{font-size:10.5px;text-anchor:middle;fill:var(--muted)}
.l-rate{fill:var(--ink);font-weight:600}
.l-bps{fill:var(--muted)}
.l-snr{fill:var(--faint)}
.l-fade{fill:var(--fadeink);font-weight:600;font-size:9.5px}
.leader{stroke:var(--faint);stroke-width:1;fill:none}
.mk{cursor:default;outline:none}
.mk:hover .dot,.mk:focus-visible .dot{stroke:var(--ink)}
.mk:focus-visible .hit{stroke:var(--ink);stroke-width:1;stroke-dasharray:2 2}
.svgwrap:has(.mk:hover) .mk:not(:hover) .dot{opacity:.55}
#tip{position:absolute;pointer-events:none;background:var(--paper);border:1px solid var(--rule);border-radius:6px;box-shadow:0 6px 18px rgba(0,0,0,.14);padding:9px 11px;font-size:12.5px;min-width:220px;z-index:5}
#tip h3{margin:0 0 2px;font-family:var(--mono);font-size:13px;font-weight:600}
#tip .sub{color:var(--muted);margin-bottom:6px}
#tip table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums}
#tip td{padding:1px 6px 1px 0;text-align:right}#tip td:first-child{text-align:left;color:var(--muted);font-family:var(--sans)}
#tip tr.hl td{color:var(--fadeink);font-weight:600}
#tip .fadenote{margin-top:6px;color:var(--fadeink);font-size:12px}
.bandname.c{font-size:15px}.modname.c{font-size:12.5px}.bar.c{stroke-width:7}.lbl.c{font-size:10px}.lbl.c.l-snr{fill:var(--muted)}
main.c{padding-block:16px 32px}main.c h1{font-size:21px;margin:0 0 2px}main.c .lede{font-size:13.5px;margin:0 0 8px;max-width:none}main.c .key{font-size:12.5px;margin:0 0 8px}
#tip th{font-family:var(--sans);font-weight:600;font-size:10.5px;color:var(--faint);text-align:right;padding:0 6px 2px 0;text-transform:uppercase;letter-spacing:.05em}
.notes{color:var(--muted);font-size:13px;margin-top:16px;max-width:84ch;display:grid;gap:6px}
.notes p{margin:0}
details{margin-top:22px}
summary{cursor:pointer;font-weight:600;font-size:14px}
.twrap{overflow-x:auto;background:var(--paper);border:1px solid var(--rule);border-radius:8px;margin-top:10px}
.twrap table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.twrap th,.twrap td{padding:5px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule)}
.twrap th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);background:var(--band)}
.twrap td{font-family:var(--mono);font-size:12.5px}
.twrap td.nm,.twrap td.tl,.twrap th.tl{text-align:left}
.twrap td.tl{font-family:var(--sans)}
.rk{color:var(--faint);font-size:10.5px;margin-left:6px;display:inline-block;min-width:2ch}
tr.faded td.nm{color:var(--fadeink);font-weight:600}
@media (prefers-reduced-motion:no-preference){.dot{transition:opacity .12s}}
</style>
<main{MAINCLS}>
<h1>{TITLE}</h1>
<p class="lede">{LEDE}</p>
<div class="key">
<span><span class="sw" style="background:var(--qpsk)"></span>QPSK</span>
<span><span class="sw" style="background:var(--q16)"></span>QAM16</span>
<span><span class="sw" style="background:var(--q64)"></span>QAM64</span>
<span><span class="sw" style="background:var(--q256)"></span>QAM256</span>
<span><span class="sw" style="background:var(--f8)"></span>8-FSK</span>
<span><span class="sw" style="background:var(--f16)"></span>16-FSK</span>
<span><span class="sw" style="background:var(--f32)"></span>32-FSK</span>
<span><span class="sw dia" style="background:var(--spec)"></span>{SPEC}</span>
<span><span class="ring"></span>Ranks {LEAD}+ places higher on a fading channel than on AWGN ({NF} modes)</span>
<span class="stack">{STACK}</span>
</div>
<div class="wrap svgwrap" id="chart">{SVG}<div id="tip" hidden></div></div>
<div class="notes">
<p>Thresholds: the smallest ARQ data burst (a control codeword plus one data codeword) through the receiver the ARQ uses, 200 trials a point, 10% failure. SNR is average transmitted power over noise in 2500 Hz. MPG, MPP and MPD are ITU-R F.1487 channels: 0.1/0.5, 1/2 and 2/4 Hz Doppler / ms delay.</p>
<p>Fading rank: every mode is ranked 1 to {N} by threshold on each channel (1 is most robust). {FADENOTE} Hover or focus a dot for all four channels.</p>
<p>Raw bps is payload rate without CRC or burst overhead, as on the submode ladder. Bands: the FSK modes sit with their nearest neighbor (8-FSK and 16-FSK at 500 Hz, 32-FSK, 2300 Hz wide, at 2400 Hz). QAM64 and QAM256 are learned, non-square constellations.</p>
</div>
<details><summary>Table view</summary>
<div class="twrap"><table>
<thead><tr><th class="tl">Mode</th><th>Hz</th><th class="tl">Modulation</th><th>Rate</th><th>bps</th><th>AWGN · rank</th><th>MPG · rank</th><th>MPP · rank</th><th>MPD · rank</th></tr></thead>
<tbody>
{ROWS}
</tbody></table></div>
</details>
</main>
<script>
const D={DATA};
const CHS=["awgn","mpg","mpp","mpd"];
const wrap=document.getElementById("chart"),tip=document.getElementById("tip");
const f=v=>v==null?"never":v.toFixed(2).replace("-","\\u2212");
function show(g){
  const m=g.dataset.m,d=D[m];if(!d)return;
  const rows=CHS.map(c=>`<tr class="${d.fade===c?"hl":""}"><td>${c.toUpperCase()}</td><td>${f(d.th[c])} dB</td><td>#${d.rk[c]}</td></tr>`).join("");
  tip.innerHTML=`<h3>${m}</h3><div class="sub">${d.hz} Hz · ${d.mod} · rate ${d.rate} · ${Math.round(d.bps)} bps</div>`+
    `<table><tr><th></th><th>10% point</th><th>rank</th></tr>${rows}</table>`+
    (d.fade?`<div class="fadenote">Ranks ${d.lead} places higher on ${d.fade.toUpperCase()} than on AWGN</div>`:"");
  tip.hidden=false;
  const r=g.getBoundingClientRect(),w=wrap.getBoundingClientRect();
  let x=r.left-w.left+wrap.scrollLeft+r.width/2+14,y=r.top-w.top+r.height/2-20;
  const tw=tip.offsetWidth;
  if(x+tw>wrap.scrollLeft+wrap.clientWidth-8)x=r.left-w.left+wrap.scrollLeft-tw-4;
  tip.style.left=x+"px";tip.style.top=Math.max(4,y)+"px";
}
document.querySelectorAll(".mk").forEach(g=>{
  g.addEventListener("mouseenter",()=>show(g));g.addEventListener("focus",()=>show(g));
  g.addEventListener("mouseleave",()=>tip.hidden=true);g.addEventListener("blur",()=>tip.hidden=true);
});
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/ladder_dd_10pct.csv", help="ladder_study's 10%% csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compact", action="store_true", help="one screen high; bps and fading tags on hover only")
    a = ap.parse_args()
    Path(a.out).write_text(build(load(a.csv), a.compact))


if __name__ == "__main__":
    main()

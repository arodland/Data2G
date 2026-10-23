"""Render the speed-test comparison charts and tables in this directory.

Writes one SVG per channel and bandwidth, plus the Markdown pages that embed
them. No dependencies beyond the standard library:

    python docs/speedtests/make_charts.py
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent

# B/min, mean of the middle 3 of 5 seeds. Points are [snr, value] or
# [snr, value, seeds, note]; fewer than 5 seeds marks an interim point, drawn
# hollow and left off the line.
#
# data2g: speedtrials.py (simulated) with the paper's message sizes.
# vara: VARA HF 4.8.9 through our real-time PipeWire loopback (2300 only).
# paper / pactor: the IONOS SIM paper, sim-mar-2021.xlsx (WGN / MPG 2 / MPP 2).
BW2300 = {
    "AWGN": {
        "data2g": [[40, 47909], [35, 47909], [30, 47909], [25, 45645], [20, 33335], [15, 21000], [10, 12175], [5, 5915], [0, 2534], [-5, 798]],
        "vara": [[40, 42159], [35, 41942], [30, 42046], [25, 29924], [20, 20034], [15, 8605], [10, 4599], [5, 1803, 5, "5 of 6 transfers completed"], [0, 1121], [-5, 342]],
        "paper": [[40, 46227], [35, 46055], [30, 46208], [25, 37490], [20, 23511], [15, 11446], [10, 5456], [5, 2331], [0, 1158], [-5, 514]],
        "pactor": [[40, 40326], [35, 40162], [30, 40336], [25, 40286], [20, 40249], [15, 24261], [10, 15872], [5, 3994], [0, 1635], [-5, 621]],
    },
    "MPG": {
        "data2g": [[40, 47558], [35, 47448], [30, 43567], [25, 30361], [20, 21445], [15, 13170], [10, 7644], [5, 3130], [0, 1222], [-5, 358]],
        "vara": [[40, 26037], [35, 25599], [30, 20566], [25, 15512], [20, 8604], [15, 4153], [10, 1897], [5, 900], [0, 333, 5, "5 of 6 transfers completed"], [-5, 88, 1, "1 of 2 transfers completed, then cancelled"]],
        "paper": [[40, 31158], [35, 31284], [30, 28972], [25, 19282], [20, 11745], [15, 6829], [10, 3060], [5, 1482], [0, 839], [-5, 214]],
        "pactor": [[40, 39762], [35, 40375], [30, 37572], [25, 28527], [20, 21421], [15, 15653], [10, 7565], [5, 2232], [0, 1361], [-5, 455]],
    },
    "MPP": {
        "data2g": [[40, 44837], [35, 43800], [30, 39727], [25, 33489], [20, 23891], [15, 17042], [10, 8816], [5, 3566], [0, 1482], [-5, 430]],
        "vara": [[40, 25493], [35, 22750], [30, 20589], [25, 14970], [20, 9171], [15, 5511], [10, 2006], [5, 1073], [0, 465], [-5, 120, 2, "2 of 2 transfers completed, then cancelled"]],
        "paper": [[40, 19711], [35, 21031], [30, 17911], [25, 17694], [20, 10945], [15, 7215], [10, 3672], [5, 1421], [0, 988], [-5, 326]],
        "pactor": [[40, 38515], [35, 35766], [30, 34175], [25, 24062], [20, 22078], [15, 15161], [10, 10134], [5, 2762], [0, 1121], [-5, 431]],
    },
}

BW500 = {
    "AWGN": {
        "data2g": [[40, 10096], [35, 10096], [30, 10096], [25, 10096], [20, 7482], [15, 7526], [10, 4969], [5, 3245], [0, 1527], [-5, 717]],
        "paper": [[40, 10377], [35, 10388], [30, 10402], [25, 10322], [20, 7853], [15, 5897], [10, 3199], [5, 1183], [0, 462], [-5, 211]],
        "pactor": [[40, 4060], [35, 4052], [30, 4074], [25, 4163], [20, 4183], [15, 4049], [10, 2372], [5, 2362], [0, 1161], [-5, 541]],
    },
    "MPG": {
        "data2g": [[40, 10096], [35, 10094], [30, 9435], [25, 7275], [20, 5535], [15, 4059], [10, 2786], [5, 1489], [0, 690], [-5, 169, 5, "5 of 11 transfers completed; 6 hit the 1-hour simulation limit mid-transfer"]],
        "paper": [[40, 7640], [35, 7344], [30, 6955], [25, 5443], [20, 4749], [15, 2389], [10, 1382], [5, 526], [0, 193], [-5, 84]],
        "pactor": [[40, 3416], [35, 3450], [30, 3544], [25, 3283], [20, 2691], [15, 2480], [10, 1731], [5, 1353], [0, 726], [-5, 355]],
    },
    "MPP": {
        "data2g": [[40, 8001], [35, 7965], [30, 7847], [25, 7090], [20, 6537], [15, 4900], [10, 3215], [5, 1973], [0, 814], [-5, 237]],
        "paper": [[40, 6269], [35, 5883], [30, 5072], [25, 4897], [20, 4336], [15, 2590], [10, 1601], [5, 839], [0, 312], [-5, 104]],
        "pactor": [[40, 1031], [35, 1037], [30, 2028], [25, 1825], [20, 1497], [15, 1432], [10, 1282], [5, 894], [0, 672], [-5, 250]],
    },
}

ORANGE, BLUE, GREEN = "#eb6834", "#2a78d6", "#1baf7a"
SNRS = [40, 35, 30, 25, 20, 15, 10, 5, 0, -5]

W, H = 760, 450
ML, MR, MT, MB = 52, 24, 40, 70
X0, X1 = -5, 40


def norm(p):
    s, v, *rest = p
    n = rest[0] if rest else 5
    note = rest[1] if len(rest) > 1 else None
    return s, v, n, note


def fmt(v):
    if v >= 1000:
        return f"{v / 1000:g}k"
    return str(v)


def svg_chart(title, series, y0, y1, yticks):
    """series: [(label, color, dashed, points)]"""
    lx = lambda s: ML + (s - X0) / (X1 - X0) * (W - ML - MR)
    ly = lambda v: H - MB - (math.log10(v) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) * (H - MT - MB)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
        f'<text x="{ML}" y="22" font-size="15" font-weight="600" fill="#0b0b0b">{title}</text>',
        f'<text x="{W - MR}" y="22" font-size="12" fill="#898781" text-anchor="end">B/min, log scale</text>',
    ]
    for v in yticks:
        y = ly(v)
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e1e0d9"/>')
        out.append(f'<text x="{ML - 6}" y="{y + 4:.1f}" font-size="11" fill="#898781" text-anchor="end">{fmt(v)}</text>')
    for s in range(X0, X1 + 1, 5):
        x = lx(s)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{H - MB}" stroke="#e1e0d9" stroke-dasharray="2 3"/>')
        out.append(f'<text x="{x:.1f}" y="{H - MB + 16}" font-size="11" fill="#898781" text-anchor="middle">{s}</text>')
    out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{H - MB}" y2="{H - MB}" stroke="#c3c2b7"/>')
    out.append(f'<text x="{(ML + W - MR) / 2:.1f}" y="{H - MB + 34}" font-size="11" fill="#898781" text-anchor="middle">SNR, dB (PEP over noise in 3 kHz)</text>')

    for label, color, dashed, pts in series:
        pts = sorted((norm(p) for p in pts), key=lambda p: p[0])
        full = [p for p in pts if p[2] >= 5]
        d = "".join(f'{"L" if i else "M"}{lx(s):.1f},{ly(v):.1f}' for i, (s, v, _, _) in enumerate(full))
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        for s, v, n, _ in pts:
            if n < 5:
                out.append(f'<circle cx="{lx(s):.1f}" cy="{ly(v):.1f}" r="4" fill="#fcfcfb" stroke="{color}" stroke-width="2"/>')
            else:
                out.append(f'<circle cx="{lx(s):.1f}" cy="{ly(v):.1f}" r="4" fill="{color}" stroke="#fcfcfb" stroke-width="2"/>')

    # legend row centred under the axis title: a line-and-dot swatch, then the label
    sw, gap, pad = 28, 6, 24
    widths = [sw + gap + len(label) * 6.5 for label, *_ in series]
    x = (ML + W - MR - sum(widths) - pad * (len(series) - 1)) / 2
    y = H - 14
    for (label, color, dashed, _), w in zip(series, widths):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out.append(f'<line x1="{x:.1f}" x2="{x + sw:.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2" stroke-linecap="round"{dash}/>')
        out.append(f'<circle cx="{x + sw / 2:.1f}" cy="{y}" r="4" fill="{color}" stroke="#fcfcfb" stroke-width="2"/>')
        out.append(f'<text x="{x + sw + gap:.1f}" y="{y + 4}" font-size="11" font-weight="600" fill="#52514e">{label}</text>')
        x += w + pad
    out.append("</svg>")
    return "\n".join(out) + "\n"


def legend_md(entries):
    return " · ".join(entries)


def table_md(data, cols):
    chans = list(data)
    head = "| SNR | " + " | ".join(f"{c} {h}" for c in chans for _, h in cols) + " |"
    sep = "|---:|" + "---:|" * (len(chans) * len(cols))
    rows = [head, sep]
    for snr in SNRS:
        cells = []
        for c in chans:
            for key, _ in cols:
                pt = next((norm(p) for p in data[c][key] if p[0] == snr), None)
                if pt is None:
                    cells.append("–")
                else:
                    _, v, n, _ = pt
                    cells.append(f"{v:,}" + (f" ({n})" if n < 5 else ""))
        rows.append(f"| {snr} dB | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def page_2300():
    series = lambda d: [
        ("PACTOR 4", GREEN, True, d["pactor"]),
        ("VARA, paper", BLUE, True, d["paper"]),
        ("VARA, ours", BLUE, False, d["vara"]),
        ("Data2G", ORANGE, False, d["data2g"]),
    ]
    charts = []
    for chan, d in BW2300.items():
        name = f"bw2300-{chan.lower()}.svg"
        (OUT / name).write_text(svg_chart(f"{chan} — Data2G, VARA HF 2300, PACTOR 4", series(d), 60, 60000,
                                          [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]))
        charts.append(f"### {chan}\n\n![{chan}: Data2G, VARA and PACTOR 4 throughput against SNR]({name})")
    return f"""# Data2G and VARA HF 2300 against the IONOS SIM paper

<!-- Generated by make_charts.py; edit the data there and re-run. -->

Payload throughput against SNR (PEP over noise in 3 kHz), mean of the middle 3
of 5 seeds.

- **Data2G** at its 2400 Hz cap is our speed test: the same channel model
  (`noise.py`, `channel.py`) simulated, with the host's turnaround latency.
- **VARA, ours** is VARA HF 4.8.9 measured through our real-time PipeWire
  loopback on that model.
- **VARA, paper** is VARA 4.0 on the IONOS SIM, from the March 2021
  spreadsheet, with the same spreadsheet's **PACTOR 4** (SCS DR-7800) for
  reference.

Solid lines are ours (Data2G simulated, VARA 4.8.9 loopback); dashed lines are
the paper (VARA 4.0, PACTOR 4 on the IONOS SIM). Hollow points average fewer
than 5 good seeds (mostly failed, or cancelled) and are left off the line.
The y axis is log scale: a horizontal gap between the lines is the same number
of dB at any rate.

AWGN is additive white Gaussian noise (the paper's WGN); MPG and MPP are CCIR
"good" and "poor" multipath, simulated with 2 rays (the paper's MPG 2 and
MPP 2).

{chr(10).join(c + chr(10) for c in charts)}
## Notes

- Message sizes are the paper's for each point (205,451 B at high SNR, down to
  9,732 B), so every curve includes the same start-up share. Data2G completed
  every transfer but one: at −5 dB on MPP, 5 of 6.
- VARA at 5 dB on AWGN and 0 dB on MPG completed 5 of 6 transfers.
- VARA at −5 dB on MPG and MPP was cancelled after 2 transfers (runs of hours
  each), so those points are hollow: MPG 1 of 2 completed, MPP 2 of 2.

## Table

B/min. A count in brackets marks a VARA figure averaged over fewer than 5
seeds.

{table_md(BW2300, [("data2g", "Data2G"), ("vara", "VARA ours"), ("paper", "VARA paper"), ("pactor", "PACTOR 4")])}
"""


def page_500():
    series = lambda d: [
        ("PACTOR 2", GREEN, True, d["pactor"]),
        ("VARA 500", BLUE, True, d["paper"]),
        ("Data2G", ORANGE, False, d["data2g"]),
    ]
    charts = []
    for chan, d in BW500.items():
        name = f"bw500-{chan.lower()}.svg"
        (OUT / name).write_text(svg_chart(f"{chan} — Data2G BW500, VARA HF 500, PACTOR 2", series(d), 60, 20000,
                                          [100, 200, 500, 1000, 2000, 5000, 10000, 20000]))
        charts.append(f"### {chan}\n\n![{chan}: Data2G, VARA HF 500 and PACTOR 2 throughput against SNR]({name})")
    return f"""# Data2G and VARA HF 500 against the IONOS SIM paper

<!-- Generated by make_charts.py; edit the data there and re-run. -->

Payload throughput against SNR (PEP over noise in 3 kHz), mean of the middle 3
of 5 seeds.

- **Data2G** at its 500 Hz cap (BW500) is our speed test: the same channel
  model (`noise.py`, `channel.py`) simulated, with the host's turnaround
  latency.
- **VARA HF 500** (VARA 4.0) and **PACTOR 2** are the IONOS SIM paper's
  figures, from the March 2021 spreadsheet, HF 500 sheet, on WGN, MPG 2 and
  MPP 2.

The solid line is Data2G (simulated); dashed lines are the paper (IONOS SIM).
Hollow points average fewer than 5 good seeds and are left off the line. The
y axis is log scale: a horizontal gap between the lines is the same number of
dB at any rate.

AWGN is additive white Gaussian noise (the paper's WGN); MPG and MPP are CCIR
"good" and "poor" multipath, simulated with 2 rays (the paper's MPG 2 and
MPP 2).

{chr(10).join(c + chr(10) for c in charts)}
## Notes

- Message sizes are the paper's VARA 500 sizes for each point (137,007 B at
  high SNR, down to 8,640 B), so Data2G and VARA include the same start-up
  share. PACTOR 2 used smaller files in the paper (69,311 B on WGN, 19,451 B
  on fading, down to 8,654 B), which costs it some rate to start-up.
- At −5 dB on MPG, 5 of 10 Data2G transfers completed: the other 5 were still
  running when the simulation's 1-hour limit stopped them (8,640 B at ~160
  B/min takes about an hour).

## Table

B/min. A count in brackets marks a Data2G figure averaged over fewer than 5
seeds.

{table_md(BW500, [("data2g", "Data2G"), ("paper", "VARA 500"), ("pactor", "PACTOR 2")])}
"""


INDEX = """# Speed tests

Data2G's throughput against VARA HF and PACTOR on the channel models of the
IONOS SIM paper, from 40 dB down to −5 dB SNR:

- **AWGN**: additive white Gaussian noise (the paper's WGN).
- **MPG**: CCIR "good" multipath, simulated with 2 rays (the paper's MPG 2).
- **MPP**: CCIR "poor" multipath, simulated with 2 rays (the paper's MPP 2).

## Results

- [2300 Hz: Data2G vs VARA HF 2300 and PACTOR 4](vara-2300.md)
- [500 Hz: Data2G BW500 vs VARA HF 500 and PACTOR 2](vara-500.md)

The charts and tables are generated by [`make_charts.py`](make_charts.py).
"""


if __name__ == "__main__":
    (OUT / "vara-2300.md").write_text(page_2300())
    (OUT / "vara-500.md").write_text(page_500())
    (OUT / "README.md").write_text(INDEX)

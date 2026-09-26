"""The ladder artifact (HTML) from the ladder study's CSVs.

    uv run python scripts/ladder_page.py runs/ladder_10pct.csv --out <page.html>

10% column: scripts/ladder_study.py (the smallest ARQ data burst through
the ARQ's receiver). 1% column: the freeze (runs/ladder_final.csv and
runs/sync_floor.csv: the larger of the code's and the band's sync point);
CPM's from the prototype study, before the c8r50 sync change.
"""

import argparse
import csv
import html
import sys
from fractions import Fraction
from pathlib import Path

from data2g import codes, cpm
from data2g.arq import policy as G
from data2g.arq.modes import MODES, is_cpm
from data2g.config import FRAME_SAMPLES, FS

sys.path.insert(0, str(Path(__file__).parent))
import ir_study as I  # noqa: E402
import outcome_data as O  # noqa: E402

CH = ("awgn", "mpg", "mpp", "mpd")
# another mode of no greater width does as well at every channel x 2 dB cell
# of the outcome data (greedy envelope, <= 0.2% loss; README, CPM section)
COVERED = {"polar-k96-f8", "polar-k192-f8", "n4-16qam-r1/3", "n4-qpsk-r1/2", "n4-qpsk-r2/3", "n4-qpsk-r1/5",
           "n4-ack-8f", "n4-ack-2f", "fsk16r25-r1/3", "fsk8r50-r1/3", "fsk8r50-r1/2", "w48-qpsk-r2/3",
           "w48-16qam-r3/4"}
ROLES = {m: ["reply"] for m in ("ack-1f", "ack-4f", "n10-ack-4f", "n4-ack-2f", "n4-ack-8f")}
for cap, m in G.CONNECT.items():
    ROLES.setdefault(m, []).append("connect")
ROLES.setdefault(G.ROBUST_CONNECT, []).append("connect retry")
ROLES = {m: sorted(set(v)) for m, v in ROLES.items()}


def const_name(s) -> str:
    c = s.constellation
    if c.startswith("fsk"):
        return f"{c[3:]}-FSK"
    if "qam4" in c or c == "qpsk":
        return "QPSK"
    for q in ("256", "64", "16"):
        if q in c:
            return f"QAM{q}" + (" (learned)" if q in ("64", "256") else "")
    return c


def rate(s) -> tuple[str, float]:
    f = Fraction(s.k, s.coded_bits)
    return (f"{f.numerator}/{f.denominator}" if f.denominator <= 20 else f"{float(f):.2f}"), float(f)


def bps(s) -> float:
    if is_cpm(s):
        g = cpm.GRIDS[s.grid]
        return codes.payload_bytes(s) * 8 / (cpm.DATA_N / g.bits / g.rate)
    return codes.payload_bytes(s) * 8 / (s.frames_per_cw * FRAME_SAMPLES / FS)


def width(s) -> int:
    return round(G.width_hz(s))


def freeze_1pct() -> dict:
    ladder = I.thresholds()
    floor = {(r["band"], r["channel"]): float(r["sync_threshold_db"]) for r in csv.DictReader(open("runs/sync_floor.csv"))}
    out = {}
    for m, s in MODES.items():
        for c in CH:
            if is_cpm(s):
                v = O.CPM_THRESHOLDS[m][CH.index(c)]
            else:
                t = ladder.get((I.ladder_name(s), c))
                v = None if t is None else max(t, floor.get((s.band, c), -99))
            out[(m, c)] = v
    return out


def fmt(v) -> str:
    if v is None or v != v:
        return "—"
    return f"{v:.2f}".rstrip("0").rstrip(".").replace("-", "−")


def page(p10: dict, p1: dict) -> str:
    rows = sorted(MODES, key=lambda m: (bps(MODES[m]), width(MODES[m])))
    body = []
    for m in rows:
        s = MODES[m]
        w = width(s)
        chip = "bw200" if w <= 200 else "bw500" if w <= 500 else "bw1200" if w <= 1200 else "bw2400"
        fam = "CPM" if is_cpm(s) else ""
        code = ("Polar" if s.code == "polar" else "LDPC")
        rtxt, _ = rate(s)
        tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in ROLES.get(m, []))
        cls = ' class="covered"' if m in COVERED and m not in ROLES else ""
        cells10 = "".join(f"<td>{fmt(p10.get((m, c)))}</td>" for c in CH)
        cells1 = "".join(f'<td class="p1">{fmt(p1.get((m, c)))}{"†" if is_cpm(s) else ""}</td>' for c in CH)
        body.append(f'<tr{cls}><td class="name">{html.escape(m)}{tags}</td><td>{bps(s):.0f}</td>'
                    f'<td><span class="bw {chip}">{w}</span></td><td>{code}{" · " + fam if fam else ""}</td>'
                    f"<td>{rtxt}</td><td>{const_name(s)}</td>{cells10}{cells1}</tr>")
    n10 = sum(1 for m in MODES for c in CH if (m, c) in p10)
    return TEMPLATE.replace("{ROWS}", "\n".join(body)).replace("{N}", str(len(MODES))).replace(
        "{DONE}", "" if n10 == 4 * len(MODES) else f'<p class="pending">10% points measured so far: {n10} of {4 * len(MODES)} cells; the rest show “—”.</p>')


TEMPLATE = """<title>Data2G Submode Ladder</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--ground:#F4F6F5;--paper:#FFFFFF;--ink:#1C2428;--muted:#5D6A70;--faint:#9AA6AA;--rule:#DCE2E1;--head:#EBEFEE;--tag:#E4EAE9;
--b200:#E7DCF3;--b200i:#5B3A86;--b500:#DCEBF6;--b500i:#1F5C85;--b1200:#DDF0E6;--b1200i:#23694A;--b2400:#F6E6D6;--b2400i:#8A4A16;
--sans:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--ground:#111618;--paper:#171D20;--ink:#E2E8EA;--muted:#93A1A7;--faint:#5E6B70;--rule:#283236;--head:#1D2528;--tag:#243034;
--b200:#2E2440;--b200i:#CDB6EE;--b500:#1C2E3C;--b500i:#9CCCEE;--b1200:#1B3128;--b1200i:#9BD9B8;--b2400:#3A2A1C;--b2400i:#EDBE92}}
:root[data-theme="dark"]{color-scheme:dark;--ground:#111618;--paper:#171D20;--ink:#E2E8EA;--muted:#93A1A7;--faint:#5E6B70;--rule:#283236;--head:#1D2528;--tag:#243034;
--b200:#2E2440;--b200i:#CDB6EE;--b500:#1C2E3C;--b500i:#9CCCEE;--b1200:#1B3128;--b1200i:#9BD9B8;--b2400:#3A2A1C;--b2400i:#EDBE92}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5}
main{max-width:1180px;margin:0 auto;padding-block:32px 48px;padding-inline:16px}
h1{font-size:26px;font-weight:600;margin:0 0 6px;text-wrap:balance;letter-spacing:-0.01em}
.lede{color:var(--muted);margin:0 0 20px;max-width:72ch}
.wrap{overflow-x:auto;background:var(--paper);border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;white-space:nowrap}
th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;font-weight:600;color:var(--muted);background:var(--head);border-bottom:1px solid var(--rule)}
th.grp{text-align:center}
td{font-family:var(--mono);font-size:13px;border-bottom:1px solid var(--rule)}
td.name,th.name{text-align:left}
td.name{font-size:12.5px}
td:nth-child(4),td:nth-child(6){text-align:left;font-family:var(--sans);font-size:13.5px}
th:nth-child(4),th:nth-child(6){text-align:left}
td:nth-child(7),td:nth-child(11),th.c10,th.c1{border-left:1px solid var(--rule)}
td.p1{color:var(--muted)}
tr:last-child td{border-bottom:0}
tr.covered td{color:var(--faint)}
tr.covered .bw{opacity:.55}
.bw{display:inline-block;min-width:3.4em;text-align:center;padding:1px 6px;border-radius:3px;font-size:12px}
.bw200{background:var(--b200);color:var(--b200i)}.bw500{background:var(--b500);color:var(--b500i)}
.bw1200{background:var(--b1200);color:var(--b1200i)}.bw2400{background:var(--b2400);color:var(--b2400i)}
.tag{display:inline-block;margin-left:6px;padding:0 5px;border-radius:3px;background:var(--tag);color:var(--muted);font-family:var(--sans);font-size:11px}
.key{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:13px;margin:0 0 12px}
.key .sw{display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--faint);margin-right:6px;vertical-align:-1px}
.notes{color:var(--muted);font-size:13px;margin-top:16px;max-width:78ch;display:grid;gap:6px}
.notes p,.pending{margin:0}
.pending{color:var(--muted);font-size:13px;margin-bottom:10px}
</style>
<main>
<h1>Data2G Submode Ladder</h1>
<p class="lede">All {N} modes the gear shifter can choose from (OFDM, and constant-envelope CPM), by payload rate. Each threshold is the SNR where bursts fail at most the stated fraction of the time; lower is more robust.</p>
<div class="key"><span><span class="sw"></span>Greyed: another mode no wider does as well everywhere, so the shifter rarely picks it</span><span><span class="tag" style="margin:0 6px 0 0">reply</span>role besides data</span></div>
{DONE}
<div class="wrap"><table>
<thead><tr><th rowspan="2" class="name">Mode</th><th rowspan="2">bps</th><th rowspan="2">Hz</th><th rowspan="2">Code</th><th rowspan="2">FEC rate</th><th rowspan="2">Const.</th><th class="grp c10" colspan="4">10% failure, dB</th><th class="grp c1" colspan="4">1% failure, dB (freeze)</th></tr>
<tr><th class="c10">AWGN</th><th>MPG</th><th>MPP</th><th>MPD</th><th class="c1">AWGN</th><th>MPG</th><th>MPP</th><th>MPD</th></tr></thead>
<tbody>
{ROWS}
</tbody></table></div>
<div class="notes">
<p>SNR is average transmitted power over noise in 2500 Hz. MPG, MPP and MPD are ITU-R F.1487 channels: 0.1/0.5, 1/2 and 2/4 Hz Doppler / ms delay.</p>
<p><b>10%:</b> the smallest ARQ data burst (a control codeword plus one data codeword) through the receiver the ARQ uses: header right and both codewords decoded, 200 trials a point, 0.25 dB steps, ±50 Hz and 10 ppm offsets.</p>
<p><b>1% (freeze):</b> 16-frame bursts; the larger of the code’s 1% packet-error point and the band’s 1% sync point. † CPM: the prototype study’s one-codeword bursts, measured before the c8r50 sync fix.</p>
<p>Payload rate excludes CRC and burst overhead. QAM64 and QAM256 are learned (non-square) constellations. “—”: not measured, or never reaches the point on that channel.</p>
</div>
</main>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p10 = {}
    for r in csv.DictReader(open(a.p10)):
        for c in CH:
            if r.get(c) not in (None, ""):
                p10[(r["name"], c)] = float(r[c])
    Path(a.out).write_text(page(p10, freeze_1pct()))


if __name__ == "__main__":
    main()

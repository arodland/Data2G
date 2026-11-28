"""sc_diag CSVs side by side: per mode, channel and cell, each arm/flag's
codewords decoded (no DD / DD), estimate error before / after DD, measured /
predicted data error."""
import csv
import sys

R = [r for p in sys.argv[1:] for r in csv.DictReader(open(p))]
arms = list(dict.fromkeys((r["arm"], r["flag"]) for r in R))
cells = sorted({(r["mode"], r["chan"], float(r["snr"])) for r in R})
I = {(r["arm"], r["flag"], r["mode"], r["chan"], float(r["snr"])): r for r in R}
print("mode chan snr | " + " | ".join(f"{a}{'+sc' if f == '1' else ''}" for a, f in arms) + "   [ok/okDD of cw; mse/mseDD; meas/pred]")
for c in cells:
    out = []
    for a, f in arms:
        r = I.get((a, f, *c))
        out.append("-" if r is None else f"{int(r['ok']) / int(r['cw']):.2f}/{int(r['ok_dd']) / int(r['cw']):.2f} {r['mse_db']}/{r['mse_dd_db']} {r['meas_db']}/{r['pred_db']}")
    print(f"{c[0]} {c[1]} {c[2]:g} | " + " | ".join(out))
print("peak dB: " + "; ".join(f"{m} " + " ".join(f"{a}{'+sc' if f == '1' else ''} {next(r['peak_db'] for r in R if (r['arm'], r['flag'], r['mode']) == (a, f, m))}" for a, f in arms if any((r['arm'], r['flag'], r['mode']) == (a, f, m) for r in R)) for m in dict.fromkeys(r["mode"] for r in R)))

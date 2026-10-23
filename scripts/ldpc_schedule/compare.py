"""python scripts/ldpc_schedule/compare.py RESULTDIR (decode.py's): per mode/channel, the SNR of the 10% and 1% FER points (interpolated in log FER)
for flooding at cap 40 and layered at several caps, as dB against flooding; and iterations spent per
codeword (converged: its iteration; failed: the cap) around the cliff."""
import sys, collections
from pathlib import Path
import numpy as np

R = Path(sys.argv[1])
CAPS = [10, 15, 20, 25, 30, 40]
pts = collections.defaultdict(list)
for f in R.glob("*.npz"):
    mode, chan, snr = f.stem.split("__")
    z = np.load(f)
    pts[(mode, chan)].append((float(snr), z))


def fer(conv, right, cap):
    return 1 - np.mean((conv <= cap) & right)


def crossing(snrs, fers, target):
    """Lowest SNR where FER falls to target, interpolated in log10 FER; None if not bracketed."""
    for i in range(len(snrs) - 1):
        a, b = fers[i], fers[i + 1]
        if a > target >= b:
            la, lb = np.log10(max(a, 1e-6)), np.log10(max(b, 1e-6))
            return snrs[i] + (np.log10(target) - la) / (lb - la) * (snrs[i + 1] - snrs[i])
    return None


print(f"{'mode':16s} {'chan':5s} {'FER':>4s} {'flood40':>8s} " + " ".join(f"{'L' + str(c):>6s}" for c in CAPS)
      + "   (layered: dB vs flood40, + = needs more SNR)")
spent_rows = []
for (mode, chan), rows in sorted(pts.items()):
    rows.sort(key=lambda r: r[0])
    snrs = [r[0] for r in rows]
    for target in (0.1, 0.01):
        f40 = crossing(snrs, [fer(z["flood_conv"], z["flood_right"], 40) for _, z in rows], target)
        cells = []
        for c in CAPS:
            x = crossing(snrs, [fer(z["layer_conv"], z["layer_right"], c) for _, z in rows], target)
            cells.append(f"{x - f40:+6.2f}" if x is not None and f40 is not None else f"{'-':>6s}")
        print(f"{mode:16s} {chan:5s} {target:4.0%} {f40 if f40 is not None else float('nan'):8.2f} " + " ".join(cells))
    # work at the cliff: points with flood40 FER in [1%, 50%]
    fl_sp, la_sp = {40: []}, {c: [] for c in CAPS}
    n = 0
    for _, z in rows:
        f = fer(z["flood_conv"], z["flood_right"], 40)
        if 0.01 <= f <= 0.5:
            fl_sp[40].append(np.minimum(z["flood_conv"], 40).mean())
            for c in CAPS:
                la_sp[c].append(np.minimum(z["layer_conv"], c).mean())
            n += 1
    if n:
        spent_rows.append((mode, chan, n, np.mean(fl_sp[40]), [np.mean(la_sp[c]) for c in CAPS]))
print()
print(f"iterations spent per codeword at the cliff (points with flood40 FER 1-50%)")
print(f"{'mode':16s} {'chan':5s} {'pts':>3s} {'flood40':>8s} " + " ".join(f"{'L' + str(c):>6s}" for c in CAPS))
for mode, chan, n, f, ls in spent_rows:
    print(f"{mode:16s} {chan:5s} {n:3d} {f:8.1f} " + " ".join(f"{x:6.1f}" for x in ls))
# overall paired discordance at cap 40 vs layered caps
tot = collections.Counter()
for rows in pts.values():
    for _, z in rows:
        fo = (z["flood_conv"] <= 40) & z["flood_right"]
        for c in (20, 25, 40):
            lo = (z["layer_conv"] <= c) & z["layer_right"]
            tot[(c, "flood only")] += int((fo & ~lo).sum())
            tot[(c, "layer only")] += int((lo & ~fo).sum())
        tot["n"] += len(fo)
        tot["undet_flood"] += int(((z["flood_conv"] <= 40) & ~z["flood_right"]).sum())
        tot["undet_layer"] += int(((z["layer_conv"] <= 40) & ~z["layer_right"]).sum())
print()
print("codewords", tot["n"], "| converged-but-wrong flood40", tot["undet_flood"], "layer40", tot["undet_layer"])
for c in (20, 25, 40):
    print(f"layered cap {c}: decoded only by flooding-40 {tot[(c, 'flood only')]}, only by layered {tot[(c, 'layer only')]}")

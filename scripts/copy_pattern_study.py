"""modem.find_copy's pattern gate (COPY_PATTERN, 2026-11-10). A copy lock
needed a header copy over the floor and coherent frame pilots. A CW ID's
key-downs and an n10 burst's pilots are coherent too.

  false  10 s of noise, a tone, a CW ID, a Hand-like 8-FSK, a CPM burst or an
         n10 burst at 0/10/20 dB: copy locks per band, and their pattern
  true   real w / w48 bursts with the head (lead-in, preamble, header) wiped,
         -8..8 dB on AWGN/MPP/MPD: the right locks' pattern, and what decodes

    python -m scripts.copy_pattern_study false 60
    python -m scripts.copy_pattern_study true 800

Found, ungated: false locks on w from CW 6 of 90 and n10 20 of 90 (pattern
0.03-0.51), none on w48 or from the other sources. Right locks' pattern: w
662, lowest 0.65; w48 543, lowest 0.53, 2 under 0.6 (neither decoded a
codeword). At 0.6: 0 false locks of 60 a cell.
"""
import sys

from data2g import threads

threads.limit(1)

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm, cwid, hfchannel, modem  # noqa: E402
from data2g.config import BANDS, FS, SUBMODES  # noqa: E402

KINDS = ("noise", "tone", "cw", "hand", "cpm", "n10")

def sig(kind, rng, n):
    f0 = float(rng.uniform(1440, 1560))
    if kind == "noise":
        return None
    if kind == "tone":
        return np.sin(2 * np.pi * f0 * np.arange(n) / FS)
    if kind == "cw":
        a = cwid.audio(cwid.id_text("W1AW"), 20, freq=f0)
        return np.tile(a, n // len(a) + 1)[:n]
    if kind == "hand":
        return cpm.tones(cpm.Grid(f"hand{f0:g}", 8, 6.25, center=f0), rng.integers(0, 8, int(n / FS * 6.25)))[:n]
    if kind == "cpm":
        spec = cpm.grid_specs("c16r25")[0]
        ctl = codes.encode(cpm.CTL["c16r25"], rng.bytes(codes.payload_bytes(cpm.CTL["c16r25"])), 0, 0)
        return np.pad(cpm.modulate(spec, [ctl] + [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0)] * 2, dup=False), (0, n))[:n]
    s = SUBMODES["n10-qpsk-r1/2"]
    return np.pad(modem.modulate([rng.bytes(codes.payload_bytes(s)) for _ in range(6)], s), (0, n))[:n]

def false_job(a):
    kind, snr, band, seed = a
    rng = np.random.default_rng([seed, KINDS.index(kind), snr + 50])
    n = 10 * FS
    s = sig(kind, rng, n)
    x = rng.normal(size=n) if s is None else hfchannel.awgn(np.pad(s, (0, n - len(s))), snr, seed=seed, s_power=hfchannel.active_power(s))
    lock = modem.find_copy(x, band)
    if lock is None:
        return a, None
    c, pat = modem.pilot_coherence(x, lock, n_max=10**4, pattern=True)
    return a, (lock["spec"].name, lock["n_cw"], round(lock["score"], 2), round(float(np.mean(c)), 2), round(pat, 2))

def false_main(N):
    cells = [(k, s, b) for b in ("w", "w48") for k in KINDS for s in ((0,) if k == "noise" else (0, 10, 20))]
    with Pool(4) as pool:
        res = pool.map(false_job, [c + (k,) for c in cells for k in range(N)], chunksize=2)
    for c in cells:
        got = [r for a, r in res if a[:3] == c and r is not None]
        print(f"{c[2]:3s} {c[0]:5s} {c[1]:+3d} dB: {len(got)} locks of {N}", "; pattern over its floor:",
              sum(g[4] > modem.PILOT_PATTERN[c[2]] for g in got), got[:3])


TRUE_MODES = {"w": ("qpsk-r1/5", "qpsk-r1/2", "16qam-r1/2", "ack-4f"), "w48": ("w48-qpsk-r1/5", "w48-qpsk-r1/2", "w48-16qam-r1/2")}

def true_job(a):
    band, seed = a
    rng = np.random.default_rng([seed, band == "w"])
    spec = SUBMODES[TRUE_MODES[band][int(rng.integers(len(TRUE_MODES[band])))]]
    n = int(rng.integers(1, 7))
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n)]
    x = modem.modulate(pl, spec)
    chan = (None, "mpp", "mpd")[int(rng.integers(3))]
    snr = float(rng.uniform(-8, 8))
    y = hfchannel.apply_channel(x, None, float(rng.uniform(-30, 30)), fading_preset=chan, seed=seed)
    head = modem.LEADIN_SAMPLES + BANDS[band].preamble_samples + modem.header_samples(band)
    y[:head] = 0.0
    y = hfchannel.awgn(np.concatenate([np.zeros(FS // 2), y, np.zeros(FS // 2)]), snr, seed=seed + 1, s_power=hfchannel.active_power(x))
    lock = modem.find_copy(y, band)
    if lock is None:
        return None
    right = lock["spec"].name == spec.name and lock["n_cw"] == n
    c, pat = modem.pilot_coherence(y, lock, n_max=10**4, pattern=True)
    dec = 0
    if right:
        try:
            r = modem.receive(y, [band], copy=lock)
            dec = sum(p is not None and ok for (p, ok) in codes.decode_many(spec, modem.burst_soft(r)))
        except Exception:
            dec = 0
    return dict(band=band, chan=chan or "awgn", snr=snr, right=right, pat=pat, coh=float(np.mean(c)), dec=dec, n=n)

def true_main(N):
    with Pool(4) as pool:
        res = [r for r in pool.map(true_job, [(b, k) for b in ("w", "w48") for k in range(N)], chunksize=2) if r]
    for band in ("w", "w48"):
        t = [r for r in res if r["band"] == band]
        thr = modem.PILOT_PATTERN[band]
        rt = [r for r in t if r["right"]]
        wr = [r for r in t if not r["right"]]
        lost = [r for r in rt if r["pat"] <= thr]
        print(f"{band}: {len(t)} copy locks of {N}; right {len(rt)} (bursts with a codeword decoded: {sum(r['dec'] > 0 for r in rt)}),"
              f" wrong {len(wr)}")
        print(f"   pattern gate {thr}: right locks under it {len(lost)}, of them with a codeword decoded {sum(r['dec'] > 0 for r in lost)};"
              f" wrong locks under it {sum(r['pat'] <= thr for r in wr)}")
        print("   decodable right locks under 0.6/0.65:", [sum(r["pat"] < t and r["dec"] > 0 for r in rt) for t in (0.6, 0.65)], "all right under:", [sum(r["pat"] < t for r in rt) for t in (0.6, 0.65)], "min", min(r["pat"] for r in rt)); print("   right locks pattern percentiles 1/5/25/50:", np.round(np.percentile([r["pat"] for r in rt], (1, 5, 25, 50)), 2) if rt else "-")
        for r in lost[:8]:
            print("     lost:", r["chan"], round(r["snr"], 1), "pat", round(r["pat"], 2), "coh", round(r["coh"], 2), "decoded", r["dec"], "of", r["n"])


if __name__ == "__main__":
    {"false": false_main, "true": true_main}[sys.argv[1]](int(sys.argv[2]))

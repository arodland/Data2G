"""Short bursts under scattered pilots (round 6): fixed bursts of the control-sized modes through two-path fading
with the taps kept (sc_diag's channel, PEP 5 on each burst's peak), 0.3 s of noise either side (phy_session's).
Per mode, cell and arm:
  hdr     the right header found by receive_any (the session receiver's offline search)
  hdr_g   header symbols' BMI with genie timing (modem._read_header at the true start): the header's own estimate
  ctl     the first codeword (a control) decoded, of bursts sent (ModemRx, DD on)
  ctl_g   the same with genie timing (receive(known=)): what the data estimate gives when the header is not the loss
  est     channel estimate error on the first frame's data cells against the true channel (genie timing), dB

    DATA2G_PEP_REF_DB=5 python -m scripts.sc_short --arms off,mix --out runs/sc8_short.csv
"""
from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from data2g import codes, constellation, hfchannel, modem, receiver  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.config import FS, LEADIN_SAMPLES, M, NCP, NSYM, SNR_REF_BW_HZ, SUBMODES  # noqa: E402
from data2g.waveform import dsp, ofdm  # noqa: E402
from scripts.sc_diag import cells as data_cells  # noqa: E402

PRESETS = {**hfchannel.FADING_PRESETS, **hfchannel.ITU_PRESETS}
PAD = int(0.3 * FS)
MODES = {"ack-1f": 1, "ack-4f": 2, "qpsk-r1/5": 1, "n10-ack-4f": 2, "n4-ack-2f": 1}


def setup(arm: str):
    """off; <layout>; <layout>+<remedy>[+<remedy>]: short<N> (bursts of at most N data frames keep pilot rows), spreadall
    (modem.SC_SPREAD_ALL), loo, coh, rows, restoreq[x], openpilot, closepilot (the copy frame's ends).
    The round-6 defaults are <layout>+loo+coh+rows."""
    base = arm.split("+")[0]
    modem.SC_BANDS, modem.SC_LAYOUT = ((), "") if base == "off" else (("w", "w48", "n10", "n4"), base)
    modem.SC_SHORT_FRAMES, modem.SC_SPREAD_ALL, modem.SC_SPREAD_LOO, modem.SC_COPY_COHERENT = 0, False, False, False
    modem.SC_COPY_ROWS, modem.SC_RESTORE_QPSK = False, 0.0
    modem.copy_open = modem.copy_close = lambda b: np.conj(b.pilot)
    for r in arm.split("+")[1:]:
        if r.startswith("short"):
            modem.SC_SHORT_FRAMES = int(r[5:])
        elif r == "spreadall":
            modem.SC_SPREAD_ALL = True
        elif r == "loo":
            modem.SC_SPREAD_LOO = True
        elif r == "coh":
            modem.SC_COPY_COHERENT = True
        elif r == "rows":
            modem.SC_COPY_ROWS = True
        elif r in ("openpilot", "closepilot"):  # the copy frame's open or close symbol the pilot, not its conjugate
            setattr(modem, "copy_" + r[:-5], lambda b: b.pilot)
        elif r.startswith("restoreq"):
            modem.SC_RESTORE_QPSK = float(r[8:] or 1)
        else:
            raise ValueError(r)


def one(args):
    arm, name, chan, snr, seed = args
    setup(arm)
    spec = SUBMODES[name]
    n_cw = MODES[name]
    rng = np.random.default_rng([seed, 3])
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
    x = modem.modulate(pl, name)
    if CUT:  # the preamble lost (its samples silent): only the header copy finds the burst
        x[LEADIN_SAMPLES:LEADIN_SAMPLES + modem.BANDS[spec.sync_band].preamble_samples] = 0
    x = np.concatenate([np.zeros(PAD), x, np.zeros(PAD)])
    z = signal.hilbert(x)
    peak = float(np.max(np.abs(z) ** 2) / 2)
    if chan == "awgn":
        y, taps = x.copy(), None
    else:
        p = PRESETS[chan]
        g = np.random.default_rng(seed)
        d = int(round(p.delay_ms * 1e-3 * FS))
        g1, g2 = (hfchannel._gaussian_taps(len(z), p.doppler_hz, g) for _ in range(2))
        y = np.real((z * g1 + np.concatenate([np.zeros(d, dtype=complex), z[:len(z) - d]]) * g2) / np.sqrt(2))
        taps = (g1, g2, d)
    sigma = np.sqrt(peak / 10 ** 0.5 * (FS / 2) / SNR_REF_BW_HZ / 10 ** (snr / 10))
    y = y + np.random.default_rng([seed, 5]).normal(scale=sigma, size=len(y))
    start = PAD + LEADIN_SAMPLES
    out = dict(hdr=0, ctl=0, ctl_g=0)
    try:
        r = receiver.receive_any(y, lead=PAD + FS // 2, cpm_grids=())
    except modem.SyncError:
        r = None
    if r is not None and r.get("family") != "cpm" and r["spec"].name == name and r["n_cw"] == n_cw:
        out["hdr"] = 1
        out["ctl"] = int(PHY.ModemRx(r, {}).decode(0, (0, 0, 0), 0, None) == pl[0])
    # genie timing: the header's own reading, and the burst
    sb = spec.sync_band
    zb = dsp.to_baseband(y)
    hd = modem._read_header(zb, start, sb)
    word = int((spec.index << modem.CW_BITS.get(sb, 6) | (n_cw - 1)) << 6 | modem.crc6(spec.index << modem.CW_BITS.get(sb, 6) | (n_cw - 1)))
    bits = modem._word_bits(word) @ modem.header_code(sb) % 2
    b = ofdm.band(sb)
    layout = modem.header_layout(sb)
    # the header symbols against the channel _read_header interpolated: its soft bits are not returned, so
    # read them again the way decode_header does (QPSK LLRs at unit noise), scaled for BMI
    soft = _header_soft(zb, start, sb)
    s = soft * (1 - 2 * bits)
    out["hdr_g"] = max(1 - np.mean(np.logaddexp(0, -a * s)) / np.log(2) for a in np.geomspace(0.01, 3, 30))
    try:
        rg = modem.receive(y, known=dict(start=start, cfo=0.0, spec=spec, n_cw=n_cw))
        out["ctl_g"] = int(PHY.ModemRx(rg, {}).decode(0, (0, 0, 0), 0, None) == pl[0])
        e = rg["est"]
        sym, k = data_cells(rg)
        n = rg["p0"] + sym * NSYM + NCP + M // 2
        db = ofdm.band(spec.band)
        H = np.ones(len(sym), dtype=complex) if taps is None else \
            (taps[0][n] + taps[1][n] * np.exp(-2j * np.pi * db.freqs[k] * taps[2] / FS)) / np.sqrt(2)
        first = np.arange(len(sym)) < len(sym) // (spec.frames_per_cw * n_cw)
        h = e["h"].ravel()
        refs = [H[first] * np.exp(2j * np.pi * db.bb[k[first]] * tau / FS) for tau in np.arange(-48, 48.01, 0.5)]
        ref = max(refs, key=lambda v: abs(np.vdot(v, h[first])))
        ref = ref * np.vdot(ref, h[first]) / np.vdot(ref, ref)
        out["est"] = float(np.mean(np.abs(h[first] - ref) ** 2) / np.mean(np.abs(ref) ** 2))
        out["spread"] = float(e.get("filter_spread_hz", e["spread_hz"]))
    except modem.SyncError:
        pass
    return args, out


def _header_soft(zb, start, band):
    got = []
    orig = modem.decode_header

    def spy(soft, *a, **k):
        if not got:
            got.append(np.array(soft))
        return orig(soft, *a, **k)

    modem.decode_header = spy
    try:
        modem._read_header(zb, start, band)
    finally:
        modem.decode_header = orig
    return got[0]


CUT = False
CELLS = "mpp:-8,mpp:-4,awgn:-15,awgn:-12,hilat-m:-6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sc8_short.csv")
    ap.add_argument("--arms", default="off,mix")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--bursts", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cut", action="store_true", help="preamble silent: header copies alone")
    a = ap.parse_args()
    global CUT
    CUT = a.cut
    jobs = [(arm, m, c.split(":")[0], float(c.split(":")[1]), seed) for arm in a.arms.split(",")
            for m in a.modes.split(",") for c in a.cells.split(",") for seed in range(a.bursts)]
    acc = {}
    with Pool(a.jobs, maxtasksperchild=200) as pool:
        for args, o in pool.imap_unordered(one, jobs, chunksize=4):
            acc.setdefault(args[:4], []).append(o)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "mode", "chan", "snr", "n", "hdr", "hdr_g", "ctl", "ctl_g", "est_db", "spread"])
        for k, v in sorted(acc.items()):
            est = [o["est"] for o in v if "est" in o]
            w.writerow([*k, len(v), np.mean([o["hdr"] for o in v]).round(3), np.mean([o["hdr_g"] for o in v]).round(3),
                        np.mean([o["ctl"] for o in v]).round(3), np.mean([o["ctl_g"] for o in v]).round(3),
                        round(10 * np.log10(np.median(est)), 1) if est else "",
                        round(float(np.median([o["spread"] for o in v if "spread" in o])), 2) if est else ""])


if __name__ == "__main__":
    main()

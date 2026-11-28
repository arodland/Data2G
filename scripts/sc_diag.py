"""Why higher-order modes cost more with scattered pilots (round 3): fixed
bursts, the receiver given the burst's timing (no search), the channel's taps
known (hfchannel's two paths), one cause per arm. Per mode, channel, cell and
flag (modem.SC_BANDS patched): codewords decoded without and with decision-
directed passes, the channel estimate's error against the true channel
(before and after DD), and the data cells' error measured against what the
LLRs were told (thermal + clip + estimate).

Arms (transmitter / receiver options, each on top of the plain one):
  plain     as the modem is
  noclip    no clipping at either end (clip gain 1, clip noise 0)
  genie     scattered pilots' own clip distortion removed at the receiver
  restore   scattered pilots put back after each clip pass (SC_RESTORE)
  boost3    scattered pilots 3 dB over the data (SC_BOOST), boost1.5 likewise
  widen1    the slow spread not widened (SC_WIDEN 1); slack0 / slack8: support padding
SNR is PEP-referenced (5 dB) on each burst's own peak, so a transmit option's
peak cost is in its numbers.

    python -m scripts.sc_diag --arms plain,noclip --out runs/sc3_diag.csv
"""

from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from data2g import codes, config, constellation, equalizer, hfchannel, modem  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.config import FS, LEADIN_SAMPLES, M, NCP, NSYM, SNR_REF_BW_HZ, SUBMODES, SYMS_PER_FRAME  # noqa: E402
from data2g.waveform import dsp, ofdm  # noqa: E402

PAD = 2400
PRESETS = {**hfchannel.FADING_PRESETS, **hfchannel.ITU_PRESETS}
_TX = dsp.tx_condition
_CLIP = config.SubmodeSpec.clip
FIELDS = ["arm", "flag", "mode", "chan", "snr", "bursts", "cw", "ok", "ok_dd", "mse_db", "mse_dd_db", "meas_db",
          "pred_db", "peak_db", "spread"]


def setup(arm: str, flag: bool):
    if arm in ("mix", "mixs", "k2d3", "fk2d3", "fk3d2", "gold", "k3d2"):
        modem.SC_LAYOUT = arm
    elif arm.startswith("gold"):
        modem.SC_LAYOUT = "gold"
    else:
        modem.SC_LAYOUT = "k3d2"
    modem.SC_BANDS = (("w", "w48", "n10", "n4") if modem.SC_LAYOUT != "k3d2" else ("w", "w48")) if flag else ()
    modem.SC_BOOST, modem.SC_PILOT_FIX, modem.SC_SLACK, modem.SC_WIDEN = 1.0, None, None, 1.25
    modem.SC_RESTORE_ACE, modem.SC_RESTORE_PLAIN, modem.SC_MODES = 1.0, 0.0, "all"
    modem.SC_COMB_SPREAD, modem.SC_SPREADS = arm == "comb", (2.0, 3.0, 4.5, 7.0, 10.0, 15.0)
    modem.SC_STRIDE = {}
    if arm.startswith("gold-"):  # gold-<carriers>.<stride>[-...]: other strides, the spread by mutual prediction
        modem.SC_STRIDE = {int(v.split(".")[0]): int(v.split(".")[1]) for v in arm.split("-")[1:]}
    if arm == "goldtrue":  # gold with the 10 Hz channels' true spread fed in
        modem.SC_SPREADS = (10.0,)
    dsp.tx_condition, config.SubmodeSpec.clip = _TX, _CLIP
    if arm == "noclip":
        dsp.tx_condition = lambda x, headroom, *a, **k: _TX(x, 40.0, *a, **{**k, "project": None})
        config.SubmodeSpec.clip = property(lambda self: config.ClipConsts({}, 1.0, 0.0))
    elif arm == "ace0":  # ACE modes' pilots left as clipped (round 2's transmitter), the receiver told what they are
        modem.SC_RESTORE_ACE = 0.0
    elif arm.startswith("plainres"):  # the modes without ACE: pilots put back this far
        modem.SC_RESTORE_PLAIN = float(arm[8:])
    elif arm == "rows":  # pilot rows in the higher orders (the band's header pilot and copy frame kept)
        modem.SC_MODES = "qpsk"
    elif arm.startswith("boost"):
        modem.SC_BOOST = 10 ** (float(arm[5:]) / 20)
    elif arm == "widen1":
        modem.SC_WIDEN = 1.0
    elif arm.startswith("slack"):
        modem.SC_SLACK = int(arm[5:])


def cells(r):
    """Per data cell of est["h"] (flattened): its on-air symbol index and carrier."""
    n_f, kc = len(r["raw"]), r["kc"]
    nc = r["est"]["h"].shape[2]
    air = np.arange(n_f) + (0 if kc is None else (np.arange(n_f) >= kc))
    if "sc" in r:
        Dc = np.array([modem.sc_frame_cells(nc, int(fa))[2] for fa in air])
        return (air[:, None] * SYMS_PER_FRAME + Dc // nc).ravel(), (Dc % nc).ravel()
    sym = air[:, None, None] * SYMS_PER_FRAME + np.arange(1, SYMS_PER_FRAME)[None, :, None] + np.zeros((1, 1, nc), int)
    return sym.ravel(), np.tile(np.arange(nc), n_f * (SYMS_PER_FRAME - 1))


def one(args):
    arm, flag, name, chan, snr, seed = args
    setup(arm, flag)
    spec = SUBMODES[name]
    b = ofdm.band(spec.band)
    rng = np.random.default_rng([seed, 3])
    n_cw = max(1, int((4.0 - modem.burst_seconds(spec, 0)) / (modem.burst_seconds(spec, 1) - modem.burst_seconds(spec, 0))))
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
    x = np.concatenate([np.zeros(PAD), modem.modulate(pl, name), np.zeros(PAD)])
    known = dict(start=PAD + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=n_cw)
    if arm == "genie" and flag:
        modem.SC_PILOT_FIX = None
        ls = modem.receive(x, known=known)["sc"]["ls"]  # through no channel: the filter's shape times (gain + distortion)
        modem.SC_PILOT_FIX = np.mean(ls, axis=0) / ls
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
    out = dict(cw=n_cw, ok=0, ok_dd=0, peak=10 * np.log10(peak / np.mean(x[PAD:-PAD] ** 2)))
    try:
        r = modem.receive(y, known=known)
    except modem.SyncError:
        return args, out
    e = r["est"]
    out["ok"] = sum(modem.decode_received(r).crc_ok)
    rx = PHY.ModemRx(r, {})
    out["ok_dd"] = sum(rx.decode(i, (0, 0, 0), 0, None) == p_ for i, p_ in enumerate(pl))
    bits = np.stack([codes.encode(spec, p_, 0, index=i) for i, p_ in enumerate(pl)])
    X = constellation.modulate(codes.spread(bits, spec.bits_per_cu), constellation.load(spec.constellation)).reshape(e["h"].shape)
    Y = modem.data_symbols(r["raw"])
    sig = np.mean(np.abs(e["h"] * X) ** 2)
    out.update(meas=np.mean(np.abs(Y - e["h"] * X) ** 2) / sig, pred=np.mean(modem.noise_var(e["h"], e) + e["mse"]) / sig,
               spread=e["filter_spread_hz"])
    sym, k = cells(r)
    if taps is None:
        H = np.ones(len(sym), dtype=complex)
    else:
        n = r["p0"] + sym * NSYM + NCP + M // 2
        H = (taps[0][n] + taps[1][n] * np.exp(-2j * np.pi * b.freqs[k] * taps[2] / FS)) / np.sqrt(2)
        H = H * np.exp(-2j * np.pi * r["cfo"] * n / FS)  # the receiver took its residual CFO (the fade's own) out
    # the estimate against the true channel: one complex gain and the window's delay fitted per burst
    best = None
    for tau in np.arange(-48, 48.01, 0.5):
        ref = H * np.exp(2j * np.pi * b.bb[k] * tau / FS)
        c = np.vdot(ref, e["h"].ravel()) / np.vdot(ref, ref)
        if best is None or abs(c) > abs(best[0]):
            best = (c, ref)
    ref = best[0] * best[1]
    out["mse"] = float(np.mean(np.abs(e["h"].ravel() - ref) ** 2) / np.mean(np.abs(ref) ** 2))
    e2 = rx._est or e
    out["mse_dd"] = float(np.mean(np.abs(e2["h"].ravel() - ref) ** 2) / np.mean(np.abs(ref) ** 2))
    return args, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sc3_diag.csv")
    ap.add_argument("--arms", default="plain")
    ap.add_argument("--flags", default="0,1")
    ap.add_argument("--plan", default="16qam-r1/2:awgn:2,3,4;16qam-r1/2:mpp:5,7,9;16qam-r1/2:mpd:6,8,10")
    ap.add_argument("--bursts", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(arm, f == "1", m, c, float(s), seed) for arm in a.arms.split(",") for f in a.flags.split(",")
            for m, c, ss in (x.split(":") for x in a.plan.split(";")) for s in ss.split(",") for seed in range(a.bursts)]
    acc = {}
    with Pool(a.jobs, maxtasksperchild=200) as pool:
        for args, o in pool.imap_unordered(one, jobs, chunksize=4):
            acc.setdefault(args[:5], []).append(o)
    db = lambda v: round(10 * np.log10(np.median(v)), 1) if len(v) else ""  # noqa: E731
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for (arm, flag, m, c, s), rows in sorted(acc.items()):
            got = [o for o in rows if "mse" in o]
            w.writerow(dict(arm=arm, flag=int(flag), mode=m, chan=c, snr=s, bursts=len(rows), cw=sum(o["cw"] for o in rows),
                            ok=sum(o["ok"] for o in rows), ok_dd=sum(o["ok_dd"] for o in rows),
                            mse_db=db([o["mse"] for o in got]), mse_dd_db=db([o["mse_dd"] for o in got]),
                            meas_db=db([o["meas"] for o in got]), pred_db=db([o["pred"] for o in got]),
                            peak_db=round(float(np.mean([o["peak"] for o in rows])), 2),
                            spread=round(float(np.median([o["spread"] for o in got])), 2) if got else ""))


if __name__ == "__main__":
    main()

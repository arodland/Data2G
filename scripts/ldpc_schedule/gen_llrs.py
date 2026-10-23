"""Paired decoder studies, step 1: channel LLRs over channel_torch (CPU), saved per SNR point, from the
waterfall into the floor (scripts/thresholds.py's Sim). The native decoder (exact phi) only picks the points:
up from where FER < 90%, 0.5 dB steps, until zero errors or a flat floor. Each point keeps up to NMAX
codewords or NERR errors.

    DATA2G_EXACT_PHI=1 python scripts/ldpc_schedule/gen_llrs.py OUT w48-16qam-r2/3,n10-64l-r3/4 awgn,mpp,mpd 20000 200

Used for the fast-phi check (PR #106) and the layered-schedule study (decode.py, compare.py).
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]
from data2g import threads
threads.limit(4)
import numpy as np, torch
torch.set_num_threads(4)
import native_subs
nat, why = native_subs.import_native()
assert nat, why
from data2g import codes
from data2g.config import SUBMODES, DATA_SYMS_PER_FRAME
from data2g.channel_torch import CHANNELS, llr as llr_fn
from thresholds import Sim, _payloads

OUT = Path(sys.argv[1]); modes = sys.argv[2].split(","); chans = sys.argv[3].split(",")
NMAX, NERR = int(sys.argv[4]), int(sys.argv[5])

def batch(sim, chan, snr, rng, g):
    s, b = sim.spec, sim.batch
    info = _payloads(s, b * sim.n_cw, rng)
    coded = codes.spread(codes.encode_info(s, info).reshape(b, sim.n_cw, -1), sim.m)
    cb = torch.tensor(coded.reshape(b, sim.n_f, DATA_SYMS_PER_FRAME, sim.ch.nc, sim.m))
    c = CHANNELS[chan]
    with torch.no_grad():
        y, h, var = sim.ch.receive(sim.ch.channel(sim.ch.transmit(sim.points[(cb * sim.w).sum(-1)]), c, snr, g), c)
        l = llr_fn(y, h, var, sim.points).reshape(b, -1)
        l = codes.despread(l, sim.n_cw, sim.m).reshape(b * sim.n_cw, -1)
    return codes._deinterleave(s, np.asarray(l, np.float32)).astype(np.float32), info

def point(sim, dec, chan, snr, nmax, nerr, seed, save):
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    L, I, n, e = [], [], 0, 0
    while n < nmax and e < nerr:
        l, info = batch(sim, chan, snr, rng, g)
        out, ok = dec.decode(l, iters=codes.LDPC_ITERS)
        e += int((~(np.asarray(ok) & (np.asarray(out) == info).all(1))).sum()); n += len(l)
        if save: L.append(l); I.append(info)
    if save:
        np.savez(OUT / f"{sim.spec.name.replace('/', '_')}__{chan}__{snr:+.1f}.npz", llr=np.concatenate(L), info=np.concatenate(I))
    return n, e

for name in modes:
    s = SUBMODES[name]
    burst = max(1, 8 // s.frames_per_cw)
    sim = Sim(s, "cpu", burst, batch=max(1, 128 // burst))
    dec = nat.ldpc.Decoder(nat.ldpc.qc_code(s.k, s.coded_bits))
    for chan in chans:
        t0 = time.time()
        # coarse: 1 dB up from -10 until FER < 0.9, then fine 0.5 dB points
        snr = -10.0
        while True:
            n, e = point(sim, dec, chan, snr, 128, 10**9, 1, False)
            if e / n < 0.9 or snr >= 45: break
            snr += 1.0
        snr -= 0.5
        prev = None
        while snr <= 45:
            n, e = point(sim, dec, chan, snr, NMAX, NERR, 2, True)
            print(json.dumps(dict(mode=name, chan=chan, snr=snr, n=n, err=e)), flush=True)
            # done at zero errors, or on a floor: FER < 1% and not falling (fading outage)
            if e == 0 or (e / n < 2e-3 and n >= NMAX): break
            if e / n < 0.1 and prev is not None and e / n >= 0.85 * prev: break
            prev = e / n
            snr += 0.5
        print(f"# {name} {chan} {time.time() - t0:.0f}s", flush=True)

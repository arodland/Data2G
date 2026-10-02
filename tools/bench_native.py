"""Timings of the C++ core's hot paths (docs/native-port-plan.md,
"Performance follow-ups"), per pool size. Run it with the core substituted:

    python tools/with_native.py tools/bench_native.py [--pool 1,4] [--reps 5]

Wall time is the best of --reps; CPU is process time (all threads) of that
same run. The machine may be shared: compare runs made back to back.
"""

import argparse
import time

import numpy as np

import data2g_native
from data2g import codes, hfchannel, modem, tnc
from data2g.arq import phy
from data2g.config import BANDS, FS, NSYM, SUBMODES
from data2g.hfchannel import FadingPreset
from data2g.waveform import ofdm, sync

set_threads = getattr(data2g_native, "set_threads", lambda n: None)  # before the pool existed


def best(fn, reps, setup=None):
    """(wall s, cpu s) of the fastest of `reps` runs of fn(setup())."""
    out = []
    for _ in range(reps):
        arg = setup() if setup else None
        w, c = time.perf_counter(), time.process_time()
        fn(arg)
        out.append((time.perf_counter() - w, time.process_time() - c))
    return min(out)


def burst(spec, n_cw, rng, mids=None):
    pays = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
    mids = mids or [(7, 1, i % 256) for i in range(n_cw)]
    bits = np.stack([codes.encode(spec, q, 0, phy.mask_value(m), i) for i, (q, m) in enumerate(zip(pays, mids))])
    return modem.modulate_bits(codes.spread(bits, spec.bits_per_cu), spec), mids


def cases(rng):
    """name -> (fn, setup) pairs, inputs built once."""
    out = {}
    # decode_many: 64 codewords of w48-16qam-r1/2, BPSK-like LLRs
    big = SUBMODES["w48-16qam-r1/2"]
    sent = np.stack([codes.encode(big, rng.bytes(codes.payload_bytes(big)), 0, 0, i) for i in range(64)])
    sigma = 0.8
    conv = (2 / sigma**2 * (1 - 2.0 * sent + sigma * rng.normal(size=sent.shape))).astype(np.float32)
    fail = rng.normal(scale=2.0, size=sent.shape).astype(np.float32)
    assert all(ok for _, ok in codes.decode_many(big, conv)), "converging case must converge"
    out["decode_many w48 64cw converging"] = (lambda _: codes.decode_many(big, conv), None)
    out["decode_many w48 64cw failing"] = (lambda _: codes.decode_many(big, fail), None)

    # a failed DD pass: w48-qpsk-r1/2, 16 codewords = 64 frames, slot 0
    spec = SUBMODES["w48-qpsk-r1/2"]
    x, mids = burst(spec, 16, rng)
    x = np.concatenate([np.zeros(3000), x, np.zeros(3000)])
    r = modem.receive(hfchannel.apply_channel(x, snr_db=-2.0, fading_preset=FadingPreset("mpd", 2.0, 4.0), seed=1))
    assert r["n_cw"] == 16
    rx0 = phy.ModemRx(dict(r), {}, dd_budget=None)
    assert rx0.decode(0, mids[0], 0, None) is None, "DD case must fail"
    out["DD pass w48 64 frames (failed)"] = (lambda rx: rx.decode(0, mids[0], 0, None),
                                             lambda: phy.ModemRx(dict(r), {}, dd_budget=None))

    # live receive, head-limited as tnc.Receiver does, of the same burst at 10 dB
    accept = modem.Accept.of()
    lead = 2000
    seg = np.concatenate([np.zeros(lead), x[3000:-3000], np.zeros(2000)])
    seg = hfchannel.apply_channel(seg, snr_db=10.0, seed=2)
    head = lead + modem.head_samples("w48") + NSYM + 3 * modem.M
    assert modem.receive(seg, ["w48"], accept, head=head)["n_cw"] == 16
    out["live receive w48 16cw (head)"] = (lambda _: modem.receive(seg, ["w48"], accept, head=head), None)
    out["receive w48 16cw (whole buffer)"] = (lambda _: modem.receive(seg), None)

    # StreamDetector: one hop (tnc.Receiver.HOP) of new audio per band
    noise = rng.normal(size=FS * 30)
    for b in sorted({BANDS[b].sync_band for b in BANDS}):
        def setup(b=b):
            d = sync.StreamDetector(ofdm.band(b))
            d.feed(modem.to_baseband(noise[:FS], 0))
            return d

        def hop(d, b=b):
            d.feed(modem.to_baseband(noise[FS:FS + tnc.Receiver.HOP], FS))

        out[f"StreamDetector hop {b}"] = (hop, setup)
    return out


def receiver_cpu(rng, threads):
    """CPU per second of audio, tnc.Receiver on 60 s: bursts in w, n10 and
    w48 at 3-12 dB with noise between, fed 1024 samples at a time."""
    parts = []
    for name, n_cw in (("qpsk-r1/2", 8), ("n10-qpsk-r1/2", 2), ("w48-16qam-r1/2", 8), ("polar-k192-f8", 4),
                       ("w48-qpsk-r1/3", 16), ("n10-qpsk-r1/3", 1)):
        x, _ = burst(SUBMODES[name], n_cw, rng)
        parts += [np.zeros(int(FS * 2.5)), x * 0.5]
    audio = np.concatenate(parts)
    audio = np.concatenate([audio, np.zeros(max(0, 60 * FS - len(audio)))])[:60 * FS]
    audio = hfchannel.apply_channel(audio, snr_db=6.0, fading_preset=FadingPreset("mpg", 0.5, 1.0), seed=3)
    rx = tnc.Receiver(modem.Accept.of())
    set_threads(threads)
    bursts = 0
    w, c = time.perf_counter(), time.process_time()
    for i in range(0, len(audio), 1024):
        bursts += sum(k == "burst" and d["rx"] is not None for k, d in rx.feed(audio[i:i + 1024]))
    return (time.perf_counter() - w) / 60, (time.process_time() - c) / 60, bursts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="1,4")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", default="", help="substring filter on case names")
    a = ap.parse_args()
    threads = [int(t) for t in a.pool.split(",")]
    rng = np.random.default_rng(0)
    table = cases(rng)
    for name, (fn, setup) in table.items():
        if a.only not in name:
            continue
        cells = []
        for n in threads:
            set_threads(n)
            fn(setup() if setup else None)  # warm-up
            w, c = best(fn, a.reps, setup)
            cells.append(f"{n}t {w * 1e3:8.2f} ms (cpu {c * 1e3:7.2f})")
        print(f"{name:34s} " + "  ".join(cells), flush=True)
    if not a.only or a.only in "receiver":
        for n in threads:
            w, c, k = receiver_cpu(np.random.default_rng(1), n)
            print(f"{'tnc Receiver 60 s, per s audio':34s} {n}t wall {w * 1e3:.1f} ms cpu {c * 1e3:.1f} ms "
                  f"({k} bursts)", flush=True)


if __name__ == "__main__":
    main()

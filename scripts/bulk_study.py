"""data2g-bulk on simulated HF: whole transfers through continuous fading,
noise against the bursts' peak (PEP-fair, as DATA2G_PEP_REF_DB=5), the same
trial seed at every SNR.

Per cell: blocks delivered, and blocks decoded from their first send alone
(what the mode would deliver at twice the throughput, without copies).

    python -m scripts.bulk_study --out runs/bulk_study.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import logging  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import bulk, codes, hfchannel, modem  # noqa: E402
from data2g.config import FS  # noqa: E402

PEP_REF_DB = 5.0
PPM = 20.0  # the sender's sample clock
TEXT = open(os.path.join(os.path.dirname(__file__), "..", "docs", "arq.md"), "rb").read()


def trial(args):
    label, chan, snr, seed, nbytes, h, only = args
    genie = label.endswith("!")  # the receiver is told the stream, its timing and CFO: no acquisition
    mode, _, passes = label.rstrip("!").partition("*")  # "qpsk-r1/2*6": six passes, combined
    passes = int(passes or 1)
    h = min(h, bulk.max_h(bulk.MODES[mode]))  # CPM: 4 at most
    rng = np.random.default_rng(seed)
    a = int(rng.integers(0, len(TEXT) - nbytes))
    text = TEXT[a:a + nbytes]
    x = bulk.tx_audio(bulk.bursts(text, mode, h, passes))
    y = np.concatenate([np.zeros(FS), x, np.zeros(FS)])
    peak = np.max(np.abs(hfchannel._analytic(y)) ** 2) / 2 / 10 ** (PEP_REF_DB / 10)
    if chan != "awgn":
        y = hfchannel.fading(y, chan, seed=seed)
    cfo = float(rng.uniform(-50, 50))
    y = hfchannel.freq_shift(y, cfo)
    y = hfchannel.sample_clock_offset(y, PPM)
    y = hfchannel.awgn(y, snr, seed=seed + 1, s_power=peak)
    rx = bulk.Rx()
    if genie:
        spec = bulk.MODES[mode]
        blocks = bulk.pack(text, codes.payload_bytes(spec))
        lay = bulk.Layout(spec, len(blocks), h)
        st = bulk.Stream(lay, bulk.stream_id(blocks), cfo=cfo)
        last = passes * lay.per_pass
        lead = FS + (0 if bulk.is_cpm(spec) else modem.LEADIN_SAMPLES)  # where the receiver's start is
        st.fit = [(lay.offset(g), (lead + lay.offset(g)) / (1 + PPM * 1e-6)) for g in (0, last)]
        st.heard = {last}  # bulk.Rx.missed receives every burst before it at its known position
        rx.streams[st.stream], rx.cur = st, st
    rx = bulk.receive((y[i:i + FS] for i in range(0, len(y), FS)), mode if only else None, rx=rx)
    n = len(bulk.pack(text, codes.payload_bytes(bulk.MODES[mode])))
    st = next(iter(rx.streams.values()), None)
    s = st.stats if st else {}
    return dict(mode=label, chan=chan, snr=snr, seed=seed, blocks=n, got=len(st.blocks) if st else 0,
                first=s.get("rv0", 0), heard=s.get("heard", 0), headerless=s.get("headerless", 0),
                control_lost=s.get("control_lost", 0), bursts=n and -(-n // h) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="qpsk-r1/5,qpsk-r1/3,qpsk-r1/2,qpsk-r3/4")
    ap.add_argument("--chans", default="mpp,mps,awgn")
    ap.add_argument("--snrs", default="-6:10:2", help="lo:hi:step, or per mode: mode=lo:hi:step,mode=...")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--bytes", type=int, default=6000)
    ap.add_argument("--h", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-mode", action="store_true", help="the receiver listens for the mode sent only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    grid = lambda g: np.arange(*(lambda lo, hi, st: (lo, hi + 1e-9, st))(*map(float, g.split(":"))))
    per = dict(x.split("=") for x in a.snrs.split(",")) if "=" in a.snrs else {}
    modes = list(per) if per else a.modes.split(",")
    jobs = [(m, c, float(s), 1000 + t, a.bytes, a.h, a.only_mode)
            for m in modes for c, t in itertools.product(a.chans.split(","), range(a.trials))
            for s in grid(per.get(m, a.snrs))]
    logging.basicConfig(level=logging.ERROR)
    with Pool(a.workers) as p, open(a.out, "w", newline="") as f:
        w = None
        for i, r in enumerate(p.imap_unordered(trial, jobs)):
            if w is None:
                w = csv.DictWriter(f, list(r))
                w.writeheader()
            w.writerow(r)
            f.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()

"""How much sync costs at the low end: scripts/loss_study.py's real-modem
ARQ sessions, with every OFDM burst also received by a genie (the true
start, CFO and header, then receive()'s own channel estimate and
decode) on the same audio. Per cell, how many bursts the real receiver
lost at sync (missed, or wrong header) that the genie decoded: an upper
bound on what better acquisition could win. CPM bursts are counted but
get no genie.

    uv run --no-sync python scripts/sync_loss_study.py --out runs/sync_loss.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data2g import modem
from data2g.arq import phy as PHY
from data2g.arq.modes import MODES, is_cpm
from data2g.config import FS, LEADIN_SAMPLES
from data2g.tnc import receive_any
from data2g.waveform.dsp import freq_correct, to_baseband

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import loss_study as LS  # noqa: E402
import phy_session as G  # noqa: E402

CELLS = [("mpg", -4.0), ("mpp", -4.0), ("mpd", 0.0), ("mpg", 0.0), ("mpp", 0.0), ("awgn", -4.0)]


def genie_receive(y: np.ndarray, spec, n_cw: int) -> dict | None:
    """receive() with acquisition and header replaced by the truth."""
    start = int(G.PAD_S * FS) + LEADIN_SAMPLES
    z = freq_correct(to_baseband(y), G.CFO_HZ)
    hd = modem._read_header(z, start, spec.sync_band)
    v = (spec.index << 6) | (n_cw - 1)
    hd.update(hdr=(spec, n_cw), word=(v << 6) | modem._crc6(v))
    acq = SimpleNamespace(freq_offset=G.CFO_HZ, preamble_start=start)
    real = modem._best_header
    modem._best_header = lambda *a, **k: (hd, acq, z)
    try:
        return modem.receive(y)
    except modem.SyncError:
        return None
    finally:
        modem._best_header = real


class GeniePhy(LS.AuditPhy):
    def hear(self, x, t0):
        self.y = self.ch.apply(x, t0)
        try:
            return receive_any(self.y, lead=int(G.PAD_S * FS) + FS // 2)
        except modem.SyncError:
            return None

    def send(self, burst, t0):
        out = super().send(burst, t0)
        row, spec = self.rows[-1], MODES[burst.submode]
        if not is_cpm(spec):
            g = genie_receive(self.y, spec, len(burst.slots))
            row["genie_ctl"] = int(g is not None and LS.ctl_decoded(g, burst.slots))
            rx = PHY.ModemRx(g, {}) if g is not None else None
            row["genie_data_ok"] = sum(1 for i, s in enumerate(burst.slots)
                                       if rx is not None and s.mask_id[2] < 128 and not s.rv
                                       and rx.decode(i, s.mask_id, 0, None) == s.payload)
        return out


def one(args):
    chan, snr, seed, horizon, policy = args
    rows = []
    ch = G.ContinuousChannel(chan, snr, seed, horizon)
    tag = dict(channel=chan, snr=snr, seed=seed)
    res = L.run(L.make_policy(policy), L.make_policy(policy), None, L.WORKLOADS["bulk"](random.Random(seed + 7)),
                seed=seed, horizon=horizon, phy=GeniePhy(ch, rows, tag))
    for r in rows:
        r["delivered_Bps"] = round(res["delivered"] / horizon, 1)
    return rows


FIELDS = ["channel", "snr", "seed", "submode", "band", "n_cw", "ctl_slots", "dup", "kind", "outcome", "data_sent",
          "data_ok", "genie_ctl", "genie_data_ok", "snr_est", "delivered_Bps"]


def summarize(path):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r["channel"], float(r["snr"]))].append(r)
    print("cell | bursts (OFDM / CPM) | real: missed, header, ctl_lost, ok | genie ctl ok | "
          "sync-lost the genie decodes: bursts, data cw | data cw real / genie")
    for k in sorted(g):
        rs = g[k]
        of = [r for r in rs if r["genie_ctl"] != ""]
        c = defaultdict(int)
        for r in of:
            c[r["outcome"]] += 1
        n = max(len(of), 1)
        sync_lost = [r for r in of if r["outcome"] in ("missed", "header")]
        won = [r for r in sync_lost if r["genie_ctl"] == "1"]
        d_real = sum(int(r["data_ok"]) for r in of if r["outcome"] == "ok")
        d_gen = sum(int(r["genie_data_ok"]) for r in of if r["genie_ctl"] == "1")
        print(f"{k[0]} {k[1]:+.0f} dB | {len(of)} / {len(rs) - len(of)} | "
              + ", ".join(f"{c[o] / n:.1%}" for o in ("missed", "header", "ctl_lost", "ok"))
              + f" | {sum(r['genie_ctl'] == '1' for r in of) / n:.1%} | {len(won)} ({len(won) / n:.1%}), "
              f"{sum(int(r['genie_data_ok']) for r in won)} | {d_real} / {d_gen}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sync_loss.csv")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--horizon", type=float, default=600.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--policy", default="shift+cpm")
    ap.add_argument("--cells", default=None, help="channel:snr,... (default CELLS)")
    a = ap.parse_args()
    cells = [(c, float(s)) for c, s in (x.split(":") for x in a.cells.split(","))] if a.cells else CELLS
    jobs = [(c, s, seed, a.horizon, a.policy) for c, s in cells for seed in range(a.seeds)]
    with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for rows in pool.imap_unordered(one, jobs):
            w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
            f.flush()
    summarize(a.out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        main()

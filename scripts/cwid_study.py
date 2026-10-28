"""CW ID measurements (docs/arq.md §7b):

  sense     CwSense: whether a waiting peer holds through an ID (its reply
            deadline, 2.2 s left when the ID starts, never lapses before the
            next header could be found), per key-down SNR (PEP, 2500 Hz) and
            channel; never-active on noise and on every mode's bursts
  locks     the live receiver's headers on CW IDs in noise, through an Engine
            (its false-lock drop) and through the Receiver alone (none)
  sessions  paired whole sessions (scripts/cpu_profile/speedtest.py: both
            engines, the loopback path, fading, PEP SNR over 3 kHz): CwSense
            out (master's behaviour), in with CW ID off, and CW ID "only" every
            --interval s; time to move --bytes, timeouts, CW IDs sent

    python -m scripts.cwid_study sense|locks|sessions [--out runs/cwid-sounding]
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm, cwid  # noqa: E402
from data2g import hfchannel as H  # noqa: E402
from data2g.arq import engine as E  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.arq.burst import Slot, TxBurst, ctl_mask, data_mask  # noqa: E402
from data2g.arq.modes import MODES, ctl_spec  # noqa: E402
from data2g.config import FS  # noqa: E402
from data2g.receiver import Receiver  # noqa: E402

B = 80  # 10 ms blocks
CALLS = ["W1AW", "K2XYZ", "VK2ABC/P", "G4ABC", "JA1XYZ", "N0CALL", "DL1ABC", "W6ABC"]


def active(y):
    s = cwid.CwSense()
    return np.array([s.feed(y[i:i + B]) for i in range(0, len(y), B)])


def held(a, t0, t1, left=2.2, hold=2.5, header=0.85):
    """A waiting peer's deadline, `left` s off as the ID starts and pushed to
    t + hold while CwSense is active, outlasts the ID and the following
    header's search (`header` s)."""
    d = t0 + left
    for i, on in enumerate(a):
        t = (i + 1) * B / FS
        if on:
            d = max(d, t + hold)
        if d <= t < t1 + header:
            return False
    return True


def sense_cell(args):
    fading, snr, n = args
    ok = 0
    for seed in range(n):
        rng = np.random.default_rng(seed)
        x = cwid.audio(cwid.id_text(CALLS[seed % len(CALLS)]), (15, 20, 25, 30)[seed % 4])
        y = np.concatenate([np.zeros(FS // 2), x, np.zeros(FS)])
        y = H.apply_channel(y, None, freq_offset_hz=rng.uniform(-40, 40), fading_preset=fading, seed=seed)
        y = H.awgn(y, snr, seed=seed + 1000, s_power=0.5)
        ok += held(active(y), 0.5, 0.5 + len(x) / FS)
    return fading or "awgn", snr, ok / n


def burst_cell(name):
    """Seconds CwSense is active over 4 bursts of `name` per (SNR, channel)."""
    spec, rng, out = MODES[name], np.random.default_rng(1), {}
    for snr in (30, 10, 0, -5):
        for fading in (None, "mpp", "mpd"):
            on = 0
            for seed in range(4):
                r = lambda sp: bytes(rng.integers(0, 256, codes.payload_bytes(sp), dtype=np.uint8))  # noqa: E731
                slots = [Slot(ctl_mask(0, 0, 5), 0, r(ctl_spec(spec)))] + [Slot(data_mask(0, i, 5), 0, r(spec))
                                                                           for i in (1, 2, 3)]
                cap = 0 if "r25" in name or "r50" in name else 2
                x = PHY.tx_audio(TxBurst(name, slots, 0, cap))
                y = H.apply_channel(np.concatenate([np.zeros(800), x, np.zeros(800)]), snr, fading_preset=fading,
                                    seed=seed, freq_offset_hz=rng.uniform(-40, 40))
                on += active(y).sum() * B / FS
            out[f"{fading or 'awgn'} {snr}"] = on
    return name, out


def sense(a):
    cells = [(f, s, a.n) for f in (None, "mpp", "mpd") for s in (-14, -12, -10, -8, -6, -4, 0, 10, 30)]
    with Pool(a.workers) as p:
        rows = p.map(sense_cell, cells)
        bursts = p.map(burst_cell, list(MODES))
    noise = active(np.random.default_rng(9).normal(size=1200 * FS)).sum() * B / FS
    print("held through the ID (fraction), key-down SNR dB in 2500 Hz (PEP):")
    for ch in ("awgn", "mpp", "mpd"):
        print(f"  {ch:5s}", " ".join(f"{s:+d}:{p:.2f}" for c, s, p in rows if c == ch))
    print(f"noise, 1200 s: active {noise:.2f} s")
    bad = {n: {k: v for k, v in d.items() if v} for n, d in bursts if any(d.values())}
    print(f"bursts (4 per cell, every mode, AWGN/MPP/MPD at 30/10/0/-5 dB): active in {len(bad)} modes")
    for n, d in bad.items():
        print(f"  {n}: " + ", ".join(f"{k} {v:.2f} s" for k, v in d.items()))
    json.dump(dict(held=rows, noise_s=noise, bursts=dict(bursts)), open(a.out / "cwid_sense.json", "w"), indent=1)


def lock_cell(args):
    """Headers on one ID in noise: the Receiver's, an Engine's not dropped (_cw_false_lock), seconds."""
    i, wpm, snr = args
    x = cwid.audio(cwid.id_text(CALLS[i % len(CALLS)]), wpm, freq=1500 + 20 * ((i % 5) - 2))
    y = H.awgn(np.concatenate([np.zeros(FS), x, np.zeros(2 * FS)]), snr, seed=i, s_power=0.5)
    r = Receiver(E.receive_accept(), cpm_grids=tuple(cpm.GRIDS))
    raw = sum(k == "header" for j in range(0, len(y), 800) for k, _ in r.feed(y[j:j + 800]))
    e = E.Engine("W1AW", seed=i)
    e.listen()
    heard, dropped, drop = [], [], e.receiver.drop
    e.session.on_header = lambda *h: heard.append(h)
    e.receiver.drop = lambda: dropped.append(1) or drop()
    for j in range(0, len(y), 800):
        e.step(y[j:j + 800])
    return raw, len(heard) - len(dropped), len(y) / FS


def locks(a):
    cells = [(i, wpm, snr) for i in range(a.n) for wpm in (15, 20, 30) for snr in (30, 20, 10, 0, -5)]
    with Pool(a.workers) as p:
        rows = p.map(lock_cell, cells)
    raw, eng, sec = (sum(r[k] for r in rows) for k in range(3))
    print(f"{len(cells)} IDs, {sec:.0f} s of audio: Receiver headers {raw}, Engine headers past its drop {eng}")
    noise = Receiver(E.receive_accept(), cpm_grids=tuple(cpm.GRIDS))
    y = np.random.default_rng(3).normal(size=int(sec) * FS)
    n = sum(k == "header" for j in range(0, len(y), 800) for k, _ in noise.feed(y[j:j + 800]))
    print(f"noise alone, {sec:.0f} s: Receiver headers {n}")
    json.dump(dict(ids=len(cells), seconds=sec, receiver=raw, engine=eng, noise=n),
              open(a.out / "cwid_locks.json", "w"), indent=1)


ARMS = ("nosense", "off", "only")


def session_cell(args):
    arm, chan, snr, seed, nbytes, interval = args
    sys.path.insert(0, str(Path(__file__).parent / "cpu_profile"))
    import speedtest as ST

    if arm == "nosense":  # master's engine: no CwSense
        cwid.CwSense.feed = lambda self, x: False
    made, ids = [], []
    init, cw_burst = ST.Side.__init__, E.Engine._cw_burst
    E.Engine._cw_burst = lambda self, call: ids.append(1) or cw_burst(self, call)

    def side_init(self, *a, **kw):
        init(self, *a, **kw)
        self.eng.cw_id = "only" if arm == "only" else "off"
        self.eng.id_interval_s = interval
        made.append(self)
    ST.Side.__init__ = side_init
    r = ST.run(chan, snr, nbytes, seed, limit_s=3600.0, direction="both")
    st = [s.eng.session.station for s in made]
    return dict(arm=arm, chan=chan, snr=snr, seed=seed, phase=r["phase"], seconds=r.get("seconds"),
                t=r.get("t"), timeouts=sum(s.stats.get("timeouts", 0) for s in st if s is not None),
                cw_n=len(ids), cw_s=len(ids) * cwid.seconds(cwid.id_text("K2XYZ")))


def sessions(a):
    cells = [(arm, c, s, seed, a.bytes, a.interval) for c, s in (("awgn", 0.0), ("mpp", 6.0), ("mpd", 10.0), ("mpg", 3.0))
             for seed in range(a.n) for arm in ARMS]
    with Pool(a.workers, maxtasksperchild=1) as p:
        rows = p.map(session_cell, cells, chunksize=1)
    with open(a.out / "cwid_sessions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for c in sorted({(r["chan"], r["snr"]) for r in rows}):
        line = f"{c[0]} {c[1]:+.0f} dB:"
        for arm in ARMS:
            rr = [r for r in rows if (r["chan"], r["snr"]) == c and r["arm"] == arm]
            done = [r["seconds"] for r in rr if r["phase"] == "done"]
            line += (f"  {arm}: {len(done)}/{len(rr)} done, median {np.median(done):.0f} s, timeouts "
                     f"{sum(r['timeouts'] for r in rr)}, CW {sum(r['cw_s'] for r in rr):.0f} s")
        print(line)
    # paired: each seed's time with CW (only) less without (off), and off less nosense
    for x, y in (("off", "nosense"), ("only", "off")):
        d = []
        for r in rows:
            if r["arm"] != x or r["phase"] != "done":
                continue
            m = [q for q in rows if q["arm"] == y and q["phase"] == "done" and (q["chan"], q["snr"], q["seed"])
                 == (r["chan"], r["snr"], r["seed"])]
            if m:
                d.append(r["seconds"] - m[0]["seconds"])
        print(f"paired {x} - {y}: n {len(d)}, median {np.median(d):+.1f} s, mean {np.mean(d):+.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("sense", "locks", "sessions"))
    ap.add_argument("--out", type=Path, default=Path("/home/andrew/code/Data2G/runs/cwid-sounding"))
    ap.add_argument("--n", type=int, default=40, help="trials (sense), calls (locks), seeds (sessions) per cell")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bytes", type=int, default=20000)
    ap.add_argument("--interval", type=float, default=60.0, help="sessions: the ID interval, s (600 on air)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    globals()[a.what](a)


if __name__ == "__main__":
    main()

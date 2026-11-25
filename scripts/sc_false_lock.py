"""False locks under scattered pilots: the Python streaming receiver (receiver.Receiver, engine.receive_accept) on
a station's own traffic, every arm on the same draws (submodes, payloads, gaps, noise; each arm transmits its own
waveform). Traffic: random OFDM bursts of every band (12 s class), half whole, half with preamble and header cut
(false_lock_study's data source), and CW IDs and Hand-like 8-FSK (its other sources), each followed by a 0.5-2 s
gap of noise. Per arm:
  false   headers reported that are no burst sent (a cut burst's own data included): per hour of audio
  busy    seconds the receiver is busy with nothing on air: per hour
  got     whole bursts received (their header kept to the end and received), of those sent

Arms: off (flag off); mix (DATA2G_PILOTS=mix); mix-pilotclose (mix with the header-copy frame closed by the pilot,
as before copy_close).

    python -m scripts.sc_false_lock --minutes 20 --jobs 6
"""
from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm, cwid, hfchannel, modem, receiver  # noqa: E402
from data2g.arq.engine import receive_accept  # noqa: E402
from data2g.arq.policy import slots_for  # noqa: E402
from data2g.config import BANDS, FS, LEADIN_SAMPLES  # noqa: E402

BLK = FS // 50
CALLS = ["W1AW", "K2XYZ", "VK2ABC/P", "G4ABC", "JA1XYZ", "N0CALL"]
ARMS = ("off", "mix", "mix-pilotclose")
_CLOSE = modem.copy_close


def setup(arm):
    """off, mix, mix-pilotclose, and mix+<remedy>... as scripts/sc_short.py's."""
    from scripts import sc_short

    sc_short.setup(arm.replace("mix-pilotclose", "mix"))
    modem.copy_close = (lambda b: b.pilot) if arm == "mix-pilotclose" else _CLOSE


def job(args):
    arm, seed, minutes = args
    setup(arm)
    rng = np.random.default_rng(seed)
    names = list(modem.SUBMODES)
    parts, sent, on = [np.zeros(FS)], [], []  # sent: (start, spec name, whole); on: (start, end) signal on air
    t = FS
    while t < minutes * 60 * FS:
        kind = rng.choice(["burst", "cut", "cw", "hand"], p=[0.4, 0.4, 0.1, 0.1])
        snr = float(rng.choice([0.0, 10.0, 20.0]))
        if kind in ("burst", "cut"):
            pool = names if kind == "burst" else [n for n in names if "ack" not in n]
            s = modem.SUBMODES[pool[rng.integers(len(pool))]]
            x = modem.modulate([rng.bytes(codes.payload_bytes(s)) for _ in range(slots_for(s, 12.0))], s)
            if kind == "cut":
                x = x[LEADIN_SAMPLES + BANDS[s.sync_band].preamble_samples + modem.header_samples(s.sync_band):]
            sent.append((t + LEADIN_SAMPLES, s.name, kind == "burst"))
        elif kind == "cw":
            x = cwid.audio(cwid.id_text(CALLS[rng.integers(len(CALLS))]), (15, 20, 30)[rng.integers(3)])
        else:
            g = cpm.Grid("hand", 8, 6.25, center=float(rng.choice([1470.0, 1530.0])))
            x = cpm.tones(g, rng.integers(0, 8, 66))
        x = hfchannel.awgn(x / np.sqrt(hfchannel.active_power(x)), snr, seed=int(rng.integers(1 << 30)), s_power=1.0)
        gap = int(rng.uniform(0.5, 2.0) * FS)
        sigma = np.sqrt((FS / 2) / 2500.0 / 10 ** (snr / 10))  # hfchannel.awgn's at unit signal power
        parts += [x, rng.normal(scale=sigma, size=gap)]
        on.append((t, t + len(x)))
        t += len(x) + gap
    y = np.concatenate(parts)
    y[:FS] = rng.normal(scale=np.sqrt((FS / 2) / 2500.0 / 10), size=FS)
    onair = np.zeros(len(y) // BLK + 1, bool)
    for a, b in on:
        onair[a // BLK:b // BLK + 1] = True
    rx = receiver.Receiver(receive_accept(), cpm_grids=tuple(cpm.GRIDS))
    starts = {s: (n, w) for s, n, w in sent}
    false = got = busy = 0
    seen = set()
    for i in range(0, len(y) - BLK + 1, BLK):
        for k, d in rx.feed(y[i:i + BLK]):
            h = d["header"] if k == "burst" else d if k == "header" else None
            if h is None or h.get("family") == "cpm":
                continue
            match = next((s for s in starts if abs(h["start"] - s) < FS // 2 and starts[s][0] == h["spec"].name
                          and starts[s][1]), None)
            if k == "header" and match is None and h["start"] not in seen:
                seen.add(h["start"])
                false += 1
            if k == "burst" and match is not None and d["rx"] is not None:
                got += 1
                starts[match] = (starts[match][0], False)  # counted once
        if rx.busy and not onair[i // BLK]:
            busy += BLK / FS
    hours = len(y) / FS / 3600
    return arm, hours, false, busy, got, sum(w for _, _, w in sent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    jobs = [(arm, 1000 + s, a.minutes) for s in range(a.seeds) for arm in a.arms.split(",")]
    with Pool(a.jobs) as p:
        res = p.map(job, jobs, chunksize=1)
    for arm in a.arms.split(","):
        r = [x for x in res if x[0] == arm]
        h = sum(x[1] for x in r)
        print(f"{arm:15s} {h:5.2f} h: false headers {sum(x[2] for x in r):4d} ({sum(x[2] for x in r) / h:6.1f}/h), "
              f"busy off air {sum(x[3] for x in r):7.1f} s ({sum(x[3] for x in r) / h:6.1f} s/h), "
              f"whole bursts received {sum(x[4] for x in r)}/{sum(x[5] for x in r)}; per seed false "
              + " ".join(str(x[2]) for x in r))


if __name__ == "__main__":
    main()

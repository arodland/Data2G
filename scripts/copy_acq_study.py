"""Mid-burst acquisition from the header copy (w, w48), prototype: for
the bursts the real receiver missed, lock on the frame pilots after the
fade and read the header from its copy.

acquire_from_copy:
1. Frame-pilot detector: the preamble's matched filter (sync._repeat_corrs:
   the frame pilot is the preamble's repeat symbol), its outputs one frame
   apart correlated as the preamble's are one repeat apart, over PAIRS
   consecutive pilot pairs, noise-normalized as detection_stat, then
   folded over every frame per grid phase. The best CANDIDATES phases, a
   CP apart at least.
2. On each peak's frame grid, the header copy read (modem._copy_llr,
   decode_header) at every frame; a word counts if its own length puts its copy frame
   where it was read (copy_frame) and it clears the band's header floor.
3. The best such word gives the preamble start; receive() takes over, its
   phase reference from the copy frame's pilot, not the faded preamble.

Run inside scripts/sync_loss_study.py's sessions: per cell, the bursts
lost at sync, how many the genie decodes, and how many the copy lock
decodes (and wrong headers it takes). Then false locks on noise.

    uv run --no-sync python scripts/copy_acq_study.py --out runs/copy_acq.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

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
from data2g.config import BANDS, FRAME_SAMPLES, FS, NCP
from data2g.waveform import ofdm, sync
from data2g.tnc import receive_any
from data2g.waveform.dsp import freq_correct, to_baseband

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import loss_study as LS  # noqa: E402
import phy_session as G  # noqa: E402
import sync_loss_study as S  # noqa: E402

PAIRS = 3  # frame-pilot pairs per detection
CANDIDATES = 4  # pilot peaks tried per buffer
EARLIER = (0, NCP // 2, NCP)  # samples before a peak the copy is read at
# mean frame-pilot coherence a copy lock needs over its claimed burst. On
# noise, locks read 0.11-0.23 (47 in 2 h, w and w48); true locks that
# decoded 0.41-0.73 (58). The live BUSY floors (modem.PILOT_NOISE: w48
# 0.20) would pass noise.
COPY_COHERENCE = 0.35


def pilot_stat(z0: np.ndarray, band: str):
    """-> (D (freqs, starts), freqs): the frame-pilot detection statistic
    before its magnitude (its phase: the CFO left in each bin, modulo
    1 / FRAME_S), start = the first pilot's useful window."""
    b = ofdm.band(band)
    t = b.preamble_template()[sync.PREAMBLE_CP:sync.PREAMBLE_CP + sync.M]
    t = t / np.linalg.norm(t)
    freqs = sync._cfo_grid()
    C = sync._repeat_corrs(z0, t, list(freqs))
    n = C.shape[1] - PAIRS * FRAME_SAMPLES
    if n <= 0:
        return None, freqs
    q = np.quantile(np.abs(C) ** 2, sync.NOISE_QUANTILE, axis=1) / -np.log(1 - sync.NOISE_QUANTILE)
    d = C[:, FRAME_SAMPLES:] * np.conj(C[:, :-FRAME_SAMPLES])
    return sum(d[:, j * FRAME_SAMPLES:j * FRAME_SAMPLES + n] for j in range(PAIRS)) / q.min(), freqs


def acquire_from_copy(y: np.ndarray, band: str):
    """-> (hd, acq, z) for receive(), or None."""
    z0 = to_baseband(y)
    D, freqs = pilot_stat(z0, band)
    if D is None:
        return None
    # folded over every frame of the buffer, per grid phase: a burst's
    # pilots add coherently (each pair turns by the same CFO), its data
    # symbols (M-periodic too: their CP) at random. Ranking 3-pair windows
    # instead, data slots outranked the pilots in bursts whose copy read
    # well at its true place.
    m = D.shape[1] // FRAME_SAMPLES
    fold = D[:, :m * FRAME_SAMPLES].reshape(len(freqs), m, FRAME_SAMPLES).sum(axis=1)
    Sm = np.abs(fold)
    best_t = Sm.max(axis=0)
    # CANDIDATES grid phases, a CP apart at least
    peaks = []
    for n in np.argsort(-best_t):
        if all(min(abs(int(n) - q), FRAME_SAMPLES - abs(int(n) - q)) >= NCP for q in peaks):
            peaks.append(int(n))
            if len(peaks) == CANDIDATES:
                break
    sb = BANDS[band]
    floor = modem.HEADER_MIN_SCORE[band]
    n_hdr = sb.header_syms
    best = None
    alias = FS / FRAME_SAMPLES  # 6.94 Hz
    tries = []
    for n0 in peaks:
        i = int(np.argmax(Sm[:, n0]))
        # the strongest path isn't the first: MPP and MPD peaked 16 and 32
        # samples late, and first_path's half-power rule missed the early
        # path in this statistic. Try the reads a CP's span earlier too.
        # the bin is 12.5 Hz wide, and the copy read interpolates its channel
        # between pilots 144 ms apart: refine by the pilot pairs' phase
        # (modulo 6.94 Hz), every alias the bin allows
        frac = float(np.angle(fold[i, n0])) / (2 * np.pi) * alias
        for k in (-1, 0, 1):
            if abs(frac + k * alias) <= sync.STEP_HZ / 2 + 0.5:
                tries += [(n0 - early, float(freqs[i]) + frac + k * alias) for early in EARLIER]
    for n, f in tries:
        z = freq_correct(z0, f)
        p = (n - NCP) % FRAME_SAMPLES  # the grid's first frame start in the buffer
        for pc in range(p, len(z0) - FRAME_SAMPLES, FRAME_SAMPLES):  # the copy at every frame of the grid
            llr = modem._copy_llr(z, pc, band, n_hdr)
            if llr is None:
                continue
            word, (spec, n_cw), score = modem.decode_header(llr, band)
            kc = modem.copy_frame(band, n_cw * spec.frames_per_cw)
            if kc is None or score < floor:
                continue
            start = pc - kc * FRAME_SAMPLES - modem.header_samples(band) - sb.preamble_samples
            p0 = pc - kc * FRAME_SAMPLES
            if start < 0 or modem.burst_end(p0, spec, n_cw) > len(z0):
                continue
            if best is None or score > best[0]:
                best = (score, word, spec, n_cw, start, pc, f, z)
    if best is None:
        return None
    score, word, spec, n_cw, start, pc, f, z = best
    b = ofdm.band(band)
    hd = modem._read_header(z, start, band)
    hd.update(hdr=(spec, n_cw), word=word, score=score,
              h_pre=b.demod_window(z, pc + NCP, modem.HEADER_BACKOFF) / b.pilot)
    return hd, SimpleNamespace(freq_offset=f, preamble_start=start), z


def coherence(y: np.ndarray, got) -> float:
    """Mean frame-pilot coherence over the claimed burst (modem.pilot_coherence,
    the live receiver's BUSY check)."""
    hd, acq, _ = got
    spec, n_cw = hd["hdr"]
    c = modem.pilot_coherence(y, dict(spec=spec, n_cw=n_cw, p0=hd["p0"], cfo=acq.freq_offset), n_max=10_000)
    return float(np.mean(c)) if c else 0.0


def receive_from_copy(y: np.ndarray, band: str, info: dict | None = None) -> dict | None:
    got = acquire_from_copy(y, band)
    if got is None:
        return None
    if info is not None:
        info.update(score=round(got[0]["score"], 3), coh=round(coherence(y, got), 3))
    real = modem._best_header
    modem._best_header = lambda *a, **k: got
    try:
        return modem.receive(y)
    except modem.SyncError:
        return None
    finally:
        modem._best_header = real


class CopyPhy(S.GeniePhy):
    def send(self, burst, t0):
        out = super().send(burst, t0)
        row, spec = self.rows[-1], MODES[burst.submode]
        if not is_cpm(spec) and spec.sync_band in modem.HEADER_COPY_BANDS and row["outcome"] in ("missed", "header"):
            info = {}
            r = receive_from_copy(self.y, spec.sync_band, info)
            row["copy_score"], row["copy_coh"] = info.get("score", ""), info.get("coh", "")
            right = r is not None and r["spec"].name == spec.name and r["n_cw"] == len(burst.slots)
            row["copy"] = "none" if r is None else ("wrong" if not right else
                                                     ("ok" if LS.ctl_decoded(r, burst.slots) else "ctl_lost"))
        return out


class FallbackPhy(G.RealPhy):
    """The real receiver, and when it finds nothing, the gated copy lock
    on either copy band (the one a live receiver would run)."""

    def hear(self, x, t0):
        y = self.ch.apply(x, t0)
        try:
            r = receive_any(y, lead=int(G.PAD_S * FS) + FS // 2)
        except modem.SyncError:
            r = None
        if r is not None:
            return r
        best = None
        for band in modem.HEADER_COPY_BANDS:
            got = acquire_from_copy(y, band)
            if got is not None and coherence(y, got) >= COPY_COHERENCE and (best is None or got[0]["score"] > best[0]["score"]):
                best = got
        if best is None:
            return None
        self.copy_locks += 1
        real = modem._best_header
        modem._best_header = lambda *a, **k: best
        try:
            return modem.receive(y)
        except modem.SyncError:
            return None
        finally:
            modem._best_header = real


def one_bps(args):
    """Delivered bytes/s of a session with the copy fallback, and its copy locks."""
    chan, snr, seed, horizon, policy = args
    phy = FallbackPhy(G.ContinuousChannel(chan, snr, seed, horizon))
    phy.copy_locks = 0
    res = L.run(L.make_policy(policy), L.make_policy(policy), None, L.WORKLOADS["bulk"](random.Random(seed + 7)),
                seed=seed, horizon=horizon, phy=phy)
    return dict(channel=chan, snr=snr, seed=seed, delivered_Bps=round(res["delivered"] / horizon, 1),
                copy_locks=phy.copy_locks)


def one(args):
    chan, snr, seed, horizon, policy = args
    rows = []
    ch = G.ContinuousChannel(chan, snr, seed, horizon)
    tag = dict(channel=chan, snr=snr, seed=seed)
    res = L.run(L.make_policy(policy), L.make_policy(policy), None, L.WORKLOADS["bulk"](random.Random(seed + 7)),
                seed=seed, horizon=horizon, phy=CopyPhy(ch, rows, tag))
    for r in rows:
        r["delivered_Bps"] = round(res["delivered"] / horizon, 1)
    return rows


def noise_locks(args):
    """Copy locks on `seconds` of noise, per sync band: [(score, coherence)]."""
    band, seconds, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(seconds // 10)):
        y = rng.normal(size=10 * FS)
        got = acquire_from_copy(y, band)
        if got is not None:
            out.append((round(got[0]["score"], 3), round(coherence(y, got), 3)))
    return out


FIELDS = S.FIELDS + ["copy", "copy_score", "copy_coh"]


def summarize(path):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r["channel"], float(r["snr"]))].append(r)
    print("cell | w/w48 bursts lost at sync | genie decodes | copy lock: decodes, ctl lost, wrong header, none "
          "| copy decodes that the genie didn't")
    for k in sorted(g):
        lost = [r for r in g[k] if r["copy"] != ""]
        c = defaultdict(int)
        for r in lost:
            c[r["copy"]] += 1
        gen = sum(r["genie_ctl"] == "1" for r in lost)
        extra = sum(r["copy"] == "ok" and r["genie_ctl"] != "1" for r in lost)
        print(f"{k[0]} {k[1]:+.0f} dB | {len(lost)} | {gen} | {c['ok']}, {c['ctl_lost']}, {c['wrong']}, {c['none']} "
              f"| {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/copy_acq.csv")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--horizon", type=float, default=600.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--policy", default="shift+cpm")
    ap.add_argument("--cells", default=None, help="channel:snr,... (default sync_loss_study.CELLS)")
    ap.add_argument("--bps", action="store_true", help="sessions with the gated copy fallback: delivered bps only")
    ap.add_argument("--noise-s", type=float, default=600.0, help="noise per worker and band, for false locks")
    a = ap.parse_args()
    cells = [(c, float(s)) for c, s in (x.split(":") for x in a.cells.split(","))] if a.cells else S.CELLS
    jobs = [(c, s, seed, a.horizon, a.policy) for c, s in cells for seed in range(a.seeds)]
    if a.bps:
        with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, ["channel", "snr", "seed", "delivered_Bps", "copy_locks"])
            w.writeheader()
            for row in pool.imap_unordered(one_bps, jobs):
                w.writerow(row)
                f.flush()
        return
    with Pool(a.jobs) as pool:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            for rows in pool.imap_unordered(one, jobs):
                w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
                f.flush()
        summarize(a.out)
        for band in modem.HEADER_COPY_BANDS:
            locks = [x for xs in pool.map(noise_locks, [(band, a.noise_s, 1000 * j + 3) for j in range(a.jobs)])
                     for x in xs]
            print(f"false copy locks on noise, {band}: {len(locks)} in {a.noise_s * a.jobs / 60:.0f} min; "
                  f"(score, coherence): {sorted(locks)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        main()

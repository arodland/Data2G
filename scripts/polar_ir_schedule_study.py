"""Polar IR decode schedule, chosen at the receiver (option 2): through the
real PHY on HF fading, does a blind per-half quality estimate pick the
right one of the two schedules?

  ir      the IR code, standard SC order (copies decided from both halves)
  one     RV 0's half alone: the IR code with u decoded first (the reverse
          schedule at the top node; Zhang et al., arXiv:2605.30885) reduces
          to this, since the copies then just repeat u's decisions
  select  one decode, of whichever schedule predict() favours
  either  ir or one succeeded (try both, CRC picks): what any choice
          between the two can reach at best
  chase   RV 0 resent instead, combined (the old polar HARQ)

predict(): each half's LLRs -> its mutual information, blind (1 - h2 of
each bit's |LLR|) -> the Gaussian-consistent mean LLR with that MI ->
polar.de_reliability over the code -> P(block) = 1 - prod(1 - Q(sqrt(m/2)))
over the bits the schedule decides.

Scenarios, codeword by codeword:
  harq  OFDM polar modes: an RV 0 burst, then a resend burst through an
        independent stretch of channel (the same stretch for the RV 1 and
        the Chase resend: paired)
  dup   CPM control under ARQ_DUP: RV 0 and RV 1 in consecutive slots of
        one burst (Chase: RV 0 twice)

    uv run python scripts/polar_ir_schedule_study.py --out runs/polar_ir_schedule.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native" / "build" / "python"))
import data2g_native as NAT  # noqa: E402
import ir_study as IRS  # noqa: E402

from data2g import codes, cpm, hfchannel, polar  # noqa: E402
from data2g.config import SUBMODES  # noqa: E402

OFDM_MODES = ("ack-4f", "ack-1f", "polar-k96-f4", "polar-k96-f8", "polar-k192-f8", "n4-ack-8f", "n10-ack-4f",
              "n4-ack-2f")
CHANNELS = ("awgn", "mpg", "mpp", "mpd", "mps")
OFFSETS_DB = (-6.0, -4.5, -3.0, -1.5, 0.0)  # from the single-shot ladder threshold
CPM_SNRS = {"c8r50": (-12.0, -10.0, -8.0, -6.0, -4.0), "c16r25": (-14.0, -12.0, -10.0, -8.0, -6.0),
            "c32r62": (-13.0, -11.0, -9.0, -7.0, -5.0)}
N_CW = 8


# --- the selector ----------------------------------------------------------------------

def _j_table():
    """Gaussian-consistent mean m -> MI J(m), Y ~ N(m, 2m)."""
    m = np.logspace(-4, 3, 3000)
    y, w = np.polynomial.hermite_e.hermegauss(160)
    s = m[:, None] + np.sqrt(2 * m[:, None]) * y
    j = 1 - (w * np.logaddexp(0, -s) / np.log(2)).sum(1) / np.sqrt(2 * np.pi)
    return np.maximum.accumulate(j), m


_J, _M = _j_table()


def mean_llr(llr: np.ndarray) -> float:
    """Blind: MI of LLRs taken as calibrated -> the Gaussian mean with it."""
    a = np.abs(llr)
    p = 1 / (1 + np.exp(np.minimum(a, 60)))
    h = -(p * np.log2(np.maximum(p, 1e-300)) + (1 - p) * np.log2(np.maximum(1 - p, 1e-300)))
    return float(np.interp(1 - h.mean(), _J, _M))


@lru_cache(maxsize=None)
def _structure(name: str):
    spec = _spec(name)
    base = codes.polar_code(spec)
    ir = codes.polar_ir_code(spec)
    free = np.setdiff1d(ir.info_pos, ir.copies[:, 1])
    return base, ir, np.concatenate([free, ir.copies[:, 0]])


def _p_block(m: np.ndarray, pos: np.ndarray) -> float:
    pe = 0.5 * erfc(np.sqrt(np.maximum(m[pos], 0) / 2) / np.sqrt(2))
    return float(1 - np.prod(1 - pe))


def predict(name: str, m0: float, m1: float) -> tuple[float, float]:
    """(P(block) of the IR schedule, of RV 0 alone) at half means m0, m1."""
    base, ir, decided = _structure(name)
    mean = np.zeros(ir.n)
    mean[ir.sent[: base.e]], mean[ir.sent[base.e:]] = m0, m1
    one = np.zeros(base.n)
    one[base.sent] = m0
    return _p_block(polar.de_reliability(mean), decided), _p_block(polar.de_reliability(one), base.info_pos)


# --- decoding --------------------------------------------------------------------------

def _spec(name: str):
    return cpm.CTL[name[4:]] if name.startswith("cpm:") else SUBMODES[name]


@lru_cache(maxsize=None)
def _decoders(name: str):
    spec = _spec(name)
    base = NAT.polar.polar_code(spec.name) if spec.name in SUBMODES else NAT.polar.PolarCode(spec.k, spec.coded_bits)
    ir = NAT.polar.PolarCode.ir(base, NAT.polar.ir_copies(spec.k, spec.coded_bits))
    return NAT.polar.SCLDecoder(base, codes.POLAR_LIST), NAT.polar.SCLDecoder(ir, codes.POLAR_LIST)


def _ok(spec, dec, llr, bits, index) -> np.ndarray:
    paths, _ = dec.decode(np.asarray(llr, np.float32))
    B, L, k = paths.shape
    crc = np.asarray(NAT.codes.crc_ok(spec.name, paths.reshape(-1, k), np.zeros(B * L, np.uint32),
                                      np.repeat(np.asarray(index, np.int32), L))).reshape(B, L).astype(bool)
    pick = np.where(crc.any(1), crc.argmax(1), 0)
    return crc.any(1) & (paths[np.arange(B), pick] == bits).all(1)


def score(name, s0, s1, s0b, bits, index) -> list[dict]:
    """Per codeword: soft bits (mapping order) of RV 0, RV 1 and the Chase
    resend -> outcomes and the selector's choice."""
    spec = _spec(name)
    dec_base, dec_ir = _decoders(name)
    E = spec.coded_bits
    buf = codes.combine(spec, None, s0, 0)
    buf = codes.combine(spec, buf, s1, 1)
    chase = codes.combine(spec, codes.combine(spec, None, s0, 0), s0b, 0)[:, :E]
    one = _ok(spec, dec_base, buf[:, :E], bits, index)
    ir = _ok(spec, dec_ir, buf, bits, index)
    ch = _ok(spec, dec_base, chase, bits, index)
    out = []
    for i in range(len(bits)):
        m0, m1 = mean_llr(buf[i, :E]), mean_llr(buf[i, E:])
        p_ir, p_one = predict(name, m0, m1)
        pick_ir = p_ir <= p_one
        out.append(dict(m0=round(m0, 4), m1=round(m1, 4), p_ir=p_ir, p_one=p_one, one=int(one[i]), ir=int(ir[i]),
                        chase=int(ch[i]), either=int(one[i] | ir[i]), pick_ir=int(pick_ir),
                        select=int(ir[i] if pick_ir else one[i])))
    return out


# --- scenarios -------------------------------------------------------------------------

def harq_trial(args):
    name, chan, snr, seed = args
    spec = SUBMODES[name]
    rng = np.random.default_rng(seed)
    payloads = [rng.bytes(codes.payload_bytes(spec)) for _ in range(N_CW)]
    s0 = IRS.burst_soft(spec, payloads, [0] * N_CW, chan, snr, seed)
    s1 = IRS.burst_soft(spec, payloads, [1] * N_CW, chan, snr, seed + 1_000_003)
    s0b = IRS.burst_soft(spec, payloads, [0] * N_CW, chan, snr, seed + 1_000_003)  # the same stretch as s1
    if s0 is None or s1 is None or s0b is None:
        return []
    bits = np.stack([codes.info_bits(spec, p, 0, i) for i, p in enumerate(payloads)])
    rows = score(name, s0, s1, s0b, bits, np.arange(N_CW))
    return [dict(scenario="harq", mode=name, channel=chan, snr=snr, seed=seed, **r) for r in rows]


def _cpm_soft(spec, ctl_spec, payload, rvs, data, chan, snr, seed):
    coded = [codes.encode(ctl_spec, payload, rv, 0, 0) for rv in rvs] + data
    x = cpm.modulate(spec, coded, dup=True)
    x = np.concatenate([np.zeros(2400), x, np.zeros(2400)])
    y = hfchannel.apply_channel(x, snr_db=snr, fading_preset=None if chan == "awgn" else chan, seed=seed)
    g = cpm.GRIDS[spec.grid]
    lock = cpm.find(g, y)
    if lock is None or lock["spec"] != spec or not lock["dup"]:
        return None
    return cpm.receive(y, lock)["soft"][:2]


def dup_trial(args):
    name, chan, snr, seed = args
    grid = name[4:]
    ctl = cpm.CTL[grid]
    spec = cpm.grid_specs(grid)[0]
    rng = np.random.default_rng(seed)
    payload = rng.bytes(codes.payload_bytes(ctl))
    data = [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0, 2)]
    ir = _cpm_soft(spec, ctl, payload, (0, 1), data, chan, snr, seed)
    ch = _cpm_soft(spec, ctl, payload, (0, 0), data, chan, snr, seed)  # same channel and noise
    if ir is None or ch is None:
        return []
    bits = codes.info_bits(ctl, payload, 0, 0)[None]
    rows = score(name, ir[0][None], ir[1][None], ch[1][None], bits, [0])
    return [dict(scenario="dup", mode=name, channel=chan, snr=snr, seed=seed, **r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/polar_ir_schedule.csv")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--trials", type=int, default=30, help="per point: bursts (harq), control words (dup x8)")
    ap.add_argument("--modes", default=",".join(OFDM_MODES + tuple(f"cpm:{g}" for g in CPM_SNRS)))
    ap.add_argument("--channels", default=",".join(CHANNELS))
    a = ap.parse_args()
    thr = IRS.thresholds()
    jobs = []
    for name in a.modes.split(","):
        for chan in a.channels.split(","):
            if name.startswith("cpm:"):
                jobs += [(dup_trial, (name, chan, s, 7919 * i + j)) for i, s in enumerate(CPM_SNRS[name[4:]])
                         for j in range(8 * a.trials)]
                continue
            t = thr.get((IRS.ladder_name(SUBMODES[name]), "mpp" if chan == "mps" else chan))
            if t is None:
                print("no threshold", name, chan)
                continue
            jobs += [(harq_trial, (name, chan, t + d, 1000 * i + j)) for i, d in enumerate(OFFSETS_DB)
                     for j in range(a.trials)]
    rows = []
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(_run, jobs, chunksize=4):
            rows += r
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(a.out)


def _run(job):
    f, args = job
    return f(args)


def summarize(path):
    from collections import defaultdict

    keys = ("one", "ir", "select", "either", "chase", "pick_ir", "wrong")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["wrong"] = int(r["select"] == "0" and r["either"] == "1")

    def table(title, key):
        g = defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        print(f"\n{title}\n{'':40s} {'n':>6s} " + " ".join(f"{k:>7s}" for k in keys))
        for k in sorted(g):
            rs = g[k]
            print(f"{' '.join(map(str, k)):40s} {len(rs):6d} " +
                  " ".join(f"{np.mean([float(r[c]) for r in rs]):7.3f}" for c in keys))

    print("success rates; pick_ir: share where the selector chose IR; wrong: it chose the schedule that failed "
          "where the other succeeded")
    table("per mode and channel, SNR points pooled", lambda r: (r["scenario"], r["mode"], r["channel"]))
    table("per channel", lambda r: (r["scenario"], r["channel"]))
    table("overall", lambda r: (r["scenario"],))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        main()

"""False accepts of primed control decodes (docs/arq.md §3a), measured.

    python -m scripts.primed_fa --trials 200000 --out runs/primed_fa.csv

A primed decode freezes what the hypothesis knows and takes the first of
the polar list's first phy.PRIMED_PATHS (4) paths whose 24-bit masked CRC
checks, then holds it to phy.PRIMED_SCORE. Per source and hypothesis, every trial decodes once
(native codes when built) and counts the paths passing the CRC truncated
to its low w bits (w = 8, 12, 16: the rate scales as PRIMED_PATHS x 2^-w, so the
24-bit rate is read off them), whether the full CRC passed, and the score
of the first path passing w = 8. Sources, CPM c16r25 control codewords
(cpm.llrs of tone energies), the hypothesis an ACK's (8 B) with its
cumulative ACK unknown (7 bits) or wholly known:
  noise    no signal at all
  other    a real control (0 dB per tone) whose bytes differ from the
           hypothesis in its pinned bits (another reply of the peer's, a
           stale or repeated burst, another station's under my mask)
  wrongkey the hypothesis' own control under another session's mask
  true     the hypothesis' own control at -1 dB per tone (the right
           decodes' scores, against PRIMED_SCORE)
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, cpm
from data2g.arq import frames as F
from data2g.arq import phy as PHY
from data2g.arq.burst import ctl_mask

GRID = "c16r25"
SPEC = cpm.CTL[GRID]
MASK = PHY.mask_value(ctl_mask(1, 0, 0x5A5A))
OTHER_MASK = PHY.mask_value(ctl_mask(1, 0, 0x1234))
WIDTHS = (8, 12, 16)


def native():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    import native_subs

    n, _ = native_subs.import_native()
    return n


def ack(rng) -> bytes:
    core = F.Core(burst_seq=int(rng.integers(8)), acted_on=int(rng.integers(8)), cum=int(rng.integers(128)),
                  recommend=int(rng.integers(64)), size_hint=int(rng.integers(4)))
    return F.Control(core, {F.T_REPLY: bytes([int(rng.integers(64))]), F.T_DUPCTL: b""}).pack(codes.payload_bytes(SPEC))[0]


def tone_soft(rng, ctl: bytes | None, mask: int, snr_db: float) -> np.ndarray:
    """Control codeword soft bits off tone energies: noise only (ctl None) or the codeword at snr_db per tone."""
    g = cpm.GRIDS[GRID]
    n_sym = SPEC.coded_bits // g.bits
    E = rng.exponential(1.0, (n_sym, g.m))
    if ctl is not None:
        tones = cpm.to_tones(g, codes.encode(SPEC, ctl, 0, mask, 0))
        amp = np.sqrt(10 ** (snr_db / 10))
        z = (rng.normal(size=(n_sym, g.m)) + 1j * rng.normal(size=(n_sym, g.m))) / np.sqrt(2)
        z[np.arange(n_sym), tones] += amp
        E = np.abs(z) ** 2
    return np.clip(cpm.llrs(g, E), -30, 30)


def trial(args):
    src, hyp, seed = args
    n = native()
    rng = np.random.default_rng(seed)
    h = ack(rng)
    known_mask = bytearray(b"\xff" * len(h))
    if hyp == "cum":
        known_mask[1] &= 0xC0
        known_mask[2] &= 0x7F
    if src == "noise":
        soft = tone_soft(rng, None, MASK, 0.0)
    elif src == "other":
        other = bytearray(h)
        other[6] ^= 1 + int(rng.integers(63))  # another T_REPLY value: a pinned byte differs
        soft = tone_soft(rng, bytes(other), MASK, 0.0)
    elif src == "true":  # the hypothesis' own control near the primed decode's floor
        soft = tone_soft(rng, h, MASK, -1.0)
    else:
        soft = tone_soft(rng, h, OTHER_MASK, 0.0)
    known = codes.known_bits(SPEC, h, bytes(known_mask))
    cands, usable = n.codes.decode_raw_pinned(SPEC.name, soft[None].astype(np.float32), known.astype(np.int64), 0)
    nb = codes.payload_bytes(SPEC)
    out = {f"w{w}": 0 for w in WIDTHS}
    out.update(full=0, score8=np.nan, true_score=np.nan)
    for c in cands[0][:PHY.PRIMED_PATHS]:
        data = np.packbits(c[:8 * nb + 24]).tobytes()
        want = (codes.crc24(data[:nb]) ^ MASK) & 0xFFFFFF
        got = int.from_bytes(data[nb:nb + 3], "big")
        for w in WIDTHS:
            if (want ^ got) & ((1 << w) - 1) == 0:
                out[f"w{w}"] += 1
                if w == 8 and np.isnan(out["score8"]):
                    out["score8"] = _score_any(c, soft)
        if want == got and np.isnan(out["true_score"]) and data[:nb] == h:
            out["true_score"] = _score_any(c, soft)
        out["full"] += want == got
    return src, hyp, seed, out


def _score_any(bits, soft):
    """The score of a path's own codeword (its CRC bits as decoded)."""
    x = 1.0 - 2.0 * codes.encode_info(SPEC, codes.descramble(SPEC, bits, 0)[None], 0)[0]
    return float(np.sum(np.log(2.0) - np.logaddexp(0.0, -x * soft)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100000)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="runs/primed_fa.csv")
    a = ap.parse_args()
    jobs = [(src, hyp, 1_000_000 * i + k) for i, (src, hyp) in enumerate(
        [(s, h) for s in ("noise", "other", "wrongkey", "true") for h in ("cum", "full")]) for k in range(a.trials)]
    tot = {}
    with Pool(a.jobs) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "hyp", "seed", *(f"w{x}" for x in WIDTHS), "full", "score8", "true_score"])
        for src, hyp, seed, o in p.imap_unordered(trial, jobs, chunksize=256):
            w.writerow([src, hyp, seed, *(o[f"w{x}"] for x in WIDTHS), o["full"], o["score8"], o["true_score"]])
            t = tot.setdefault((src, hyp), dict(n=0, full=0, s=[], **{f"w{x}": 0 for x in WIDTHS}))
            t["n"] += 1
            t["full"] += o["full"]
            for x in WIDTHS:
                t[f"w{x}"] += o[f"w{x}"]
            if not np.isnan(o["score8"]):
                t["s"].append(o["score8"])
            if not np.isnan(o["true_score"]):
                t.setdefault("ts", []).append(o["true_score"])
    print(f"per decode: paths passing the CRC's low w bits (expected {PHY.PRIMED_PATHS} x 2^-w on noise), full 24-bit passes,")
    print("and of the w=8 passes, the share scoring >= PRIMED_SCORE")
    for (src, hyp), t in sorted(tot.items()):
        rates = "  ".join(f"w{x} {t[f'w{x}'] / t['n']:.2e} (x{t[f'w{x}'] / t['n'] / (PHY.PRIMED_PATHS * 2.0 ** -x):.2f})" for x in WIDTHS)
        s = np.array(t["s"])
        print(f"  {src:8s} {hyp:4s} n={t['n']}: {rates}  full {t['full']}  "
              f"score>={PHY.PRIMED_SCORE:.0f}: {np.mean(s >= PHY.PRIMED_SCORE) if len(s) else float('nan'):.4f} of {len(s)}"
              + (f"  right decodes' score p1/p10/p50 {np.percentile(t['ts'], [1, 10, 50]).round(1)} (n={len(t['ts'])})"
                 if t.get("ts") else "")
              + (f"  w8 passes' score p50/p90/p99 {np.percentile(s, [50, 90, 99]).round(1)}" if len(s) > 20 else ""))


if __name__ == "__main__":
    main()

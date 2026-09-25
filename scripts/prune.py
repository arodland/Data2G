"""Prune the ladder.

1. Domination: A dominates B when A carries at least B's payload rate
   and is no worse on every judged channel (awgn, mpg, mpd, as
   specified; mpp is printed, not judged), "worse" meaning by more than
   the 0.25 dB bisection step -- except that a longer codeword never
   dominates a shorter one it near-ties with.
2. Near-ties, among what is left: two candidates whose payload rates are
   within 15% and whose thresholds are within 0.5 dB on every judged
   channel are one step to a gear-shifter. The shorter codeword stays
   (finer retransmission granularity for the ARQ layer; decided
   2026-09-23), the faster one on equal lengths.
3. The <=500 Hz bands (NARROW) also get a ladder of their own, pruned
   by the same rules among themselves, whatever wider modes dominate
   them: fitting a 500 Hz band-plan segment is a feature (decided
   2026-09-23).
4. `--keep` names are kept whatever dominates them: latency exceptions
   (ack-1f, the shortest wide burst, 2026-09-23), which pruning by
   payload rate can not value.

Rate is payload bits per second (CRC excluded, so a CRC-32 codeword
pays for it), which compares bands of different widths. Burst
overhead is the same for every submode and does not change the order.
A candidate needs a finite threshold on at least one judged channel (it
is then optimized for that one); infinite thresholds compare as worst.

    uv run python scripts/prune.py runs/ladder.csv
"""

import argparse
import csv
from collections import defaultdict

from data2g.config import FRAME_SAMPLES, FS

JUDGED = ("awgn", "mpg", "mpd")
INF = float("inf")
# Thresholds are bisected to 0.25 dB; a difference of one step is within
# measurement resolution and does not save a candidate from domination.
RESOLUTION = 0.25
NARROW = ("n10", "n4")


def load(path, rows_out=None, sync=None):
    """Judged thresholds are PEP-referenced where the CSV says how the
    submode clips (a peak_db column): average-power threshold + post-clip
    peak-to-average. Submodes clipping with different headroom differ by up
    to ~3.4 dB of PAPR, so on a peak-limited transmitter only the
    PEP-referenced figure compares across them.

    `sync` ({(band, channel): dB}, scripts/sync_floor.py) makes them end to
    end: max(code threshold, the band's sync threshold), both average
    power, before the PAPR is added. Each fails <= 1% there, so the
    burst fails <= ~2%."""
    thr = defaultdict(dict)
    meta = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if rows_out is not None:
                rows_out.setdefault(r["name"], r)
            t = float(r["threshold_db"])
            if sync is not None:
                t = max(t, sync[(r.get("band") or "w", r["channel"])])
            thr[r["name"]][r["channel"]] = t + float(r.get("peak_db") or 0)
            k, frames = int(r["k"]), int(r["frames"])
            crc = 32 if r["code"] == "ldpc" and k >= 512 else 16
            # payload bits/s, so bands of different widths compare
            meta[r["name"]] = ((k - crc) / (frames * FRAME_SAMPLES / FS), int(r["n"]))
    return {n: (*meta[n], t) for n, t in thr.items() if all(c in t for c in JUDGED)}


def near_tie(a, b):
    (ra, _, ta), (rb, _, tb) = a, b
    lo, hi = sorted((ra, rb))
    return hi <= 1.15 * lo and all(
        (ta[c] == tb[c] == INF) or abs(ta[c] - tb[c]) <= 0.5 for c in JUDGED
    )


def dominates(a, b):
    (ra, _, ta), (rb, _, tb) = a, b
    return ra >= rb and all(ta[c] <= tb[c] + RESOLUTION for c in JUDGED) and (
        ra > rb or any(ta[c] < tb[c] - RESOLUTION for c in JUDGED)
    )


def band_of(name: str) -> str:
    b = name.split("-")[0]
    return b if b in ("n10", "n4", "w48") else "w"


def prune(c: dict, keep=()) -> tuple[list, dict]:
    """-> (kept names by rate, {dropped name: reason}): the overall
    ladder plus the <=500 Hz ladder plus `keep`."""
    kept, why = prune_one(c)
    nk, nwhy = prune_one({n: v for n, v in c.items() if band_of(n) in NARROW})
    for n in nk:
        if n in why:
            why.pop(n)
            kept.append(n)
    for n, r in nwhy.items():
        why[n] = r
    for n in keep:
        if why.pop(n, None) is not None and n not in kept:
            kept.append(n)
    return sorted(kept, key=lambda n: c[n][0]), why


def prune_one(c: dict) -> tuple[list, dict]:
    """-> (kept names by rate, {dropped name: reason}).

    Domination first, except that a longer codeword never dominates a
    shorter one it near-ties with; then near-ties among the survivors.
    (Near-ties first let a tie remove the longer codeword and domination
    then remove the shorter, losing both.)"""
    why = {n: "never decodes on any judged channel" for n, v in c.items()
           if all(v[2][ch] == INF for ch in JUDGED)}
    live = [n for n in c if n not in why]

    def shorter_protected(a, b):  # b shorter than a and tied with it
        return near_tie(c[a], c[b]) and c[b][1] < c[a][1]

    for n in live:
        by = [m for m in live if m != n and dominates(c[m], c[n]) and not shorter_protected(m, n)]
        if by:
            why[n] = f"dominated by {by[0]}"
    live = [n for n in live if n not in why]
    for a in live:
        for b in live:
            if a < b and near_tie(c[a], c[b]):
                # shorter codeword wins; equal length: faster wins
                ka = (c[a][1], -c[a][0])
                kb = (c[b][1], -c[b][0])
                lose, win = (b, a) if ka < kb else (a, b)
                why.setdefault(lose, f"near-tie with {win} (shorter codeword kept)")
    kept = sorted((n for n in live if n not in why), key=lambda n: c[n][0])
    return kept, why


def min_burst_ms(band: str, frames: int) -> float:
    """On-air signal of a one-codeword burst, lead-in/out silence excluded."""
    from data2g.config import BANDS, NSYM
    from data2g.modem import header_samples

    sb = BANDS[band].sync_band
    return (BANDS[sb].preamble_samples + header_samples(sb) + frames * FRAME_SAMPLES + NSYM) / FS * 1000


def load_sync(path):
    with open(path) as f:
        return {(r["band"], r["channel"]): float(r["sync_threshold_db"]) for r in csv.DictReader(f)}


def write_table(path, src, c, rows, why, kept_only=False, sync=None, keep=()):
    band_hz = {"n4": 200, "n10": 500, "w": 1200, "w48": 2400}
    fmt = lambda v: "fails" if v == INF else f"{v:g}"  # noqa: E731
    lines = [
        "# Submode candidates",
        "",
        f"{'The kept' if kept_only else 'Every measured'} candidate{'s' if kept_only else ''}, by payload bits/s (CRC excluded). Generated by",
        f"`scripts/prune.py {src}{f' --sync {sync}' if sync else ''}{''.join(f' --keep {k}' for k in keep)} --markdown {path}"
        f"{' --kept-only' if kept_only else ''}`; do not edit by hand.",
        "",
        "Thresholds in dB SNR (2500 Hz reference), 16-frame bursts with codewords",
        "spread over the burst, "
        + ("end to end: the larger of the code's threshold (PER <= 1e-2) and the "
           "band's sync threshold (acquisition + header fail <= 1e-2), so the "
           "burst fails <= ~2%; a sync-limited cell is marked *. "
           if sync else "PER <= 1e-2, acquisition assumed (sync floor not included). ")
        + "Each cell is `PEP / avg`: PEP-referenced (average +",
        "the submode's post-clip peak PAPR, what a peak-limited transmitter",
        "compares on; pruning and ordering use it) / average power (what the",
        "receiver's SNR estimate reads). Each submode clips with its own",
        "headroom (scripts/pick_headroom.py). Pruning judges awgn, mpg and mpd;",
        "mpp is shown for reference. The <=500 Hz bands (n10, n4) are also",
        "pruned among themselves and keep that ladder whatever wider modes",
        "dominate them.",
        "",
        "Min burst: on-air signal time of a one-codeword burst (preamble, the",
        "band's header, the codeword's frames, closing pilot); the 100 ms of",
        "silence either side is not included.",
        "",
        "| bps | band | code | constellation | frames/cw | k | n | min burst (ms) | clip headroom | peak PAPR "
        "| awgn | mpg | mpd | mpp | status |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    code_avg = defaultdict(dict)
    if sync:
        with open(src) as f:
            for r in csv.DictReader(f):
                code_avg[r["name"]][r["channel"]] = float(r["threshold_db"])
    for n in sorted(c, key=lambda n: c[n][0]):
        if kept_only and n in why:
            continue
        r, t = rows[n], c[n][2]
        name = r["name"]
        band = band_of(name)
        status = ("**kept** (latency exception)" if n in keep else "**kept**") if n not in why else why[n]
        pk = float(r.get("peak_db") or 0)
        code = code_avg.get(n, {})
        mark = lambda ch: "*" if ch in code and t[ch] - pk > code[ch] + 1e-6 else ""  # noqa: E731
        both = lambda ch: fmt(t.get(ch, float("nan"))) if t.get(ch) == INF else (  # noqa: E731
            f"{t[ch]:.2f} / {t[ch] - pk:g}{mark(ch)}" if ch in t else "")
        lines.append(
            f"| {c[n][0]:.0f} | {band} ({band_hz[band]} Hz) | {r['code']} | {r['constellation']} | "
            f"{r['frames']} | {r['k']} | {r['n']} | {min_burst_ms(band, int(r['frames'])):.0f} | "
            f"{r.get('headroom', '1') or '1'} dB | {pk:.2f} | "
            f"{both('awgn')} | {both('mpg')} | {both('mpd')} | {both('mpp')} | {status} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--markdown", help="also write every candidate as a table here")
    ap.add_argument("--kept-only", action="store_true", help="--markdown: only the kept candidates")
    ap.add_argument("--sync", help="sync thresholds CSV (scripts/sync_floor.py): end-to-end thresholds")
    ap.add_argument("--keep", nargs="+", action="extend", default=[], help="candidates kept whatever dominates them (latency exceptions)")
    a = ap.parse_args()
    rows = {}
    c = load(a.csv, rows, load_sync(a.sync) if a.sync else None)
    kept, why = prune(c, a.keep)
    if a.markdown:
        write_table(a.markdown, a.csv, c, rows, why, a.kept_only, a.sync, a.keep)
    print(f"KEPT\n{'candidate':32s} {'bps':>6s} {'n':>5s} " + " ".join(f"{ch:>6s}" for ch in (*JUDGED, "mpp")))
    for n in kept:
        r, nb, t = c[n]
        print(f"{n:32s} {r:6.0f} {nb:5d} " + " ".join(f"{t.get(ch, float('nan')):6.2f}" for ch in (*JUDGED, "mpp")))
    print("\nDROPPED")
    for n in sorted(why, key=lambda n: c[n][0]):
        print(f"  {n:32s} {c[n][0]:6.0f} bps: {why[n]}")


if __name__ == "__main__":
    main()

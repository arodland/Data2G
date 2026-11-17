"""Expected-header decoding for OFDM replies, burst level.

A reply's header is one of a few words its listener expects (the modes
of link.Station.expected_modes, a few codewords). Per trial, one ACK
burst through the channel, received five ways:

  real     modem.receive as the engine does (receive_accept)
  data     the burst with its true start, CFO and header (receive's
           `known`): does the data decode? (the ceiling)
  g_full   genie start and CFO, today's header decode and floor
  g_exp    genie start and CFO, ML over the expected words only, the
           expected floor
  r_exp    modem.receive over the expected words, the expected floor

The expected floor holds the false-accept chance per header read on
noise to today's. From `--noise` reads of white noise, P_full(t), the
chance the best of today's W words scores t or more; a word's own
p(t) = 1 - (1 - P_full(t))^(1/W); the expected set's W_e words pass t
at 1 - (1 - p(t))^W_e, and its floor is the t where that equals
P_full(HEADER_MIN_SCORE). (Today's rate itself is a tail estimate:
how many noise reads passed is printed.)

    uv run --no-sync python scripts/expect_header_study.py --jobs 2
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import os
from multiprocessing import Pool

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.arq.engine import receive_accept
from data2g.config import CW_BITS, FS, LEADIN_SAMPLES, SUBMODES
from data2g.waveform.dsp import freq_correct, to_baseband

# per sync band: the expected set, a station's two ACK modes there and
# its data mode, each up to EXP_CW codewords
EXP_CW = 4
EXPECTED = {"w": ("ack-4f", "ack-1f", "qpsk-r1/2"), "n10": ("n10-ack-4f", "n4-ack-2f", "n4-ack-8f")}


def expected_accept(band: str) -> modem.Accept:
    return modem.Accept(tuple((n, EXP_CW) for n in EXPECTED[band] if n in SUBMODES), 0.0)


def word_of(spec, n_cw: int) -> int:
    v = (spec.index << CW_BITS.get(spec.sync_band, 6)) | (n_cw - 1)
    return (v << 6) | modem.crc6(v)


def noise_scores(args):
    band, seed, n = args
    rng = np.random.default_rng(seed)
    full, exp = receive_accept(), expected_accept(band)
    start = FS // 4
    out = []
    for _ in range(n):
        z = freq_correct(to_baseband(rng.standard_normal(3 * FS)), 0.0)
        out.append((modem._read_header(z, start, band, full)["score"], modem._read_header(z, start, band, exp)["score"]))
    return out


def trial(args):
    mode, n_cw, chan, snr, seed, floors = args
    spec = SUBMODES[mode]
    band = spec.sync_band
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)], spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    f0 = rng.uniform(-50, 50)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=f0, ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    truth, start = word_of(spec, n_cw), lead + LEADIN_SAMPLES
    full, exp = receive_accept(), expected_accept(band)
    row = {}

    try:
        r = modem.receive(y, accept=full)
        row["real"] = "ok" if (r["spec"].name, r["n_cw"]) == (mode, n_cw) else "wrong"
    except modem.SyncError as e:
        row["real"] = "missed" if "no preamble" in str(e) else "header"

    try:
        b = modem.decode_received(modem.receive(y, known=dict(start=start, cfo=f0, spec=spec, n_cw=n_cw)))
        row["data"] = all(b.crc_ok)
    except modem.SyncError:
        row["data"] = False

    z = freq_correct(to_baseband(y), f0)
    hd = modem._read_header(z, start, band, full)
    row["g_full"] = hd["word"] == truth and hd["hdr"] is not None
    he = modem._read_header(z, start, band, exp)
    row["g_exp"] = he["word"] == truth and he["score"] >= floors[band]
    row["s_exp"] = he["score"]

    real_floor = dict(modem.HEADER_MIN_SCORE)
    try:
        modem.HEADER_MIN_SCORE.update({k: min(v, floors.get(k, v)) for k, v in real_floor.items()})
        r = modem.receive(y, accept=exp)
        row["r_exp"] = "ok" if (r["spec"].name, r["n_cw"]) == (mode, n_cw) else "wrong"
    except modem.SyncError as e:
        row["r_exp"] = "missed" if "no preamble" in str(e) else "header"
    finally:
        modem.HEADER_MIN_SCORE.update(real_floor)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["ack-4f", "n10-ack-4f", "n4-ack-8f"])
    ap.add_argument("--n-cw", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--at", nargs="+", default=["awgn:-10", "awgn:-8", "mpp:-6", "mpp:-4", "mpd:-6", "mpd:-4"])
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--noise", type=int, default=4000)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 2)))
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        floors = {}
        for band in sorted({SUBMODES[m].sync_band for m in a.modes}):
            per = a.noise // (4 * a.jobs)
            s = np.array([v for part in pool.map(noise_scores, [(band, 90_000 + i, per) for i in range(4 * a.jobs)])
                          for v in part])
            f0 = modem.HEADER_MIN_SCORE[band]
            w_full = len(modem._valid_words(band, receive_accept()))
            w_exp = len(modem._valid_words(band, expected_accept(band)))
            passed = int(np.sum(s[:, 0] >= f0))
            rate = max(passed, 1) / len(s)
            ts = np.linspace(0.10, f0, 300)
            p_full = np.array([np.mean(s[:, 0] >= t) for t in ts])
            p_one = 1 - (1 - p_full) ** (1 / w_full)
            p_exp = 1 - (1 - p_one) ** w_exp
            q = float(ts[np.argmax(p_exp <= rate)])
            # check: the expected set's own noise reads over that floor
            seen = float(np.mean(s[:, 1] >= q))
            floors[band] = q
            print(f"{band}: {len(s)} noise reads, {passed} over today's floor {f0:.3f} ({w_full} words); "
                  f"expected set {w_exp} words, floor {q:.3f} (its own reads over it: {seen:.4f})", flush=True)
        for mode in a.modes:
            for n_cw in a.n_cw:
                for at in a.at:
                    chan, snr = at.split(":")
                    rows = pool.map(trial, [(mode, n_cw, chan, float(snr), 40_000 + i, floors)
                                            for i in range(a.trials)], chunksize=4)
                    n = len(rows)
                    real = {k: sum(r["real"] == k for r in rows) / n for k in ("ok", "missed", "header", "wrong")}
                    rexp = {k: sum(r["r_exp"] == k for r in rows) / n for k in ("ok", "missed", "header", "wrong")}
                    won = sum(r["data"] and r["real"] != "ok" and r["r_exp"] == "ok" for r in rows) / n
                    print(f"{mode:11s} x{n_cw} {chan:4s} {float(snr):+5.1f} | data {np.mean([r['data'] for r in rows]):.2f}"
                          f" | real ok {real['ok']:.2f} miss {real['missed']:.2f} hdr {real['header']:.2f}"
                          f" wrong {real['wrong']:.2f} | genie full {np.mean([r['g_full'] for r in rows]):.2f}"
                          f" exp {np.mean([r['g_exp'] for r in rows]):.2f} | real exp ok {rexp['ok']:.2f}"
                          f" miss {rexp['missed']:.2f} hdr {rexp['header']:.2f} wrong {rexp['wrong']:.2f}"
                          f" | won (data ok) {won:.2f}", flush=True)


if __name__ == "__main__":
    main()

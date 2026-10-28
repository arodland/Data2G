"""Over-the-air sounding: every mode Data2G uses, in one WAV, and an
offline report of what a recording of it heard (docs/sounding.md).

make: a CW ID, then numbered bursts of every OFDM submode (config.SUBMODES)
and CPM mode (cpm.SPECS), R rounds, then the ID again with SK. Each slot's
payload is known from (seed, burst number, slot) and carries the burst
number; every codeword's CRC is masked with SOUND_KEY, so a listening
station's mask-0 CQ/ID check fails and it ignores them.

decode: the live receiver (receiver.Receiver, as engine.Engine feeds it)
over any WAV, each heard burst matched to the sequence (its payload's
number, else mode and time), every slot decoded under its known mask,
the receiver's own measurements, a genie channel from the known symbols
(SNR, delay and Doppler spread, fades), waveform checks (clipping, band
edges, head and tail), and a pass/fail prediction per mode from
codes_data/mode_thresholds.json at the measured SNR.

    python -m scripts.sounding make runs/cwid-sounding/seq.wav --call W1AW --rate 48000
    python -m scripts.sounding decode rec.wav --call W1AW --out runs/cwid-sounding/rec
"""

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from data2g import codes, cpm, cwid, modem
from data2g.arq import burst as B
from data2g.arq import phy as PHY
from data2g.arq.engine import receive_accept
from data2g.arq.modes import MODES, ctl_spec, is_cpm
from data2g.arq.policy import MODE_THRESHOLDS
from data2g.config import FS, LEADIN_SAMPLES, NSYM, RS, SNR_REF_BW_HZ, SYMS_PER_FRAME, max_codewords
from data2g.receiver import Receiver
from data2g.waveform import dsp

SOUND_KEY = 0x50DA  # every slot's CRC mask key: no live session (key 0 for CQ/ID) accepts these
MAGIC = 0xD2  # a payload's first byte; then the burst number (2 bytes) and the slot
GAP_S = 1.0  # silence between bursts (a receiver hands one burst over before the next starts)
PTT_S = 0.1  # engine.Engine's default ptt_delay_s, ahead of each burst
DATA_S = 2.0  # data airtime per burst, about: whole codewords, at least one
CHANNELS = ("awgn", "mpg", "mpp", "mpd")  # mode_thresholds.json's columns
FADE_DB = 6.0  # a burst this far under the median path SNR counts as faded
BLOCK = FS // 10  # receiver feed size


def family(spec) -> str:
    return "cpm" if is_cpm(spec) else spec.band


def cap_of(spec) -> int:
    """The narrowest bandwidth cap the mode fits (its TX filter, for CPM)."""
    if is_cpm(spec):
        return 2 if spec.grid == "c32r62" else 0
    return {"n4": 0, "n10": 0, "w": 1, "w48": 2}[spec.band]


def n_codewords(spec) -> int:
    """CPM: the control and one data codeword (5.5-15 s); OFDM: about DATA_S of codewords."""
    if is_cpm(spec):
        return 2
    cw_s = spec.frames_per_cw * SYMS_PER_FRAME * NSYM / FS
    return int(min(max_codewords(spec.sync_band), max(1, round(DATA_S / cw_s))))


def order(modes=None) -> list[str]:
    """One round: each family's modes floor first (mode_thresholds' AWGN
    column), families interleaved by their modes' fractional position, so
    each family is spread over the round and one fade takes one of each."""
    names = [n for n in MODES if modes is None or n in modes]
    fams = {}
    for n in sorted(names, key=lambda n: MODE_THRESHOLDS[n][0]):
        fams.setdefault(family(MODES[n]), []).append(n)
    fam_rank = {f: i for i, f in enumerate(("n10", "w", "cpm", "w48", "n4"))}
    key = [((i + 0.5) / len(v), fam_rank[f], n) for f, v in fams.items() for i, n in enumerate(v)]
    return [n for *_, n in sorted(key)]


def masks(spec, n_cw: int) -> list[tuple]:
    """Each slot's codeword identity: CPM's slot 0 is its control codeword."""
    return [B.ctl_mask(0, 0, SOUND_KEY) if is_cpm(spec) and i == 0 else B.data_mask(0, i, SOUND_KEY)
            for i in range(n_cw)]


def payload(seed: int, n: int, slot: int, nbytes: int) -> bytes:
    head = bytes([MAGIC, n >> 8 & 255, n & 255, slot])
    rng = np.random.default_rng([seed, n, slot])
    return (head + bytes(rng.integers(0, 256, max(0, nbytes - 4), dtype=np.uint8)))[:nbytes]


def tx_burst(seed: int, n: int, name: str) -> B.TxBurst:
    spec = MODES[name]
    n_cw = n_codewords(spec)
    slots = []
    for i, m in enumerate(masks(spec, n_cw)):
        sp = ctl_spec(spec) if is_cpm(spec) and i == 0 else spec
        slots.append(B.Slot(m, 0, payload(seed, n, i, codes.payload_bytes(sp))))
    return B.TxBurst(name, slots, n, cap_of(spec))


def sequence(call: str, seed: int = 0, reps: int = 2, modes=None) -> tuple[np.ndarray, list[dict]]:
    """-> (audio at FS, peak 1.0; one dict per burst with its TxBurst and
    unit-RMS audio). Burst audio is scaled to envelope peak 1.0 as
    engine.Engine._start_tx does."""
    parts = [cwid.audio(cwid.id_text(call)), np.zeros(int(GAP_S * FS))]
    t = sum(len(p) for p in parts)
    bursts = []
    rnd = order(modes)
    for r in range(reps):
        for name in rnd:
            n = len(bursts)
            b = tx_burst(seed, n, name)
            x = PHY.tx_audio(b)
            lead = int(PTT_S * FS)
            start = t + lead
            spec = MODES[name]
            bursts.append(dict(n=n, round=r, submode=name, family=family(spec), n_cw=len(b.slots), cap=b.cap,
                               start_s=start / FS, end_s=(start + len(x)) / FS,
                               # where the receiver puts the burst's start: OFDM's preamble, past its lead-in
                               sync_s=(start + (0 if is_cpm(spec) else LEADIN_SAMPLES)) / FS,
                               tx=b, audio=x))
            parts += [np.zeros(lead), x / np.max(dsp.envelope(x)), np.zeros(int(GAP_S * FS))]
            t += lead + len(x) + int(GAP_S * FS)
    parts.append(cwid.audio(cwid.id_text(call) + " SK"))
    return np.concatenate(parts), bursts


def public(b: dict) -> dict:
    return {k: v for k, v in b.items() if k not in ("tx", "audio")}


# --- make ----------------------------------------------------------------------------

def make(a):
    audio, bursts = sequence(a.call, a.seed, a.reps, a.modes)
    if a.rate != FS:
        audio = signal.resample_poly(audio, a.rate // FS, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, a.rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    meta = dict(call=a.call, seed=a.seed, reps=a.reps, modes=a.modes, rate=a.rate, key=SOUND_KEY,
                seconds=len(audio) / a.rate, bursts=[public(b) for b in bursts])
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: {len(bursts)} bursts ({len(bursts) // a.reps} modes x {a.reps}), "
          f"{len(audio) / a.rate / 60:.1f} min at {a.rate} Hz")


# --- decode: reading -----------------------------------------------------------------

def read_wav(path) -> tuple[np.ndarray, int]:
    """-> (first channel scaled to full scale 1.0, sample rate)."""
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128) / 128
    elif np.issubdtype(x.dtype, np.integer):  # 24-bit comes as left-justified int32
        x = x.astype(np.float64) / -np.iinfo(x.dtype).min
    return x.astype(np.float64), fs


def to_fs(x: np.ndarray, fs: int) -> np.ndarray:
    r = Fraction(FS, fs)
    return x if r == 1 else signal.resample_poly(x, r.numerator, r.denominator)


def listen(y: np.ndarray) -> list[dict]:
    """The live receiver over the whole recording -> heard bursts: stream
    start (preamble or CPM front), header, rx (None if lost)."""
    rx = Receiver(receive_accept(), cpm_grids=tuple(cpm.GRIDS))
    out = []
    for i in range(0, len(y), BLOCK):
        for kind, ev in rx.feed(y[i:i + BLOCK]):
            if kind == "burst":
                h = ev["header"]
                out.append(dict(start=h["start"], end=h["end"], submode=h["spec"].name, n_cw=h["n_cw"],
                                score=float(h["score"]), r=ev["rx"]))
    return out


def decode_slots(ev: dict) -> None:
    """Every slot under the sounding's masks: payloads, and the burst number they carry."""
    ev["payloads"], ev["numbers"] = [], []
    if ev["r"] is None:
        return
    spec = MODES[ev["submode"]]
    rx = PHY.ModemRx(ev["r"], {})
    for i, m in enumerate(masks(spec, ev["n_cw"])):
        p = rx.decode(i, m, 0, None)
        ev["payloads"].append(p)
        if p is not None and len(p) >= 4 and p[0] == MAGIC and p[3] == i:
            ev["numbers"].append(p[1] << 8 | p[2])


def match(heard: list[dict], bursts: list[dict], window_s: float = 1.0) -> float | None:
    """Assign heard bursts to the sequence (ev["n"]): by a decoded number,
    then the rest by mode and time once the sequence's offset in the
    recording is known. -> that offset (s), or None if nothing decoded."""
    offs = []
    for ev in heard:
        ev["n"] = None
        if ev["numbers"]:
            n = max(set(ev["numbers"]), key=ev["numbers"].count)
            if n < len(bursts):
                ev["n"] = n
                offs.append(ev["start"] / FS - bursts[n]["sync_s"])
    if not offs:
        return None
    off = float(np.median(offs))
    taken = {ev["n"] for ev in heard if ev["n"] is not None}
    for ev in heard:
        if ev["n"] is not None:
            continue
        t = ev["start"] / FS - off
        near = [b for b in bursts if b["n"] not in taken and b["submode"] == ev["submode"]
                and abs(b["sync_s"] - t) <= window_s]
        if near:
            ev["n"] = min(near, key=lambda b: abs(b["sync_s"] - t))["n"]
            taken.add(ev["n"])
    return off


# --- decode: the genie ---------------------------------------------------------------

def _db(v):
    return None if v is None or not np.isfinite(v) or v <= 0 else round(float(10 * np.log10(v)), 2)


LAGS = (1, 3, 6, 12, 24, 48, 96, 192)  # symbols, for the time correlation


def doppler(rho_t: dict) -> float | None:
    """Two-sided Doppler spread (2 sigma, hfchannel's doppler_hz) of a
    Gaussian Doppler spectrum, whose |correlation| at lag dt is
    exp(-2 pi^2 sigma^2 dt^2): least squares through the origin over the
    lags up to its first fall under 0.5, those under 0.95 (else the
    longest of them, or the shortest lag). Past that fall a 2-3 s burst's
    |mean| floors out and wanders back up: it read 2 Hz as 0.3."""
    pts = []
    for lag, r in sorted(rho_t.items()):
        if r < 0.5 and pts:
            break
        pts.append((lag * NSYM / FS, r))
    use = [(dt, r) for dt, r in pts if r <= 0.95] or pts[-1:]
    if not use:
        return None
    a = np.array([2 * math.pi ** 2 * dt ** 2 for dt, _ in use])
    v = np.array([-math.log(min(max(r, 1e-3), 1.0)) for _, r in use])
    return round(2 * math.sqrt(max(0.0, float(a @ v / (a @ a)))), 3)


def genie_ofdm(r: dict, b: dict) -> dict:
    """Channel per data and pilot cell from the known burst: the clean
    burst demodulated at its own timing (modem.receive known=) gives X, the
    transmitted cells after the clipper and TX filter; H = Y / X.
    - noise: each cell's H against the mean of its neighbours in time (a
      second difference: a linear change cancels), median-based;
    - SNR: the cells' power over that, in the SNR_REF_BW_HZ convention
      (modem.snr_db's);
    - delay spread: |correlation| of H on adjacent carriers (50 Hz), each
      carrier normalized to unit power, as two equal paths' RMS spread
      (arccos(rho) / (2 pi 50 Hz): 1 ms on mpp, 2 ms on mpd);
    - Doppler spread: |correlation| of H in time per lag (rho_t, each
      carrier normalized), fitted as a Gaussian Doppler spectrum (doppler());
    - per carrier and per frame SNR (band edges; the burst's head and tail)."""
    spec = r["spec"]
    clean = b["audio"] + np.random.default_rng(0).normal(0, 1e-4, len(b["audio"]))
    rc = modem.receive(clean, known=dict(start=LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=r["n_cw"]))
    Y, X = r["raw"], rc["raw"]
    n_f, _, nc = Y.shape
    kc = r["kc"]
    air = np.arange(n_f) + (0 if kc is None else (np.arange(n_f) >= kc))
    ti = (air[:, None] * SYMS_PER_FRAME + np.arange(SYMS_PER_FRAME)).ravel()  # symbol index on air
    Y, X = Y.reshape(-1, nc), X.reshape(-1, nc)
    H = Y / X
    inv = 1 / np.abs(X) ** 2
    ok = (ti[2:] - ti[:-2]) == 2  # three consecutive symbols
    d2 = H[1:-1] - (H[:-2] + H[2:]) / 2
    w = inv[1:-1] + (inv[:-2] + inv[2:]) / 4
    n0 = float(np.median((np.abs(d2) ** 2 / w)[ok]) / np.log(2))  # |d2|^2 / w: exponential, mean n0
    if not n0 > 0:
        return {}
    p = np.mean(np.abs(Y) ** 2, axis=0) - n0  # per carrier
    snr = float(np.mean(p)) / n0
    out = dict(snr_db=_db(snr * nc * RS / SNR_REF_BW_HZ),
               carrier_db=[_db(v / n0) for v in p])
    frames = np.mean(np.abs(Y.reshape(n_f, SYMS_PER_FRAME, nc)) ** 2, axis=(1, 2)) - n0
    out["frame_db"] = [_db(v / n0) for v in frames]
    g = p / np.mean(np.abs(X) ** 2, axis=0)  # each carrier's channel power
    if np.all(g > 0) and snr > 0.3:
        Hn = H / np.sqrt(g)
        if nc > 1:
            f = Hn[:, 1:] * np.conj(Hn[:, :-1])
            rho = float(np.mean([abs(f[ti // SYMS_PER_FRAME == k].mean()) for k in np.unique(ti // SYMS_PER_FRAME)]))
            out["delay_ms"] = round(float(np.arccos(np.clip(rho, 0, 1)) / (2 * np.pi * RS) * 1000), 2)
        pos = {t: i for i, t in enumerate(ti)}
        out["rho_t"] = {}
        for lag in LAGS:
            pairs = [(pos[t], pos[t + lag]) for t in ti if t + lag in pos]
            if len(pairs) < 20:
                break
            i, j = np.array(pairs).T
            out["rho_t"][lag] = round(float(np.mean(np.abs(np.mean(Hn[j] * np.conj(Hn[i]), axis=0)))), 4)
        out["doppler_hz"] = doppler(out["rho_t"])
    return out


def genie_cpm(r: dict, b: dict) -> dict:
    """Known tones' energy against the other tones' (median-based, as
    cpm.measure), SNR in the SNR_REF_BW_HZ convention; per tone (band
    edges) and over the burst (head and tail, in tenths)."""
    spec = MODES[b["submode"]]
    g = cpm.GRIDS[spec.grid]
    coded = [codes.encode(ctl_spec(spec) if i == 0 else spec, s.payload, s.rv, PHY.mask_value(s.mask_id), i)
             for i, s in enumerate(b["tx"].slots)]
    tones = np.concatenate([cpm.to_tones(g, c) for c in coded])
    E = r["E"][:len(tones)]
    k = np.arange(len(E))
    known = E[k, tones]
    rest = E.copy()
    rest[k, tones] = np.nan
    n0 = float(np.nanmedian(rest) / np.log(2))
    if not n0 > 0:
        return {}
    es = float(np.mean(known)) - n0
    per_tone = [float(np.mean(known[tones == t])) - n0 if np.any(tones == t) else None for t in range(g.m)]
    tenths = [float(np.mean(c)) - n0 for c in np.array_split(known, 10)]
    return dict(snr_db=_db(es / n0 * g.rate / SNR_REF_BW_HZ),
                carrier_db=[None if v is None else _db(v / n0) for v in per_tone],
                frame_db=[_db(v / n0) for v in tenths])


def integrity(ev: dict, y: np.ndarray, x0: np.ndarray, fs0: int) -> dict:
    """The recording around a heard burst: clipping in the original
    samples (|x| >= 0.99 of full scale), and the burst's first and last 40
    ms against its median power (noise before it subtracted), in 20 ms
    windows: a radio that eats the 20 ms lead-in shows a low head."""
    a, e = ev["start"], ev["end"]
    seg0 = x0[int(a * fs0 / FS):int(e * fs0 / FS)]
    out = dict(clip_frac=round(float(np.mean(np.abs(seg0) >= 0.99)), 5) if len(seg0) else None,
               peak_dbfs=_db(float(np.max(seg0 ** 2))) if len(seg0) else None)
    w = FS // 50
    seg = y[a:e]
    noise = y[max(0, a - int(0.9 * FS)):max(0, a - FS // 10)]
    if len(seg) >= 6 * w and len(noise):
        p = np.mean(seg[:len(seg) // w * w].reshape(-1, w) ** 2, axis=1) - np.mean(noise ** 2)
        med = np.median(p)
        if med > 0:
            out["head_db"] = _db(max(np.mean(p[:2]), 1e-12 * med) / med)
            out["tail_db"] = _db(max(np.mean(p[-2:]), 1e-12 * med) / med)
    return out


# --- decode: report ------------------------------------------------------------------

def channel_class(dop: float | None, delay: float | None) -> str:
    """The mode_thresholds column nearest the measured channel. Slow but
    selective (hfchannel's mps) goes to mpp: selectivity costs more."""
    dop, delay = dop or 0.0, delay or 0.0
    if dop < 0.05 and delay < 0.2:
        return "awgn"
    if dop < 0.4:
        return "mpp" if delay >= 0.6 else "mpg"
    return "mpp" if dop < 1.5 else "mpd"


def _stat(v):
    v = [x for x in v if x is not None]
    return round(float(np.median(v)), 2) if v else None


def analyse(x0: np.ndarray, fs0: int, call: str, seed: int, reps: int, modes=None) -> dict:
    _, bursts = sequence(call, seed, reps, modes)
    y = to_fs(x0, fs0)
    heard = listen(y)
    for ev in heard:
        decode_slots(ev)
    off = match(heard, bursts)
    by_n = {ev["n"]: ev for ev in heard if ev["n"] is not None}
    rows = []
    for b in bursts:
        row = dict(public(b), heard=False, decoded=False, cw_ok=0)
        ev = by_n.get(b["n"])
        if ev is not None:
            r = ev["r"]
            want = [s.payload for s in b["tx"].slots]
            got = ev["payloads"] + [None] * (len(want) - len(ev["payloads"]))
            row.update(heard=True, header_score=round(ev["score"], 3), decoded=r is not None,
                       rec_s=round(ev["start"] / FS, 3), cw_ok=sum(g == w for g, w in zip(got, want)),
                       cw_wrong=sum(g is not None and g != w for g, w in zip(got, want)),
                       header_n_cw=ev["n_cw"])
            row.update(integrity(ev, y, x0, fs0))
            if r is not None:
                row["meas"] = {k: round(float(v), 3) for k, v in PHY.measure(r).items() if not k.startswith("mi_")}
                row["cfo_hz"] = round(float(r["cfo"]), 2)
                if ev["n_cw"] == b["n_cw"]:
                    row["genie"] = genie_cpm(r, b) if is_cpm(MODES[b["submode"]]) else genie_ofdm(r, b)
                    s = row["genie"].get("snr_db")
                    if s is not None:
                        row["genie"]["snr_pep_db"] = round(s + PHY.peak_db(b["submode"], b["cap"]), 2)
        rows.append(row)
    extra = [dict(submode=ev["submode"], rec_s=round(ev["start"] / FS, 3), score=round(ev["score"], 3),
                  decoded=ev["r"] is not None) for ev in heard if ev["n"] is None]

    # the path: PEP-referred SNR over every burst (each mode's own power taken out), fades
    g = [r_ for r_ in rows if r_.get("genie", {}).get("snr_pep_db") is not None]
    pep = np.array([r_["genie"]["snr_pep_db"] for r_ in g])
    path = {}
    if len(pep):
        med = float(np.median(pep))
        path = dict(bursts=len(pep), snr_pep_median_db=round(med, 2), snr_pep_p10_db=round(float(np.percentile(pep, 10)), 2),
                    snr_pep_p90_db=round(float(np.percentile(pep, 90)), 2), snr_pep_min_db=round(float(pep.min()), 2),
                    snr_pep_max_db=round(float(pep.max()), 2), snr_pep_std_db=round(float(pep.std()), 2),
                    faded_frac=round(float(np.mean(pep < med - FADE_DB)), 3), fade_db=FADE_DB,
                    # pooled: each lag's median over the OFDM bursts (one 2-3 s burst sees a fade or two)
                    doppler_hz=doppler({lag: float(np.median(v)) for lag in LAGS
                                        if len(v := [r_["genie"]["rho_t"][lag] for r_ in g
                                                     if lag in r_["genie"].get("rho_t", {})]) >= 3}),
                    delay_ms=_stat([r_["genie"].get("delay_ms") for r_ in g if r_["family"] in ("w", "w48")]),
                    cfo_hz=_stat([r_.get("cfo_hz") for r_ in rows]))
        path["channel"] = channel_class(path["doppler_hz"], path["delay_ms"])
        # band response: per carrier (tone) SNR over the burst's median, averaged per band
        path["band_db"] = {}
        for fam in sorted({r_["family"] for r_ in g}):
            if fam == "cpm":
                continue
            c = np.array([r_["genie"]["carrier_db"] for r_ in g if r_["family"] == fam
                          and None not in r_["genie"]["carrier_db"]])
            if len(c):
                c = c - np.median(c, axis=1, keepdims=True)
                path["band_db"][fam] = [round(float(v), 1) for v in c.mean(axis=0)]

    # per mode: what it did, and what the ladder study predicts at the path's SNR on that channel
    per_mode = []
    for name in order(modes):
        rs = [r_ for r_ in rows if r_["submode"] == name]
        sent = sum(r_["n_cw"] for r_ in rs)
        ok = sum(r_["cw_ok"] for r_ in rs)
        m = dict(submode=name, bursts=len(rs), heard=sum(r_["heard"] for r_ in rs), cw_ok=ok, cw_sent=sent,
                 snr_db=_stat([r_.get("genie", {}).get("snr_db") for r_ in rs]),
                 snr_est_db=_stat([r_.get("meas", {}).get("snr_est") for r_ in rs]),
                 thresholds_db=dict(zip(CHANNELS, MODE_THRESHOLDS[name])))
        if path:
            thr = MODE_THRESHOLDS[name][CHANNELS.index(path["channel"])]
            pred = path["snr_pep_median_db"] - PHY.peak_db(name, rs[0]["cap"])
            m.update(predicted_snr_db=round(pred, 2), margin_db=round(pred - thr, 2) if thr < 90 else None,
                     predicted_pass=bool(thr < 90 and pred >= thr), passed=bool(sent and ok >= 0.9 * sent))
        per_mode.append(m)
    return dict(call=call, seed=seed, reps=reps, modes=modes, rate=fs0, seconds=len(x0) / fs0,
                offset_s=None if off is None else round(off, 3), path=path, per_mode=per_mode, bursts=rows, extra=extra)


def _f(v, fmt="{:.1f}"):
    return "-" if v is None else fmt.format(v)


def report_text(rep: dict) -> str:
    rows, path = rep["bursts"], rep["path"]
    heard = sum(r["heard"] for r in rows)
    out = [f"# Sounding report: {rep['call']} seed {rep['seed']}", "",
           f"- recording: {rep['seconds']:.1f} s at {rep['rate']} Hz; sequence found at "
           f"{_f(rep['offset_s'], '{:.2f}')} s",
           f"- bursts: {len(rows)} sent, {heard} headers heard, {sum(r['decoded'] for r in rows)} received, "
           f"codewords {sum(r['cw_ok'] for r in rows)}/{sum(r['n_cw'] for r in rows)}; "
           f"{len(rep['extra'])} heard outside the sequence"]
    if path:
        out += [f"- path SNR (PEP, genie, {path['bursts']} bursts): median {path['snr_pep_median_db']:.1f} dB, "
                f"10-90% {path['snr_pep_p10_db']:.1f}..{path['snr_pep_p90_db']:.1f}, min {path['snr_pep_min_db']:.1f}, "
                f"max {path['snr_pep_max_db']:.1f}; {100 * path['faded_frac']:.0f}% of bursts over {FADE_DB:g} dB under the median",
                f"- channel: Doppler {_f(path['doppler_hz'], '{:.2f}')} Hz, delay spread {_f(path['delay_ms'], '{:.2f}')} ms, "
                f"CFO {_f(path['cfo_hz'], '{:.1f}')} Hz; thresholds column: {path['channel']}"]
        for fam, v in path["band_db"].items():
            out.append(f"- {fam} carrier response (dB re median, low to high): {' '.join(f'{x:+.1f}' for x in v)}")
        clip = [r["clip_frac"] for r in rows if r.get("clip_frac")]
        out.append(f"- clipping: {len(clip)} bursts with samples at full scale (worst {max(clip, default=0):.2%}); "
                   f"peak {_f(max((r['peak_dbfs'] for r in rows if r.get('peak_dbfs') is not None), default=None))} dBFS")
        heads = [r["head_db"] for r in rows if r.get("head_db") is not None]
        tails = [r["tail_db"] for r in rows if r.get("tail_db") is not None]
        if heads:
            out.append(f"- burst head/tail power re median (first/last 40 ms): head median {np.median(heads):+.1f} dB "
                       f"(worst {min(heads):+.1f}), tail median {np.median(tails):+.1f} dB (worst {min(tails):+.1f})")
    out += ["", "## Per mode", "",
            "| mode | heard | cw ok | genie SNR | snr_est | predicted | 10% thr | margin | pred | got |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m in rep["per_mode"]:
        thr = m["thresholds_db"][path["channel"]] if path else None
        out.append(f"| {m['submode']} | {m['heard']}/{m['bursts']} | {m['cw_ok']}/{m['cw_sent']} | {_f(m['snr_db'])} | "
                   f"{_f(m['snr_est_db'])} | {_f(m.get('predicted_snr_db'))} | {_f(thr if thr is None or thr < 90 else None)} | "
                   f"{_f(m.get('margin_db'))} | {'pass' if m.get('predicted_pass') else 'fail' if path else '-'} | "
                   f"{'pass' if m.get('passed') else 'fail'} |")
    out += ["", "## Per burst", "",
            "| n | mode | rec s | hdr | cw ok | genie SNR | PEP | snr_est | Doppler | delay | head | tail | clip |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g, me = r.get("genie", {}), r.get("meas", {})
        out.append(f"| {r['n']} | {r['submode']} | {_f(r.get('rec_s'), '{:.2f}')} | {_f(r.get('header_score'), '{:.2f}')} | "
                   f"{r['cw_ok']}/{r['n_cw']} | {_f(g.get('snr_db'))} | {_f(g.get('snr_pep_db'))} | {_f(me.get('snr_est'))} | "
                   f"{_f(g.get('doppler_hz'), '{:.2f}')} | {_f(g.get('delay_ms'), '{:.2f}')} | {_f(r.get('head_db'))} | "
                   f"{_f(r.get('tail_db'))} | {_f(r.get('clip_frac'), '{:.3f}')} |")
    return "\n".join(out) + "\n"


def decode(a):
    x0, fs0 = read_wav(a.rec)
    rep = analyse(x0, fs0, a.call, a.seed, a.reps, a.modes)
    out = Path(a.out) if a.out else Path(a.rec).with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = report_text(rep)
    Path(f"{out}.report.md").write_text(text)
    Path(f"{out}.report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(text)
    print(f"wrote {out}.report.md, {out}.report.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make", "decode"):
        p = sub.add_parser(name)
        p.add_argument("out" if name == "make" else "rec")
        p.add_argument("--call", required=name == "make", default="N0CALL")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--reps", type=int, default=2)
        p.add_argument("--modes", nargs="+", choices=list(MODES), help="a subset (decode: the same as make's)")
    sub.choices["make"].add_argument("--rate", type=int, choices=(8000, 48000), default=8000)
    sub.choices["decode"].add_argument("--out", help="report path prefix (default: the recording's, less .wav)")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else decode(a)


if __name__ == "__main__":
    main()

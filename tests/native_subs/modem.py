"""`pytest --native` substitutions for data2g.modem. See __init__.py."""


def substitutions(native):
    """data2g.modem: header codes and ML decode, modulation, the burst and
    header-copy searches, receive and decode. Submodes cross by name; a spec
    that isn't the configured one, a patched PROTOCOL_VERSION or
    HEADER_MIN_SCORE (both compiled in), or a non-array input stays in
    Python. Results come back as Python's dicts, SubmodeSpecs restored."""
    import functools

    import numpy as np

    from data2g import config, modem
    from data2g.waveform import sync

    N, W = native.modem, native.waveform
    names = ("crc6", "header_bits", "_signs", "_valid_words", "_valid_signs", "decode_header", "modulate",
             "modulate_bits", "burst_waveform", "ace_cells", "_bin_phase_step", "_demod_frames", "_read_header",
             "_copy_llr", "_best_header", "find_burst", "pilot_coherence", "find_copy", "_cfo_aliases",
             "_copy_header", "receive", "resolve_alias", "data_channel", "noise_var", "soft_bits", "burst_soft",
             "slot_soft", "snr_db", "symbol_times", "demodulate", "decode_received")
    py = {k: getattr(modem, k) for k in names}
    min_score = dict(modem.HEADER_MIN_SCORE)

    def same_config():
        return modem.PROTOCOL_VERSION == config.PROTOCOL_VERSION and modem.HEADER_MIN_SCORE == min_score

    def own(spec):
        return spec in config.SUBMODES if isinstance(spec, str) else config.SUBMODES.get(spec.name) == spec

    def specs(d):
        return None if d is None else dict(d, spec=config.SUBMODES[d["spec"]])

    def header(t):
        word, (name, n_cw), score = t
        return word, (config.SUBMODES[name], n_cw), score

    def hdr_dict(d):
        return dict(d, hdr=None if d["hdr"] is None else (config.SUBMODES[d["hdr"][0]], d["hdr"][1]))

    def acq(t):
        start, f, metric, alts = t
        return sync.Acquisition(preamble_start=start, freq_offset=f, metric=metric, alternatives=alts)

    def guarded(name, test=lambda *a, **k: True):
        """The decorated native function unless the config is patched or
        test(...) fails; C++ SyncError becomes modem.SyncError."""
        def wrap(fn):
            @functools.wraps(py[name])
            def call(*a, **k):
                if not (same_config() and test(*a, **k)):
                    return py[name](*a, **k)
                try:
                    return fn(*a, **k)
                except W.SyncError as e:
                    raise sync.SyncError(str(e)) from None
            return call
        return wrap

    def bands_arg(bands):
        return None if bands is None else list(bands)

    def lock_ok(lock):
        return own(lock["spec"])

    def f64(x):
        return np.asarray(x, dtype=np.float64)

    @guarded("crc6")
    def crc6(v):
        return N.crc6(int(v))

    @guarded("header_bits")
    def header_bits(submode, n_cw, band="w"):
        return N.header_bits(int(submode), int(n_cw), band)

    @guarded("_signs")
    def _signs(words, band):
        return N.signs(np.asarray(words, dtype=np.int64), band)

    @functools.lru_cache(maxsize=None)
    def _valid_words(band, accept=None):
        if not same_config():
            return py["_valid_words"](band, accept)
        out = N.valid_words(band, accept)
        out.setflags(write=False)
        return out

    @functools.lru_cache(maxsize=None)
    def _valid_signs(band, accept=None):
        if not same_config():
            return py["_valid_signs"](band, accept)
        out = N.signs(_valid_words(band, accept), band)
        out.setflags(write=False)
        return out

    @guarded("decode_header")
    def decode_header(soft, band="w", accept=None):
        return header(N.decode_header(f64(soft).reshape(-1), band, accept))

    @guarded("modulate", lambda payloads, submode, rvs=None: own(submode))
    def modulate(payloads, submode, rvs=None):
        top = config.max_codewords((config.SUBMODES[submode] if isinstance(submode, str) else submode).sync_band)
        if not 1 <= len(payloads) <= top:
            raise ValueError(f"1..{top} codewords per burst, got {len(payloads)}")
        return N.modulate([bytes(p) for p in payloads], submode, list(rvs or [0] * len(payloads)))

    @guarded("modulate_bits", lambda bits, spec: own(spec) and isinstance(bits, np.ndarray))
    def modulate_bits(bits, spec):
        return N.modulate_bits(np.asarray(bits, dtype=np.uint8).reshape(-1), spec)

    @guarded("burst_waveform", lambda data, spec: own(spec))
    def burst_waveform(data, spec):
        return N.burst_waveform(np.asarray(data, dtype=complex), spec)

    @guarded("ace_cells", lambda spec, n_f: own(spec))
    def ace_cells(spec, n_f):
        full = N.ace_cells(spec, int(n_f))
        return full[:, config.NCP:], full

    @guarded("_demod_frames")
    def _demod_frames(z, p, n_f, shift, phi_ref, steps_in=None, band="w", slips_in=None):
        return N.demod_frames(z, int(p), int(n_f), int(shift), float(phi_ref), steps_in, band, slips_in)

    @guarded("_read_header")
    def _read_header(z, start, band="w", accept=None):
        return hdr_dict(N.read_header(z, int(start), band, accept))

    @guarded("_copy_llr")
    def _copy_llr(z, p, band, n_hdr):
        return N.copy_llr(z, int(p), band, int(n_hdr))

    @guarded("_best_header")
    def _best_header(z0, bands=None, complete=True, accept=None, stats=None, final=False):
        hd, a, z = N.best_header(z0, bands_arg(bands), complete, accept, stats, final)
        return hdr_dict(hd), acq(a), z

    @guarded("find_burst")
    def find_burst(x, bands=None, accept=None, stats=None):
        return specs(N.find_burst(f64(x), bands_arg(bands), accept, stats))

    @guarded("pilot_coherence", lambda x, lock, *a, **k: lock_ok(lock))
    def pilot_coherence(x, lock, n_max=8, latest=False):
        return N.pilot_coherence(f64(x), lock, int(n_max), bool(latest))

    @guarded("find_copy")
    def find_copy(x, band, accept=None, C=None, level=None):
        lock, peak = N.find_copy(f64(x), band, accept, C, level)
        if peak is not None:
            py["find_copy"].peak = find_copy.peak = peak
        return specs(lock)

    @guarded("_copy_header", lambda z, lock: lock_ok(lock))
    def _copy_header(z, lock):
        return hdr_dict(N.copy_header(z, lock))

    # `known` (data2g-bulk's receive at a known position) has no native port: Python's
    @guarded("receive", lambda x, bands=None, accept=None, head=None, copy=None, known=None:
             known is None and (copy is None or lock_ok(copy)))
    def receive(x, bands=None, accept=None, head=None, copy=None):
        d = N.receive(f64(x), bands_arg(bands), accept, None if head is None else int(head), copy)
        return dict(d, spec=config.SUBMODES[d["spec"]], acq=acq(d["acq"]))

    @guarded("data_channel", lambda h_pilot, support, band="w", *a, **k: band in config.BANDS)
    def data_channel(h_pilot, support, band="w", n0_pre=np.inf, clip=None, n0_pre_k=None, n_frames=None):
        return N.data_channel(h_pilot, tuple(map(int, support)), band, float(n0_pre), clip, n0_pre_k, n_frames)

    @guarded("noise_var", lambda h, est: isinstance(h, np.ndarray) and h.ndim >= 1)
    def noise_var(h, est):
        n0 = np.broadcast_to(f64(est.get("n0_k", est["n0"])), h.shape[-1:])
        return N.noise_var(h, n0, float(est["clip_ratio"]))

    @guarded("soft_bits", lambda raw, h, var, spec: own(spec) and all(isinstance(a, np.ndarray) for a in (raw, h)))
    def soft_bits(raw, h, var, spec):
        return N.soft_bits(raw, h, np.broadcast_to(var, h.shape), spec)

    def arrays(r, est=None):
        return own(r["spec"]) and all(isinstance(e["h"], np.ndarray) for e in (r["est"], est or r["est"]))

    @guarded("burst_soft", arrays)
    def burst_soft(r, est=None):
        return N.burst_soft(r, est)

    @guarded("slot_soft", lambda r, est, slot: arrays(r, est))
    def slot_soft(r, est, slot):
        return N.slot_soft(r, est, int(slot))

    @guarded("snr_db", lambda est, band: band in config.BANDS)
    def snr_db(est, band):
        return N.snr_db(est, band)

    @guarded("symbol_times", lambda r: own(r["spec"]))
    def symbol_times(r):
        return N.symbol_times(r)

    def burst(t):
        name, payloads, ok, f, start, snr, soft = t
        return modem.Burst(submode=config.SUBMODES[name], payloads=payloads, crc_ok=ok, freq_offset=f,
                           preamble_start=start, snr_db=snr, soft=soft)

    @guarded("decode_received", lambda r: own(r["spec"]) and isinstance(r["est"].get("h"), np.ndarray))
    def decode_received(r):
        return burst(N.decode_received(r))

    @guarded("demodulate")
    def demodulate(x, bands=None, accept=None):
        return burst(N.demodulate(f64(x), bands_arg(bands), accept))

    small = {
        "_bin_phase_step": lambda h: N.bin_phase_step(np.asarray(h, complex)),
        "_cfo_aliases": lambda d, centre: N.cfo_aliases(complex(d), float(centre)),
        "resolve_alias": lambda fine, coarse: N.resolve_alias(float(fine), float(coarse)),
    }
    return {
        **{(modem, k): guarded(k)(fn) for k, fn in small.items()},
        **{(modem, k): v for k, v in {
            "crc6": crc6, "header_bits": header_bits, "_signs": _signs, "_valid_words": _valid_words,
            "_valid_signs": _valid_signs, "decode_header": decode_header, "modulate": modulate,
            "modulate_bits": modulate_bits, "burst_waveform": burst_waveform, "ace_cells": ace_cells,
            "_demod_frames": _demod_frames, "_read_header": _read_header, "_copy_llr": _copy_llr,
            "_best_header": _best_header, "find_burst": find_burst, "pilot_coherence": pilot_coherence,
            "find_copy": find_copy, "_copy_header": _copy_header, "receive": receive, "data_channel": data_channel,
            "noise_var": noise_var, "soft_bits": soft_bits, "burst_soft": burst_soft, "slot_soft": slot_soft,
            "snr_db": snr_db, "symbol_times": symbol_times, "decode_received": decode_received,
            "demodulate": demodulate}.items()},
    }

"""`pytest --native` substitutions for data2g.cpm. See __init__.py."""


def substitutions(native):
    """data2g.cpm. Grids and specs go by name; one that isn't the frozen one
    stays in Python, as does every grid while a test patches modem's
    PROTOCOL_VERSION (the header words' CRC-6 seed, compiled in) or the leak
    model's switches (LEAK, LEAK_K, LEAK_Z, CTL_SQ, SPILL: C++ reads the
    environment once, native.cpm.study())."""
    n = native.cpm
    import functools

    from data2g import config, cpm, modem

    py = {k: getattr(cpm, k) for k in ("header_symbols", "layout", "burst_seconds", "modulate", "detect",
                                       "read_header", "soft", "find", "late_lock", "find_mid", "measure", "refine",
                                       "prime_lock", "CoarseTrack")}

    def frozen_grid(name):  # an on-air grid or a wide toolbox one (compiled in either way), or None
        return cpm.GRIDS.get(name, cpm.WIDE_GRIDS.get(name))

    uses = ("lock", "ctl", "data", "dd")

    def own_grid(g):
        return frozen_grid(g.name) == g and modem.PROTOCOL_VERSION == config.PROTOCOL_VERSION

    def own_spec(s):
        frozen = cpm.SPECS.get(s.name, cpm.WIDE_SPECS.get(s.name, cpm.CTL.get(s.grid, cpm.WIDE_CTL.get(s.grid))))
        return frozen == s and modem.PROTOCOL_VERSION == config.PROTOCOL_VERSION

    def header_symbols(grid, word):
        # the native one recomputes the CRC-6 from word >> 6, with its own seed
        if word < 1 << 16 and modem.PROTOCOL_VERSION == config.PROTOCOL_VERSION:
            return n.header_symbols(grid, word >> 6)
        return py["header_symbols"](grid, word)

    @functools.lru_cache(maxsize=None)
    def layout(grid, n_sym):
        d = n.layout(grid, n_sym)
        return cpm.Layout(d["n"], d["sync_rows"], d["sync_tones"], d["hdr_rows"], d["data_rows"], d["front"])

    def burst_seconds(spec, n_cw, dup=False):
        return n.burst_seconds(spec.name, n_cw, dup) if own_spec(spec) else py["burst_seconds"](spec, n_cw, dup)

    def modulate(spec, coded, dup, cap=0, connect_hash=None):
        if not own_spec(spec):
            return py["modulate"](spec, coded, dup, cap, connect_hash=connect_hash)
        return n.bandpass(spec.grid, n.modulate(spec.name, list(coded), dup, cap, connect_hash), cap)

    def CoarseTrack(g, reach_hz=150.0, source=None):  # a compiled-in grid's: native, for native detect to read
        # (a wide grid's sums of another track's rows, `source`: native from a native track)
        if own_grid(g) and (source is None or isinstance(source, n.CoarseTrack)):
            return n.CoarseTrack(g.name, reach_hz, source)
        return py["CoarseTrack"](g, reach_hz, source)

    def detect(g, x, reach_hz=150.0, fine=True, front_only=False, n_sym=0, floor=-1.0, mid=False, track=None, x0=0):
        if not own_grid(g):
            return py["detect"](g, x, reach_hz, fine, front_only, n_sym, floor, mid, track, x0)
        return n.detect(g.name, x, reach_hz, fine, front_only, n_sym, floor, mid, track, x0)

    def read_header(g, x, s0, cfo, copies=cpm.HDR_COPIES):
        if not own_grid(g):
            return py["read_header"](g, x, s0, cfo, copies)
        name, nd, d, score, h2, front = n.read_header(g.name, x, s0, cfo, copies)
        return cpm.SPECS[name], nd, d, score, h2, front

    def refine(g, x, s0, cfo, spec, n_data, dup, connect_hash=None):
        if not own_grid(g) or not own_spec(spec):
            return py["refine"](g, x, s0, cfo, spec, n_data, dup, connect_hash=connect_hash)
        return n.refine(g.name, x, s0, cfo, spec.name, n_data, dup, connect_hash)

    def soft(g, spec, x, s0, cfo, n_data, dup):
        if not own_grid(g):
            return py["soft"](g, spec, x, s0, cfo, n_data, dup)
        slots, E = n.soft(g.name, x, s0, cfo, n_data, dup)
        return list(slots), E

    def lock(g, d):
        if d is None:
            return None
        if not d.get("wait"):
            d["spec"] = cpm.SPECS[d["spec"]]
        return {**d, "band": g.name, "family": "cpm"}

    def find(g, x, threshold=None, reach_hz=150.0, front_only=True, lo=0, hi=None, wait=False, track=None, x0=0):
        if not own_grid(g):
            return py["find"](g, x, threshold, reach_hz, front_only, lo, hi, wait, track, x0)
        return lock(g, n.find(g.name, x, threshold, reach_hz, front_only, lo, hi, wait, track, x0))

    def late_lock(g, x, cand):
        if not own_grid(g):
            return py["late_lock"](g, x, cand)
        return lock(g, n.late_lock(g.name, x, cand["start"], cand["cfo"], cand["score"], cand["header_end"]))

    def find_mid(g, x, lo=0, hi=None, track=None, x0=0):
        if not own_grid(g):
            return py["find_mid"](g, x, lo, hi, track, x0)
        return lock(g, n.find_mid(g.name, x, lo, hi, track, x0))

    def prime_lock(g, x, lo, hi, cfo):
        if not own_grid(g):
            return py["prime_lock"](g, x, lo, hi, cfo)
        d = lock(g, n.prime_lock(g.name, x, lo, hi, cfo))
        return None if d is None else dict(d, primed=True)

    def measure(g, E, n_sym):
        if not own_grid(g):
            return py["measure"](g, E, n_sym)
        m = n.measure(g.name, E, n_sym)
        return dict(snr_est=m["snr_est"], spread_est=m["spread_est"], delay_est_ms=0.0, headroom=0.0,
                    frames=m["frames"], snr=m["snr"])

    def by_grid(native_fn, py_fn):
        return lambda g, *a, **k: native_fn(g.name, *a, **k) if own_grid(g) else py_fn(g, *a, **k)

    py.update({k: getattr(cpm, k) for k in ("coh_read_header", "dd_metric", "dd_soft", "slot_rows", "bins",
                                            "tone_metric", "spill_significant", "ctl_square")})

    def tone_metric(g, E, use="lock"):
        return n.tone_metric(g.name, E, use) if own_grid(g) and use in uses else py["tone_metric"](g, E, use)

    def spill_significant(E):
        return n.spill_significant(E)

    def ctl_square(g, E, slot):
        return n.ctl_square(g.name, E, slot) if own_grid(g) else py["ctl_square"](g, E, slot)

    def coh_read_header(g, x, s0, cfo, copies=cpm.HDR_COPIES):
        if not own_grid(g):
            return py["coh_read_header"](g, x, s0, cfo, copies)
        name, nd, d, score, h2, front = n.coh_read_header(g.name, x, s0, cfo, copies)
        return cpm.SPECS[name], nd, d, score, h2, front

    def dd_metric(r, post, slot):
        g = cpm.GRIDS[r["band"]]
        if not own_grid(g) or not own_spec(r["spec"]):
            return py["dd_metric"](r, post, slot)
        Z = r["Z"].reshape(-1, g.m)  # (a wide grid's sub-windows: L rows a symbol)
        return n.dd_metric(r["spec"].name, r["dup"], r["n_cw"], Z, {int(k): v for k, v in post.items()}, slot,
                           r.get("connect_hash"))

    def dd_soft(r, metric, slot):
        g = cpm.GRIDS[r["band"]]
        if not own_grid(g) or not own_spec(r["spec"]):
            return py["dd_soft"](r, metric, slot)
        return n.dd_soft(r["spec"].name, r["dup"], r["n_cw"], metric, slot)

    def bins(g, x, start, n_sym, cfo):
        if not own_grid(g):
            return py["bins"](g, x, start, n_sym, cfo)
        Z = n.bins(g.name, x, start, n_sym, cfo)
        return Z if g.L == 1 else Z.reshape(n_sym, g.L, g.m)

    def slot_rows(g, n_data, dup):
        if not own_grid(g):
            return py["slot_rows"](g, n_data, dup)
        return tuple(n.slot_rows(g.name, n_data, dup))

    return {
        (cpm, "header_symbols"): header_symbols,
        (cpm, "layout"): layout,
        (cpm, "stream_symbols"): n.stream_symbols,
        (cpm, "burst_seconds"): burst_seconds,
        (cpm, "to_tones"): by_grid(n.to_tones, cpm.to_tones),
        (cpm, "tones"): by_grid(n.tones, cpm.tones),
        (cpm, "modulate"): modulate,
        (cpm, "bandpass"): by_grid(n.bandpass, cpm.bandpass),
        (cpm, "_energies"): by_grid(n.energies, cpm._energies),
        (cpm, "_shares"): n.shares,
        (cpm, "presence"): by_grid(n.presence, cpm.presence),
        (cpm, "present"): by_grid(n.present, cpm.present),
        (cpm, "detect"): detect,
        (cpm, "CoarseTrack"): CoarseTrack,
        (cpm, "llrs"): by_grid(n.llrs, cpm.llrs),
        (cpm, "symbol_ll"): by_grid(n.symbol_ll, cpm.symbol_ll),
        (cpm, "read_header"): read_header,
        (cpm, "tone_llr"): by_grid(n.tone_llr, cpm.tone_llr),
        (cpm, "refine"): refine,
        (cpm, "soft"): soft,
        (cpm, "_peak_ratio"): by_grid(n.peak_ratio, cpm._peak_ratio),
        (cpm, "find"): find,
        (cpm, "late_lock"): late_lock,
        (cpm, "find_mid"): find_mid,
        (cpm, "prime_lock"): prime_lock,
        (cpm, "hdr_span"): lambda grid, copies: n.hdr_span(grid, copies),
        (cpm, "measure"): measure,
        (cpm, "bins"): bins,
        (cpm, "coh_metric"): n.coh_metric,
        # (a wide grid's k-sub-window sums: the median of their gamma(k), cpm._noise_median, from Python)
        (cpm, "_n0"): lambda E, k=1: n.noise_n0(E, cpm._noise_median(k)),
        (cpm, "bit_llrs"): by_grid(n.bit_llrs, cpm.bit_llrs),
        (cpm, "coh_read_header"): coh_read_header,
        (cpm, "slot_rows"): slot_rows,
        (cpm, "dd_metric"): dd_metric,
        (cpm, "dd_soft"): dd_soft,
        (cpm, "tone_metric"): tone_metric,
        (cpm, "leak_metric"): n.leak_metric,
        (cpm, "spill_significant"): spill_significant,
        (cpm, "ctl_square"): ctl_square,
    }

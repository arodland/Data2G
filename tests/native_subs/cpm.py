"""`pytest --native` substitutions for data2g.cpm. See __init__.py."""


def substitutions(native):
    """data2g.cpm. Grids and specs go by name; one that isn't the frozen one
    stays in Python, as does every grid while a test patches modem's
    PROTOCOL_VERSION (the header words' CRC-6 seed, compiled in)."""
    n = native.cpm
    import functools

    from data2g import config, cpm, modem

    py = {k: getattr(cpm, k) for k in ("header_symbols", "layout", "burst_seconds", "modulate", "detect",
                                       "read_header", "soft", "find", "late_lock", "find_mid", "measure", "refine")}

    def own_grid(g):
        return cpm.GRIDS.get(g.name) == g and modem.PROTOCOL_VERSION == config.PROTOCOL_VERSION

    def own_spec(s):
        return cpm.SPECS.get(s.name, cpm.CTL.get(s.grid)) == s and modem.PROTOCOL_VERSION == config.PROTOCOL_VERSION

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

    def modulate(spec, coded, dup, cap=0):
        if not own_spec(spec):
            return py["modulate"](spec, coded, dup, cap)
        return n.bandpass(spec.grid, n.modulate(spec.name, list(coded), dup, cap), cap)

    def detect(g, x, reach_hz=150.0, fine=True, front_only=False, n_sym=0, floor=-1.0, mid=False):
        if not own_grid(g):
            return py["detect"](g, x, reach_hz, fine, front_only, n_sym, floor, mid)
        return n.detect(g.name, x, reach_hz, fine, front_only, n_sym, floor, mid)

    def read_header(g, x, s0, cfo, copies=cpm.HDR_COPIES):
        if not own_grid(g):
            return py["read_header"](g, x, s0, cfo, copies)
        name, nd, d, score, h2, front = n.read_header(g.name, x, s0, cfo, copies)
        return cpm.SPECS[name], nd, d, score, h2, front

    def refine(g, x, s0, cfo, spec, n_data, dup):
        if not own_grid(g) or not own_spec(spec):
            return py["refine"](g, x, s0, cfo, spec, n_data, dup)
        return n.refine(g.name, x, s0, cfo, spec.name, n_data, dup)

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

    def find(g, x, threshold=None, reach_hz=150.0, front_only=True, lo=0, hi=None, wait=False):
        if not own_grid(g):
            return py["find"](g, x, threshold, reach_hz, front_only, lo, hi, wait)
        return lock(g, n.find(g.name, x, threshold, reach_hz, front_only, lo, hi, wait))

    def late_lock(g, x, cand):
        if not own_grid(g):
            return py["late_lock"](g, x, cand)
        return lock(g, n.late_lock(g.name, x, cand["start"], cand["cfo"], cand["score"], cand["header_end"]))

    def find_mid(g, x, lo=0, hi=None):
        if not own_grid(g):
            return py["find_mid"](g, x, lo, hi)
        return lock(g, n.find_mid(g.name, x, lo, hi))

    def measure(g, E, n_sym):
        if not own_grid(g):
            return py["measure"](g, E, n_sym)
        m = n.measure(g.name, E, n_sym)
        return dict(snr_est=m["snr_est"], spread_est=m["spread_est"], delay_est_ms=0.0, headroom=0.0,
                    frames=m["frames"], snr=m["snr"])

    def by_grid(native_fn, py_fn):
        return lambda g, *a, **k: native_fn(g.name, *a, **k) if own_grid(g) else py_fn(g, *a, **k)

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
        (cpm, "detect"): detect,
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
        (cpm, "hdr_span"): lambda grid, copies: n.hdr_span(grid, copies),
        (cpm, "measure"): measure,
    }

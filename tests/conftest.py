"""`pytest --native` runs this suite with C++ functions (native/bindings)
substituted into the reference modules, so the whole suite is the native
port's acceptance test (docs/native-port-plan.md). Build the module with
tools/build_native.sh. A skip is not a pass: with --native, a missing or
stale module is an error.
"""

import functools
import sys
from pathlib import Path

import pytest

NATIVE_MODULE_DIR = Path(__file__).resolve().parent.parent / "native" / "build" / "python"
NATIVE_ABI = 1
_import_error = None
_originals = {}  # (module, attribute) -> the Python function --native replaced
_PROVIDERS = []  # more substitutions: functions native -> {(module, attr): replacement}


def provider(fn):
    """Register a module's substitutions. One decorated function per ported
    module, appended at the end of this file, so ports merge cleanly."""
    _PROVIDERS.append(fn)
    return fn


def import_native():
    global _import_error
    if str(NATIVE_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(NATIVE_MODULE_DIR))
    try:
        import data2g_native
    except ImportError as e:  # not built, or built for another interpreter
        _import_error = f"{e} (looked in {NATIVE_MODULE_DIR})"
        return None
    if getattr(data2g_native, "__abi__", None) != NATIVE_ABI:
        _import_error = f"{data2g_native.__file__} has ABI {getattr(data2g_native, '__abi__', '?')}, want {NATIVE_ABI}"
        return None
    return data2g_native


# Every data2g.equalizer function but _support_basis (its cache is internal
# to the C++ one).
EQUALIZER_NATIVE = ("residual_cfo", "delay_profile", "delay_support", "window_shift", "_freq_smooth",
                    "preamble_noise", "preamble_noise_k", "per_carrier_noise", "_doppler_corr", "measure_spread",
                    "estimate", "time_shift_phase", "refine")


def _substitutions(native):
    """(module, attribute) -> the native replacement. Specs are passed to C++
    by name, so a spec that isn't exactly the frozen one stays in Python."""
    import functools

    import numpy as np

    from data2g import codes, config, constellation, equalizer, ldpc

    py_frozen = codes.frozen
    nc = native.constellation
    py = {f: getattr(constellation, f) for f in ("load", "ace_dirs", "modulate", "llr", "ace_project")}
    names = set(nc.names())
    by_bytes = {nc.points(n).tobytes(): n for n in names}

    def frozen_array(a):
        a.setflags(write=False)
        return a

    @functools.lru_cache(maxsize=None)
    def load(name):
        return frozen_array(nc.points(name)) if name in names else py["load"](name)

    @functools.lru_cache(maxsize=None)
    def ace_dirs(name):
        return frozen_array(nc.ace_dirs(name)) if name in names else py["ace_dirs"](name)

    # Points cross to C++ by name: an array equal to a frozen set bit for bit
    # is that set (modem.QPSK is gray-qam4); any other stays in Python.
    def name_of(points):
        return by_bytes.get(points.tobytes()) if isinstance(points, np.ndarray) and points.dtype == complex else None

    def modulate(bits, points):
        name = name_of(points)
        return py["modulate"](bits, points) if name is None else nc.modulate(np.asarray(bits).reshape(-1), name)

    def llr(y, h, var, points):
        name = name_of(points)
        return py["llr"](y, h, var, points) if name is None else nc.llr(*np.broadcast_arrays(y, h, var), name)

    def ace_project(got, want, dirs):  # torch (channel_torch) stays in Python
        if not all(isinstance(a, np.ndarray) for a in (got, want, dirs)):
            return py["ace_project"](got, want, dirs)
        got, want = np.broadcast_arrays(got, want)
        return nc.ace_project(got, want, np.broadcast_to(dirs, got.shape + (2,))).reshape(got.shape)

    def frozen(spec):
        if config.SUBMODES.get(spec.name) != spec:
            return py_frozen(spec)
        d = {"fingerprint": codes._fingerprint(spec), "perm": native.codes.interleaver(spec.name)}
        if spec.code == "polar":
            d["info_pos"] = native.codes.info_pos(spec.name)
        return d

    py_polar_code, py_decoder = codes.polar_code, codes._decoder

    @functools.lru_cache(maxsize=None)
    def polar_code(spec):
        if config.SUBMODES.get(spec.name) == spec:
            return native.polar.polar_code(spec.name)
        py = py_polar_code(spec)  # e.g. the CPM control codeword: GA-designed
        if py.frozen_override is None and py.design_snr_db == codes.POLAR_DESIGN_SNR_DB:
            try:
                return native.polar.PolarCode(spec.k, spec.coded_bits)
            except IndexError:  # no frozen GA design for this (k, e)
                pass
        return py

    @functools.lru_cache(maxsize=None)
    def decoder(spec, device=None):
        code = codes.polar_code(spec) if spec.code == "polar" and device is None else None
        if isinstance(code, native.polar.PolarCode):
            return native.polar.SCLDecoder(code, codes.POLAR_LIST)
        return py_decoder(spec, device)

    return {
        (codes, "polar_code"): polar_code,
        (codes, "_decoder"): decoder,
        (codes, "crc24"): native.codes.crc24,
        (codes, "_with_crc"): native.codes.with_crc,
        (codes, "scramble_seed"): native.codes.scramble_seed,
        (codes, "scrambler"): native.codes.scrambler,
        (codes, "frozen"): frozen,
        (constellation, "load"): load,
        (constellation, "ace_dirs"): ace_dirs,
        (constellation, "modulate"): modulate,
        (constellation, "llr"): llr,
        (constellation, "ace_project"): ace_project,
        **{(equalizer, name): getattr(native.equalizer, name) for name in EQUALIZER_NATIVE},
        # every LDPC code (codes.ldpc_code, mother(), IR extents) and decoder
        # (codes._decoder, _ext_decoder, arq.phy's posteriors) is then C++
        (ldpc, "qc_code"): native.ldpc.qc_code,
        (ldpc, "MinSumDecoder"): native.ldpc.MinSumDecoder,
    }


@provider
def _waveform_substitutions(native):
    """data2g.waveform. Bands cross by name; a band whose spec isn't the
    configured one stays in Python. Modules that from-imported a name hold
    their own reference, so modem is listed too."""
    import functools

    import numpy as np

    from data2g import config, modem
    from data2g.waveform import dsp, ofdm, sync

    W = native.waveform
    py_band, py_detection_stat, py_raw_stat = ofdm.band, sync.detection_stat, sync._raw_stat
    py_refine, py_acquire = sync._refine, sync.acquire

    def name_of(band):
        if band is None:
            return "w"
        return band.spec.name if config.BANDS.get(band.spec.name) == band.spec else None

    class NativeBand(ofdm.Band):
        def modulate_symbols(self, symbols):
            return self._n.modulate_symbols(np.atleast_2d(symbols))

        def demod_window(self, z, start, backoff=0):
            return self._n.demod_window(z, int(start), int(backoff))

        def preamble_waveform(self):
            return self._n.preamble_waveform()

        def preamble_template(self):
            return self._template

    @functools.lru_cache(maxsize=None)
    def band(name="w"):
        if config.BANDS.get(name) is None:
            return py_band(name)
        n = W.band(name)
        b = NativeBand(spec=config.BANDS[name], freqs=n.freqs, bb=n.bb, mod=n.mod, demod=n.demod, pilot=n.pilot)
        t = n.preamble_template
        t.flags.writeable = False
        object.__setattr__(b, "_n", n)
        object.__setattr__(b, "_template", t)
        return b

    w = band("w")

    def detection_stat(z, band=None, reach=config.ACQUIRE_REACH_HZ, repeats=None):
        if (n := name_of(band)) is None:
            return py_detection_stat(z, band, reach, repeats)
        return W.detection_stat(z, n, reach, repeats or 0)

    def _raw_stat(z, band=None, reach=config.ACQUIRE_REACH_HZ, repeats=None, levels_from=None, outs=None):
        if (n := name_of(band)) is None:
            return py_raw_stat(z, band, reach, repeats, levels_from, outs)
        S, q, freqs, c = W.raw_stat(z, n, reach, repeats or 0, levels_from, outs is not None)
        if outs is not None:
            outs.extend(c)
        return S, q, freqs

    def _refine(z, band, n, f):
        if (name := name_of(band)) is None:
            return py_refine(z, band, n, f)
        return W.refine(z, name, int(n), float(f))

    def acquire(z, threshold=None, reach=config.ACQUIRE_REACH_HZ, search=None, band=None, S=None):
        if (n := name_of(band)) is None:
            return py_acquire(z, threshold, reach, search, band, S)
        try:
            start, f, metric, alts = W.acquire(z, n, threshold, reach,
                                               None if search is None else (int(search[0]), int(search[1])), S)
        except W.SyncError as e:
            raise sync.SyncError(str(e)) from None
        return sync.Acquisition(preamble_start=start, freq_offset=f, metric=metric, alternatives=alts)

    class StreamDetector(W.StreamDetector):
        def __init__(self, band, reach=config.ACQUIRE_REACH_HZ):
            assert name_of(band), "StreamDetector: native bands only"
            super().__init__(band.spec.name, reach)
            self.band = band

    def to_baseband(x, n0=0):
        return W.to_baseband(x, int(n0))

    def freq_correct(z, f_hz):
        return W.freq_correct(z, float(f_hz))

    def tx_condition(x, clip_headroom_db, overshoot=config.CLIP_OVERSHOOT, active=slice(None),
                     bandpass=config.TX_BANDPASS, project=None, closing=()):
        lo, hi, step = active.indices(len(x))
        assert step == 1
        return W.tx_condition(x, float(clip_headroom_db), list(overshoot), lo, hi, tuple(map(float, bandpass)),
                              project, list(closing))

    return {
        (ofdm, "band"): band,
        (ofdm, "_W"): w,
        (ofdm, "MOD_MATRIX"): w.mod,
        (ofdm, "DEMOD_MATRIX"): w.demod,
        (ofdm, "modulate_symbols"): w.modulate_symbols,
        (ofdm, "demod_window"): w.demod_window,
        (ofdm, "preamble_waveform"): w.preamble_waveform,
        (ofdm, "preamble_template"): w.preamble_template,
        (ofdm, "pilot_sequence"): lambda: w.pilot,
        (modem, "PILOT"): w.pilot,
        (dsp, "to_baseband"): to_baseband,
        (dsp, "freq_correct"): freq_correct,
        (dsp, "tx_condition"): tx_condition,
        (dsp, "papr_db"): W.papr_db,
        (modem, "to_baseband"): to_baseband,
        (modem, "freq_correct"): freq_correct,
        (modem, "tx_condition"): tx_condition,
        (modem, "acquire"): acquire,
        (sync, "detection_stat"): detection_stat,
        (sync, "_raw_stat"): _raw_stat,
        (sync, "_repeat_corr"): W.repeat_corr,
        (sync, "_repeat_corrs"): lambda z, t, freqs: W.repeat_corrs(z, t, list(freqs)),
        (sync, "first_path"): lambda power, peak, search=config.FIRST_PATH_SEARCH, frac=config.FIRST_PATH_FRAC,
        cyclic=False: W.first_path(power, int(peak), search, frac, cyclic),
        (sync, "_refine"): _refine,
        (sync, "_crossings"): lambda D, threshold, span, limit: W.crossings(D, threshold, span, limit),
        (sync, "acquire"): acquire,
        (sync, "StreamDetector"): StreamDetector,
    }


@provider
def _cpm_substitutions(native):
    """data2g.cpm. Grids and specs go by name; one that isn't the frozen one
    stays in Python. The TX bandpass (dsp) and the MI features (predictor)
    are still Python's, applied to the native results."""
    n = native.cpm
    import functools

    import numpy as np

    from data2g import cpm
    from data2g.arq import predictor as P

    py = {k: getattr(cpm, k) for k in ("header_symbols", "layout", "burst_seconds", "modulate", "detect",
                                       "read_header", "soft", "find", "measure")}

    def own_grid(g):
        return cpm.GRIDS.get(g.name) == g

    def own_spec(s):
        return cpm.SPECS.get(s.name, cpm.CTL.get(s.grid)) == s

    def header_symbols(grid, word):
        return n.header_symbols(grid, word >> 6) if word < 1 << 16 else py["header_symbols"](grid, word)

    @functools.lru_cache(maxsize=None)
    def layout(grid, n_sym):
        d = n.layout(grid, n_sym)
        return cpm.Layout(d["n"], d["sync_rows"], d["sync_tones"], d["hdr_rows"], d["data_rows"], d["front"])

    def burst_seconds(spec, n_cw, dup=False):
        return n.burst_seconds(spec.name, n_cw, dup) if own_spec(spec) else py["burst_seconds"](spec, n_cw, dup)

    def modulate(spec, coded, dup):
        if not own_spec(spec):
            return py["modulate"](spec, coded, dup)
        return cpm.bandpass(cpm.GRIDS[spec.grid], n.modulate(spec.name, list(coded), dup))

    def detect(g, x, reach_hz=150.0, fine=True, front_only=False, n_sym=0, floor=-1.0):
        if not own_grid(g):
            return py["detect"](g, x, reach_hz, fine, front_only, n_sym, floor)
        return n.detect(g.name, x, reach_hz, fine, front_only, n_sym, floor)

    def read_header(g, x, s0, cfo, copies=cpm.HDR_COPIES):
        if not own_grid(g):
            return py["read_header"](g, x, s0, cfo, copies)
        name, nd, d, score, h2 = n.read_header(g.name, x, s0, cfo, copies)
        return cpm.SPECS[name], nd, d, score, h2

    def soft(g, spec, x, s0, cfo, n_data, dup):
        if not own_grid(g):
            return py["soft"](g, spec, x, s0, cfo, n_data, dup)
        slots, E = n.soft(g.name, x, s0, cfo, n_data, dup)
        return list(slots), E

    def find(g, x, threshold=None, reach_hz=150.0, front_only=True, lo=0, hi=None):
        if not own_grid(g):
            return py["find"](g, x, threshold, reach_hz, front_only, lo, hi)
        d = n.find(g.name, x, threshold, reach_hz, front_only, lo, hi)
        if d is None:
            return None
        d["spec"] = cpm.SPECS[d["spec"]]
        return {**d, "band": g.name, "family": "cpm"}

    def measure(g, E, n_sym):
        if not own_grid(g):
            return py["measure"](g, E, n_sym)
        m = n.measure(g.name, E, n_sym)
        snr = m.pop("snr")
        out = dict(snr_est=m["snr_est"], spread_est=m["spread_est"], delay_est_ms=0.0, headroom=0.0, frames=m["frames"])
        for c in P.CONSTS:
            out[f"mi_{c}"] = P.effective_mi(np.sqrt(snr), np.ones_like(snr), c)
        return out

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
        (cpm, "_energies"): by_grid(n.energies, cpm._energies),
        (cpm, "_shares"): n.shares,
        (cpm, "detect"): detect,
        (cpm, "llrs"): by_grid(n.llrs, cpm.llrs),
        (cpm, "read_header"): read_header,
        (cpm, "soft"): soft,
        (cpm, "_peak_ratio"): by_grid(n.peak_ratio, cpm._peak_ratio),
        (cpm, "find"): find,
        (cpm, "measure"): measure,
    }


@provider
def _codes_substitutions(native):
    """data2g.codes' codec (encode, combine, decode) in C++. Specs go by
    name: a frozen submode, a CPM mode or a CPM control codeword; any other
    spec, torch (decode_llrs' device) and dtypes the binding lacks stay in
    Python. codes.decode routes through decode_many; _decode_code_order
    takes a decoder object and is C++-internal."""
    import functools

    import numpy as np

    from data2g import codes, config, cpm

    n = native.codes
    py = {k: getattr(codes, k) for k in (
        "crc_bits", "payload_bytes", "rv_cycle", "buffer_len", "rv_positions", "info_bits", "encode", "encode_info",
        "flip", "spread", "despread", "combine", "decode_buffer", "_payloads", "decode_many", "decode_llrs",
        "decode_raw", "descramble", "check", "crc_ok", "interleaver")}

    @functools.lru_cache(maxsize=None)
    def own(spec):
        return (config.SUBMODES.get(spec.name) == spec or cpm.SPECS.get(spec.name) == spec
                or cpm.CTL.get(getattr(spec, "grid", None)) == spec)

    def by_name(attr):
        f, p = getattr(n, attr), py[attr]
        return lambda spec, *a, **k: f(spec.name, *a, **k) if own(spec) else p(spec, *a, **k)

    @functools.lru_cache(maxsize=None)
    def flip(spec, index, rv=0):
        out = n.flip(spec.name, index, rv) if own(spec) else py["flip"](spec, index, rv)
        out.setflags(write=False)
        return out

    @functools.lru_cache(maxsize=None)
    def interleaver(spec):
        if not own(spec):
            return py["interleaver"](spec)
        out = n.interleaver(spec.name)
        out.setflags(write=False)
        return out

    def native_array(x, ndim):
        return isinstance(x, np.ndarray) and x.ndim >= ndim and x.dtype in (np.uint8, np.float64)

    def spread(coded, m):
        if not native_array(coded, 2):
            return py["spread"](coded, m)
        *lead, n_cw, N = coded.shape
        return n.spread(coded.reshape(-1, n_cw * N), n_cw, m).reshape(*lead, n_cw * N)

    def despread(x, n_cw, m):
        if not native_array(x, 1):
            return py["despread"](x, n_cw, m)
        *lead, total = x.shape
        return n.despread(x.reshape(-1, total), n_cw, m).reshape(*lead, n_cw, total // n_cw)

    def info_bits(spec, payload, crc_mask=0, index=0):
        if not own(spec):
            return py["info_bits"](spec, payload, crc_mask, index)
        return n.info_bits(spec.name, bytes(payload), crc_mask, index)

    def encode(spec, payload, rv=0, crc_mask=0, index=0):
        if not own(spec):
            return py["encode"](spec, payload, rv, crc_mask, index)
        return n.encode(spec.name, bytes(payload), rv, crc_mask, index)

    def combine(spec, buf, soft, rvs):
        if not own(spec):
            return py["combine"](spec, buf, soft, rvs)
        return n.combine(spec.name, buf, np.atleast_2d(soft), rvs)

    def _payloads(spec, bits, converged, masks=None, index=None):
        if not own(spec):
            return py["_payloads"](spec, bits, converged, masks, index)
        return n.payloads(spec.name, np.asarray(bits), np.asarray(converged, np.uint8), masks, index)

    def decode_llrs(spec, llr, iters=40, device=None, crc_mask=0, index=0):
        if device is not None or not own(spec):
            return py["decode_llrs"](spec, llr, iters, device, crc_mask, index)
        return n.decode_llrs(spec.name, llr, iters, crc_mask, index)

    def descramble(spec, bits, index):
        if not own(spec):
            return py["descramble"](spec, bits, index)
        return n.descramble(spec.name, np.asarray(bits), int(index))

    return {
        **{(codes, a): by_name(a) for a in ("crc_bits", "payload_bytes", "rv_cycle", "buffer_len", "rv_positions",
                                             "encode_info", "decode_buffer", "decode_many", "decode_raw", "check",
                                             "crc_ok")},
        (codes, "flip"): flip,
        (codes, "interleaver"): interleaver,
        (codes, "spread"): spread,
        (codes, "despread"): despread,
        (codes, "info_bits"): info_bits,
        (codes, "encode"): encode,
        (codes, "combine"): combine,
        (codes, "_payloads"): _payloads,
        (codes, "decode_llrs"): decode_llrs,
        (codes, "descramble"): descramble,
    }


# Study toggles: with any set, the predictor and the shifter stay Python
# (C++ has the installed model, its LOGIT_OFFSETS, every mode, BIAS_MAX 6).
GEAR_STUDY_ENV = ("DATA2G_OUTCOME_MODEL", "DATA2G_OUTCOME_LCB", "DATA2G_LOGIT_OFFSETS", "DATA2G_DROP_MODES",
                  "DATA2G_BIAS_FIX")


@provider
def _gear_substitutions(native):
    """modem's burst timing, data2g.arq.modes, .predictor and .policy (the
    gear shifter: GearShifter is a Python subclass whose state and methods
    live in C++). Specs go by name; one that isn't the configured one stays
    in Python. Modules that from-imported a name are listed too."""
    import dataclasses
    import os

    import numpy as np

    from data2g import config, kisslink, modem
    from data2g.arq import engine, modes
    from data2g.arq import policy as G
    from data2g.arq import predictor as P

    T, A = native.timing, native.arq
    py = {k: getattr(modem, k) for k in ("_hosts", "header_layout", "header_samples", "copy_frame", "frames_on_air",
                                         "burst_end", "head_samples", "burst_seconds")}
    py.update({f"modes.{k}": getattr(modes, k) for k in ("burst_seconds", "ctl_payload_bytes", "max_ctl", "min_cw")})
    py.update({f"G.{k}": getattr(G, k) for k in ("width_hz", "ctl_slots", "slots_for")})

    def own(spec):
        return config.SUBMODES.get(spec.name) == spec

    def own_mode(spec):
        return modes.MODES.get(spec.name) == spec

    def by_band(name):
        native_fn, py_fn = getattr(T, name.lstrip("_")), py[name]
        return lambda band, *a: native_fn(band, *a) if band in config.BANDS else py_fn(band, *a)

    def by_spec(native_fn, py_fn, test=own_mode):
        return lambda spec, *a: native_fn(spec.name, *a) if test(spec) else py_fn(spec, *a)

    def burst_end(p0, spec, n_cw):
        return T.burst_end(int(p0), spec.name, n_cw) if own(spec) else py["burst_end"](p0, spec, n_cw)

    def burst_seconds(spec, n_cw, dup=False):
        return A.burst_seconds(spec.name, int(n_cw), bool(dup)) if own_mode(spec) else \
            py["modes.burst_seconds"](spec, n_cw, dup)

    ctl_payload_bytes = by_spec(A.ctl_payload_bytes, py["modes.ctl_payload_bytes"])
    max_ctl = by_spec(A.max_ctl, py["modes.max_ctl"])
    min_cw = by_spec(lambda n, data: A.min_cw(n, bool(data)), py["modes.min_cw"])
    subs = {
        **{(modem, k): by_band(k) for k in ("_hosts", "header_layout", "header_samples", "copy_frame", "head_samples")},
        (modem, "frames_on_air"): by_spec(T.frames_on_air, py["frames_on_air"], own),
        (modem, "burst_end"): burst_end,
        (modem, "burst_seconds"): by_spec(T.burst_seconds, py["burst_seconds"], own),
        **{(m, "burst_seconds"): burst_seconds for m in (modes, G)},
        **{(m, "ctl_payload_bytes"): ctl_payload_bytes for m in (modes, G, kisslink)},
        **{(m, "max_ctl"): max_ctl for m in (modes, G, kisslink)},
        **{(m, "min_cw"): min_cw for m in (modes, G)},
    }
    if any(v in os.environ for v in GEAR_STUDY_ENV):
        return subs

    def effective_mi(h, var, const):
        h, var = np.broadcast_arrays(np.asarray(h, dtype=complex), np.asarray(var, dtype=float))
        return A.effective_mi(h.reshape(-1), var.reshape(-1), const)

    def capacity(snr_db, const):
        out = A.capacity(np.asarray(snr_db, dtype=float).reshape(-1), const).reshape(np.shape(snr_db))
        return out[()] if out.ndim == 0 else out

    def outcome_inputs(measured, band, gap, seconds, prev=None, bands=P.BANDS):
        return A.outcome_inputs(measured, band, gap, seconds, prev, list(bands))

    def predict_outcome(measured, band, gap, seconds, submodes=None, prev=None):
        d = A.predict_outcome(measured, band, gap, seconds, prev)
        return {s.name: d[s.name] for s in (submodes or config.SUBMODES.values())}

    class GearShifter(G.GearShifter):
        """State in the C++ object; dict and list fields cross by copy (assign
        them whole, as link.py and the tests do)."""

        def __init__(self, *a, **k):
            object.__setattr__(self, "_n", A.GearShifter())
            super().__init__(*a, **k)

        def choose(self, station, escalation):
            return self._n.choose(station, int(escalation))

        def next_capacity(self, station):
            return self._n.next_capacity(station)

        def observe(self, measured, submode, now):
            self._n.observe(measured, submode, float(now))

        def outcome(self, submode, decoded, sent, usable=None):
            self._n.outcome(submode, int(decoded), int(sent), None if usable is None else bool(usable))

        def recommend(self, station):
            return self._n.recommend(station)

        def payload_bytes(self, m):
            return A.payload_bytes(m)

        def ctl_payload_bytes(self, m):
            return A.ctl_payload_bytes(m)

        def max_ctl(self, m):
            return A.max_ctl(m)

        def rv_cycle(self, m):
            return A.rv_cycle(m)

        def connect_mode(self, cap, tries=0):
            return A.connect_mode(cap, tries)

        def airtime(self, m, n_cw, dup=False):
            return A.burst_seconds(m, int(n_cw), bool(dup))

        def mode_name(self, rec):
            return A.decode(int(rec)) or f"?{rec}"

    for f in dataclasses.fields(G.GearShifter):
        setattr(GearShifter, f.name, property(lambda s, f=f.name: getattr(s._n, f),
                                              lambda s, v, f=f.name: setattr(s._n, f, v)))

    def decode(rec):
        return A.decode(int(rec))

    return {
        **subs,
        (P, "effective_mi"): effective_mi,
        (P, "capacity"): capacity,
        (P, "const_family"): A.const_family,
        (P, "outcome_inputs"): outcome_inputs,
        (P, "outcome_knows"): A.outcome_knows,
        (P, "predict_outcome"): predict_outcome,
        (G, "allowed"): lambda cap: [modes.MODES[n] for n in A.allowed(cap)],
        (G, "width_hz"): by_spec(A.width_hz, py["G.width_hz"]),
        (G, "encode"): A.encode,
        (G, "decode"): decode,
        (G, "ctl_slots"): by_spec(A.ctl_slots, py["G.ctl_slots"]),
        (G, "slots_for"): lambda spec, seconds, data=True, dup=False: (
            A.slots_for(spec.name, float(seconds), bool(data), bool(dup)) if own_mode(spec)
            else py["G.slots_for"](spec, seconds, data, dup)),
        (G, "GearShifter"): GearShifter,
        (engine, "GearShifter"): GearShifter,
    }


def pytest_addoption(parser):
    parser.addoption("--native", action="store_true", default=False,
                     help="run the suite against the C++ core (tools/build_native.sh builds it)")


def pytest_configure(config):
    if not config.getoption("--native"):
        return
    native = import_native()
    if native is None:
        raise pytest.UsageError(f"--native: cannot import data2g_native: {_import_error}")
    subs = _substitutions(native)
    for p in _PROVIDERS:
        subs.update(p(native))
    for (module, attr), fn in subs.items():
        _originals[module, attr] = getattr(module, attr)
        setattr(module, attr, fn)
    config._native_count = len(subs)


def pytest_report_header(config):
    if config.getoption("--native"):
        return f"native: {config._native_count} substitutions from {NATIVE_MODULE_DIR}"


@pytest.fixture(scope="session")
def native():
    """The module for direct parity tests, with or without --native."""
    module = import_native()
    if module is None:
        pytest.skip(f"data2g_native not built: {_import_error}")
    return module


@pytest.fixture(scope="session")
def reference():
    """reference(module, "name"): the Python function, substituted or not."""
    return lambda module, attr: _originals.get((module, attr), getattr(module, attr))

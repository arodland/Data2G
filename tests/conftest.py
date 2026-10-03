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

    py_outcome_inputs = P.outcome_inputs

    def outcome_inputs(measured, band, gap, seconds, prev=None, bands=P.BANDS, noise=False):
        if noise:  # the noise profile's inputs: Python only, until a model with them ships
            return py_outcome_inputs(measured, band, gap, seconds, prev, bands, noise)
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

        def reply_hold(self, station, burst):
            return self._n.reply_hold(station, burst.submode)

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


@provider
def _audio_substitutions(native):
    """The host's rate conversion and input conditioning. host.py
    from-imports Decimator, so it is listed there too; tnc.Receiver looks
    Blanker up at construction, so it gets the native one."""
    from data2g import host, tnc

    A = native.audio
    return {
        (tnc, "Decimator"): A.Decimator,
        (host, "Decimator"): A.Decimator,
        (host, "Interpolator"): A.Interpolator,
        (tnc, "Blanker"): A.Blanker,
    }

@provider
def _arq_substitutions(native):
    """data2g.arq frames, link and session. Station and Session are the C++
    classes behind Python-compatible wrappers (native/bindings/bind_arq.cpp):
    the policy, the received burst and the session's random.Random stay the
    Python objects, called in the same order as the Python does."""
    from data2g.arq import frames, link, session

    a = native.arq
    out = {(frames, f): getattr(a, f) for f in ("Core", "Control", "pack_bitmap", "unpack_bitmap", "pack_rv", "unpack_rv",
                                                 "pack_flags", "unpack_flags", "deflate", "deflate_fit", "inflate",
                                                 "pack_call", "unpack_call", "pack_connect", "unpack_connect", "to_records",
                                                 "RecordReader")}
    out.update({(link, f): getattr(a, f) for f in ("unwrap", "ctl_mask", "data_mask", "Station")})
    out.update({(session, f): getattr(a, f) for f in ("session_key", "Session", "_frame_desc")})
    return out



@provider
def _modem_substitutions(native):
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
    names = ("_crc6", "header_bits", "_signs", "_valid_words", "_valid_signs", "decode_header", "modulate",
             "modulate_bits", "burst_waveform", "ace_cells", "_bin_phase_step", "_demod_frames", "_read_header",
             "_copy_llr", "_best_header", "find_burst", "pilot_coherence", "find_copy", "_cfo_aliases",
             "_copy_header", "receive", "resolve_alias", "data_channel", "noise_var", "soft_bits", "demodulate",
             "decode_received")
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

    @guarded("_crc6")
    def _crc6(v):
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
        if not 1 <= len(payloads) <= config.MAX_CODEWORDS:
            raise ValueError(f"1..{config.MAX_CODEWORDS} codewords per burst, got {len(payloads)}")
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
    def _demod_frames(z, p, n_f, shift, phi_ref, steps_in=None, band="w"):
        return N.demod_frames(z, int(p), int(n_f), int(shift), float(phi_ref), steps_in, band)

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

    @guarded("receive", lambda x, bands=None, accept=None, head=None, copy=None: copy is None or lock_ok(copy))
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
            "_crc6": _crc6, "header_bits": header_bits, "_signs": _signs, "_valid_words": _valid_words,
            "_valid_signs": _valid_signs, "decode_header": decode_header, "modulate": modulate,
            "modulate_bits": modulate_bits, "burst_waveform": burst_waveform, "ace_cells": ace_cells,
            "_demod_frames": _demod_frames, "_read_header": _read_header, "_copy_llr": _copy_llr,
            "_best_header": _best_header, "find_burst": find_burst, "pilot_coherence": pilot_coherence,
            "find_copy": find_copy, "_copy_header": _copy_header, "receive": receive, "data_channel": data_channel,
            "noise_var": noise_var, "soft_bits": soft_bits, "decode_received": decode_received,
            "demodulate": demodulate}.items()},
    }


@provider
def _tnc_substitutions(native):
    """data2g.tnc: KISS framing, burst packing, search_span, receive_any and
    the streaming Receiver. Receiver is the C++ one behind tnc.Receiver's
    interface (engine from-imports it, so it is listed there too); it is the
    Python one when a test patches its search (_stats, _searched) or modem's
    compiled-in constants. Specs come back by name and are restored here."""
    import numpy as np

    from data2g import config, cpm, modem, tnc
    from data2g.arq import engine
    from data2g.waveform import sync

    T = native.tnc
    PyReceiver = tnc.Receiver
    py = {k: getattr(tnc, k) for k in ("capacity", "pack", "receive_any")}
    min_score = dict(modem.HEADER_MIN_SCORE)

    def same_config():
        return modem.PROTOCOL_VERSION == config.PROTOCOL_VERSION and modem.HEADER_MIN_SCORE == min_score

    def own(spec):
        return config.SUBMODES.get(spec.name) == spec

    def header(d):
        return dict(d, spec=(cpm.SPECS if d.get("family") == "cpm" else config.SUBMODES)[d["spec"]])

    def rx(d):
        if d is None:
            return None
        if d.get("family") == "cpm":
            return dict(d, spec=cpm.SPECS[d["spec"]])
        start, f, metric, alts = d["acq"]
        return dict(d, spec=config.SUBMODES[d["spec"]],
                    acq=sync.Acquisition(preamble_start=start, freq_offset=f, metric=metric, alternatives=alts))

    def event(kind, d):
        return (kind, header(d)) if kind == "header" else (kind, dict(d, header=header(d["header"]), rx=rx(d["rx"])))

    class Receiver(PyReceiver):
        __doc__ = PyReceiver.__doc__

        def __init__(self, accept, cpm_grids=(), blank=True):
            cls = type(self)
            self._n = None
            if cls._stats is PyReceiver._stats and cls._searched is PyReceiver._searched and same_config():
                self.accept, self.bands = accept, accept.bands
                self._n = T.Receiver(accept, list(cpm_grids), blank)
            else:
                super().__init__(accept, cpm_grids, blank)

        def reset(self):
            return super().reset() if self._n is None else self._n.reset()

        def feed(self, x):
            if self._n is None:
                return super().feed(x)
            return [event(k, d) for k, d in self._n.feed(np.asarray(x, dtype=np.float64))]

        busy = property(lambda s: PyReceiver.busy.fget(s) if s._n is None else s._n.busy)
        channel_busy = property(lambda s: PyReceiver.channel_busy.fget(s) if s._n is None else s._n.channel_busy)
        on_air = property(lambda s: PyReceiver.on_air.fget(s) if s._n is None else s._n.on_air)

    def capacity(spec, max_cw=config.MAX_CODEWORDS):
        return T.capacity(spec.name, int(max_cw)) if own(spec) else py["capacity"](spec, max_cw)

    def pack(packets, spec):
        return T.pack([bytes(p) for p in packets], spec.name) if own(spec) else py["pack"](packets, spec)

    def receive_any(y, lead=0, cpm_grids=None):
        if not same_config():
            return py["receive_any"](y, lead, cpm_grids)
        return rx(T.receive_any(np.asarray(y, dtype=np.float64), int(lead),
                                None if cpm_grids is None else list(cpm_grids)))

    return {
        (tnc, "kiss_encode"): lambda data, port=0: T.kiss_encode(bytes(data), int(port)),
        (tnc, "KissDecoder"): T.KissDecoder,
        (tnc, "capacity"): capacity,
        (tnc, "pack"): pack,
        (tnc, "unpack"): lambda payloads, ok: T.unpack([bytes(p) for p in payloads], [bool(o) for o in ok]),
        (tnc, "search_span"): lambda bands, cpm_grids=(): T.search_span(list(bands), list(cpm_grids)),
        (tnc, "receive_any"): receive_any,
        (tnc, "Receiver"): Receiver,
        (engine, "Receiver"): Receiver,
    }


@provider
def _phy_substitutions(native):
    """data2g.arq.phy (TX audio, ModemRx with DD, measure) and
    data2g.kisslink (KissLink, AX.25 parsing) in C++. ModemRx stays Python
    for a mode that isn't the configured one, or while a test patches a
    codec function it calls (C++ never calls back into them); it reads
    phy.DD when made, and its store stays the caller's dict."""
    import time

    from data2g import codes, kisslink
    from data2g.arq import modes
    from data2g.arq import phy as PHY

    P, K = native.phy, native.kisslink
    py_rx, py_tx, py_soft, py_measure = PHY.ModemRx, PHY.tx_audio, PHY.soft_bits, PHY.measure
    watched = ("decode_raw", "decode_buffer", "decode_llrs", "_payloads", "check", "combine", "flip", "encode",
               "encode_info", "descramble")

    def own(spec):
        return modes.MODES.get(spec.name) == spec

    def substituted(f):  # the codes provider's, not a test's patch
        return getattr(f, "__module__", None) == __name__

    class ModemRx:
        """A class, as phy.ModemRx is, so a study can wrap its methods
        (scripts/crc_exposure.py patches __init__ and decode); the decoding
        is the C++ object's."""
        __doc__ = py_rx.__doc__

        def __new__(cls, r, store, dd_budget=None):
            if not own(r["spec"]) or not all(substituted(getattr(codes, f)) for f in watched):
                return py_rx(r, store, dd_budget)
            return super().__new__(cls)

        def __init__(self, r, store, dd_budget=None):
            self._n = P.ModemRx(r, store, dd_budget, bool(PHY.DD))
            self.spec, self.n_cw, self.submode = r["spec"], r["n_cw"], r["spec"].name
            self.n_ctl_slots, self.store, self.r = r.get("n_ctl_slots", 0), store, r

        _spec = py_rx._spec

        def decode(self, slot, mask_id, rv, key):
            return self._n.decode(slot, mask_id, rv, key)

        def forget(self, key):
            return self._n.forget(key)

    def tx_audio(burst):
        return P.tx_audio(burst) if burst.submode in modes.MODES else py_tx(burst)

    def soft_bits(r):
        return P.soft_bits(r) if own(r["spec"]) else py_soft(r)

    def measure(r):
        return P.measure(r) if own(r["spec"]) else py_measure(r)

    def parse_ax25(frame):
        t = K.parse_ax25(bytes(frame))
        return None if t is None else kisslink.Ax25(*t)

    def KissLink(cap=2, queue=None, peers=None, me=None, clock=time.monotonic, n_sent=0, broadcast=None, **kw):
        if queue or peers or me:
            raise NotImplementedError("--native: a KissLink starts with no queue, peers or me")
        return K.KissLink(cap, clock=None if clock is time.monotonic else clock, n_sent=n_sent, broadcast=broadcast,
                          **kw)

    return {
        (PHY, "mask_value"): P.mask_value,
        (PHY, "tx_audio"): tx_audio,
        (PHY, "soft_bits"): soft_bits,
        (PHY, "measure"): measure,
        (PHY, "ModemRx"): ModemRx,
        (kisslink, "parse_ax25"): parse_ax25,
        (kisslink, "station_hash"): K.station_hash,
        (kisslink, "KissLink"): KissLink,
    }


@provider
def _engine_substitutions(native):
    """data2g.arq.engine.Engine: the C++ Engine in its deterministic (sync)
    mode behind engine.Engine's interface (native/bindings/bind_engine.cpp).
    The policy factory, random.Random(seed), the session's Python view and
    phy.tx_audio stay the Python objects, called as engine.py calls them; a
    KissLink that isn't the C++ one falls back to the Python Engine. host.py
    from-imports Engine, so it is listed there too."""
    import random

    import numpy as np

    from data2g import host, modem
    from data2g.arq import engine as E
    from data2g.arq import phy as PHY

    PyEngine, N, KL = E.Engine, native.engine, native.kisslink.KissLink

    class _Receiver:  # what the host reads of engine.receiver
        def __init__(self, n):
            self._n = n

        busy = property(lambda s: s._n.busy)
        channel_busy = property(lambda s: s._n.channel_busy)

    class Engine:
        __doc__ = PyEngine.__doc__

        def __new__(cls, call, policy=None, ptt_delay_s=0.1, record_dir=None, seed=None, min_header_score=0.0,
                    kiss=None, stats_interval_s=60.0):
            if kiss is not None and not isinstance(kiss, KL):
                return PyEngine(call, policy, ptt_delay_s, record_dir, seed, min_header_score, kiss, stats_interval_s)
            return super().__new__(cls)

        def __init__(self, call, policy=None, ptt_delay_s=0.1, record_dir=None, seed=None, min_header_score=0.0,
                     kiss=None, stats_interval_s=60.0):
            self._n = N.Engine(call, policy or E.GearShifter, ptt_delay_s, record_dir, None, min_header_score, kiss,
                               stats_interval_s, rng=random.Random(seed), tx_audio=lambda b: PHY.tx_audio(b),
                               dd=bool(PHY.DD))
            self.accept = modem.Accept.of(None, E.MAX_BURST_S, min_header_score)
            self._kiss_rx = []

        def __getattr__(self, name):  # step, session, connect, events, tx, _extra, ...
            return getattr(self._n, name)

        def step(self, x):
            return self._n.step(np.asarray(x, dtype=np.float64))

        n = property(lambda s: s._n.n, lambda s, v: setattr(s._n, "n", v))
        id_interval_s = property(lambda s: s._n.id_interval_s, lambda s, v: setattr(s._n, "id_interval_s", v))

        @property
        def kiss_rx(self):
            self._kiss_rx += self._n.take_kiss_rx()
            return self._kiss_rx

        @property
        def receiver(self):
            r = self._n.receiver
            return _Receiver(self._n) if r is None else r

        @receiver.setter
        def receiver(self, r):
            self._n.receiver = r

    return {(E, "Engine"): Engine, (host, "Engine"): Engine}


@provider
def _host_substitutions(native):
    """data2g.host.Host: the C++ Host (native/bindings/bind_host.cpp) when
    its engine is the C++ one (_engine_substitutions' Engine); over a Python
    Engine it stays Python. out_cmd and out_data are this wrapper's list and
    bytearray, filled after each call, so tests read and clear them as
    host.py's."""
    from data2g import host

    PyHost, N = _originals.get((host, "Host"), host.Host), native.host

    class Host:
        __doc__ = PyHost.__doc__

        def __new__(cls, engine, buffer_credit=None):
            if not isinstance(getattr(engine, "_n", None), native.engine.Engine):
                return PyHost(engine, buffer_credit)
            return super().__new__(cls)

        def __init__(self, engine, buffer_credit=None):
            self.engine, self._n = engine, N.Host(engine._n, buffer_credit)
            self.out_cmd, self.out_data = [], bytearray()

        def _take(self, result=None):
            self.out_cmd += self._n.take_cmd()
            self.out_data += self._n.take_data()
            return result

        def command(self, line):
            return self._take(self._n.command(line))

        def client_gone(self):
            return self._take(self._n.client_gone())

        def data_in(self, data):
            return self._take(self._n.data_in(bytes(data)))

        def after_step(self, ptt):
            return self._take(self._n.after_step(bool(ptt)))

        cap = property(lambda s: s._n.cap, lambda s, v: setattr(s._n, "cap", v))
        listening = property(lambda s: s._n.listening, lambda s, v: setattr(s._n, "listening", v))
        buffer_credit = property(lambda s: s._n.buffer_credit, lambda s, v: setattr(s._n, "buffer_credit", v))

    return {(host, "Host"): Host}

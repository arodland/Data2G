"""`pytest --native` substitutions for data2g.waveform (ofdm, dsp, sync). See __init__.py."""


def substitutions(native):
    """data2g.waveform. Bands cross by name; a band whose spec isn't the
    configured one stays in Python."""
    import functools

    import numpy as np

    from data2g import config
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
        (ofdm, "modulate_symbols"): w.modulate_symbols,
        (ofdm, "demod_window"): w.demod_window,
        (ofdm, "preamble_waveform"): w.preamble_waveform,
        (ofdm, "preamble_template"): w.preamble_template,
        (ofdm, "pilot_sequence"): lambda: w.pilot,
        (dsp, "to_baseband"): to_baseband,
        (dsp, "freq_correct"): freq_correct,
        (dsp, "tx_condition"): tx_condition,
        (dsp, "papr_db"): W.papr_db,
        (sync, "detection_stat"): detection_stat,
        (sync, "_raw_stat"): _raw_stat,
        (sync, "repeat_corr"): W.repeat_corr,
        (sync, "repeat_corrs"): lambda z, t, freqs: W.repeat_corrs(z, t, list(freqs)),
        (sync, "noise_level"): lambda C: W.noise_level(np.asarray(C)),
        (sync, "first_path"): lambda power, peak, search=config.FIRST_PATH_SEARCH, frac=config.FIRST_PATH_FRAC,
        cyclic=False: W.first_path(power, int(peak), search, frac, cyclic),
        (sync, "_refine"): _refine,
        (sync, "_crossings"): lambda D, threshold, span, limit: W.crossings(D, threshold, span, limit),
        (sync, "acquire"): acquire,
        (sync, "StreamDetector"): StreamDetector,
    }

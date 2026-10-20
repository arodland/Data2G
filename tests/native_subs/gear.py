"""`pytest --native` substitutions for the gear shifter: modem's burst timing,
data2g.arq.modes, .predictor and .policy. See __init__.py."""

# Study toggles: with any set, the predictor and the shifter stay Python
# (C++ has the installed model, its LOGIT_OFFSETS, every mode, BIAS_MAX 6).
GEAR_STUDY_ENV = ("DATA2G_OUTCOME_MODEL", "DATA2G_OUTCOME_GATE", "DATA2G_OUTCOME_FAST", "DATA2G_CPM_FLOOR", "DATA2G_OUTCOME_LCB", "DATA2G_LOGIT_OFFSETS", "DATA2G_DROP_MODES",
                  "DATA2G_BIAS_FIX", "DATA2G_NOISE_RULE")
# GearShifter fields the C++ shifter doesn't keep: study inputs (link history,
# energy) that models with those inputs read in Python only
PY_ONLY_GEAR = {"turns", "timeouts_seen", "link_now"}


def substitutions(native):
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
                                         "burst_end", "head_samples", "receive_head", "burst_seconds")}
    py.update({f"modes.{k}": getattr(modes, k) for k in ("burst_seconds", "ctl_payload_bytes", "max_ctl", "min_cw")})
    py.update({f"G.{k}": getattr(G, k) for k in ("width_hz", "ctl_slots", "slots_for")})
    py["predict_outcome"] = P.predict_outcome

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
        **{(modem, k): by_band(k) for k in ("_hosts", "header_layout", "header_samples", "copy_frame", "head_samples",
                                               "receive_head")},
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

    def outcome_inputs(measured, band, gap, seconds, prev=None, bands=P.BANDS, noise=False, link=False, energy=False):
        if noise or link:  # the noise profile's and link history's inputs: Python only (studies)
            return py_outcome_inputs(measured, band, gap, seconds, prev, bands, noise, link, energy)
        return A.outcome_inputs(measured, band, gap, seconds, prev, list(bands), bool(energy))

    def predict_outcome(measured, band, gap, seconds, submodes=None, prev=None, model=None):
        if model is not None:  # another model than the installed one (a gate's): Python
            return py["predict_outcome"](measured, band, gap, seconds, submodes, prev, model)
        d = A.predict_outcome(measured, band, gap, seconds, prev)
        return {s.name: d[s.name] for s in (submodes or config.SUBMODES.values()) if s.name in d}

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

        def observe_energy(self, snr_db, now):
            if snr_db is not None:
                self._n.observe_energy(float(snr_db), float(now))

        def energy_features(self):
            e = self._n.energy_features()
            return None if e is None else list(e)

        def expected_reply(self):
            return self._n.expected_reply()

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
        if f.name in PY_ONLY_GEAR:  # study state the C++ shifter doesn't keep (link history, energy)
            continue
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

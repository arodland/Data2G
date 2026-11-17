"""`pytest --native` substitutions for data2g.arq frames, burst, link and
session. See __init__.py."""


def substitutions(native):
    """data2g.arq frames, link and session. Station and Session are the C++
    classes behind Python-compatible wrappers (native/bindings/bind_arq.cpp):
    the policy, the received burst and the session's random.Random stay the
    Python objects, called in the same order as the Python does."""
    import functools

    from data2g.arq import burst, frames, link, nod, onod, session

    a = native.arq
    out = {(frames, f): getattr(a, f) for f in ("Core", "Control", "pack_bitmap", "unpack_bitmap", "pack_rv", "unpack_rv",
                                                 "pack_flags", "unpack_flags", "deflate", "deflate_fit", "inflate",
                                                 "pack_call", "unpack_call", "pack_connect", "unpack_connect", "to_records",
                                                 "RecordReader")}
    out.update({(link, f): getattr(a, f) for f in ("unwrap", "ctl_mask", "data_mask", "Station")})
    out.update({(burst, f): getattr(a, f) for f in ("ctl_mask", "data_mask")})
    out.update({(session, f): getattr(a, f) for f in ("session_key", "Session", "_frame_desc")})
    out[(nod, "family")] = a.nod_family
    out[(nod, "detect")] = a.nod_detect
    # OFDM nods (onod.py): the family (cached, read-only, as Python's), a sequence's audio, the slot's scores
    thr = dict(onod.THR)

    @functools.lru_cache(maxsize=512)
    def family(band, k, key, direction, bn8, acted8):
        f = a.onod_family(band, k, key, direction, bn8, acted8)
        f.flags.writeable = False
        return f

    def detect(x, lo, hi, cfo, band, k, fam, thr_=None):
        return a.onod_detect(x, int(lo), int(hi), float(cfo), band, k, fam, onod.THR[(band, k)] if thr_ is None else thr_)

    out[(onod, "family")] = family
    out[(onod, "waveform")] = a.onod_waveform
    out[(onod, "scores")] = lambda x, lo, hi, cfo, band, k, fam: a.onod_scores(x, int(lo), int(hi), float(cfo), band, k, fam)
    out[(onod, "detect")] = detect
    return out

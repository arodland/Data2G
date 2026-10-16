"""`pytest --native` substitutions for data2g.arq frames, burst, link and
session. See __init__.py."""


def substitutions(native):
    """data2g.arq frames, link and session. Station and Session are the C++
    classes behind Python-compatible wrappers (native/bindings/bind_arq.cpp):
    the policy, the received burst and the session's random.Random stay the
    Python objects, called in the same order as the Python does."""
    from data2g.arq import burst, frames, link, session

    a = native.arq
    out = {(frames, f): getattr(a, f) for f in ("Core", "Control", "pack_bitmap", "unpack_bitmap", "pack_rv", "unpack_rv",
                                                 "pack_flags", "unpack_flags", "deflate", "deflate_fit", "inflate",
                                                 "pack_call", "unpack_call", "pack_connect", "unpack_connect", "to_records",
                                                 "RecordReader")}
    out.update({(link, f): getattr(a, f) for f in ("unwrap", "ctl_mask", "data_mask", "Station")})
    out.update({(burst, f): getattr(a, f) for f in ("ctl_mask", "data_mask")})
    out.update({(session, f): getattr(a, f) for f in ("session_key", "Session", "_frame_desc")})
    return out

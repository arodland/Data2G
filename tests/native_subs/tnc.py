"""`pytest --native` substitutions for data2g.tnc's KISS byte codec. See __init__.py."""


def substitutions(native):
    """data2g.tnc: the KISS byte codec."""
    from data2g import tnc

    T = native.tnc
    return {
        (tnc, "kiss_encode"): lambda data, port=0, cmd=0: T.kiss_encode(bytes(data), int(port), int(cmd)),
        (tnc, "KissDecoder"): T.KissDecoder,
    }

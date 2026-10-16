"""`pytest --native` substitutions for data2g.audio's filters. See __init__.py."""


def substitutions(native):
    """data2g.audio: the host's rate conversion and input conditioning.
    host.py and receiver.Receiver look them up in data2g.audio when they
    make one, so they get the native ones."""
    from data2g import audio

    A = native.audio
    return {
        (audio, "Decimator"): A.Decimator,
        (audio, "Interpolator"): A.Interpolator,
        (audio, "Blanker"): A.Blanker,
    }

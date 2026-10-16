"""Data2G: an HF data modem. This package is the normative reference: the
on-air format, the receiver and the protocols are defined here, and the
C++ port (native/) is tested against it. docs/ARCHITECTURE.md is the map;
each module's docstring has the detail.

Modules, bottom up (each imports only those above it, but for config's
lazy look-up of a constellation's size):

    config          on-air constants: bands, the SUBMODES ladder, clip constants
    constellation   point sets, mapping, bit LLRs
    waveform        OFDM: dsp (baseband, TX clipper), ofdm (bands), sync (preamble)
    equalizer       burst-wide pilot channel estimation
    ldpc, polar     the codes, knowing nothing of submodes
    codes           CRC framing, scrambling, interleaving, HARQ combining,
                    the decode entry points, keyed by submode
    modem           the OFDM burst modem: header, modulate, receive, soft bits
    cpm             the constant-envelope M-FSK modes (the ARQ's bottom rungs)
    audio           sound card <-> FS: rate conversion, impulse blanking
    receiver        the streaming receiver: audio in, bursts out
    arq             the connected ARQ, the gear shifter and the station
                    runtime (arq.engine.Engine); see arq/__init__.py
    kisslink        broadcast over KISS (named groups)
    tnc             the KISS byte codec and rigctld PTT
    host            data2g-host: the VARA-compatible server around an Engine

Beside them: threads (BLAS thread caps, set before numpy loads). Studies
only, never on the live path or ported: hfchannel and interference
(channel simulators), channel_torch and decoders_torch (torch, for
training and GPU studies).
"""

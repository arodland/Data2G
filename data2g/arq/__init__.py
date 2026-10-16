"""The connected, half-duplex ARQ (docs/arq.md) and the live station
stack around it. Modules, bottom up:

    frames     wire codecs: the core word, TLV extensions, session/CQ/ID
               bodies, callsigns, compression, stream records
    burst      what a burst is to the PHY: Slot, TxBurst, RxBurst and the
               codeword identities (ctl_mask, data_mask)
    link       one station's transfer state, both directions (Station),
               and the Policy interface it asks for modes
    session    connect, disconnect, keepalive and wake around a Station,
               on its caller's clock
    modes      every mode the ARQ can use, its bandwidth and the
               recommendation code that names it on air
    predictor  the outcome model and its measurement features
    policy     the gear shifter (GearShifter): link.Policy on air
    phy        bursts on the real modem: TxBurst -> audio, a received
               burst -> RxBurst (masked CRCs, HARQ combining, DD), measure()
    engine     Engine.step(audio): receiver, session, shifter, phy, KISS
               and ID frames, clocked by audio samples

frames, burst, link and session know no modem: the two seams are
burst.RxBurst (phy.ModemRx on air, fakes in tests and scripts/linksim.py)
and link.Policy (policy.GearShifter, or a fixed one). phy is the only
module that reaches into codes and the modem; policy, predictor and modes
are the gear shifter's.

Native: native/core/arq/<module>.hpp|.cpp, each header citing its Python
file; tests/conftest.py's _arq_, _gear_ and _phy_substitutions run this
package's tests on it under pytest --native.
"""

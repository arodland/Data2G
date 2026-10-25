"""modem.receive(head=...) at a burst's end: the burst has wholly arrived,
so a weak header is not refused as "still arriving"."""

from pathlib import Path

import numpy as np

from data2g import modem
from data2g.arq import link as L
from data2g.arq import phy as PHY
from data2g.arq.engine import MAX_BURST_S
from data2g.config import LEADIN_SAMPLES


def test_weak_header_at_the_burst_end_is_received(protocol):
    """AG7EW's ack-4f x3 on air (recordings/20261002-232711, t 387.6), the
    segment the streaming receiver passed: header score 0.48, under
    STREAM_COMMIT_SCORE, with an acquisition alternative near the head
    window's end. The live receiver lost it ("a header is still arriving");
    its three control codewords decode (session key 4818, the callee's).
    Recorded before codes.polar_order reversed polar payloads, so they are
    decoded in the order they were sent: no reversal, the CRC checked as
    sent."""
    protocol(12)  # recorded under it
    from data2g import codes, polar

    seg = np.load(Path(__file__).parent / "data" / "ag7ew_ack4f.npz")["audio"].astype(np.float64)
    head = LEADIN_SAMPLES + modem.receive_head("w")
    r = modem.receive(seg, ["w"], modem.Accept.of(None, MAX_BURST_S, 0.0), head=min(head, len(seg)))
    assert r["spec"].name == "ack-4f" and r["n_cw"] == 3 and r["score"] < modem.STREAM_COMMIT_SCORE
    spec = r["spec"]  # ack-4f: its control is its own codeword
    code = polar.PolarCode(spec.k, spec.coded_bits, codes.POLAR_DESIGN_SNR_DB,
                           tuple(int(i) for i in codes.polar_code(spec).info_pos))
    soft = np.asarray(PHY.soft_bits(r), np.float32)
    deint = np.empty_like(soft)
    deint[:, np.asarray(codes.interleaver(spec))] = soft
    paths, _ = polar.SCLDecoder(code, codes.POLAR_LIST).decode(deint)
    for i in range(3):
        m = PHY.mask_value(L.ctl_mask(1, i, 4818))
        assert codes.crc_ok(spec, paths[i], np.full(len(paths[i]), m), np.full(len(paths[i]), i)).any()

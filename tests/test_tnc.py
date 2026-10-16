"""The TNC's links: KISS byte framing and rigctld PTT."""

from data2g import tnc


def test_kiss_roundtrip_with_escapes_across_reads():
    frames = [bytes([0xC0, 0xDB, 0xDC, 0xDD, 1, 2]), b"hello", bytes([0xDB, 0xDC])]
    wire = b"".join(tnc.kiss_encode(f) for f in frames)
    dec, got = tnc.KissDecoder(), []
    for i in range(0, len(wire), 3):  # frames split across reads
        got += dec.feed(wire[i : i + 3])
    assert got == [(0, f) for f in frames]


def test_kiss_decoder_discards_a_frame_that_never_ends():
    dec = tnc.KissDecoder()
    assert dec.feed(b"\x00" + b"a" * (1 << 17)) == []  # no FEND: junk, not buffered forever
    assert dec.feed(b"zz\xc0\x00hi\xc0") == [(0, b"hi")]  # the next frame is read normally


def _fake_rigctld():
    """A rigctld stand-in on a free port: answers RPRT 0, records commands."""
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    got = []

    def serve():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            with c:
                while data := c.recv(64):
                    got.append(data)
                    c.sendall(b"RPRT 0\n")

    threading.Thread(target=serve, daemon=True).start()
    return srv, got


def test_exit_unkeys_only_a_radio_we_keyed(caplog):
    import socket
    import time

    # never connected: no attempt, no wait, no warning
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    rig, t = tnc.Rigctld("127.0.0.1", closed), time.monotonic()
    rig.release()
    assert time.monotonic() - t < 0.5 and not caplog.records

    # connected, never keyed: no PTT off
    srv, got = _fake_rigctld()
    rig = tnc.Rigctld("127.0.0.1", srv.getsockname()[1])
    rig.ptt(False)  # an unkey while running (connects, never keys)
    got.clear()
    rig.release()
    assert got == [] and not rig.keyed

    # keyed once: PTT off at exit
    rig = tnc.Rigctld("127.0.0.1", srv.getsockname()[1])
    rig.ptt(True)
    rig.release()
    time.sleep(0.2)
    assert got[-1] == b"T 0\n"
    srv.close()

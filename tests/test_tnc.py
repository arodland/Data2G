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


def test_vara_send_commands_are_data():
    """VARA's KISS sends frames with command 1 or 2 (issue #283); standard
    TXDELAY and P carry one byte and stay commands."""
    assert tnc.is_data(0x00, b"") and tnc.is_data(0x30, b"x")
    assert tnc.is_data(0x01, b"ab") and tnc.is_data(0x12, b"ab")
    assert not tnc.is_data(0x01, b"\x32") and not tnc.is_data(0x02, b"\x3f")
    assert not tnc.is_data(0x03, b"ab") and not tnc.is_data(0x06, b"MODE x")


def test_kiss_server_preserves_command_nibble():
    """KissServer forwards the data frame's command nibble to on_packet and
    encodes broadcast() with the command nibble it is given (#283)."""
    import socket
    import threading
    import time

    packets, cmds = [], []
    srv = tnc.KissServer(("127.0.0.1", 0), lambda port, f, ack, cmd: packets.append((port, f, ack is not None, cmd)),
                         lambda cmd, p, port: cmds.append((cmd, p, port)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(srv.server_address, timeout=2) as c:
            c.settimeout(2)
            c.sendall(tnc.kiss_encode(b"std", 0, 0)
                      + tnc.kiss_encode(b"vara7", 0, 1)
                      + tnc.kiss_encode(b"beacon", 2, 2)
                      + tnc.kiss_encode(b"\x12\x34acked", 1, tnc.ACKMODE)
                      + tnc.kiss_encode(b"\x3f", 0, 2))  # 1-byte P command, not data
            deadline = time.monotonic() + 2
            while (len(packets) < 4 or len(cmds) < 1) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert packets == [(0, b"std", False, 0), (0, b"vara7", False, 1),
                               (2, b"beacon", False, 2), (1, b"acked", True, 0)]
            assert cmds == [(2, b"\x3f", 0)]
            srv.broadcast(b"rx0", 0, 0)
            srv.broadcast(b"rx1", 0, 1)
            srv.broadcast(b"rx2", 2, 2)
            dec, got = tnc.KissDecoder(), []
            while len(got) < 3:
                got += dec.feed(c.recv(4096))
            assert got == [(0x00, b"rx0"), (0x01, b"rx1"), (0x22, b"rx2")]
    finally:
        srv.shutdown()
        srv.close_clients()
        srv.server_close()

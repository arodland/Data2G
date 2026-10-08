"""data2g.host's TCP ports and audio loop, with a fake sound card and no PTT."""

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from data2g import host


class FakeStream:
    def __init__(self, rate, channels):
        self.rate, self.channels = rate, channels

    def write(self, data):
        assert len(data) % (4 * self.channels) == 0

    def get_output_latency(self):
        return 0.0

    def close(self):
        pass


class FakePA:
    """A stereo sound card (the host opens stereo where it can)."""

    def get_default_input_device_info(self):
        return {"maxInputChannels": 2, "maxOutputChannels": 0}

    def get_default_output_device_info(self):
        return {"maxInputChannels": 0, "maxOutputChannels": 2}

    def open(self, rate, channels, stream_callback=None, input=False, **kw):
        if stream_callback:  # record noise / play what the host queues, in real time
            rng = np.random.default_rng(0)

            def card():
                while True:
                    x = rng.normal(0, 0.01, 1024 * channels).astype(np.float32).tobytes() if input else None
                    stream_callback(x, 1024, None, 0)
                    time.sleep(1024 / rate)
            threading.Thread(target=card, daemon=True).start()
        return FakeStream(rate, channels)


def _connect(port, tries=50):
    for _ in range(tries):
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=5)
        except OSError:
            time.sleep(0.1)
    return socket.create_connection(("127.0.0.1", port), timeout=5)


def test_commands_over_tcp(tmp_path):
    a = SimpleNamespace(mycall="W1AW", host="127.0.0.1", command_port=18310, sample_rate=48000, output_volume=0.0,
                        rigctld_host="localhost", rigctld_port=0, ptt_on_delay_ms=100, ptt_off_delay_ms=0, tx_lead_ms=100,
                        min_header_score=0.0, record_dir=tmp_path, input_device=None, output_device=None,
                        buffer_credit=-1, vara=True, kiss=True, kiss_port=18320, kiss_address="127.0.0.1",
                        kiss_bw=2400, broadcast_mode=None, kiss_busy_limit=60.0, stats_interval=60.0)
    stop = threading.Event()
    th = threading.Thread(target=host.serve, args=(a, FakePA(), stop), daemon=True)
    th.start()
    c, d = _connect(18310), _connect(18311)  # the data port binds after the command port
    c.sendall(b"VERSION\rMYCALL K2XYZ\rLISTEN ON\rBW500\rFOO\r")
    got = b""
    deadline = time.time() + 20
    while got.count(b"\r") < 5 and time.time() < deadline:
        got += c.recv(4096)
    stop.set()
    th.join(timeout=20)
    c.close()
    d.close()
    assert got.split(b"\r")[:5] == [f"VERSION {host.VERSION}".encode(), b"OK", b"OK", b"OK", b"WRONG"]
    assert (tmp_path / "events.jsonl").exists()


def test_broadcast_port_and_ackmode_over_tcp(tmp_path):
    """A group opened on the command port, a frame sent on its KISS port with
    ACKMODE: the tag comes back on the KISS port once the burst has gone."""
    from data2g import tnc

    a = SimpleNamespace(mycall="W1AW", host="127.0.0.1", command_port=18330, sample_rate=48000, output_volume=0.0,
                        rigctld_host="localhost", rigctld_port=0, ptt_on_delay_ms=100, ptt_off_delay_ms=0, tx_lead_ms=100,
                        min_header_score=0.0, record_dir=tmp_path, input_device=None, output_device=None,
                        buffer_credit=-1, kiss_port=18340, kiss_address="127.0.0.1",
                        kiss_bw=2400, broadcast_mode=None, kiss_busy_limit=60.0, stats_interval=60.0)
    stop = threading.Event()
    th = threading.Thread(target=host.serve, args=(a, FakePA(), stop), daemon=True)
    th.start()
    c, k = _connect(18330), _connect(18340)
    try:
        c.sendall(b"BCAST OPEN APRS\r")
        got = b""
        deadline = time.time() + 20
        while b"\r" not in got and time.time() < deadline:
            got += c.recv(4096)
        assert got.split(b"\r")[0] == b"BCAST PORT 1"
        k.sendall(tnc.kiss_encode(b"\x12\x34" + b"!beacon", 1, tnc.ACKMODE))
        dec, frames = tnc.KissDecoder(), []
        k.settimeout(1.0)
        while not frames and time.time() < deadline + 20:
            try:
                frames += dec.feed(k.recv(4096))
            except socket.timeout:
                pass
        assert frames == [(1 << 4 | tnc.ACKMODE, b"\x12\x34")]
    finally:
        stop.set()
        th.join(timeout=20)
        c.close()
        k.close()


def test_capture_keeps_every_sample():
    """Input queued by the callback while the loop is busy comes out whole, in order, left channel."""
    class PA:
        def open(self, stream_callback, **kw):
            self.cb = stream_callback
            return FakeStream(48000, 2)

    pa = PA()
    cap = host.Capture(pa, 2, 48000, None)
    ramp = np.arange(10_000, dtype=np.float32)
    for i in range(0, len(ramp), 512):  # the loop stalled: 10k frames arrive before any read
        pa.cb(np.repeat(ramp[i:i + 512], 2).tobytes(), 512, None, 2 if i == 0 else 0)
    stop = threading.Event()
    got = np.concatenate([cap.read(4800, stop), cap.read(4800, stop)])
    assert np.array_equal(got, ramp[:9600])
    assert cap.overflows == 1
    stop.set()
    assert cap.read(4800, stop) is None  # 400 frames left: not a block, and stopping


def test_bad_callsigns_are_wrong_not_fatal():
    from data2g.arq.engine import Engine

    h = host.Host(Engine("W1AW", seed=1))
    h.command("CONNECT W1AW SOMETHINGLONG")
    h.command("MYCALL BAD!CALL")
    assert h.out_cmd == ["WRONG", "WRONG"]
    assert h.engine.call == "W1AW" and h.engine.session.state == "idle"


def test_empty_kiss_frame_is_not_queued():
    from data2g.kisslink import KissLink

    k = KissLink()
    k.enqueue(b"")
    assert not k.queue

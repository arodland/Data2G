"""data2g.host's TCP ports and audio loop, with a fake sound card and no PTT."""

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from data2g import host


class FakeStream:
    def __init__(self, rate, channels):
        self.rate, self.channels, self.rng = rate, channels, np.random.default_rng(0)

    def read(self, n, exception_on_overflow=True):
        time.sleep(0.002)
        return (self.rng.normal(0, 0.01, n * self.channels)).astype(np.float32).tobytes()

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

    def open(self, rate, channels, stream_callback=None, **kw):
        if stream_callback:  # play what the host queues, in real time
            def card():
                while True:
                    stream_callback(None, 1024, None, 0)
                    time.sleep(1024 / rate)
            threading.Thread(target=card, daemon=True).start()
        return FakeStream(rate, channels)


def test_commands_over_tcp(tmp_path):
    a = SimpleNamespace(mycall="W1AW", host="127.0.0.1", command_port=18310, sample_rate=48000, output_volume=0.0,
                        rigctld_host="localhost", rigctld_port=0, ptt_on_delay_ms=100, ptt_off_delay_ms=0, tx_lead_ms=100,
                        min_header_score=0.0, record_dir=tmp_path, input_device=None, output_device=None,
                        buffer_credit=-1, vara=True, kiss=True, kiss_port=18320, kiss_address="127.0.0.1",
                        kiss_bw=2400, broadcast_mode=None, kiss_busy_limit=60.0)
    stop = threading.Event()
    th = threading.Thread(target=host.serve, args=(a, FakePA(), stop), daemon=True)
    th.start()
    for _ in range(50):
        try:
            c = socket.create_connection(("127.0.0.1", 18310), timeout=5)
            break
        except OSError:
            time.sleep(0.1)
    d = socket.create_connection(("127.0.0.1", 18311), timeout=5)
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

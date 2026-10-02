"""Two data2g-host processes (the C++ app) cross-connected through two named
pipes (--audio-io pipe:IN,OUT): a VARA session driven over the TCP ports
as a client would (LISTEN ON, CONNECT, ~2 kB each way, DISCONNECT), then
KISS frames both ways. Neither host may log an audio overflow, underrun,
backlog or drop. Skips if the binary isn't built; `pytest --native` errors
instead."""

import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from data2g import tnc

BINARY = Path(__file__).resolve().parent.parent / "native" / "build" / ("data2g-host.exe" if os.name == "nt" else "data2g-host")
BAD = re.compile(r"overflow|underrun|behind the card|dropped|CRITICAL|Traceback")


def free_ports(n):
    """A base port with n consecutive free ports after it."""
    for _ in range(100):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = s.getsockname()[1]
        if base + n > 65535:
            continue
        socks = []
        try:
            for p in range(base, base + n):
                s = socket.socket()
                socks.append(s)
                s.bind(("127.0.0.1", p))
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free ports")


class Client:
    """A VARA client: the command port's lines and the data port's bytes."""

    def __init__(self, port, proc, seconds=60):
        # 60 s: a first launch on macOS waits for the binary and Qt's
        # frameworks to be assessed before main runs.
        deadline = time.monotonic() + seconds
        while True:
            try:
                self.cmd = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError as e:
                if proc.poll() is not None:
                    raise AssertionError(f"host on port {port} exited {proc.returncode} before listening") from e
                if time.monotonic() > deadline:
                    raise AssertionError(f"host on port {port} not listening after {seconds} s") from e
                time.sleep(0.1)
        self.data = socket.create_connection(("127.0.0.1", port + 1), timeout=1)
        self.cmd.settimeout(0.05)
        self.data.settimeout(0.05)
        self.lines, self.got, self._buf = [], bytearray(), b""

    def send(self, line):
        self.cmd.sendall(line.encode() + b"\r")

    def poll(self):
        for sock, sink in ((self.cmd, None), (self.data, self.got)):
            try:
                d = sock.recv(65536)
            except (socket.timeout, BlockingIOError):
                continue
            if sink is not None:
                sink += d
                continue
            self._buf += d
            *done, self._buf = self._buf.split(b"\r")
            self.lines += [x.decode() for x in done]

    def close(self):
        self.cmd.close()
        self.data.close()


def wait(clients, until, seconds, what):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for c in clients:
            c.poll()
        if until():
            return time.monotonic()
        time.sleep(0.02)
    raise AssertionError(f"timed out: {what}; " + " | ".join(str(c.lines[-20:]) for c in clients))


def kiss_read(sock, dec, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            d = sock.recv(65536)
        except socket.timeout:
            continue
        for cmd, payload in dec.feed(d):
            if cmd == 0:
                return payload
    return None


def ui_frame(dst, src, info):
    def addr(call, last):
        return bytes(ord(c) << 1 for c in call.ljust(6)) + bytes([0x60 | last])
    return addr(dst, False) + addr(src, True) + b"\x03\xf0" + info


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes (POSIX mkfifo)")
@pytest.mark.parametrize("worker", ["--decode-worker", "--no-decode-worker"])
def test_two_hosts_over_named_pipes(tmp_path, request, worker):
    if not BINARY.exists():
        if request.config.getoption("--native"):
            pytest.fail(f"--native: {BINARY} not built (tools/build_native.sh)")
        pytest.skip(f"{BINARY} not built")
    a2b, b2a = tmp_path / "a2b", tmp_path / "b2a"
    os.mkfifo(a2b)
    os.mkfifo(b2a)
    # Diagnostic: how long a bare start takes here (a first launch on macOS
    # can take seconds), printed if the test fails.
    t = time.monotonic()
    subprocess.run([str(BINARY), "--help"], capture_output=True, timeout=120)
    t_help = time.monotonic() - t
    ports = {}
    procs, logs = {}, {}
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for name, inp, out in (("A", b2a, a2b), ("B", a2b, b2a)):
        base = free_ports(3)
        ports[name] = base
        logs[name] = tmp_path / f"{name}.log"
        procs[name] = subprocess.Popen(
            [str(BINARY), "--mycall", name, "--command-port", str(base), "--kiss-port", str(base + 2),
             "--audio-io", f"pipe:{inp},{out}", "--rigctld-port", "0", "--record-dir", str(tmp_path / f"rec{name}"),
             "--stats-interval", "0", worker],
            stdout=subprocess.DEVNULL, stderr=open(logs[name], "w"), env=env)
    clients = {}
    try:
        a = clients["A"] = Client(ports["A"], procs["A"])
        b = clients["B"] = Client(ports["B"], procs["B"])
        b.send("MYCALL B")
        b.send("LISTEN ON")
        a.send("MYCALL A")
        a.send("VERSION")
        # a reply needs audio flowing both ways: the engine steps on capture
        wait([a, b], lambda: len(b.lines) >= 2 and "VERSION Data2G 0.1" in a.lines, 30, "replies")
        assert b.lines[:2] == ["OK", "OK"]
        t0 = time.monotonic()
        a.send("CONNECT A B")
        t_conn = wait([a, b], lambda: "CONNECTED A B 2300" in a.lines and "CONNECTED A B 2300" in b.lines, 60,
                      "CONNECTED") - t0
        rng = np.random.default_rng(1)
        up, down = rng.bytes(2048), rng.bytes(2048)
        t1 = time.monotonic()
        a.data.sendall(up)
        b.data.sendall(down)
        t_data = wait([a, b], lambda: len(b.got) >= len(up) and len(a.got) >= len(down), 120, "data") - t1
        assert bytes(b.got) == up and bytes(a.got) == down
        a.send("DISCONNECT")
        wait([a, b], lambda: "DISCONNECTED" in a.lines and "DISCONNECTED" in b.lines, 60, "DISCONNECTED")
        for c in (a, b):
            assert any(x.startswith("BUFFER") for x in c.lines) and "PTT ON" in c.lines and "BUSY ON" in c.lines

        # KISS, between sessions: a UI frame each way
        ka = socket.create_connection(("127.0.0.1", ports["A"] + 2), timeout=0.1)
        kb = socket.create_connection(("127.0.0.1", ports["B"] + 2), timeout=0.1)
        time.sleep(0.3)
        fa, fb = ui_frame("APRS", "A", b"hello from A"), ui_frame("APRS", "B", b"hello from B")
        t2 = time.monotonic()
        ka.sendall(tnc.kiss_encode(fa))
        assert kiss_read(kb, tnc.KissDecoder(), 30) == fa
        kb.sendall(tnc.kiss_encode(fb))
        assert kiss_read(ka, tnc.KissDecoder(), 30) == fb
        t_kiss = time.monotonic() - t2
        ka.close()
        kb.close()
    finally:
        for c in clients.values():
            c.close()
        for p in procs.values():
            p.send_signal(signal.SIGTERM)
        codes = {}
        for n, p in procs.items():
            try:
                codes[n] = p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                codes[n] = f"killed after SIGTERM was ignored for 20 s ({p.wait()})"
        # shown by pytest only on failure
        print(f"\n{worker}: data2g-host --help took {t_help:.1f} s; exit codes {codes}")
        for n in logs:
            print(f"--- host {n} stderr ---\n{logs[n].read_text()[-4000:]}")
    text = {n: logs[n].read_text() for n in logs}
    print(f"\n{worker}: connect {t_conn:.1f} s; 2 kB each way {t_data:.1f} s "
          f"({8 * 2 * 2048 / t_data:.0f} bit/s both ways); KISS both ways {t_kiss:.1f} s")
    for n, t in text.items():
        bad = [line for line in t.splitlines() if BAD.search(line)]
        assert not bad, (n, bad)
        assert "shutting down" in t, (n, t[-2000:])
        assert codes[n] == 0, (n, codes[n], t[-2000:])


def test_cli_has_every_host_py_flag(request):
    """data2g-host takes host.py's command line: every flag in its --help."""
    import sys

    if not BINARY.exists():
        if request.config.getoption("--native"):
            pytest.fail(f"--native: {BINARY} not built (tools/build_native.sh)")
        pytest.skip(f"{BINARY} not built")
    flags = re.compile(r"--[a-z][a-z0-9-]*")
    py = subprocess.run([sys.executable, "-m", "data2g.host", "--help"], capture_output=True, text=True, check=True).stdout
    cpp = subprocess.run([str(BINARY), "--help"], capture_output=True, text=True, check=True).stdout
    assert set(flags.findall(py)) - set(flags.findall(cpp)) == set()
    bad = subprocess.run([str(BINARY), "--kiss-bw", "600"], capture_output=True, text=True)
    assert bad.returncode == 2 and "invalid choice" in bad.stderr
    out = subprocess.run([str(BINARY), "--list-modes", "--kiss-bw", "500"], capture_output=True, text=True, check=True).stdout
    ref = subprocess.run([sys.executable, "-m", "data2g.host", "--list-modes", "--kiss-bw", "500"], capture_output=True,
                         text=True, check=True).stdout
    assert out == ref

"""run.sh's raw phases: a Pat-free exchange over the VARA ports, so data2g
sees the bytes as written (Pat's B2F LZHUF-compresses every message first,
leaving nothing for T_COMP). W1AW connects K2XYZ, both send a file at once
as Pat's P2P exchange does, then W1AW disconnects. Also vara_ref.sh's
transfer, straight into VARA (Pat writes at most 7 x 127 bytes between
BUFFER reports, and VARA reports BUFFER only after each burst, so a Pat
session gets 889 bytes an over).

    python raw.py <a command port> <b command port> <a->b file> <b->a file> <timeout s>

RAW_CALLS: the two calls (default "W1AW K2XYZ"), each sent as MYCALL
first; RAW_BW: a bandwidth command for both (e.g. BW2300). Prints bytes,
seconds, rates and whether each direction arrived exact, and the median of
any SN reports (VARA's SNR estimate); exits 1 if not exact (or on
timeout)."""

import os
import select
import socket
import statistics
import sys
import time


def main():
    port_a, port_b, up, down, timeout = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5])
    up, down = open(up, "rb").read(), open(down, "rb").read()
    deadline = time.monotonic() + timeout
    cmd = {k: socket.create_connection(("127.0.0.1", p)) for k, p in (("a", port_a), ("b", port_b))}
    data = {k: socket.create_connection(("127.0.0.1", p + 1)) for k, p in (("a", port_a), ("b", port_b))}
    lines = {"a": [], "b": []}
    pending = {"a": b"", "b": b""}
    got = {"a": bytearray(), "b": bytearray()}

    def pump(until) -> bool:
        """Read both hosts' ports until until() or the deadline."""
        while not until():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            ready, _, _ = select.select([*cmd.values(), *data.values()], [], [], min(left, 1.0))
            for s in ready:
                chunk = s.recv(65536)
                if not chunk:
                    raise SystemExit(f"host closed {s.getpeername()}")
                k = next(k for k in "ab" if s in (cmd[k], data[k]))
                if s is data[k]:
                    got[k] += chunk
                    continue
                pending[k] += chunk
                *done, pending[k] = pending[k].split(b"\r")
                lines[k] += [x.decode() for x in done if x]
        return True

    def heard(k, prefix):
        return lambda: any(x.startswith(prefix) for x in lines[k])

    call_a, call_b = os.environ.get("RAW_CALLS", "W1AW K2XYZ").split()
    for k, call in (("a", call_a), ("b", call_b)):
        cmd[k].sendall(f"MYCALL {call}\r".encode() + (os.environ["RAW_BW"].encode() + b"\r" if "RAW_BW" in os.environ else b""))
    cmd["b"].sendall(b"LISTEN ON\r")
    cmd["a"].sendall(f"CONNECT {call_a} {call_b}\r".encode())
    if not pump(lambda: heard("a", "CONNECTED")() and heard("b", "CONNECTED")()):
        raise SystemExit(f"no connect in {timeout:.0f} s: {lines}")
    t0 = time.monotonic()
    data["a"].sendall(up)
    data["b"].sendall(down)
    ok = pump(lambda: len(got["b"]) >= len(up) and len(got["a"]) >= len(down))
    dt = time.monotonic() - t0
    exact = bytes(got["b"]) == up and bytes(got["a"]) == down
    print(f"{'done' if ok else 'timeout'} after {dt:.1f} s: a->b {len(got['b'])}/{len(up)} B "
          f"({60 * len(got['b']) / dt:.0f} B/min), b->a {len(got['a'])}/{len(down)} B, exact {exact}")
    sn = [float(x.split()[1]) for k in "ab" for x in lines[k] if x.startswith("SN ")]
    if sn:
        print(f"SN reports: {len(sn)}, median {statistics.median(sn):.1f} dB")
    lines["a"].clear()
    cmd["a"].sendall(b"DISCONNECT\r")
    deadline = time.monotonic() + 60
    pump(heard("a", "DISCONNECTED"))
    sys.exit(0 if ok and exact else 1)


if __name__ == "__main__":
    main()

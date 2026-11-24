"""runs/onair/session_pair.py CALLER_PORT CALLER_CALL CALLEE_PORT CALLEE_CALL [BYTES] [BW] [LIMIT_S]: an ARQ
session between two data2g-hosts (VARA-style command port P, data port P+1): the callee listens, the
caller connects and sends BYTES random bytes (default 30000) on its data port; prints every command line
both sides emit, and the time from CONNECTED to the callee holding every byte, then disconnects."""
import os
import select
import socket
import sys
import time

cp, cc, ep, ec = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 30000
bw = sys.argv[6] if len(sys.argv) > 6 else "BW1200"
limit = float(sys.argv[7]) if len(sys.argv) > 7 else 300.0
payload = os.urandom(n)
cmd = {"caller": socket.create_connection(("127.0.0.1", cp)), "callee": socket.create_connection(("127.0.0.1", ep))}
data = {"caller": socket.create_connection(("127.0.0.1", cp + 1)), "callee": socket.create_connection(("127.0.0.1", ep + 1))}
for k in cmd:
    cmd[k].sendall(f"{bw}\r".encode())
cmd["callee"].sendall(b"LISTEN ON\r")
time.sleep(0.5)
cmd["caller"].sendall(f"CONNECT {cc} {ec}\r".encode())
t0 = time.time()
bufs, got, t_conn, t_done, sent = {k: b"" for k in cmd}, b"", None, None, False
while time.time() - t0 < limit:
    r, _, _ = select.select(list(cmd.values()) + [data["callee"]], [], [], 0.5)
    for k, s in cmd.items():
        if s in r:
            d = s.recv(4096)
            bufs[k] += d
            while b"\r" in bufs[k]:
                line, bufs[k] = bufs[k].split(b"\r", 1)
                line = line.decode(errors="replace")
                print(f"{time.time() - t0:7.1f} {k}: {line}", flush=True)
                if k == "caller" and line.startswith("CONNECTED") and not sent:
                    t_conn, sent = time.time(), True
                    data["caller"].sendall(payload)
    if data["callee"] in r:
        got += data["callee"].recv(65536)
        if len(got) >= n and t_done is None:
            t_done = time.time()
            ok = got[:n] == payload
            print(f"DONE {n} bytes in {t_done - t_conn:.1f} s after CONNECTED = {8 * n / (t_done - t_conn):.0f} bps, "
                  f"intact {ok}", flush=True)
            cmd["caller"].sendall(b"DISCONNECT\r")
    if t_done is not None and time.time() - t_done > 15:
        break
if t_done is None:
    print(f"NOT DONE: callee got {len(got)} of {n} bytes in {limit:.0f} s", flush=True)
    cmd["caller"].sendall(b"ABORT\r")
    time.sleep(2)

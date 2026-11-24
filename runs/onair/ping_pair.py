"""runs/onair/ping_pair.py FROM_PORT TO_PORT TO_CALL [NARROW|WIDE] [SECONDS]: send PING from the host on
FROM_PORT and print every line both hosts' command ports emit for SECONDS (default 20)."""
import select
import socket
import sys
import time

fp, tp, call = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
bw = sys.argv[4] if len(sys.argv) > 4 else "NARROW"
secs = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
socks = {name: socket.create_connection(("127.0.0.1", p)) for name, p in (("from", fp), ("to", tp))}
socks["from"].sendall(f"PING {call} {bw}\r".encode())
end, bufs = time.time() + secs, {k: b"" for k in socks}
while time.time() < end:
    r, _, _ = select.select(list(socks.values()), [], [], 0.5)
    for name, s in socks.items():
        if s in r:
            d = s.recv(4096)
            if not d:
                continue
            bufs[name] += d
            while b"\r" in bufs[name]:
                line, bufs[name] = bufs[name].split(b"\r", 1)
                print(f"{time.strftime('%H:%M:%S')} {name}: {line.decode(errors='replace')}", flush=True)

"""A VARA-style TNC's two TCP ports, forwarded and logged, to watch an app
talk to a real TNC: listen on <port> and <port>+1, forward to <tnc_port>
and <tnc_port>+1. The log has "<epoch> <host|tnc> <line>" for every command
line and "<epoch> <host|tnc>-data <bytes>" for every chunk on the data port.

    python tnc_proxy.py <port> <tnc_port> <log>
"""

import asyncio
import sys
import time


async def pipe(r, w, log, who, lines):
    buf = b""
    try:
        while d := await r.read(4096):
            w.write(d)
            await w.drain()
            if not lines:
                log.write(f"{time.time():.3f} {who}-data {len(d)}\n")
            else:
                buf += d
                *done, buf = buf.replace(b"\n", b"\r").split(b"\r")
                for line in filter(None, done):
                    log.write(f"{time.time():.3f} {who} {line.decode('ascii', 'replace')}\n")
            log.flush()
    finally:
        w.close()


def serve(port, tnc_port, log, lines):
    async def handle(r, w):
        tr, tw = await asyncio.open_connection("127.0.0.1", tnc_port)
        await asyncio.gather(pipe(r, tw, log, "host", lines), pipe(tr, w, log, "tnc", lines), return_exceptions=True)
    return asyncio.start_server(handle, "127.0.0.1", port)


async def main():
    port, tnc, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    log = open(path, "a")
    cmd = await serve(port, tnc, log, True)
    data = await serve(port + 1, tnc + 1, log, False)
    async with cmd, data:
        await asyncio.gather(cmd.serve_forever(), data.serve_forever())


if __name__ == "__main__":
    asyncio.run(main())

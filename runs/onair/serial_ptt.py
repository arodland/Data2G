"""Key a radio through serial control lines: python serial_ptt.py PORT KEY [MAX_S], KEY one of
  rts      RTS high keys (DigiRig), DTR held low
  dtr-rts  DTR high and RTS low keys (AIOC; RTS high inhibits)
for at most MAX_S seconds (default 45) or until killed. Unkeys on SIGTERM/SIGINT, at MAX_S, and (the
port closing, which drops both lines) if the process dies."""
import fcntl
import os
import signal
import struct
import sys
import termios
import time

port, key = sys.argv[1], sys.argv[2]
max_s = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
DTR, RTS = termios.TIOCM_DTR, termios.TIOCM_RTS
ON, OFF = {"rts": ((RTS, DTR), (0, RTS | DTR)), "dtr-rts": ((DTR, RTS), (RTS, DTR))}[key]
fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)


def bits(on, off):
    if on:
        fcntl.ioctl(fd, termios.TIOCMBIS, struct.pack("I", on))
    if off:
        fcntl.ioctl(fd, termios.TIOCMBIC, struct.pack("I", off))


bits(*OFF)  # opening the port may have raised both lines
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
try:
    bits(*ON)
    print("keyed", flush=True)
    time.sleep(max_s)
finally:
    bits(*OFF)
    print("unkeyed", flush=True)

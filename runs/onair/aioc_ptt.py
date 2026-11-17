"""Key the AIOC's PTT on its serial port (DTR high and RTS low keys it; RTS high inhibits) for at most MAX_S seconds,
or until killed: python aioc_ptt.py [/dev/ttyACM0] [MAX_S]. Unkeys by dropping DTR and raising RTS;
if this process dies, closing the port drops DTR, which unkeys too."""
import fcntl
import os
import signal
import struct
import sys
import termios
import time

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
max_s = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
DTR, RTS = termios.TIOCM_DTR, termios.TIOCM_RTS


def bits(on, off):
    fcntl.ioctl(fd, termios.TIOCMBIS, struct.pack("I", on))
    fcntl.ioctl(fd, termios.TIOCMBIC, struct.pack("I", off))


def unkey(*_):
    sys.exit(0)  # the finally below unkeys; closing the port at exit drops DTR as well


signal.signal(signal.SIGTERM, unkey)
signal.signal(signal.SIGINT, unkey)
bits(DTR, RTS)  # keyed
print("keyed", flush=True)
try:
    time.sleep(max_s)
finally:
    bits(RTS, DTR)
    print("unkeyed", flush=True)

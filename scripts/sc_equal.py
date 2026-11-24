"""Flag off against master: the same bursts modulated, sent through a channel and received by this tree,
dumped for comparison with another tree's (run in each from its own directory, then --compare)."""
from data2g import threads

threads.limit(1)

import sys  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, hfchannel, modem  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.config import SUBMODES  # noqa: E402

MODES = [("ack-1f", 1), ("ack-4f", 2), ("qpsk-r1/2", 3), ("16qam-r1/2", 2), ("n10-qpsk-r1/2", 3), ("n10-16qam-r3/4", 2), ("n10-256l-r3/4", 2),
         ("n4-ack-8f", 1), ("n4-qpsk-r1/2", 2), ("w48-qpsk-r1/2", 4), ("w48-16qam-r1/2", 6), ("w48-16qam-r3/4", 6), ("w48-64l-r3/4", 8),
         ("w48-256l-r5/8", 8)]
if len(sys.argv) > 2 and sys.argv[1] == "--compare":
    a, b = (np.load(p) for p in sys.argv[2:4])
    bad = [k for k in a.files if k not in b.files or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])]
    print(f"{len(a.files)} arrays, {len(bad)} differ", bad[:10])
    sys.exit(bool(bad))
out = {}
for name, n_cw in MODES:
    spec = SUBMODES[name]
    for i, (chan, snr) in enumerate((("", 25), ("mpp", 12), ("mpd", 6))):
        rng = np.random.default_rng([i, n_cw])
        pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
        x = modem.modulate(pl, name)
        y = hfchannel.apply_channel(np.concatenate([np.zeros(2000), x, np.zeros(2000)]), snr_db=snr, fading_preset=chan or None, seed=i)
        key = f"{name}|{chan}"
        out[key + "|tx"] = x
        try:
            r = modem.receive(y)
        except modem.SyncError:
            out[key + "|rx"] = np.zeros(0)
            continue
        out[key + "|soft"] = modem.burst_soft(r)
        rx = PHY.ModemRx(r, {})
        out[key + "|dec"] = np.frombuffer(b"".join((rx.decode(s, (0, 0, 0), 0, None) or b"\x00") for s in range(n_cw)), dtype=np.uint8)
        out[key + "|h"] = r["est"]["h"]
np.savez(sys.argv[1], **out)
print(len(out), "arrays")

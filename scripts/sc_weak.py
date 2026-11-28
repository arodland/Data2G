"""What the caller's weak_path rule (arq.policy: snr_est under ROBUST_START_SNR_DB, spread_est at least
ROBUST_START_SPREAD_HZ, or unmeasured) would read from a connect-mode burst, flag off against a layout (patched in
one process), bursts through phy_session's channel at PEP 5. weak_path only runs on a session made by a robust
retry, whose CONNECT_ACK is CPM on HF; this shows what an OFDM one would have read.

    DATA2G_PEP_REF_DB=5 python -m scripts.sc_weak --layout mix
"""
from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, modem  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.arq import policy as POL  # noqa: E402
from data2g.config import FS, SUBMODES  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phy_session as G  # noqa: E402

BANDS = {"mix": ("w", "w48", "n10", "n4"), "k3d2": ("w", "w48"), "gold": ("w", "w48", "n10", "n4")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="mix")
    ap.add_argument("--cells", default="awgn:-8,mpp:-4,hilat-m:0")
    ap.add_argument("--modes", default="qpsk-r1/5,n10-qpsk-r1/3")
    ap.add_argument("--bursts", type=int, default=100)
    a = ap.parse_args()
    if G.PEP_REF_DB is None:
        ap.error("set DATA2G_PEP_REF_DB=5")
    print("mode cell arm: heard | weak | snr_est median [p10, p90] | spread_est median, unmeasured share | both arms agree")
    for name in a.modes.split(","):
        spec = SUBMODES[name]
        for cell in a.cells.split(","):
            chan, snr = cell.split(":")
            res = {}
            for arm in ("off", a.layout):
                modem.SC_BANDS, modem.SC_LAYOUT = ((), "") if arm == "off" else (BANDS[arm], arm)
                out = []
                for seed in range(a.bursts):
                    rng = np.random.default_rng([seed, 9])
                    x = modem.modulate([rng.bytes(codes.payload_bytes(spec))], name)
                    y = G.ContinuousChannel(chan, float(snr), seed, 30.0).apply(x, 5.0)
                    try:
                        r = G.receive_any(y, lead=int(G.PAD_S * FS) + FS // 2)
                    except modem.SyncError:
                        r = None
                    if r is None or r.get("family") == "cpm" or r["spec"].name != name:
                        out.append(None)
                        continue
                    m = PHY.measure(r)
                    s = m["spread_est"]
                    heard = None if s == POL.UNMEASURED_SPREAD_HZ else s
                    weak = m["snr_est"] < POL.ROBUST_START_SNR_DB or (heard is not None and heard >= POL.ROBUST_START_SPREAD_HZ)
                    out.append((weak, m["snr_est"], s, heard is None))
                res[arm] = out
            for arm, out in res.items():
                h = [o for o in out if o is not None]
                sn = np.array([o[1] for o in h])
                sp = np.array([o[2] for o in h])
                print(f"{name} {cell} {arm:5s}: {len(h)}/{len(out)} | weak {np.mean([o[0] for o in h]):.2f} | snr_est "
                      f"{np.median(sn):+.1f} [{np.percentile(sn, 10):+.1f}, {np.percentile(sn, 90):+.1f}] | spread {np.median(sp):.2f}, "
                      f"unmeasured {np.mean([o[3] for o in h]):.2f}")
            both = [(x[0], y[0]) for x, y in zip(*res.values()) if x is not None and y is not None]
            print(f"   both heard {len(both)}: same weak call {np.mean([x == y for x, y in both]):.2f}, "
                  f"weak off-only {sum(x and not y for x, y in both)}, {a.layout}-only {sum(y and not x for x, y in both)}")


if __name__ == "__main__":
    main()

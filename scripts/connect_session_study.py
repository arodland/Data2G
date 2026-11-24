"""Live two-station sessions from the CONNECT on (scripts/linksim.run over the
real modem, phy_session.RealPhy): connect tries, robust start and the first
data bursts' modes, for the interactions of the CPM leak metric and
fsk32r62-r2/3 with master's connect retries (#316) and robust start (#313).
2026-11-24.

Each session: W1AW calls K2XYZ at cap 2 (2400 Hz), 64 bytes each way, written
at the start, HORIZON s at most, `shift+cpm`. SNR as loss_study's (set
DATA2G_PEP_REF_DB=5). The branch's switches come from the environment
(DATA2G_CPM_LEAK, DATA2G_CPM_R23, DATA2G_OUTCOME_MODEL), and two study ones:
  DATA2G_STUDY_NOFAST=1     no connect retry in ROBUST_FAST: every retry in
                            ROBUST_CONNECT (#316's ROBUST_CONNECT_TRIES 5, no
                            last-try flag)
  DATA2G_STUDY_ROBUST=mode  GearShifter.robust_mode() always `mode` (in a
                            session: polls, control-only bursts at the floor)

Cells: <preset or doppler/delay_ms>:<snr>[:c] (c: a steady carrier at
1573.4 Hz, 12 dB over the floor in 50 Hz, at both stations: #295's).

    DATA2G_PEP_REF_DB=5 python -m scripts.connect_session_study --cells hilat-d:0,30/12:0 --out runs/cs_x.csv
"""

from data2g import threads  # noqa: I001

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

from data2g import interference as INTF  # noqa: E402
from data2g.arq import policy as G  # noqa: E402
from data2g.arq import link  # noqa: E402
from data2g.arq import session as S  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import phy_session as PS  # noqa: E402

HORIZON = 180.0
BYTES = 64
CARRIER = INTF.Spec(qrm=(INTF.Qrm("carrier", 1573.4, 50, 12.0, 600.0, 1.0),))
FIELDS = ["cell", "seed", "connected", "t_conn", "tries", "conn_mode", "robust_start", "complete", "t_done",
          "delivered", "first_data", "first_ctl", "modes", "reason"]


def patch():
    """The study toggles, in this process (pool workers run it at start)."""
    if os.environ.get("DATA2G_STUDY_NOFAST") == "1":
        G.GearShifter.connect_mode = lambda self, cap, tries=0, last=False: G.CONNECT[cap] if tries == 0 else G.ROBUST_CONNECT
    if os.environ.get("DATA2G_STUDY_ROBUST"):
        mode = os.environ["DATA2G_STUDY_ROBUST"]
        G.GearShifter.robust_mode = lambda self: mode
    conn, start = S.Session._connected, S.Session._robust_start

    def _robust_start(self, mode, _f=start):
        self._study_mode = mode
        return _f(self, mode)

    def _connected(self, now, _f=conn):
        self._study_conn = (now, self._tries, self.station.esc_floor)
        return _f(self, now)
    S.Session._robust_start, S.Session._connected = _robust_start, _connected


class LogPhy(PS.RealPhy):
    def __init__(self, ch):
        super().__init__(ch)
        self.log = []  # (t0, submode, data?, heard?)

    def send(self, burst, t0):
        out = super().send(burst, t0)
        data = any(s.mask_id[2] < 128 for s in burst.slots) and getattr(burst, "nod", None) is None
        self.log.append((t0, burst.submode, data, out[1] is not None))
        return out


def channel(cell: str, seed: int):
    name, snr, *c = cell.split(":")
    dop = dly = None
    if "/" in name:
        dop, dly = (float(v) for v in name.split("/"))
    intf = (CARRIER, CARRIER) if c else None
    return PS.ContinuousChannel(name, float(snr), seed, HORIZON, dop, dly, interference=intf)


def one(args):
    cell, seed = args
    phy = LogPhy(channel(cell, seed))
    steps = [("a", BYTES, None, 0.0), ("b", BYTES, None, 0.0)]
    a_ref = {}
    orig = S.Session.connect

    def connect(self, *a, _f=orig, **k):  # the caller, to read its state after the run
        a_ref["a"] = self
        return _f(self, *a, **k)
    S.Session.connect = connect
    try:
        res = L.run(L.make_policy("shift+cpm"), L.make_policy("shift+cpm"), None, steps, seed=seed, horizon=HORIZON,
                    phy=phy, cap=2)
    finally:
        S.Session.connect = orig
    a = a_ref["a"]
    conn = getattr(a, "_study_conn", None)
    after = [x for x in phy.log if conn is not None and x[0] >= conn[0]]
    modes = {}
    for _, m, _, _ in phy.log:
        modes[m] = modes.get(m, 0) + 1
    return dict(cell=cell, seed=seed, connected=int(conn is not None),
                t_conn=round(conn[0], 1) if conn else "", tries=conn[1] if conn else a._tries,
                conn_mode=getattr(a, "_study_mode", ""), robust_start=int(bool(conn) and conn[2] >= link.MAX_ESCALATION),
                complete=int(res["complete"]), t_done=round(res["t"] if res["complete"] else HORIZON, 1),
                delivered=res["delivered"], first_data="|".join([m for _, m, d, _ in after if d][:5]),
                first_ctl="|".join([m for _, m, d, _ in after if not d][:5]),
                modes=";".join(f"{k}:{v}" for k, v in sorted(modes.items())), reason="|".join(map(str, res["reason"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if PS.PEP_REF_DB is None:
        ap.error("set DATA2G_PEP_REF_DB (5)")
    patch()
    jobs = [(c, s) for c in a.cells.split(",") for s in range(a.seed0, a.seed0 + a.seeds)]
    with Pool(a.jobs, initializer=patch) as pool, open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for row in pool.imap_unordered(one, jobs):
            w.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()

"""scripts/loss_study.py, plus per job (channel, snr, seed) how its sessions started: written to <out>.starts.csv,
one row a job: connects made by a robust-mode retry (robust_connects), of them the caller's starts at the robust
floor (floor_starts: session._robust_start with policy.weak_path() true). Same arguments as loss_study.

    DATA2G_PEP_REF_DB=5 python -m scripts.sc_loss --out runs/x.csv --cells mpp:-8 ...
"""
from data2g import threads

threads.limit(1)

import csv  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

from data2g.arq import session as S  # noqa: E402
from scripts import loss_study as LS  # noqa: E402

_COUNT = {"robust_connects": 0, "floor_starts": 0}
_orig = S.Session._robust_start


def _robust_start(self, mode):
    if mode != self.policy.connect_mode(self.cap, 0) and self.policy.robust(mode) and self.station.master:
        _COUNT["robust_connects"] += 1
        _COUNT["floor_starts"] += bool(self.policy.weak_path())
    return _orig(self, mode)


S.Session._robust_start = _robust_start
_one = LS.one


def one(args):
    _COUNT.update(robust_connects=0, floor_starts=0)
    rows = _one(args)
    with open(os.environ["SC_STARTS"], "a", newline="") as f:
        csv.writer(f).writerow([args[0], args[1], args[2], _COUNT["robust_connects"], _COUNT["floor_starts"]])
    return rows


LS.one = one

if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1]
    os.environ["SC_STARTS"] = out[:-4] + ".starts.csv"
    with open(os.environ["SC_STARTS"], "w") as f:
        f.write("channel,snr,seed,robust_connects,floor_starts\n")
    LS.main()

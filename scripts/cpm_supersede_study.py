"""Should a pending CPM burst be supersedable? Receiver._supersede returns at
once for one, so a false CPM lock holds the receiver for its whole claimed
span. Two candidate rules, against the shipped receiver on the same audio:

  cc M   CPM -> CPM: a later CPM lock (starting past the pending one's
         start) whose score beats the pending one's by M replaces it
  co F   CPM -> OFDM: under a pending CPM lock scoring below F, a clear
         (non-suspect) OFDM header replaces it

Streams of random bursts, OFDM and CPM (every grid), back to back with
0.3-2 s gaps, each through its own channel (AWGN, MPG or MPP, SNR uniform
-6..12 dB, CFO +-40 Hz) into one continuous noise floor. A share of them
lose their front (zeroed: a fade, or a burst begun while we transmitted),
which is what makes false locks: a receiver meets their data. Per burst
and variant: was it heard (a header at its start, its submode), and did
its control codeword decode. Per header: true or false, family, lock kind
(front, late, mid; OFDM header or copy), score, span.

    uv run python scripts/cpm_supersede_study.py --jobs 24 --workers 12 --minutes 20 --out runs/cpm_supersede
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402

from data2g import cpm, hfchannel, modem, receiver  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.arq.engine import receive_accept  # noqa: E402
from data2g.arq.modes import MODES, is_cpm  # noqa: E402
from data2g.config import FS  # noqa: E402
from data2g.receiver import _shift  # noqa: E402

BLK = FS // 50
OFDM = [n for n in MODES if not is_cpm(MODES[n]) and "ack" not in n]
CPM = [n for n in MODES if is_cpm(MODES[n]) and not n.endswith("-ctl")]
CUT_SHARE = 0.4  # bursts whose front is lost
MATCH = (-0.15, 0.35)  # a header's start against its burst's audio start, s


def _tag(fn, how):
    def f(*a, **k):
        r = fn(*a, **k)
        if isinstance(r, dict) and not r.get("wait"):
            r = dict(r, how=r.get("how", how))
        return r
    return f


cpm.find = _tag(cpm.find, "front")
cpm.late_lock = _tag(cpm.late_lock, "late")
cpm.find_mid = _tag(cpm.find_mid, "mid")


class Variant(receiver.Receiver):
    """The shipped receiver, plus a rule for a pending CPM burst."""

    def __init__(self, accept, cc=None, co=None):
        super().__init__(accept, cpm_grids=tuple(cpm.GRIDS))
        self.cc, self.co = cc, co

    def _supersede(self, out, whole=False):
        p = self.pending
        if p.get("family") != "cpm" or (self.cc is None and self.co is None):
            return super()._supersede(out, whole)
        self.fresh = 0
        q = None
        if self.cc is not None:
            c = self._find_cpm()
            if c is not None:
                c = _shift(dict(c, n_cw=1 + c["dup"] + c["n_data"]), self.off)
                # past the pending burst's own front (its mid block reads as itself)
                if c["start"] > p["start"] + FS // 10 and c["score"] >= p["score"] + self.cc:
                    q = c
        if q is None and self.co is not None and p["score"] < self.co:
            w0 = max(0, len(self.buf) - self.keep)
            try:
                o = modem.find_burst(self.buf[w0:], self.bands, self.accept, stats=self._stats(w0))
                o = _shift(o, w0 + self.off)
                if o["score"] >= self.SUSPECT_SCORE and o["start"] > p["start"]:
                    q = o
            except modem.SyncError:
                pass
        self._searched()
        if q is None:
            return False
        out.append(("burst", {"header": p, "rx": None,
                              "audio": self.buf[max(0, p["start"] - self.off):max(0, q["start"] - self.off)]}))
        self.pending = q
        self.pilots_ok = q.get("family") == "cpm" or "copy" in q or q["score"] >= self.SUSPECT_SCORE
        self.confirmed = False
        out.append(("header", dict(q, stream_end=self.off + len(self.buf))))
        return True


VARIANTS = {"base": {}, "cc0": dict(cc=0.0), "cc05": dict(cc=0.05), "cc10": dict(cc=0.10),
            "co20": dict(co=0.20), "co30": dict(co=0.30), "co99": dict(co=99.0)}


def stream(seed: int, minutes: float):
    """-> (audio, bursts): bursts as dicts with their TxBurst and placement."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FS)
    y = hfchannel.awgn(np.zeros(n), 0.0, seed=seed, s_power=1.0)  # 0 dB: a unit-RMS burst
    at, bursts = FS, []
    while True:
        cp = rng.random() < 0.5
        name = (CPM if cp else OFDM)[rng.integers(len(CPM if cp else OFDM))]
        k = O.n_for(name, float(rng.uniform(3, 20)))
        b = O.burst(name, max(1, k), rng)
        x = PHY.tx_audio(b)
        if at + len(x) + FS > n:
            break
        chan = ["awgn", "mpg", "mpp"][rng.integers(3)]
        snr = float(rng.uniform(-6, 12))
        x = hfchannel.freq_shift(x, float(rng.uniform(-40, 40)))
        if chan != "awgn":
            x = hfchannel.fading(x, chan, seed=int(rng.integers(1 << 30)))
        cut = rng.random() < CUT_SHARE
        cut_s = float(rng.uniform(0.3, min(4.0, 0.6 * len(x) / FS))) if cut else 0.0
        x[:int(cut_s * FS)] = 0.0
        y[at:at + len(x)] += x * 10 ** (snr / 20)
        bursts.append(dict(id=len(bursts), name=name, n_cw=len(b.slots), at=at, end=at + len(x), snr=round(snr, 2),
                           chan=chan, cut_s=round(cut_s, 2), family="cpm" if cp else "ofdm", tx=b))
        at += len(x) + int(rng.uniform(0.3, 2.0) * FS)
    return y, bursts


def match(h, bursts):
    for b in bursts:
        d = (h["start"] - b["at"]) / FS
        if MATCH[0] <= d <= MATCH[1] and h["spec"].name == b["name"]:
            return b
    return None


def ctl_ok(r, b) -> bool:
    try:
        s = b["tx"].slots[0]
        return PHY.ModemRx(r, {}, 0.0).decode(0, s.mask_id, 0, None) == s.payload
    except Exception:  # noqa: BLE001 (a malformed false lock's result)
        return False


def job(args):
    seed, minutes = args
    y, bursts = stream(seed, minutes)
    heads, outcome = [], []
    for v, kw in VARIANTS.items():
        rx = Variant(receive_accept(), **kw)
        heard, ok = set(), set()
        for i in range(0, len(y), BLK):
            for kind, ev in rx.feed(y[i:i + BLK]):
                h = ev if kind == "header" else ev["header"]
                b = match(h, bursts)
                if kind == "header":
                    heads.append(dict(seed=seed, variant=v, t=round(h["start"] / FS, 2), name=h["spec"].name,
                                      family=h.get("family", "ofdm"),
                                      how=h.get("how", "copy" if "copy" in h else "header"),
                                      score=round(float(h["score"]), 4), span_s=round((h["end"] - h["start"]) / FS, 2),
                                      true=b is not None, burst=-1 if b is None else b["id"]))
                    if b is not None:
                        heard.add(b["id"])
                elif b is not None and ev["rx"] is not None and ctl_ok(ev["rx"], b):
                    ok.add(b["id"])
        for b in bursts:
            outcome.append(dict(seed=seed, variant=v, burst=b["id"], name=b["name"], family=b["family"],
                                snr=b["snr"], chan=b["chan"], cut_s=b["cut_s"], dur_s=round((b["end"] - b["at"]) / FS, 2),
                                heard=b["id"] in heard, ctl_ok=b["id"] in ok))
    return heads, outcome


def write(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--out", type=Path, default=Path("runs/cpm_supersede"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as p:
        res = p.map(job, [(a.seed + k, a.minutes) for k in range(a.jobs)])
    write(a.out / "headers.csv", [h for r in res for h in r[0]])
    write(a.out / "bursts.csv", [o for r in res for o in r[1]])
    print(f"wrote {a.out}/headers.csv, bursts.csv")

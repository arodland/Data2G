"""What CPM control-only bursts say at the floor, relative to what their
receiver already knows: the census behind the reply-signature codebook.

Real-modem sessions (phy_session, as loss_study), shift+cpm, PEP-referenced.
Every control-only burst a station builds in a CPM mode is recorded with
both stations' state at that moment:
  kind       reply (ARQ, fresh), poll (PROBE), dup / repeat
  cum        the core's cum less the peer's tx base (what the peer knows is
             acked), and the peer's next less its base (outstanding)
  bitmap     T_BITMAP's set bits (out-of-order codewords held)
  rec        recommend's ladder step from the last control I sent (band
             change: 'band'), size_hint changed, reply_lost
  T_REPLY / T_DUPCTL  absent, same as my last control's, or changed
  others     the other TLVs present
    python -m scripts.sig_census --out runs/sigc/census.csv [--seeds 8]
    python -m scripts.sig_census report runs/sigc/census.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

from data2g.arq import frames as F  # noqa: E402
from data2g.arq import link as LK  # noqa: E402
from data2g.arq.modes import MODES, is_cpm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import phy_session as G  # noqa: E402

CELLS = [("awgn", s) for s in (-18, -16, -14, -12)] + [("mpp", s) for s in (-16, -14, -12, -10)] + \
        [("mpd", s) for s in (-15, -13, -11, -9)] + [("mpg", s) for s in (-14, -12, -10, -8)]
NAMES = {v: k for k, v in vars(F).items() if k.startswith("T_") and isinstance(v, int)}
STATIONS = []
ROWS = []
TAG = {}

_init, _build = LK.Station.__init__, LK.Station.build


def _init_hook(self, *a, **k):
    _init(self, *a, **k)
    STATIONS.append(self)
    self._census_last = None  # (rec, size_hint, ext) of my last control


def _band_index(rec: int) -> tuple[int, int]:
    return rec >> 4, rec & 15


def _build_hook(self, fresh: bool = True):
    burst = _build(self, fresh)
    slots = burst.slots
    ctl = [s for s in slots if s.mask_id[2] >= 128]
    if not ctl:
        return burst
    c = F.Control.unpack([ctl[0].payload] + ([s.payload for s in ctl[1:] if s.rv == 0]))
    core, ext = c.core, c.ext
    last = self._census_last
    self._census_last = (core.recommend, core.size_hint, dict(ext))
    if len(ctl) != len(slots) or not is_cpm(MODES[burst.submode]):
        return burst
    peer = next((s for s in STATIONS if s is not self), None)
    base, nxt = (peer.tx.base, peer.tx.next) if peer is not None else (0, 0)
    kind = {F.ARQ: "reply" if fresh else "stale", F.PROBE: "poll", F.ARQ_DUP: "dup"}.get(core.ftype, str(core.ftype))
    if getattr(self, "_repeated", False) and burst is self.last_sent:
        kind = "repeat"
    cum_rel = (core.cum - base) % F.SEQ_MOD
    if last is None:
        rec_step, hint_same, reply, dupctl = "first", "first", "first", "first"
    else:
        (b0, i0), (b1, i1) = _band_index(last[0]), _band_index(core.recommend)
        rec_step = "band" if b0 != b1 else str(i1 - i0)
        hint_same = str(int(last[1] == core.size_hint))
        reply = ("absent" if F.T_REPLY not in ext else
                 "same" if last[2].get(F.T_REPLY) == ext[F.T_REPLY] else "changed")
        dupctl = ("same" if (F.T_DUPCTL in ext) == (F.T_DUPCTL in last[2]) else "changed")
    bitmap = F.unpack_bitmap(ext[F.T_BITMAP], core.cum) if F.T_BITMAP in ext else set()
    others = sorted(NAMES.get(t, str(t)) for t in ext if t not in (F.T_REPLY, F.T_DUPCTL, F.T_BITMAP))
    ROWS.append(dict(TAG, station=self.direction, submode=burst.submode, kind=kind, cum_rel=cum_rel,
                     outstanding=(nxt - base), all_acked=int(core.cum % F.SEQ_MOD == nxt % F.SEQ_MOD),
                     bitmap=len(bitmap), reply_lost=int(core.reply_lost), rec_step=rec_step, hint_same=hint_same,
                     t_reply=reply, t_dupctl=dupctl, dupctl_on=int(F.T_DUPCTL in ext), others="+".join(others),
                     ctl_bytes=sum(2 + len(v) for v in ext.values()) + 4))
    return burst


LK.Station.__init__, LK.Station.build = _init_hook, _build_hook


def one(args):
    chan, snr, seed, workload = args
    STATIONS.clear()
    ROWS.clear()
    TAG.clear()
    TAG.update(channel=chan, snr=snr, seed=seed, workload=workload)
    ch = G.ContinuousChannel(chan, snr, seed, 600.0)
    L.run(L.make_policy("shift+cpm"), L.make_policy("shift+cpm"), None,
          L.WORKLOADS[workload](random.Random(seed + 7)), seed=seed, horizon=600.0, phy=G.RealPhy(ch), cap=2)
    return list(ROWS)


FIELDS = ["channel", "snr", "seed", "workload", "station", "submode", "kind", "cum_rel", "outstanding", "all_acked",
          "bitmap", "reply_lost", "rec_step", "hint_same", "t_reply", "t_dupctl", "dupctl_on", "others", "ctl_bytes"]


def report(path):
    rows = list(csv.DictReader(open(path)))
    print(f"{len(rows)} CPM control-only bursts")
    for wl in sorted({r["workload"] for r in rows}):
        rs = [r for r in rows if r["workload"] == wl]
        print(f"\n== {wl}: {len(rs)}")
        for col in ("kind", "all_acked", "cum_rel", "outstanding", "bitmap", "reply_lost", "rec_step", "hint_same",
                    "t_reply", "t_dupctl", "others"):
            c = Counter(r[col] for r in rs)
            print(f"  {col:11s} " + ", ".join(f"{k or '-'} {v / len(rs):.1%}" for k, v in c.most_common(8)))
        # the joint state a signature would have to say, most common first
        key = lambda r: (r["kind"], "all" if r["all_acked"] == "1" else f"cum+{r['cum_rel']}",  # noqa: E731
                         f"bm{r['bitmap']}" if r["bitmap"] != "0" else "", f"rl{r['reply_lost']}",
                         f"rec{r['rec_step']}", f"h{r['hint_same']}", f"reply-{r['t_reply']}",
                         f"dup-{r['t_dupctl']}", r["others"])
        c = Counter(key(r) for r in rs)
        cum = 0
        print("  joint states (cumulative share):")
        for i, (k, v) in enumerate(c.most_common(40)):
            cum += v
            print(f"   {i + 1:2d} {v / len(rs):6.1%} {cum / len(rs):6.1%}  " + " ".join(x for x in k if x))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "report":
        return report(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sigc/census.csv")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if G.PEP_REF_DB is None:
        ap.error("set DATA2G_PEP_REF_DB=5")
    jobs = [(c, float(s), seed, wl) for c, s in CELLS for seed in range(a.seeds) for wl in ("bulk", "chat")]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.jobs) as p, open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for rows in p.imap_unordered(one, jobs):
            w.writerows(rows)
            f.flush()
    report(a.out)


if __name__ == "__main__":
    main()

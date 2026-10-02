"""Replay a session recorded at both ends (data2g.arq.engine.Recorder:
data2g-host --record-dir, or Engine(record_dir=...)) with exact knowledge
of what was on air.

Every burst one station sent is matched to what the other heard (wall
clock + engine time; the sender's log holds each slot's payload, CRC
mask and RV). Then, offline:
- heard bursts: every slot re-decoded against its true identity: did the
  control decode, how many first-transmission data codewords did, and
  how many decoded in bursts whose control failed (what a more robust
  control codeword would have saved);
- missed bursts (no header heard live): the receiver's continuous audio
  at that time searched again, with the detector's peak against its
  threshold (sync near-misses) and whether an offline search finds it.

    uv run python scripts/replay.py recordings/W1AW recordings/K2XYZ --csv runs/replay.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data2g import codes, modem
from data2g.arq import phy as PHY
from data2g.config import BANDS, SUBMODES
from data2g.waveform import ofdm, sync

SLACK_S = 3.0  # a burst's end to the receiver's rx event, at most


def load(d: Path) -> dict:
    ev = [json.loads(line) for line in open(d / "events.jsonl")]
    start = next(e for e in ev if e["kind"] == "start")
    audio = np.fromfile(d / "audio_in.f16", dtype=np.float16) if (d / "audio_in.f16").exists() else None
    return dict(dir=d, call=start["call"], wall=start["wall"], fs=start.get("fs", 8000), audio=audio,
                tx=[e for e in ev if e["kind"] == "tx"], rx=[e for e in ev if e["kind"] == "rx"])


def slot_outcomes(spec, r, slots) -> list:
    """Per slot: True/False decoded against its true identity; None for a
    resend at RV > 0 (not decodable alone)."""
    soft = PHY.soft_bits(r)
    out = []
    for i, s in enumerate(slots):
        if s["rv"] or i >= len(soft):
            out.append(None)
            continue
        mask = PHY.mask_value(tuple(s["mask"]))
        payload, ok = codes.decode_many(spec, soft[i:i + 1], mask, index=i)[0]
        out.append(bool(ok and payload == bytes.fromhex(s["payload"])))
    return out


def replay(sender: dict, receiver: dict) -> list[dict]:
    rows = []
    used = set()
    for tx in sender["tx"]:
        spec = SUBMODES[tx["submode"]]
        end = sender["wall"] + tx["t"] + tx["seconds"]
        best = None
        for j, rx in enumerate(receiver["rx"]):
            d = receiver["wall"] + rx["t"] - end
            if j not in used and rx["submode"] == tx["submode"] and -0.5 <= d <= SLACK_S:
                if best is None or d < best[1]:
                    best = (j, d)
        row = dict(sender=sender["call"], t=round(tx["t"], 2), submode=spec.name, n_cw=len(tx["slots"]),
                   heard=best is not None)
        slots = tx["slots"]
        n_ctl = sum(1 for s in slots if s["mask"][2] >= 128)
        if best is not None:
            used.add(best[0])
            rx = receiver["rx"][best[0]]
            audio = np.load(receiver["dir"] / rx["file"])["audio"].astype(np.float64)
            try:
                r = modem.receive(audio, [spec.sync_band])
                oc = slot_outcomes(spec, r, slots)
            except modem.SyncError:
                oc = [False] * len(slots)
            data = [o for o, s in zip(oc, slots) if s["mask"][2] < 128 and o is not None]
            # control: each codeword alone, or with its RV 1 copy (ARQ_DUP)
            ctl_ok = all(o for o, s in zip(oc[:n_ctl], slots) if not s["rv"])
            if not ctl_ok and n_ctl >= 2 and slots[1]["rv"] == 1:
                try:
                    soft = PHY.soft_bits(r)
                    m = PHY.mask_value(tuple(slots[0]["mask"]))
                    buf = codes.combine(spec, None, codes.flip(spec, 0, 0) * soft[0:1], 0)
                    buf = codes.combine(spec, buf, codes.flip(spec, 1, 1) * soft[1:2], 1)
                    p, good = codes.decode_buffer(spec, buf, 1, m, index=codes.PLAIN)[0]
                    ctl_ok = good and p == bytes.fromhex(slots[0]["payload"])
                except (NameError, modem.SyncError):
                    pass
            row.update(ctl_ok=ctl_ok, data_first=len(data), data_ok=sum(data),
                       data_ok_ctl_lost=sum(data) if not ctl_ok else 0,
                       snr_est=round(rx["meas"]["snr_est"], 1) if rx.get("meas") else "")
        elif receiver["audio"] is not None:
            # where the burst was on the receiver's clock, +-1 s
            fs = receiver["fs"]
            a = int((sender["wall"] + tx["t"] - receiver["wall"] - 1.0) * fs)
            b = int((end - receiver["wall"] + 1.0) * fs)
            x = receiver["audio"][max(0, a):max(0, b)].astype(np.float64)
            # the recording holds exact zeros while that station transmitted
            # (and a digital loopback's silence): the detector's noise level
            # needs some noise
            x = x + np.random.default_rng(0).normal(0, 1e-5, len(x))
            row.update(ctl_ok=False)
            if len(x) > fs:
                band = ofdm.band(spec.sync_band)
                S, _ = sync.detection_stat(modem.to_baseband(x), band)
                row.update(peak=round(float(S.max()), 1), threshold=BANDS[spec.sync_band].preamble_threshold)
                try:
                    r = modem.receive(x, [spec.sync_band])
                    row.update(found_offline=r["spec"].name == spec.name)
                except modem.SyncError:
                    row.update(found_offline=False)
        rows.append(row)
    return rows


def summarize(rows):
    heard = [r for r in rows if r["heard"]]
    missed = [r for r in rows if not r["heard"]]
    print(f"{len(rows)} bursts sent, {len(heard)} heard, {len(missed)} missed")
    if heard:
        print(f"  heard: control lost in {sum(not r['ctl_ok'] for r in heard)}; first-transmission data codewords "
              f"{sum(r['data_ok'] for r in heard)}/{sum(r['data_first'] for r in heard)} decoded, "
              f"{sum(r['data_ok_ctl_lost'] for r in heard)} of them in bursts whose control failed")
    if missed:
        near = [r for r in missed if "peak" in r and r["peak"] >= 0.7 * r["threshold"]]
        print(f"  missed: {sum(bool(r.get('found_offline')) for r in missed)} found by an offline search; "
              f"{len(near)} with a detector peak within 30% of threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--csv", type=Path)
    x = ap.parse_args()
    A, B = load(x.a), load(x.b)
    rows = replay(A, B) + replay(B, A)
    rows.sort(key=lambda r: r["t"])
    for who in (A["call"], B["call"]):
        print(f"== sent by {who}")
        summarize([r for r in rows if r["sender"] == who])
    if x.csv:
        keys = sorted({k for r in rows for k in r})
        with open(x.csv, "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

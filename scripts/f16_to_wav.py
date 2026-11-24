"""scripts/f16_to_wav.py IN.f16 [OUT.wav] [--rate HZ]: a recording's audio_in.f16 (raw float16, what the
receiver was fed) -> a float32 WAV. The rate comes from the same directory's events.jsonl ("start" event's
fs), else --rate (FM 16000, HF 8000). OUT defaults to IN with .wav."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ap = argparse.ArgumentParser()
ap.add_argument("inp", type=Path)
ap.add_argument("out", type=Path, nargs="?")
ap.add_argument("--rate", type=int)
a = ap.parse_args()
fs = a.rate
ev = a.inp.parent / "events.jsonl"
if fs is None and ev.exists():
    with open(ev) as f:
        fs = next((e["fs"] for e in map(json.loads, f) if e.get("kind") == "start" and "fs" in e), None)
if fs is None:
    ap.error("no events.jsonl with the rate beside it: give --rate")
out = a.out or a.inp.with_suffix(".wav")
wavfile.write(out, fs, np.fromfile(a.inp, dtype=np.float16).astype(np.float32))
print(f"{out}: {fs} Hz")

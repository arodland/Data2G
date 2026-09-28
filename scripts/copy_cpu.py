"""Receiver CPU with and without the copy search (tnc.Receiver._find_copy),
the streaming receiver fed 20 ms blocks as the host does: on noise, and
on low-SNR traffic (w/w48/n10 bursts 0.3-1.5 s apart on MPP). CPU seconds
per audio minute, and the copy search's own share.

    uv run --no-sync python scripts/copy_cpu.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent))

from data2g import cpm, hfchannel, modem, tnc  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.config import FS  # noqa: E402

import outcome_data as O  # noqa: E402

MINUTES = 5


def traffic(seed: int, snr: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts, total = [np.zeros(2 * FS)], 2 * FS
    while total < MINUTES * 60 * FS:
        name = str(rng.choice(["ack-4f", "qpsk-r1/5", "qpsk-r1/3", "w48-qpsk-r1/3", "n10-ack-4f"]))
        x = PHY.tx_audio(O.burst(name, int(rng.integers(1, 5)), rng))
        gap = np.zeros(int(rng.uniform(0.3, 1.5) * FS))
        parts += [x, gap]
        total += len(x) + len(gap)
    y = np.concatenate(parts)[:MINUTES * 60 * FS]
    return hfchannel.apply_channel(y, snr_db=snr, freq_offset_hz=12.0, ppm=10, fading_preset="mpp", seed=seed,
                                   )


def run(y: np.ndarray, copy: bool) -> tuple[float, float, int, int]:
    """-> (CPU s per audio minute, of it the copy search's, bursts, copy locks)."""
    r = tnc.Receiver(modem.Accept.of(None, 16.0), cpm_grids=tuple(cpm.GRIDS))
    spent = [0.0]
    find = r._find_copy

    def timed():
        t = time.process_time()
        try:
            return find()
        finally:
            spent[0] += time.process_time() - t
    r._find_copy = timed if copy else (lambda: None)
    t0, bursts, copies = time.process_time(), 0, 0
    for i in range(0, len(y), FS // 50):
        for k, ev in r.feed(y[i:i + FS // 50]):
            bursts += k == "burst"
            copies += k == "header" and "copy" in ev
    mins = len(y) / FS / 60
    return (time.process_time() - t0) / mins, spent[0] / mins, bursts, copies


if __name__ == "__main__":
    noise = np.random.default_rng(1).normal(0, 0.1, MINUTES * 60 * FS)
    for tag, y in (("noise", noise), ("MPP -4 dB traffic", traffic(2, -4.0)), ("MPP 0 dB traffic", traffic(3, 0.0))):
        for copy in (False, True):
            cpu, own, n, c = run(y, copy)
            print(f"{tag}, copy search {'on ' if copy else 'off'}: {cpu:.2f} CPU s per audio min "
                  f"(copy search {own:.2f}), {n} bursts, {c} copy locks", flush=True)

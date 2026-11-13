"""Nods (docs/arq.md §3b): a CPM control-only reply or poll whose content is
one of a few likely states, sent as one of M keyed c16r25 tone sequences and
detected, not decoded, where its listener expects it: it says its meaning
without words.

The family (M sequences of L tones) is keyed by the session key, the
sender's direction, its burst number and the burst it acts on (both mod 8),
so a nod can only match the one turn it answers. Which of the M it is
carries the meaning (link.Station.nod_*: the codebook and its anchors).

scripts/sig_study.py round 2 (2026-11-12): L 64 (2.6 s), M 32, one CFO
point, the square law's 4 best rescored coherently: 10% points awgn -19.2,
mpg -11.4, mpp -16.2, mpd -16.3 dB (mean power in 2500 Hz) at 1e-6 false
accepts per slot; today's primed control-only reply (5.0 s) -14.7, -10.0,
-11.7, -11.8.
"""

from functools import lru_cache

import numpy as np

from .. import cpm

GRID = cpm.GRIDS["c16r25"]
L = 64  # tones
M = 32  # sequences in a family; meanings past N_MEANINGS are reserved (never sent: a detection of one is none)
N_MEANINGS = 26
TOPK = 4  # the square law's best, rescored coherently
# Acceptance (one slot, +-ReplySlot.TOL_S): the best meaning's coherent score
# at least THR and the runner-up under it. THR: the exponential tail over the
# 90% point of 100000 noise slots' best (runs/sig2/noise_*.npy, max 31.2),
# at 1e-6 false accepts per slot: 36.85.
THR = 37.0
POLL_TOL_S = 0.25  # a poll's window either side (its start is the master's clock and PTT, not a learned gap)
POLL_LAG_S = 0.1  # ... centred this long past session.NOD_POLL_S (the engine's PTT delay)


@lru_cache(maxsize=256)
def family(key: int, direction: int, bn8: int, acted8: int) -> np.ndarray:
    """(M, L) the keyed family's tone sequences."""
    return np.random.default_rng([0x5167, key & 0xFFFFFFFF, direction, bn8, acted8]).integers(0, GRID.m, (M, L))


def waveform(seq: np.ndarray, cap: int = 2) -> np.ndarray:
    """A nod's sequence -> audio as a CPM burst's (tones, glide, the cap's TX filter, ramps)."""
    return cpm.bandpass(GRID, cpm.tones(GRID, seq, cpm.tx_filter(GRID.name, cap).glide), cap)


def seconds() -> float:
    return L / GRID.rate + 2 * cpm.RAMP_S


def span() -> int:
    """Samples a nod takes."""
    return L * GRID.T


def detect(x: np.ndarray, lo: int, hi: int, cfo: float, fam: np.ndarray) -> tuple[int, int, float] | None:
    """A nod of `fam` starting in [lo, hi] (x's samples; x must hold
    hi + span()) at CFO cfo -> (meaning, start, score), or None: none over
    THR, a runner-up over it too, or a reserved meaning."""
    phases = range(0, GRID.T, GRID.T // 4)
    n_pos = max(0, (hi - lo) // GRID.T) + 1
    best = np.full(M, -np.inf)
    where = np.zeros(M, int)
    idx = np.arange(L)
    for p in phases:
        E = np.abs(cpm.bins(GRID, x, lo + p, n_pos + L, cfo)) ** 2
        T_ = cpm.tone_llr(GRID, E)
        for m in range(n_pos):
            s0 = lo + p + m * GRID.T
            if s0 > hi:
                continue
            sq = T_[(m + idx)[None, :], fam].sum(axis=1)
            up = sq > best
            best = np.where(up, sq, best)
            where[up] = s0
    coh = np.full(M, -np.inf)
    for k in np.argsort(best)[::-1][:TOPK]:
        Z = cpm.bins(GRID, x, int(where[k]), L, cfo)
        coh[k] = float(cpm._coh_run_llr(GRID, Z, fam[k], cpm._n0(np.abs(Z) ** 2)).sum())
    order = np.argsort(coh)[::-1]
    b, r = coh[order[0]], coh[order[1]]
    if b < THR or r >= THR or order[0] >= N_MEANINGS:
        return None
    return int(order[0]), int(where[order[0]]), float(b)


# --- the codebook (docs/arq.md §3b) ---------------------------------------------------------
# reply (a fresh control-only burst): ack all / unchanged x recommend step x T_DUPCTL;
# poll (PROBE, the master's after a timeout): its previous burst's state x step x T_DUPCTL.
REPLY_STEPS = (0, 1, -1, 2, -2)
POLL_STEPS = (0, 2, -2)
CODEBOOK = ([("reply", ack, step, dup) for ack in ("all", "unchanged") for step in REPLY_STEPS for dup in (0, 1)]
            + [("poll", "same", step, dup) for step in POLL_STEPS for dup in (0, 1)])
assert len(CODEBOOK) == N_MEANINGS
INDEX = {m: i for i, m in enumerate(CODEBOOK)}


class NodRx:
    """A detected nod as a received control-only burst (burst.RxBurst):
    its control is the one the listener synthesized from the meaning
    (link.Station.nod_payload), or none. No primed or repeat decode."""

    nod = True

    def __init__(self, submode: str, mask_id: tuple, payload: bytes | None):
        self.submode, self.n_cw = submode, 1
        self._mask, self._payload = mask_id, payload

    def decode(self, slot: int, mask_id: tuple, rv: int, key) -> bytes | None:
        return self._payload if slot == 0 and mask_id == self._mask else None

    def converged(self, slot: int, pair: bool = False) -> bool:
        return False

    def raw(self, slot: int) -> list:
        return []

    def forget(self, key) -> None:
        pass


if __name__ == "__main__":  # a self-check: a clean nod detects, noise does not
    rng = np.random.default_rng(1)
    fam = family(1234, 1, 3, 5)
    x = np.concatenate([np.zeros(2000), waveform(fam[7]), np.zeros(2000)])
    x = x + rng.normal(0, 0.05, len(x))
    got = detect(x, 1800, 2200, 0.0, fam)
    assert got is not None and got[0] == 7, got
    assert detect(rng.normal(size=len(x)), 1800, 2200, 0.0, fam) is None
    print("ok", got)
